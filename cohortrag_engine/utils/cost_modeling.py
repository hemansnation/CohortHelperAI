"""
Cost modeling and token usage tracking for CohortRAG Engine
=========================================================

This module provides comprehensive cost tracking, token usage monitoring,
and cost optimization recommendations for production RAG deployments.
"""

import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from pathlib import Path
import re

# Local imports
try:
    from ..config import get_config
except ImportError:
    from config import get_config

@dataclass
class TokenUsage:
    """Track token usage for a single operation"""
    operation_id: str
    timestamp: float
    operation_type: str  # 'query', 'embedding', 'rerank'
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str
    cost: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class CostSummary:
    """Summary of costs over a time period"""
    period_start: float
    period_end: float
    total_operations: int
    total_tokens: int
    total_cost: float
    avg_tokens_per_operation: float
    avg_cost_per_operation: float
    cost_by_operation_type: Dict[str, float]
    cost_by_model: Dict[str, float]
    peak_usage_hour: Optional[str] = None

class GeminiPricingCalculator:
    """Calculate costs for Google Gemini models"""

    # Gemini 2.5 Flash pricing (as of Nov 2024)
    PRICING = {
        "gemini-2.5-flash": {
            "input_per_1k": 0.000075,   # $0.000075 per 1K input tokens
            "output_per_1k": 0.0003,    # $0.0003 per 1K output tokens
            "context_caching_per_1k": 0.0000075  # Cached context discount
        },
        "gemini-2.0-flash-exp": {
            "input_per_1k": 0.0,        # Currently free in preview
            "output_per_1k": 0.0,
            "context_caching_per_1k": 0.0
        },
        "text-embedding-004": {
            "input_per_1k": 0.00001,    # $0.00001 per 1K tokens
            "output_per_1k": 0.0,       # No output for embeddings
            "context_caching_per_1k": 0.0
        }
    }

    @classmethod
    def calculate_cost(cls, model_name: str, input_tokens: int,
                      output_tokens: int = 0, use_caching: bool = False) -> float:
        """
        Calculate cost for token usage

        Args:
            model_name: Name of the Gemini model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            use_caching: Whether context caching is used

        Returns:
            Total cost in USD
        """
        if model_name not in cls.PRICING:
            logging.warning(f"Unknown model {model_name}, using gemini-2.5-flash pricing")
            model_name = "gemini-2.5-flash"

        pricing = cls.PRICING[model_name]

        # Calculate base cost
        input_cost = (input_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_per_1k"]

        # Apply caching discount if enabled
        if use_caching and input_tokens > 32768:  # Caching threshold
            cached_tokens = input_tokens - 32768
            input_cost = (32768 / 1000) * pricing["input_per_1k"]
            input_cost += (cached_tokens / 1000) * pricing["context_caching_per_1k"]

        return input_cost + output_cost

    @classmethod
    def get_pricing_info(cls, model_name: str) -> Dict[str, Any]:
        """Get pricing information for a model"""
        return cls.PRICING.get(model_name, cls.PRICING["gemini-2.5-flash"])

class TokenTracker:
    """Track token usage across the application"""

    def __init__(self, enable_persistence: bool = True):
        self.usage_records: List[TokenUsage] = []
        self.enable_persistence = enable_persistence
        self.lock = threading.Lock()

        # Configuration
        config = get_config()
        self.storage_path = Path(getattr(config, 'cost_tracking_path', './cost_tracking'))
        self.storage_path.mkdir(exist_ok=True)

        # Load existing data if persistence is enabled
        if self.enable_persistence:
            self._load_existing_data()

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 4 characters for English text
        # More accurate for Gemini would require tiktoken or similar
        char_count = len(text)
        estimated_tokens = max(1, char_count // 4)

        # Adjust for common patterns
        word_count = len(text.split())
        if word_count > 0:
            # Use word-based estimation as a sanity check
            word_based_estimate = int(word_count * 1.3)  # 1.3 tokens per word average
            estimated_tokens = min(estimated_tokens, word_based_estimate)

        return estimated_tokens

    def track_llm_usage(self, operation_type: str, model_name: str,
                       input_text: str, output_text: str,
                       user_id: Optional[str] = None,
                       session_id: Optional[str] = None,
                       actual_input_tokens: Optional[int] = None,
                       actual_output_tokens: Optional[int] = None) -> TokenUsage:
        """
        Track LLM usage and calculate costs

        Args:
            operation_type: Type of operation ('query', 'embedding', 'rerank')
            model_name: Name of the model used
            input_text: Input text sent to model
            output_text: Output text received from model
            user_id: Optional user identifier
            session_id: Optional session identifier
            actual_input_tokens: Actual input token count if available
            actual_output_tokens: Actual output token count if available

        Returns:
            TokenUsage record
        """
        # Estimate or use actual token counts
        input_tokens = actual_input_tokens or self.estimate_tokens(input_text)
        output_tokens = actual_output_tokens or self.estimate_tokens(output_text)
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        cost = GeminiPricingCalculator.calculate_cost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # Create usage record
        usage = TokenUsage(
            operation_id=f"{operation_type}_{int(time.time() * 1000)}",
            timestamp=time.time(),
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_name=model_name,
            cost=cost,
            user_id=user_id,
            session_id=session_id
        )

        # Store record
        with self.lock:
            self.usage_records.append(usage)

        # Persist to disk if enabled
        if self.enable_persistence:
            self._persist_usage(usage)

        return usage

    def track_embedding_usage(self, text: str, model_name: str = "text-embedding-004",
                            user_id: Optional[str] = None,
                            session_id: Optional[str] = None) -> TokenUsage:
        """Track embedding generation usage"""
        return self.track_llm_usage(
            operation_type="embedding",
            model_name=model_name,
            input_text=text,
            output_text="",  # Embeddings don't produce text output
            user_id=user_id,
            session_id=session_id
        )

    def get_usage_summary(self, hours_back: int = 24) -> CostSummary:
        """
        Get cost and usage summary for the specified time period

        Args:
            hours_back: Number of hours to look back from now

        Returns:
            CostSummary with aggregated statistics
        """
        end_time = time.time()
        start_time = end_time - (hours_back * 3600)

        with self.lock:
            # Filter records within time period
            filtered_records = [
                record for record in self.usage_records
                if start_time <= record.timestamp <= end_time
            ]

        if not filtered_records:
            return CostSummary(
                period_start=start_time,
                period_end=end_time,
                total_operations=0,
                total_tokens=0,
                total_cost=0.0,
                avg_tokens_per_operation=0.0,
                avg_cost_per_operation=0.0,
                cost_by_operation_type={},
                cost_by_model={}
            )

        # Calculate aggregates
        total_operations = len(filtered_records)
        total_tokens = sum(record.total_tokens for record in filtered_records)
        total_cost = sum(record.cost for record in filtered_records)

        # Group by operation type
        cost_by_operation_type = {}
        for record in filtered_records:
            op_type = record.operation_type
            cost_by_operation_type[op_type] = cost_by_operation_type.get(op_type, 0) + record.cost

        # Group by model
        cost_by_model = {}
        for record in filtered_records:
            model = record.model_name
            cost_by_model[model] = cost_by_model.get(model, 0) + record.cost

        # Find peak usage hour
        peak_usage_hour = self._find_peak_usage_hour(filtered_records, start_time, end_time)

        return CostSummary(
            period_start=start_time,
            period_end=end_time,
            total_operations=total_operations,
            total_tokens=total_tokens,
            total_cost=total_cost,
            avg_tokens_per_operation=total_tokens / total_operations,
            avg_cost_per_operation=total_cost / total_operations,
            cost_by_operation_type=cost_by_operation_type,
            cost_by_model=cost_by_model,
            peak_usage_hour=peak_usage_hour
        )

    def _find_peak_usage_hour(self, records: List[TokenUsage], start_time: float, end_time: float) -> Optional[str]:
        """Find the hour with highest token usage"""
        if not records:
            return None

        # Group records by hour
        hourly_usage = {}
        for record in records:
            hour_key = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:00")
            hourly_usage[hour_key] = hourly_usage.get(hour_key, 0) + record.total_tokens

        if not hourly_usage:
            return None

        # Find peak hour
        peak_hour = max(hourly_usage.items(), key=lambda x: x[1])
        return peak_hour[0]

    def get_cost_projections(self, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Project costs based on recent usage patterns

        Args:
            days_ahead: Number of days to project ahead

        Returns:
            Cost projection data
        """
        # Get recent usage (last 7 days)
        recent_summary = self.get_usage_summary(hours_back=7 * 24)

        if recent_summary.total_operations == 0:
            return {
                'projection_period_days': days_ahead,
                'projected_operations': 0,
                'projected_cost': 0.0,
                'confidence': 'low',
                'recommendations': ['Insufficient usage data for accurate projection']
            }

        # Calculate daily averages
        daily_operations = recent_summary.total_operations / 7
        daily_cost = recent_summary.total_cost / 7

        # Project forward
        projected_operations = int(daily_operations * days_ahead)
        projected_cost = daily_cost * days_ahead

        # Determine confidence based on usage consistency
        confidence = 'high' if recent_summary.total_operations > 50 else 'medium' if recent_summary.total_operations > 10 else 'low'

        # Generate recommendations
        recommendations = []
        if projected_cost > 100:  # $100+ per month
            recommendations.append('Consider enabling query caching to reduce LLM costs')
        if projected_cost > 500:  # $500+ per month
            recommendations.append('Evaluate using cheaper models for non-critical queries')
        if recent_summary.avg_tokens_per_operation > 2000:
            recommendations.append('Consider optimizing prompt length to reduce token usage')

        return {
            'projection_period_days': days_ahead,
            'recent_daily_avg_operations': daily_operations,
            'recent_daily_avg_cost': daily_cost,
            'projected_operations': projected_operations,
            'projected_cost': projected_cost,
            'confidence': confidence,
            'recommendations': recommendations
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get cost optimization recommendations"""
        summary = self.get_usage_summary(hours_back=7 * 24)
        recommendations = []

        # High-cost operations
        if summary.avg_cost_per_operation > 0.01:  # $0.01+ per operation
            recommendations.append({
                'type': 'high_cost_per_operation',
                'severity': 'high',
                'description': 'Average cost per operation is high',
                'suggestion': 'Consider implementing query caching or using smaller models',
                'potential_savings': '30-70%'
            })

        # High token usage
        if summary.avg_tokens_per_operation > 3000:
            recommendations.append({
                'type': 'high_token_usage',
                'severity': 'medium',
                'description': 'High token usage per operation',
                'suggestion': 'Optimize prompts and context to reduce token consumption',
                'potential_savings': '20-40%'
            })

        # Model usage optimization
        cost_by_model = summary.cost_by_model
        if 'gemini-2.5-flash' in cost_by_model and cost_by_model['gemini-2.5-flash'] > summary.total_cost * 0.8:
            recommendations.append({
                'type': 'model_optimization',
                'severity': 'low',
                'description': 'Heavy usage of gemini-2.5-flash',
                'suggestion': 'Consider using gemini-2.0-flash-exp for non-critical queries (currently free)',
                'potential_savings': '50-90%'
            })

        # Embedding optimization
        if 'embedding' in summary.cost_by_operation_type:
            embedding_cost = summary.cost_by_operation_type['embedding']
            if embedding_cost > summary.total_cost * 0.3:
                recommendations.append({
                    'type': 'embedding_optimization',
                    'severity': 'medium',
                    'description': 'High embedding generation costs',
                    'suggestion': 'Implement embedding caching or use local embedding models',
                    'potential_savings': '60-90%'
                })

        return recommendations

    def _persist_usage(self, usage: TokenUsage):
        """Persist usage record to disk"""
        if not self.enable_persistence:
            return

        try:
            # Create daily log file
            date_str = datetime.fromtimestamp(usage.timestamp).strftime("%Y-%m-%d")
            log_file = self.storage_path / f"usage_{date_str}.jsonl"

            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(usage)) + '\n')

        except Exception as e:
            logging.error(f"Failed to persist usage data: {e}")

    def _load_existing_data(self):
        """Load existing usage data from disk"""
        if not self.enable_persistence or not self.storage_path.exists():
            return

        try:
            # Load recent files (last 30 days)
            cutoff_time = time.time() - (30 * 24 * 3600)

            for log_file in self.storage_path.glob("usage_*.jsonl"):
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                usage_data = json.loads(line)
                                usage = TokenUsage(**usage_data)

                                # Only load recent data
                                if usage.timestamp > cutoff_time:
                                    self.usage_records.append(usage)

                except Exception as e:
                    logging.warning(f"Failed to load usage data from {log_file}: {e}")

        except Exception as e:
            logging.error(f"Failed to load existing usage data: {e}")

    def export_usage_data(self, output_file: str, days_back: int = 30) -> Dict[str, Any]:
        """Export usage data to CSV or JSON"""
        end_time = time.time()
        start_time = end_time - (days_back * 24 * 3600)

        with self.lock:
            filtered_records = [
                asdict(record) for record in self.usage_records
                if start_time <= record.timestamp <= end_time
            ]

        try:
            output_path = Path(output_file)
            if output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(filtered_records, f, indent=2)
            elif output_path.suffix.lower() == '.csv':
                import pandas as pd
                df = pd.DataFrame(filtered_records)
                df.to_csv(output_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use .json or .csv")

            return {
                'status': 'success',
                'records_exported': len(filtered_records),
                'file_path': str(output_path),
                'period_days': days_back
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'records_exported': 0
            }

# Global token tracker instance
token_tracker = TokenTracker()

def track_tokens(operation_type: str, model_name: str):
    """Decorator for automatic token tracking"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get input text from args/kwargs
            input_text = ""
            if args and isinstance(args[0], str):
                input_text = args[0]
            elif 'query' in kwargs:
                input_text = kwargs['query']
            elif 'text' in kwargs:
                input_text = kwargs['text']

            # Execute function
            result = func(*args, **kwargs)

            # Get output text
            output_text = ""
            if hasattr(result, 'answer'):
                output_text = result.answer
            elif isinstance(result, str):
                output_text = result

            # Track usage
            token_tracker.track_llm_usage(
                operation_type=operation_type,
                model_name=model_name,
                input_text=input_text,
                output_text=output_text
            )

            return result
        return wrapper
    return decorator

def main():
    """Test cost modeling functionality"""
    print("💰 Testing Cost Modeling & Token Tracking")
    print("=" * 50)

    # Test token estimation
    test_text = "What is retrieval augmented generation and how does it work in educational applications?"
    estimated_tokens = token_tracker.estimate_tokens(test_text)
    print(f"📝 Text: {test_text}")
    print(f"🔢 Estimated tokens: {estimated_tokens}")

    # Test cost calculation
    cost = GeminiPricingCalculator.calculate_cost(
        model_name="gemini-2.5-flash",
        input_tokens=estimated_tokens,
        output_tokens=150  # Assume 150 token response
    )
    print(f"💵 Estimated cost: ${cost:.6f}")

    # Test usage tracking
    usage = token_tracker.track_llm_usage(
        operation_type="query",
        model_name="gemini-2.5-flash",
        input_text=test_text,
        output_text="Retrieval Augmented Generation (RAG) is a framework that combines information retrieval with language generation..."
    )
    print(f"📊 Usage tracked: {usage.operation_id}")

    # Test summary
    summary = token_tracker.get_usage_summary(hours_back=1)
    print(f"\n📈 Usage Summary (last hour):")
    print(f"   Operations: {summary.total_operations}")
    print(f"   Total cost: ${summary.total_cost:.6f}")

    # Test projections
    projections = token_tracker.get_cost_projections(days_ahead=30)
    print(f"\n🔮 30-day projection:")
    print(f"   Estimated cost: ${projections['projected_cost']:.2f}")
    print(f"   Confidence: {projections['confidence']}")

    # Test recommendations
    recommendations = token_tracker.get_optimization_recommendations()
    print(f"\n💡 Optimization recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"   • {rec['description']}: {rec['suggestion']}")

if __name__ == "__main__":
    main()