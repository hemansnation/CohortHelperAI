import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

class CohortRAGConfig:
    """Configuration class for CohortRAG Engine"""

    def __init__(self, env_file: Optional[str] = None):
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Look for .env in current directory first, then parent
            current_dir = Path(__file__).parent
            env_path = current_dir / ".env"
            if not env_path.exists():
                env_path = current_dir.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)

        # API Configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Paths
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.data_dir = os.getenv("DATA_DIR", "./data")

        # Model Configuration
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1")
        self.llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")

        # RAG Configuration
        self.similarity_top_k = int(os.getenv("SIMILARITY_TOP_K", "3"))
        self.context_window = int(os.getenv("CONTEXT_WINDOW", "4096"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

        # Evaluation Configuration
        self.enable_evaluation = os.getenv("ENABLE_EVALUATION", "false").lower() == "true"
        self.ragas_log_level = os.getenv("RAGAS_LOG_LEVEL", "INFO")

        # Ensure directories exist
        Path(self.chroma_db_path).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"CohortRAGConfig(data_dir='{self.data_dir}', llm_model='{self.llm_model}')"

# Global config instance
config = None

def get_config() -> CohortRAGConfig:
    """Get global configuration instance"""
    global config
    if config is None:
        config = CohortRAGConfig()
    return config