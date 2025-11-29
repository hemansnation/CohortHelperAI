# 🤝 Contributing to CohortRAG Engine

Welcome to the CohortRAG Engine community! We're building the premier open-source RAG system for education, and every contribution helps millions of students get better, faster answers.

## 🌟 **Why Your Contribution Matters**

- **🎓 Impact Education**: Help millions of students worldwide get better answers
- **🚀 Learn Cutting-Edge AI**: Work with production-ready RAG technology
- **📈 Build Your Portfolio**: Contribute to a validated, enterprise-grade project
- **🤝 Join a Global Community**: Connect with educators and AI developers
- **🔮 Shape the Future**: Influence the direction of educational AI

## ⚡ **Quick Start (5 Minutes)**

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/CohortHelperAI.git
cd CohortHelperAI/cohortrag_engine

# 2. Install with dev dependencies
pip install -e ".[dev,test]"

# 3. Verify everything works
pytest tests/ --quick
cohortrag-benchmark --quick

# 4. Find your first issue
# Browse: https://github.com/YourUsername/CohortHelperAI/labels/good%20first%20issue
```

**Ready to contribute?** Jump to [🎯 Contribution Areas](#-contribution-areas) or [🛠️ Development Workflow](#-development-workflow)

## 🌟 **Code of Conduct**

We are committed to fostering a welcoming and inclusive community. By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be collaborative**: Focus on constructive discussion and feedback
- **Be professional**: Maintain professional communication in all interactions

## 🚀 **Getting Started**

### **Prerequisites**
- Python 3.8+ installed
- Git knowledge and GitHub account
- Basic understanding of RAG systems (helpful but not required)
- Interest in educational technology

### **First Steps**
1. **Fork** the repository to your GitHub account
2. **Star** the repository if you find it useful ⭐
3. **Read** the documentation in the `docs/` directory
4. **Join** our community discussions

## 🛠 **Development Setup**

### **1. Clone Your Fork**
```bash
git clone https://github.com/YOUR_USERNAME/CohortHelperAI.git
cd CohortHelperAI/cohortrag_engine
```

### **2. Set Up Python Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black isort pytest pytest-cov flake8
```

### **3. Configure Environment**
```bash
# Create development .env file
cp .env.example .env
# Edit .env with your configuration (see docs/install.md for details)
```

### **4. Verify Installation**
```bash
# Run tests to ensure everything works
python tests/run_tests.py

# Run the main application
python main.py
```

### **5. Set Up Pre-commit Hooks (Optional but Recommended)**
```bash
pip install pre-commit
pre-commit install
```

## 🎯 **How to Contribute**

We welcome various types of contributions:

### **🐛 Bug Reports**
Found a bug? Help us fix it!

**Before reporting:**
- Check if the issue already exists in our [Issue Tracker](https://github.com/YourUsername/CohortHelperAI/issues)
- Try to reproduce the bug with the latest version
- Gather relevant information (OS, Python version, error logs)

**When reporting:**
- Use a clear, descriptive title
- Provide steps to reproduce the issue
- Include error messages and stack traces
- Mention expected vs actual behavior
- Add relevant system information

### **🚀 Feature Requests**
Have an idea for improving CohortRAG Engine?

**Good feature requests include:**
- Clear description of the problem it solves
- Specific use case for educational applications
- Proposed solution or implementation approach
- Consideration of backward compatibility
- Willingness to help implement (bonus points!)

### **📝 Documentation Improvements**
Documentation is crucial for adoption:

- **User guides**: Help new users get started
- **API documentation**: Improve code documentation
- **Examples**: Add real-world usage examples
- **Tutorials**: Create step-by-step learning materials
- **Translations**: Help make docs accessible globally

### **🧪 Testing & Quality Assurance**
Help us maintain high quality:

- **Test coverage**: Add tests for uncovered code
- **Integration tests**: Test real-world scenarios
- **Performance testing**: Benchmark improvements
- **Educational content testing**: Test with real educational materials
- **Cross-platform testing**: Test on different environments

### **🔧 Code Contributions**
Ready to dive into the code? Great!

**Focus Areas:**
- **Core RAG improvements**: Enhance retrieval and generation
- **Educational optimizations**: Improve accuracy for educational content
- **Performance optimizations**: Make it faster and more efficient
- **New integrations**: Add support for more vector stores, LLMs, etc.
- **Developer experience**: Improve CLI, error messages, debugging tools

## 📬 **Pull Request Process**

### **Before You Start**
1. **Check existing issues** and PRs to avoid duplication
2. **Create or comment on an issue** to discuss your planned changes
3. **Fork and create a branch** from `main` for your work

### **Branch Naming Convention**
```bash
# Feature branches
feature/add-azure-openai-support
feature/improve-chunk-scoring

# Bug fixes
fix/memory-leak-async-ingestion
fix/error-handling-pdf-parsing

# Documentation
docs/update-installation-guide
docs/add-api-examples

# Refactoring
refactor/cleanup-evaluation-module
refactor/optimize-embedding-pipeline
```

### **Development Workflow**
```bash
# 1. Create and switch to your feature branch
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... edit files ...

# 3. Format code (required)
black .
isort .

# 4. Run tests locally
python tests/run_tests.py
pytest tests/ -v

# 5. Add and commit your changes
git add .
git commit -m "feat: add support for Azure OpenAI embeddings"

# 6. Push to your fork
git push origin feature/your-feature-name

# 7. Create Pull Request via GitHub UI
```

### **Pull Request Requirements**

✅ **Required Checklist:**
- [ ] Code follows our style guidelines (black + isort)
- [ ] All tests pass locally
- [ ] New functionality includes tests
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventional format
- [ ] PR description explains the changes clearly
- [ ] No merge conflicts with main branch

✅ **PR Description Template:**
```markdown
## Summary
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manually tested with educational content

## Educational Impact
How does this change benefit educational use cases?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### **Review Process**
1. **Automated checks** must pass (tests, linting, formatting)
2. **Code review** by at least one maintainer
3. **Educational testing** for changes affecting core RAG functionality
4. **Documentation review** for user-facing changes
5. **Final approval** and merge by maintainers

## 🎨 **Code Style Guidelines**

We maintain consistent code style across the project:

### **Python Style**
- **Formatter**: [Black](https://black.readthedocs.io/) (line length: 88)
- **Import sorting**: [isort](https://isort.readthedocs.io/)
- **Linting**: [flake8](https://flake8.pycqa.org/) (with exceptions for line length)
- **Type hints**: Encouraged, especially for public APIs

### **Running Style Checks**
```bash
# Format code
black .
isort .

# Check formatting (CI will verify)
black --check .
isort --check-only .

# Run linting
flake8 --max-line-length=88 --extend-ignore=E203,W503 .
```

### **Naming Conventions**
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`
- **Modules**: `snake_case.py`

### **Documentation Style**
```python
def process_educational_content(
    content: str,
    subject_area: Optional[str] = None
) -> ProcessedContent:
    """
    Process educational content for optimal RAG retrieval.

    Args:
        content: Raw educational content text
        subject_area: Optional subject classification for optimization

    Returns:
        ProcessedContent object with chunked and metadata-enriched content

    Raises:
        ContentProcessingError: If content cannot be processed

    Example:
        >>> processed = process_educational_content("Physics lecture notes...")
        >>> print(f"Created {len(processed.chunks)} chunks")
    """
```

## 🧪 **Testing Requirements**

Quality is paramount for educational technology:

### **Test Categories**
1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Benchmark critical paths
5. **Educational Content Tests**: Test with real educational materials

### **Running Tests**
```bash
# Run all tests
python tests/run_tests.py

# Run specific test module
python -m pytest tests/test_chunking.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run performance benchmarks
python -c "
from utils.benchmarks import ComprehensiveBenchmark
from core.retrieval import CohortRAGRetriever
benchmark = ComprehensiveBenchmark(CohortRAGRetriever())
results = benchmark.run_comprehensive_benchmark(num_queries=10)
print(f'Avg latency: {results[\"performance_metrics\"][\"avg_latency\"]:.3f}s')
"
```

### **Test Writing Guidelines**
- **Descriptive names**: `test_async_ingestion_handles_large_pdf_files()`
- **Clear assertions**: Test one thing per test method
- **Educational relevance**: Use educational content in test data
- **Performance awareness**: Mark slow tests appropriately
- **Mocking external services**: Don't rely on live APIs in tests

### **Required Tests for PRs**
- All new functions must have unit tests
- New features must have integration tests
- Bug fixes must include regression tests
- Performance improvements must include benchmarks

## 📚 **Documentation Standards**

Good documentation accelerates adoption:

### **Documentation Types**
1. **Code documentation**: Docstrings and inline comments
2. **User guides**: Step-by-step instructions
3. **API documentation**: Complete reference
4. **Examples**: Real-world usage scenarios
5. **Architecture docs**: System design explanations

### **Documentation Checklist**
- [ ] All public functions have docstrings
- [ ] User-facing changes documented in relevant guides
- [ ] Code examples are tested and working
- [ ] Markdown follows consistent formatting
- [ ] Links and references are valid

## 🐛 **Issue Guidelines**

### **Issue Types & Labels**
- **`bug`**: Something isn't working correctly
- **`enhancement`**: New feature or improvement request
- **`documentation`**: Documentation needs improvement
- **`good first issue`**: Great for newcomers
- **`help wanted`**: Community assistance needed
- **`educational-focus`**: Specifically related to educational use cases
- **`performance`**: Performance optimization opportunities
- **`question`**: Support or clarification needed

### **Issue Templates**

**Bug Report Template:**
```markdown
**Description**
Clear description of what the bug is.

**Steps to Reproduce**
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- CohortRAG Version: [e.g., 1.2.0]

**Additional Context**
- Error logs
- Screenshots
- Educational content type being processed
```

**Feature Request Template:**
```markdown
**Educational Problem**
What educational challenge does this address?

**Proposed Solution**
Describe your proposed solution.

**Alternative Solutions**
Other approaches you've considered.

**Implementation Ideas**
Any thoughts on implementation approach.

**Additional Context**
Examples, mockups, or related research.
```

## 💬 **Community & Communication**

### **Where to Get Help**
- **GitHub Discussions**: General questions and community chat
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Check `docs/` first for answers
- **Code Comments**: Inline documentation explains design decisions

### **Communication Guidelines**
- **Be specific**: Provide context and details
- **Be patient**: Maintainers volunteer their time
- **Be helpful**: Answer questions from other community members
- **Be educational**: Consider the educational technology perspective

### **Recognition**
We value all contributions! Contributors are recognized through:
- **Contributors list** in repository and releases
- **Special mentions** in significant releases
- **Maintainer opportunities** for consistent contributors
- **Speaking opportunities** at education technology events

## 🎓 **Educational Focus Areas**

As an educational RAG system, we particularly welcome contributions in:

### **Subject Area Optimizations**
- **STEM**: Mathematics, physics, chemistry, biology optimizations
- **Humanities**: Literature, history, philosophy content handling
- **Languages**: Multi-language support and language learning
- **Professional**: Business, law, medicine domain expertise

### **Educational Methodology Integration**
- **Bloom's Taxonomy**: Question classification and response optimization
- **Learning Objectives**: Alignment with curriculum standards
- **Assessment Integration**: Connection to learning assessment tools
- **Accessibility**: Support for diverse learning needs

### **Institutional Integration**
- **LMS Integration**: Canvas, Blackboard, Moodle connectors
- **SSO Support**: SAML, OAuth for institutional authentication
- **Analytics**: Learning analytics and outcome measurement
- **Compliance**: FERPA, GDPR, and other educational regulations

## 📊 **Success Metrics**

We track contribution success through:

### **Technical Metrics**
- **Test Coverage**: Maintain >90% coverage
- **Performance**: <2s latency for educational queries
- **Accuracy**: >90% RAGAS scores on educational content
- **Cost Efficiency**: <$0.05 per query

### **Community Metrics**
- **Contributor Growth**: New contributors each month
- **Issue Resolution**: Average time to close issues
- **Documentation Quality**: User feedback and usage
- **Educational Impact**: Institutions and students served

## 🏆 **Contributor Recognition**

### **Contribution Levels**
- **First-time Contributor**: Welcome package and guidance
- **Regular Contributor**: Recognition in release notes
- **Core Contributor**: Increased review permissions
- **Maintainer**: Full repository access and responsibilities

### **Special Recognition**
- **Educational Impact Award**: For contributions significantly improving educational outcomes
- **Innovation Award**: For creative solutions to educational challenges
- **Community Champion**: For exceptional help in community building

## 🚀 **Getting Started Today**

Ready to contribute? Here's how to make your first contribution:

### **Quick Start for First-Time Contributors**
1. **Browse** [good first issues](https://github.com/YourUsername/CohortHelperAI/labels/good%20first%20issue)
2. **Comment** on an issue to express interest
3. **Ask questions** if anything is unclear
4. **Submit** your first PR following our guidelines
5. **Celebrate** your contribution to educational technology! 🎉

### **Suggested First Contributions**
- **Documentation improvements**: Fix typos, add examples
- **Test additions**: Increase coverage for existing code
- **Educational content testing**: Test with your domain expertise
- **Small feature additions**: Minor enhancements to existing functionality

---

## 🤝 **Thank You**

Every contribution, no matter how small, helps make education more accessible and effective through technology. We're grateful for your participation in building the future of educational AI.

**Questions?** Don't hesitate to ask in our [Discussions](https://github.com/YourUsername/CohortHelperAI/discussions) or reach out to maintainers.

**Happy Contributing!** 🎓✨

---

*This guide is inspired by the best practices of successful open-source educational technology projects. It's a living document that evolves with our community.*