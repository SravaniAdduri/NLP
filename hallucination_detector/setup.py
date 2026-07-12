from setuptools import setup, find_packages

setup(
    name="hallucination_detector",
    version="1.0.0",
    description="An Adaptive Multi-Agent Retrieval-Augmented Framework for Hallucination Detection and Fact Verification",
    author="Research Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.2.0",
        "langchain-community>=0.2.0",
        "langchain-huggingface>=0.0.3",
        "langgraph>=0.1.0",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.2",
        "transformers>=4.36.0",
        "torch>=2.1.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "streamlit>=1.28.0",
        "pydantic>=2.5.0",
        "PyPDF2>=3.0.0",
        "beautifulsoup4>=4.12.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ]
    },
)
