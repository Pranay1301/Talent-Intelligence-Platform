"""Setup configuration for the resume screening system."""

from setuptools import setup, find_packages

setup(
    name="resume-screening-system",
    version="1.0.0",
    description="AI-Powered Resume Screening and Candidate Ranking System",
    author="Pranay Bhardwaj",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.27.0",
        "sentence-transformers>=2.5.0",
        "faiss-cpu>=1.8.0",
        "spacy>=3.7.0",
        "pdfplumber>=0.11.0",
        "PyPDF2>=3.0.0",
        "python-docx>=1.1.0",
        "scikit-learn>=1.4.0",
        "sqlalchemy>=2.0.0",
        "streamlit>=1.31.0",
        "plotly>=5.19.0",
        "pandas>=2.2.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "python-dotenv>=1.0.0",
        "python-multipart>=0.0.9",
    ],
)
