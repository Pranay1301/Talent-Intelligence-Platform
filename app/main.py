"""
FastAPI Application Entry Point

AI-Powered Resume Screening and Candidate Ranking System
Production-ready API with CORS, logging, and comprehensive endpoints.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="AI Resume Screening API",
    description=(
        "An intelligent system that reads resumes, compares them with job descriptions, "
        "ranks candidates by relevance, and explains match quality using NLP, "
        "Sentence-BERT embeddings, FAISS vector search, and hybrid scoring."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "AI Resume Screening API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "description": "Upload resumes, rank candidates against job descriptions with explainable AI.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
