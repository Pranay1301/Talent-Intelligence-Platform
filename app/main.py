"""
FastAPI Application Entry Point

AI-Powered Intelligent Talent Screening, Ranking & Hiring Recommendation Platform
Production-grade ML + NLP + Vector Search + Explainable AI system.
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
    title="AI Talent Intelligence Platform",
    description=(
        "A production-grade, enterprise-level ML platform for automated resume screening, "
        "semantic candidate ranking, and hiring recommendation. Combines NLP, Sentence-BERT "
        "embeddings, FAISS vector search, multi-stage hybrid ranking, skill ontology mapping, "
        "explainable AI, bias mitigation, and interview recommendation to deliver "
        "recruiter-grade candidate evaluation at scale."
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
    """Root endpoint with platform information."""
    return {
        "name": "AI Talent Intelligence Platform",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "description": (
            "Enterprise-grade talent screening platform with multi-stage ranking, "
            "explainable AI, bias mitigation, and interview recommendations."
        ),
        "capabilities": [
            "Resume parsing (PDF/DOCX)",
            "Semantic candidate-job matching (Sentence-BERT)",
            "FAISS vector retrieval",
            "Multi-stage hybrid ranking",
            "Skill ontology + synonym mapping",
            "Explainable AI scoring",
            "Bias-aware ranking",
            "Interview recommendations",
            "Recruiter analytics dashboard",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
