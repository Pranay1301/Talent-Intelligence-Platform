"""
FastAPI route definitions for the Talent Intelligence Platform.

Endpoints:
- POST /api/upload_resume         - Upload and parse a resume
- POST /api/upload_resume_text    - Upload resume as text
- GET  /api/analyze_resume/{id}   - Detailed resume analysis
- POST /api/parse_jd              - Parse and analyze a job description
- POST /api/rank_candidates       - Multi-stage candidate ranking
- GET  /api/candidates            - List all candidates
- GET  /api/candidates/{id}       - Get candidate details
- DELETE /api/candidates/{id}     - Delete a candidate
- GET  /api/get_candidate_score/{id}  - Score candidate for a JD
- POST /api/get_explanations      - Get detailed ranking explanations
- GET  /api/interview_recommendation/{id} - Interview recommendation
- POST /api/export_report         - Export ranking report
- GET  /api/dashboard_metrics     - Recruiter analytics data
- GET  /api/health                - System health check
"""

import os
import uuid
import shutil
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, Field

from app.models.schemas import (
    JDInput, ResumeTextInput, RankingRequest, RankingResponse,
    ResumeUploadResponse, HealthResponse,
)
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Talent Intelligence"])
settings = get_settings()

# Lazy-loaded service singleton
_service = None


def get_service():
    global _service
    if _service is None:
        from app.services.screening_service import ScreeningService
        _service = ScreeningService()
    return _service


# --- Request models for new endpoints ---

class JDParseRequest(BaseModel):
    jd_text: str = Field(..., min_length=20)


class InterviewRequest(BaseModel):
    jd_text: str = Field(..., min_length=20)


class ExplanationRequest(BaseModel):
    jd_text: str = Field(..., min_length=20)
    candidate_ids: list[str] = Field(default=[])
    top_k: int = Field(default=5, ge=1, le=50)


# --- Resume Endpoints ---

@router.post("/upload_resume", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume file (PDF/DOCX) for parsing and storage."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use PDF or DOCX.")

    candidate_id = str(uuid.uuid4())[:8]
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{candidate_id}{ext}")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        service = get_service()
        result = service.upload_resume(file_path, candidate_id)
        return ResumeUploadResponse(**result)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, f"Failed to process resume: {str(e)}")


@router.post("/upload_resume_text", response_model=ResumeUploadResponse)
async def upload_resume_text(data: ResumeTextInput):
    """Upload resume as raw text."""
    try:
        service = get_service()
        result = service.upload_resume_text(data.resume_text, data.name)
        return ResumeUploadResponse(**result)
    except Exception as e:
        logger.error(f"Text upload failed: {e}")
        raise HTTPException(500, f"Failed to process resume: {str(e)}")


@router.get("/analyze_resume/{candidate_id}")
async def analyze_resume(candidate_id: str):
    """Get detailed analysis of a parsed resume including skill categories and domains."""
    service = get_service()
    result = service.analyze_resume(candidate_id)
    if not result:
        raise HTTPException(404, f"Candidate not found: {candidate_id}")
    return result


# --- JD Endpoints ---

@router.post("/parse_jd")
async def parse_jd(request: JDParseRequest):
    """Parse and analyze a job description — extract requirements with importance classification."""
    try:
        service = get_service()
        return service.parse_jd(request.jd_text)
    except Exception as e:
        raise HTTPException(500, f"JD parsing failed: {str(e)}")


# --- Ranking Endpoints ---

@router.post("/rank_candidates")
async def rank_candidates(request: RankingRequest):
    """
    Multi-stage candidate ranking against a job description.
    
    Pipeline: FAISS Retrieval → Hybrid Scoring → Re-ranking + Explainability
    """
    try:
        service = get_service()
        result = service.rank_candidates(
            jd_text=request.jd_text,
            candidate_ids=request.candidate_ids,
            top_k=request.top_k,
        )
        return result
    except Exception as e:
        logger.error(f"Ranking failed: {e}")
        raise HTTPException(500, f"Ranking failed: {str(e)}")


@router.get("/get_candidate_score/{candidate_id}")
async def get_candidate_score(candidate_id: str, jd_text: str = Query(...)):
    """Get a specific candidate's full score breakdown against a JD."""
    try:
        service = get_service()
        result = service.rank_candidates(
            jd_text=jd_text, candidate_ids=[candidate_id], top_k=1
        )
        candidates = result.get("ranked_candidates", [])
        if candidates:
            return candidates[0]
        raise HTTPException(404, "Candidate not found or no score available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/get_explanations")
async def get_explanations(request: ExplanationRequest):
    """Get detailed explanations for candidate rankings."""
    try:
        service = get_service()
        result = service.rank_candidates(
            jd_text=request.jd_text,
            candidate_ids=request.candidate_ids,
            top_k=request.top_k,
        )
        # Extract just the explanations
        explanations = []
        for c in result.get("ranked_candidates", []):
            explanations.append({
                "candidate_id": c["candidate_id"],
                "name": c["name"],
                "rank": c["rank"],
                "final_score": c["final_score"],
                "explanation": c.get("score_breakdown", {}).get("explanation", {}),
                "interview_recommendation": c.get("interview_recommendation", {}),
            })
        return {"explanations": explanations}
    except Exception as e:
        raise HTTPException(500, str(e))


# --- Interview Endpoints ---

@router.get("/interview_recommendation/{candidate_id}")
async def interview_recommendation(candidate_id: str, jd_text: str = Query(...)):
    """Get interview recommendation and focus areas for a candidate."""
    try:
        service = get_service()
        result = service.get_interview_recommendation(candidate_id, jd_text)
        if not result:
            raise HTTPException(404, "Candidate not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# --- Candidate CRUD ---

@router.get("/candidates")
async def list_candidates():
    """Get all stored candidates."""
    service = get_service()
    candidates = service.get_all_candidates()
    return {"total": len(candidates), "candidates": candidates}


@router.get("/candidates/{candidate_id}")
async def get_candidate(candidate_id: str):
    """Get a specific candidate's details."""
    service = get_service()
    candidate = service.get_candidate(candidate_id)
    if not candidate:
        raise HTTPException(404, f"Candidate not found: {candidate_id}")
    return candidate


@router.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str):
    """Delete a candidate."""
    service = get_service()
    if service.delete_candidate(candidate_id):
        return {"message": f"Candidate {candidate_id} deleted"}
    raise HTTPException(404, f"Candidate not found: {candidate_id}")


# --- Reports & Analytics ---

@router.post("/export_report")
async def export_report(request: RankingRequest, format: str = Query(default="json")):
    """Export ranking results as JSON or CSV report."""
    try:
        service = get_service()
        result = service.rank_candidates(
            jd_text=request.jd_text,
            candidate_ids=request.candidate_ids,
            top_k=request.top_k,
        )
        filepath = service.export_results(result, format)
        return {"message": "Report exported", "file_path": filepath}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/dashboard_metrics")
async def dashboard_metrics():
    """Get aggregated analytics for the recruiter dashboard."""
    service = get_service()
    return service.get_dashboard_metrics()


# --- Health ---

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check with system statistics."""
    service = get_service()
    stats = service.get_stats()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_candidates=stats["total_candidates"],
        faiss_vectors=stats["faiss_vectors"],
    )
