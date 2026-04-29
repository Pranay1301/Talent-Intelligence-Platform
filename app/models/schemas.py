"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# --- Request Models ---

class JDInput(BaseModel):
    """Job description input for ranking."""
    title: str = Field(default="", description="Job title")
    description: str = Field(..., description="Full JD text", min_length=20)
    company: str = Field(default="", description="Company name")


class ResumeTextInput(BaseModel):
    """Resume text input (for non-file uploads)."""
    name: str = Field(default="", description="Candidate name")
    resume_text: str = Field(..., description="Raw resume text", min_length=20)


class RankingRequest(BaseModel):
    """Request to rank candidates against a job description."""
    jd_text: str = Field(..., description="Job description text")
    candidate_ids: list[str] = Field(default=[], description="Specific candidate IDs to rank (empty = all)")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top candidates to return")


# --- Response Models ---

class ScoreDetail(BaseModel):
    embedding_similarity: float = 0.0
    skill_match_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    certification_bonus: float = 0.0
    final_score: float = 0.0


class ExplanationDetail(BaseModel):
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    extra_skills: list[str] = []
    experience_match: str = ""
    education_match: str = ""
    strengths: list[str] = []
    weaknesses: list[str] = []
    recommendation: str = ""


class CandidateResult(BaseModel):
    """Single candidate ranking result."""
    candidate_id: str
    name: str
    email: str = ""
    rank: int
    final_score: float
    scores: ScoreDetail
    explanation: ExplanationDetail


class RankingResponse(BaseModel):
    """Response containing ranked candidates."""
    job_title: str = ""
    total_candidates: int
    ranked_candidates: list[CandidateResult]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ResumeUploadResponse(BaseModel):
    """Response after uploading a resume."""
    candidate_id: str
    name: str
    email: str = ""
    skills_count: int
    experience_years: float
    message: str = "Resume uploaded and parsed successfully"


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    total_candidates: int = 0
    faiss_vectors: int = 0


class ExportFormat(BaseModel):
    format: str = Field(default="json", description="Export format: json, csv")
