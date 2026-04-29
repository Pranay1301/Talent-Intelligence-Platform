"""
Screening Service - Orchestrates the entire resume screening pipeline.
Connects parsing, embedding, FAISS, and ranking into a unified workflow.
"""

import os
import uuid
import json
import logging
from datetime import datetime

from app.config import get_settings
from resume_parser.parser import ResumeParser
from jd_parser.parser import JDParser
from embedding_engine.engine import EmbeddingEngine
from embedding_engine.vector_store import FAISSVectorStore
from ranking_engine.ranker import CandidateRanker, RankedCandidate
from database.models import CandidateModel, RankingHistoryModel, get_session

logger = logging.getLogger(__name__)
settings = get_settings()


class ScreeningService:
    """
    Main orchestration service for the resume screening pipeline.
    
    Pipeline:
    1. Parse resume → extract structured data
    2. Generate embedding → store in FAISS
    3. Parse JD → extract requirements
    4. Retrieve similar candidates via FAISS
    5. Rank candidates with hybrid scoring
    6. Return ranked results with explanations
    """

    def __init__(self):
        self.resume_parser = ResumeParser(spacy_model=settings.spacy_model)
        self.jd_parser = JDParser()
        self.embedding_engine = EmbeddingEngine(model_name=settings.embedding_model)
        
        dim = self.embedding_engine.get_embedding_dimension()
        self.vector_store = FAISSVectorStore(
            dimension=dim,
            index_path=settings.faiss_index_path,
            metadata_path=settings.faiss_metadata_path,
        )
        self.ranker = CandidateRanker()
        self.db = get_session(settings.database_url)
        logger.info("ScreeningService initialized successfully")

    def upload_resume(self, file_path: str, candidate_id: str = None) -> dict:
        """
        Process and store a resume.
        
        1. Parse the resume file
        2. Generate embedding
        3. Store in FAISS index
        4. Save to database
        """
        candidate_id = candidate_id or str(uuid.uuid4())[:8]
        
        # Parse resume
        parsed = self.resume_parser.parse(file_path)
        logger.info(f"Parsed resume for: {parsed.name}")

        # Generate embedding from resume text
        embedding_text = self._build_embedding_text(parsed)
        embedding = self.embedding_engine.encode_single(embedding_text)

        # Store in FAISS
        import numpy as np
        self.vector_store.add(
            np.array([embedding]),
            [{"candidate_id": candidate_id, "name": parsed.name, "email": parsed.email}],
        )
        self.vector_store.save()

        # Save to database
        candidate = CandidateModel(
            id=candidate_id,
            name=parsed.name,
            email=parsed.email,
            phone=parsed.phone,
            linkedin=parsed.linkedin,
            github=parsed.github,
            summary=parsed.summary,
            skills=parsed.skills,
            education=parsed.education,
            experience=parsed.experience,
            projects=parsed.projects,
            certifications=parsed.certifications,
            total_experience_years=parsed.total_experience_years,
            raw_text=parsed.raw_text,
            file_path=file_path,
        )

        existing = self.db.query(CandidateModel).filter_by(id=candidate_id).first()
        if existing:
            for key, value in candidate.to_dict().items():
                if key != "candidate_id" and key != "created_at":
                    setattr(existing, key, value)
        else:
            self.db.add(candidate)
        self.db.commit()

        return {
            "candidate_id": candidate_id,
            "name": parsed.name,
            "email": parsed.email,
            "skills_count": len(parsed.skills),
            "experience_years": parsed.total_experience_years,
            "message": "Resume uploaded and parsed successfully",
        }

    def upload_resume_text(self, text: str, name: str = "", candidate_id: str = None) -> dict:
        """Process and store a resume from raw text."""
        candidate_id = candidate_id or str(uuid.uuid4())[:8]
        
        parsed = self.resume_parser.parse_from_text(text)
        if name:
            parsed.name = name

        embedding_text = self._build_embedding_text(parsed)
        embedding = self.embedding_engine.encode_single(embedding_text)

        import numpy as np
        self.vector_store.add(
            np.array([embedding]),
            [{"candidate_id": candidate_id, "name": parsed.name, "email": parsed.email}],
        )
        self.vector_store.save()

        candidate = CandidateModel(
            id=candidate_id, name=parsed.name, email=parsed.email,
            phone=parsed.phone, skills=parsed.skills,
            education=parsed.education, experience=parsed.experience,
            projects=parsed.projects, certifications=parsed.certifications,
            total_experience_years=parsed.total_experience_years,
            raw_text=parsed.raw_text,
        )
        
        existing = self.db.query(CandidateModel).filter_by(id=candidate_id).first()
        if existing:
            for key, value in candidate.to_dict().items():
                if key not in ("candidate_id", "created_at"):
                    setattr(existing, key, value)
        else:
            self.db.add(candidate)
        self.db.commit()

        return {
            "candidate_id": candidate_id,
            "name": parsed.name,
            "email": parsed.email,
            "skills_count": len(parsed.skills),
            "experience_years": parsed.total_experience_years,
            "message": "Resume processed successfully",
        }

    def rank_candidates(self, jd_text: str, candidate_ids: list[str] = None,
                        top_k: int = 10) -> dict:
        """
        Rank candidates against a job description.
        
        Pipeline:
        1. Parse JD
        2. Generate JD embedding
        3. FAISS retrieval (if no specific candidates)
        4. Hybrid scoring
        5. Return ranked results with explanations
        """
        # Parse JD
        jd_parsed = self.jd_parser.parse(jd_text)
        logger.info(f"Parsed JD: {jd_parsed.title}")

        # Get candidates
        if candidate_ids:
            candidates = []
            for cid in candidate_ids:
                c = self.db.query(CandidateModel).filter_by(id=cid).first()
                if c:
                    candidates.append(c.to_dict())
        else:
            candidates = [c.to_dict() for c in self.db.query(CandidateModel).all()]

        if not candidates:
            return {
                "job_title": jd_parsed.title,
                "total_candidates": 0,
                "ranked_candidates": [],
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Generate JD embedding
        jd_embedding_text = f"{jd_parsed.title} {jd_text}"
        jd_embedding = self.embedding_engine.encode_single(jd_embedding_text)

        # Compute similarity scores for each candidate
        similarity_scores = []
        for candidate in candidates:
            candidate_text = self._build_embedding_text_from_dict(candidate)
            sim = self.embedding_engine.hybrid_similarity(jd_embedding_text, candidate_text)
            similarity_scores.append(sim["combined_score"])

        # Rank candidates
        ranked = self.ranker.rank_candidates(
            candidates, jd_parsed.to_dict(), similarity_scores
        )

        # Limit to top_k
        ranked = ranked[:top_k]

        # Save ranking history
        history = RankingHistoryModel(
            jd_text=jd_text,
            jd_title=jd_parsed.title,
            results=[r.to_dict() for r in ranked],
            total_candidates=len(candidates),
        )
        self.db.add(history)
        self.db.commit()

        return {
            "job_title": jd_parsed.title,
            "total_candidates": len(candidates),
            "ranked_candidates": [r.to_dict() for r in ranked],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_candidate(self, candidate_id: str) -> dict:
        """Get a single candidate's details."""
        c = self.db.query(CandidateModel).filter_by(id=candidate_id).first()
        if not c:
            return None
        return c.to_dict()

    def get_all_candidates(self) -> list[dict]:
        """Get all stored candidates."""
        return [c.to_dict() for c in self.db.query(CandidateModel).all()]

    def delete_candidate(self, candidate_id: str) -> bool:
        """Delete a candidate."""
        c = self.db.query(CandidateModel).filter_by(id=candidate_id).first()
        if c:
            self.db.delete(c)
            self.db.commit()
            return True
        return False

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "total_candidates": self.db.query(CandidateModel).count(),
            "faiss_vectors": self.vector_store.total_vectors,
            "embedding_dimension": self.embedding_engine.get_embedding_dimension(),
        }

    def export_results(self, ranking_results: dict, format: str = "json") -> str:
        """Export ranking results to file."""
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        if format == "csv":
            import csv
            filepath = f"reports/ranking_{timestamp}.csv"
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Rank", "Name", "Email", "Score", "Recommendation"])
                for c in ranking_results.get("ranked_candidates", []):
                    exp = c.get("score_breakdown", {}).get("explanation", {})
                    writer.writerow([
                        c["rank"], c["name"], c.get("email", ""),
                        c["final_score"],
                        exp.get("recommendation", ""),
                    ])
        else:
            filepath = f"reports/ranking_{timestamp}.json"
            with open(filepath, "w") as f:
                json.dump(ranking_results, f, indent=2, default=str)

        return filepath

    def _build_embedding_text(self, parsed) -> str:
        """Build text for embedding from ParsedResume object."""
        parts = [parsed.summary]
        if parsed.skills:
            parts.append("Skills: " + ", ".join(parsed.skills))
        for exp in parsed.experience:
            parts.append(exp.get("title", ""))
            for r in exp.get("responsibilities", []):
                parts.append(r)
        for proj in parsed.projects:
            parts.append(proj.get("name", ""))
            for d in proj.get("details", []):
                parts.append(d)
        return " ".join(filter(None, parts))

    def _build_embedding_text_from_dict(self, candidate: dict) -> str:
        """Build text for embedding from candidate dict."""
        parts = [candidate.get("summary", "")]
        skills = candidate.get("skills", [])
        if skills:
            parts.append("Skills: " + ", ".join(skills))
        for exp in candidate.get("experience", []):
            parts.append(exp.get("title", ""))
            for r in exp.get("responsibilities", []):
                parts.append(r)
        return " ".join(filter(None, parts))
