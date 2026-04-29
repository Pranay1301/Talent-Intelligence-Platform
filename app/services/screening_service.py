"""
Screening Service - Orchestrates the full talent intelligence pipeline.

Connects parsing, embedding, FAISS, multi-stage ranking, explainability,
interview recommendations, and bias mitigation into a unified workflow.
"""

import os
import uuid
import json
import logging
from datetime import datetime

import numpy as np

from app.config import get_settings
from resume_parser.parser import ResumeParser
from jd_parser.parser import JDParser
from embedding_engine.engine import EmbeddingEngine
from embedding_engine.vector_store import FAISSVectorStore
from ranking_engine.ranker import CandidateRanker
from app.pipelines.screening_pipeline import ScreeningPipeline
from configs.skill_ontology import SkillOntology
from bias_mitigation.fairness import BiasMitigator
from interview_engine.recommender import InterviewRecommender
from database.models import CandidateModel, RankingHistoryModel, get_session

logger = logging.getLogger(__name__)
settings = get_settings()


class ScreeningService:
    """
    Main orchestration service for the talent intelligence platform.
    
    Multi-Stage Pipeline:
    1. Parse resume → extract structured data (skills, education, experience, projects, certs)
    2. Generate embedding → store in FAISS for fast retrieval
    3. Parse JD → extract requirements with importance weighting
    4. Stage 1: FAISS retrieval (fast candidate shortlisting)
    5. Stage 2: Hybrid multi-signal scoring with ontology-aware matching
    6. Stage 3: Re-ranking with explanations, interview recs, bias checks
    7. Return ranked results with full transparency
    """

    def __init__(self):
        self.resume_parser = ResumeParser(spacy_model=settings.spacy_model)
        self.jd_parser = JDParser()
        self.embedding_engine = EmbeddingEngine(model_name=settings.embedding_model)
        self.ontology = SkillOntology()
        self.bias_mitigator = BiasMitigator()
        self.interview_engine = InterviewRecommender()

        dim = self.embedding_engine.get_embedding_dimension()
        self.vector_store = FAISSVectorStore(
            dimension=dim,
            index_path=settings.faiss_index_path,
            metadata_path=settings.faiss_metadata_path,
        )
        self.ranker = CandidateRanker()

        # Multi-stage pipeline
        self.pipeline = ScreeningPipeline(
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
            ranker=self.ranker,
        )

        self.db = get_session(settings.database_url)
        logger.info("ScreeningService initialized with multi-stage pipeline")

    def upload_resume(self, file_path: str, candidate_id: str = None) -> dict:
        """
        Process and store a resume.
        
        1. Parse the resume file
        2. Normalize skills via ontology
        3. Generate embedding
        4. Store in FAISS index
        5. Save to database
        """
        candidate_id = candidate_id or str(uuid.uuid4())[:8]

        # Parse resume
        parsed = self.resume_parser.parse(file_path)
        logger.info(f"Parsed resume for: {parsed.name}")

        # Normalize skills via ontology
        normalized_skills = self.ontology.normalize_skills(parsed.skills)

        # Generate embedding from resume text
        embedding_text = self._build_embedding_text(parsed)
        embedding = self.embedding_engine.encode_single(embedding_text)

        # Store in FAISS
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
            skills=normalized_skills,
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
                if key not in ("candidate_id", "created_at"):
                    setattr(existing, key, value)
        else:
            self.db.add(candidate)
        self.db.commit()

        return {
            "candidate_id": candidate_id,
            "name": parsed.name,
            "email": parsed.email,
            "skills_count": len(normalized_skills),
            "experience_years": parsed.total_experience_years,
            "message": "Resume uploaded and parsed successfully",
        }

    def upload_resume_text(self, text: str, name: str = "", candidate_id: str = None) -> dict:
        """Process and store a resume from raw text."""
        candidate_id = candidate_id or str(uuid.uuid4())[:8]

        parsed = self.resume_parser.parse_from_text(text)
        if name:
            parsed.name = name

        normalized_skills = self.ontology.normalize_skills(parsed.skills)

        embedding_text = self._build_embedding_text(parsed)
        embedding = self.embedding_engine.encode_single(embedding_text)

        self.vector_store.add(
            np.array([embedding]),
            [{"candidate_id": candidate_id, "name": parsed.name, "email": parsed.email}],
        )
        self.vector_store.save()

        candidate = CandidateModel(
            id=candidate_id, name=parsed.name, email=parsed.email,
            phone=parsed.phone, skills=normalized_skills,
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
            "skills_count": len(normalized_skills),
            "experience_years": parsed.total_experience_years,
            "message": "Resume processed successfully",
        }

    def analyze_resume(self, candidate_id: str) -> dict:
        """Return detailed analysis of a parsed resume."""
        c = self.db.query(CandidateModel).filter_by(id=candidate_id).first()
        if not c:
            return None

        data = c.to_dict()
        skill_categories = {}
        for skill in data.get("skills", []):
            normalized = self.ontology.normalize_skill(skill)
            # Find which hierarchy category it belongs to
            from configs.skill_ontology import SKILL_HIERARCHY
            found = False
            for category, children in SKILL_HIERARCHY.items():
                if normalized in [ch.lower() for ch in children]:
                    cat_name = category.replace("_", " ").title()
                    skill_categories.setdefault(cat_name, []).append(skill)
                    found = True
                    break
            if not found:
                skill_categories.setdefault("Other", []).append(skill)

        domains = self.ontology.get_domain(data.get("raw_text", ""))

        return {
            **data,
            "skill_categories": skill_categories,
            "detected_domains": domains,
            "seniority_level": self.ontology.get_seniority_level(
                data.get("experience", [{}])[0].get("title", "") if data.get("experience") else ""
            ),
        }

    def parse_jd(self, jd_text: str) -> dict:
        """Parse and analyze a job description."""
        parsed = self.jd_parser.parse(jd_text)
        result = parsed.to_dict()

        # Classify requirement importance
        result["requirement_classification"] = {
            "critical": parsed.required_skills,
            "preferred": parsed.preferred_skills,
            "bonus": [],  # Could be extended
        }
        result["detected_domains"] = self.ontology.get_domain(jd_text)
        result["seniority_level"] = self.ontology.get_seniority_level(parsed.title)

        return result

    def rank_candidates(self, jd_text: str, candidate_ids: list[str] = None,
                        top_k: int = 10, weight_preset: str = None,
                        enable_bias_check: bool = True) -> dict:
        """
        Rank candidates using the multi-stage pipeline.
        
        Pipeline:
        1. Parse JD → extract requirements
        2. Stage 1: FAISS retrieval
        3. Stage 2: Multi-signal hybrid scoring
        4. Stage 3: Re-ranking + explanations + interview recs
        """
        # Parse JD
        jd_parsed = self.jd_parser.parse(jd_text)
        jd_data = jd_parsed.to_dict()
        logger.info(f"Ranking candidates for: {jd_parsed.title}")

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
                "pipeline_metrics": {},
                "timestamp": datetime.utcnow().isoformat(),
            }

        # Execute multi-stage pipeline
        pipeline_result = self.pipeline.execute(
            jd_parsed=jd_data,
            jd_text=jd_text,
            candidates_pool=candidates,
            top_k=top_k,
            retrieval_k=min(len(candidates), max(top_k * 3, 50)),
            weight_preset=weight_preset,
            enable_bias_check=enable_bias_check,
        )

        # Save ranking history
        history = RankingHistoryModel(
            jd_text=jd_text,
            jd_title=jd_parsed.title,
            results=pipeline_result.ranked_candidates,
            total_candidates=len(candidates),
        )
        self.db.add(history)
        self.db.commit()

        return pipeline_result.to_dict()

    def get_candidate(self, candidate_id: str) -> dict:
        """Get a single candidate's details."""
        c = self.db.query(CandidateModel).filter_by(id=candidate_id).first()
        return c.to_dict() if c else None

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

    def get_interview_recommendation(self, candidate_id: str, jd_text: str) -> dict:
        """Get interview recommendation for a specific candidate."""
        result = self.rank_candidates(jd_text, candidate_ids=[candidate_id], top_k=1)
        candidates = result.get("ranked_candidates", [])
        if candidates:
            return candidates[0].get("interview_recommendation", {})
        return None

    def get_dashboard_metrics(self) -> dict:
        """Get aggregated metrics for the dashboard."""
        total_candidates = self.db.query(CandidateModel).count()
        total_rankings = self.db.query(RankingHistoryModel).count()

        # Skill distribution
        all_skills = []
        for c in self.db.query(CandidateModel).all():
            all_skills.extend(c.skills or [])

        skill_counts = {}
        for s in all_skills:
            skill_counts[s] = skill_counts.get(s, 0) + 1

        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        # Experience distribution
        exp_ranges = {"0-2 years": 0, "2-5 years": 0, "5-10 years": 0, "10+ years": 0}
        for c in self.db.query(CandidateModel).all():
            yrs = c.total_experience_years or 0
            if yrs < 2:
                exp_ranges["0-2 years"] += 1
            elif yrs < 5:
                exp_ranges["2-5 years"] += 1
            elif yrs < 10:
                exp_ranges["5-10 years"] += 1
            else:
                exp_ranges["10+ years"] += 1

        return {
            "total_candidates": total_candidates,
            "total_rankings_performed": total_rankings,
            "faiss_vectors": self.vector_store.total_vectors,
            "top_skills": dict(top_skills),
            "experience_distribution": exp_ranges,
            "embedding_dimension": self.embedding_engine.get_embedding_dimension(),
        }

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
                writer.writerow([
                    "Rank", "Name", "Email", "Score",
                    "Decision", "Role Fit", "Focus Areas", "Recommendation",
                ])
                for c in ranking_results.get("ranked_candidates", []):
                    interview = c.get("interview_recommendation", {})
                    exp = c.get("score_breakdown", {}).get("explanation", {}).get("summary", {})
                    writer.writerow([
                        c["rank"], c["name"], c.get("email", ""),
                        c["final_score"],
                        interview.get("decision", ""),
                        interview.get("role_fit", ""),
                        "; ".join(interview.get("interview_focus_areas", [])[:2]),
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
