"""
Multi-Stage Screening Pipeline

Production-grade RAG-style retrieval and ranking pipeline:

Stage 1: Candidate Retrieval
  → FAISS nearest-neighbor retrieval (fast, broad)
  → Retrieves top-N candidates from large pool

Stage 2: Hybrid Scoring
  → Multi-signal scoring (embedding + skills + experience + education + projects + leadership + domain)
  → Ontology-aware skill matching
  → Bias-aware scoring

Stage 3: Re-ranking & Explanation
  → Contextual re-ranking with detailed analysis
  → Explainable AI layer
  → Interview recommendations
  → Fairness audit

This multi-stage approach mirrors production ML ranking systems
(similar to Google Search, LinkedIn Recruiter, Amazon hiring tools).
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field

from configs.skill_ontology import SkillOntology
from configs.scoring_weights import DEFAULT_WEIGHTS, WEIGHT_PRESETS
from bias_mitigation.fairness import BiasMitigator
from interview_engine.recommender import InterviewRecommender
from explainability.explainer import RankingExplainer

logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    stage1_retrieval_time_ms: float = 0
    stage2_scoring_time_ms: float = 0
    stage3_reranking_time_ms: float = 0
    total_time_ms: float = 0
    candidates_retrieved: int = 0
    candidates_scored: int = 0
    candidates_returned: int = 0

    def to_dict(self) -> dict:
        return {
            "stage1_retrieval_ms": round(self.stage1_retrieval_time_ms, 2),
            "stage2_scoring_ms": round(self.stage2_scoring_time_ms, 2),
            "stage3_reranking_ms": round(self.stage3_reranking_time_ms, 2),
            "total_ms": round(self.total_time_ms, 2),
            "candidates_retrieved": self.candidates_retrieved,
            "candidates_scored": self.candidates_scored,
            "candidates_returned": self.candidates_returned,
        }


@dataclass
class PipelineResult:
    """Complete result from the screening pipeline."""
    job_title: str = ""
    ranked_candidates: list[dict] = field(default_factory=list)
    total_candidates_in_pool: int = 0
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    fairness_audit: dict = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "job_title": self.job_title,
            "ranked_candidates": self.ranked_candidates,
            "total_candidates_in_pool": self.total_candidates_in_pool,
            "pipeline_metrics": self.metrics.to_dict(),
            "fairness_audit": self.fairness_audit,
            "timestamp": self.timestamp,
        }


class ScreeningPipeline:
    """
    Multi-stage candidate screening and ranking pipeline.
    
    Mirrors production ML ranking architectures:
    - Stage 1 (Retrieval): Fast, broad candidate retrieval via FAISS
    - Stage 2 (Scoring): Detailed multi-signal hybrid scoring
    - Stage 3 (Re-ranking): Contextual re-ranking with explanations
    """

    def __init__(self, embedding_engine, vector_store, ranker):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.ranker = ranker
        self.ontology = SkillOntology()
        self.bias_mitigator = BiasMitigator()
        self.interview_engine = InterviewRecommender()
        self.explainer = RankingExplainer()

    def execute(self, jd_parsed: dict, jd_text: str,
                candidates_pool: list[dict],
                top_k: int = 10,
                retrieval_k: int = 50,
                weight_preset: str = None,
                enable_bias_check: bool = True) -> PipelineResult:
        """
        Execute the full multi-stage screening pipeline.
        
        Args:
            jd_parsed: Parsed JD data
            jd_text: Raw JD text
            candidates_pool: All available candidates
            top_k: Final number of candidates to return
            retrieval_k: Number to retrieve in Stage 1
            weight_preset: Optional scoring weight preset name
            enable_bias_check: Whether to run bias audit
        """
        from datetime import datetime
        result = PipelineResult(
            job_title=jd_parsed.get("title", ""),
            total_candidates_in_pool=len(candidates_pool),
            timestamp=datetime.utcnow().isoformat(),
        )
        metrics = result.metrics
        pipeline_start = time.time()

        # Select weights
        weights = WEIGHT_PRESETS.get(weight_preset, DEFAULT_WEIGHTS)

        # --- STAGE 1: Retrieval ---
        stage1_start = time.time()
        retrieved_candidates, similarity_scores = self._stage1_retrieval(
            jd_text, candidates_pool, retrieval_k
        )
        metrics.stage1_retrieval_time_ms = (time.time() - stage1_start) * 1000
        metrics.candidates_retrieved = len(retrieved_candidates)
        logger.info(f"Stage 1: Retrieved {len(retrieved_candidates)} candidates in {metrics.stage1_retrieval_time_ms:.0f}ms")

        if not retrieved_candidates:
            return result

        # --- STAGE 2: Hybrid Scoring ---
        stage2_start = time.time()
        scored_candidates = self._stage2_scoring(
            retrieved_candidates, jd_parsed, similarity_scores, weights
        )
        metrics.stage2_scoring_time_ms = (time.time() - stage2_start) * 1000
        metrics.candidates_scored = len(scored_candidates)
        logger.info(f"Stage 2: Scored {len(scored_candidates)} candidates in {metrics.stage2_scoring_time_ms:.0f}ms")

        # --- STAGE 3: Re-ranking & Explanation ---
        stage3_start = time.time()
        final_candidates = self._stage3_reranking(
            scored_candidates, jd_parsed, top_k, enable_bias_check
        )
        metrics.stage3_reranking_time_ms = (time.time() - stage3_start) * 1000
        metrics.candidates_returned = len(final_candidates)
        logger.info(f"Stage 3: Re-ranked to top {len(final_candidates)} in {metrics.stage3_reranking_time_ms:.0f}ms")

        result.ranked_candidates = final_candidates
        metrics.total_time_ms = (time.time() - pipeline_start) * 1000

        # Fairness audit
        if enable_bias_check:
            result.fairness_audit = self.bias_mitigator.audit_ranking(final_candidates)

        return result

    def _stage1_retrieval(self, jd_text: str, candidates: list[dict],
                          retrieval_k: int) -> tuple[list[dict], list[float]]:
        """
        Stage 1: Fast retrieval using FAISS vector similarity.
        
        If FAISS index has vectors, use it for fast retrieval.
        Otherwise, fall back to computing similarities directly.
        """
        jd_embedding = self.embedding_engine.encode_single(jd_text)

        # Try FAISS retrieval first
        if self.vector_store.total_vectors > 0:
            faiss_results = self.vector_store.search(jd_embedding, top_k=retrieval_k)
            
            # Map FAISS results back to candidate data
            faiss_ids = {r["candidate_id"] for r in faiss_results}
            faiss_scores = {r["candidate_id"]: r["similarity_score"] for r in faiss_results}
            
            retrieved = []
            scores = []
            for c in candidates:
                cid = c.get("candidate_id", "")
                if cid in faiss_ids:
                    retrieved.append(c)
                    scores.append(faiss_scores.get(cid, 0))

            if retrieved:
                return retrieved, scores

        # Fallback: compute similarities directly
        if not candidates:
            return [], []

        candidate_texts = []
        for c in candidates:
            text_parts = [c.get("summary", "")]
            text_parts.append("Skills: " + ", ".join(c.get("skills", [])))
            for exp in c.get("experience", []):
                text_parts.append(exp.get("title", ""))
                text_parts.extend(exp.get("responsibilities", [])[:2])
            candidate_texts.append(" ".join(filter(None, text_parts)))

        # Encode all candidates
        candidate_embeddings = self.embedding_engine.encode(candidate_texts)
        
        if len(candidate_embeddings) == 0:
            return candidates[:retrieval_k], [0.5] * min(len(candidates), retrieval_k)

        # Compute similarities
        jd_emb = jd_embedding.reshape(1, -1)
        sim_matrix = self.embedding_engine.compute_similarity_matrix(jd_emb, candidate_embeddings)
        scores = sim_matrix[0].tolist()

        # Sort by similarity and take top-K
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_indices = indexed[:retrieval_k]

        retrieved = [candidates[i] for i, _ in top_indices]
        ret_scores = [s for _, s in top_indices]

        return retrieved, ret_scores

    def _stage2_scoring(self, candidates: list[dict], jd: dict,
                        similarity_scores: list[float],
                        weights: dict) -> list[dict]:
        """
        Stage 2: Multi-signal hybrid scoring with ontology-aware matching.
        """
        scored = []

        for i, candidate in enumerate(candidates):
            emb_sim = similarity_scores[i] if i < len(similarity_scores) else 0.0

            # Ontology-aware skill matching
            skill_overlap = self.ontology.compute_skill_overlap(
                candidate.get("skills", []),
                jd.get("required_skills", []) + jd.get("preferred_skills", [])
            )

            # Compute sub-scores
            scores = self._compute_detailed_scores(candidate, jd, emb_sim, skill_overlap)

            # Weighted final score
            final_score = sum(
                weights.get(key, 0) * scores.get(key, 0)
                for key in weights
            )
            scores["final_score"] = min(1.0, final_score)

            scored.append({
                **candidate,
                "_scores": scores,
                "_skill_overlap": skill_overlap,
                "_embedding_similarity": emb_sim,
            })

        # Sort by final score
        scored.sort(key=lambda x: x["_scores"]["final_score"], reverse=True)
        return scored

    def _compute_detailed_scores(self, candidate: dict, jd: dict,
                                  emb_sim: float, skill_overlap: dict) -> dict:
        """Compute all scoring dimensions."""
        scores = {
            "embedding_similarity": max(0, min(1, emb_sim)),
            "skill_match": skill_overlap.get("match_ratio", 0),
        }

        # Experience relevance
        cand_years = candidate.get("total_experience_years", 0)
        min_years = jd.get("min_experience_years", 0)
        max_years = jd.get("max_experience_years", 0)
        if min_years > 0:
            if cand_years >= min_years:
                if max_years > 0 and cand_years <= max_years:
                    scores["experience_relevance"] = 1.0
                elif max_years > 0:
                    scores["experience_relevance"] = max(0.6, 1.0 - (cand_years - max_years) * 0.05)
                else:
                    scores["experience_relevance"] = 1.0
            else:
                scores["experience_relevance"] = min(0.8, cand_years / max(min_years, 1))
        else:
            scores["experience_relevance"] = 0.5

        # Education relevance
        edu_levels = {"phd": 4, "doctorate": 4, "master": 3, "mba": 3,
                      "bachelor": 2, "b.tech": 2, "b.s": 2, "associate": 1, "diploma": 1}
        max_level = 0
        for edu in candidate.get("education", []):
            for kw, lvl in edu_levels.items():
                if kw in edu.get("degree", "").lower():
                    max_level = max(max_level, lvl)
        req_level = 0
        for req in jd.get("education_requirements", []):
            for kw, lvl in edu_levels.items():
                if kw in req.lower():
                    req_level = max(req_level, lvl)
        scores["education_relevance"] = 1.0 if max_level >= req_level else max(0.3, max_level / max(req_level, 1))

        # Certification bonus
        certs = candidate.get("certifications", [])
        scores["certification_bonus"] = min(1.0, len(certs) * 0.35) if certs else 0.0

        # Project impact score
        projects = candidate.get("projects", [])
        jd_text = jd.get("raw_text", "").lower()
        relevant_projects = 0
        for proj in projects:
            proj_text = f"{proj.get('name', '')} {' '.join(proj.get('details', []))}".lower()
            if any(skill.lower() in proj_text for skill in jd.get("required_skills", [])[:5]):
                relevant_projects += 1
        scores["project_impact"] = min(1.0, relevant_projects * 0.4) if projects else 0.2

        # Leadership score
        has_leadership = any(
            any(kw in exp.get("title", "").lower()
                for kw in ["lead", "senior", "manager", "director", "head", "principal"])
            for exp in candidate.get("experience", [])
        )
        scores["leadership_score"] = 0.8 if has_leadership else 0.2

        # Domain alignment
        cand_domains = self.ontology.get_domain(candidate.get("raw_text", ""))
        jd_domains = self.ontology.get_domain(jd.get("raw_text", ""))
        if jd_domains and cand_domains:
            overlap = len(set(cand_domains) & set(jd_domains))
            scores["domain_alignment"] = min(1.0, overlap / max(len(jd_domains), 1))
        else:
            scores["domain_alignment"] = 0.5

        # Seniority match
        cand_seniority = self.ontology.get_seniority_level(
            candidate.get("experience", [{}])[0].get("title", "") if candidate.get("experience") else ""
        )
        jd_seniority = self.ontology.get_seniority_level(jd.get("title", ""))
        seniority_diff = abs(cand_seniority - jd_seniority)
        scores["seniority_match"] = max(0, 1.0 - seniority_diff * 0.25)

        return scores

    def _stage3_reranking(self, scored_candidates: list[dict], jd: dict,
                          top_k: int, enable_bias_check: bool) -> list[dict]:
        """
        Stage 3: Re-ranking with explanations, interview recommendations,
        and bias checks.
        """
        final = []

        for rank, candidate in enumerate(scored_candidates[:top_k], 1):
            scores = candidate.pop("_scores", {})
            skill_overlap = candidate.pop("_skill_overlap", {})
            emb_sim = candidate.pop("_embedding_similarity", 0)

            # Generate detailed explanation
            explanation = self.explainer.explain(scores, candidate, jd, skill_overlap)

            # Generate interview recommendation
            score_breakdown_for_interview = {
                "scores": scores,
                "explanation": explanation.to_dict().get("summary", {}),
            }
            interview_rec = self.interview_engine.recommend(
                score_breakdown_for_interview, candidate, jd
            )

            # Bias check on individual candidate
            fairness_score = 1.0
            if enable_bias_check:
                _, fairness_report = self.bias_mitigator.compute_fairness_adjusted_score(
                    scores.get("final_score", 0), candidate
                )
                fairness_score = fairness_report.fairness_score

            final.append({
                "rank": rank,
                "candidate_id": candidate.get("candidate_id", ""),
                "name": candidate.get("name", ""),
                "email": candidate.get("email", ""),
                "final_score": round(scores.get("final_score", 0), 4),
                "score_breakdown": {
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "explanation": explanation.to_dict(),
                },
                "interview_recommendation": interview_rec.to_dict(),
                "fairness_score": round(fairness_score, 4),
                "resume_data": {
                    k: v for k, v in candidate.items()
                    if k not in ("raw_text",)  # Exclude raw text from response
                },
            })

        return final
