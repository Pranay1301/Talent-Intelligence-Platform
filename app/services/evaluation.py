"""
Model Evaluation Framework

Measurable KPIs for the ranking system:
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- Ranking latency
- Candidate relevance improvement vs keyword baseline

Includes benchmark comparisons: Traditional ATS vs Proposed System.
This is what interviewers want to see — measurable impact.
"""

import time
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation report for the ranking system."""
    precision_at_k: dict[int, float] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    mean_reciprocal_rank: float = 0.0
    avg_ranking_latency_ms: float = 0.0
    baseline_comparison: dict = field(default_factory=dict)
    total_candidates: int = 0
    total_relevant: int = 0

    def to_dict(self) -> dict:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mean_reciprocal_rank": round(self.mean_reciprocal_rank, 4),
            "avg_ranking_latency_ms": round(self.avg_ranking_latency_ms, 2),
            "baseline_comparison": self.baseline_comparison,
            "total_candidates": self.total_candidates,
            "total_relevant": self.total_relevant,
        }


class RankingEvaluator:
    """
    Evaluation framework for measuring ranking quality.
    
    Computes standard IR metrics and compares against a
    keyword-based baseline (traditional ATS approach).
    """

    def evaluate(self, ranked_candidates: list[dict],
                 relevant_ids: set[str],
                 k_values: list[int] = None) -> EvaluationResult:
        """
        Evaluate ranking quality against ground-truth relevant candidates.
        
        Args:
            ranked_candidates: List of ranked candidate dicts (must have 'candidate_id')
            relevant_ids: Set of candidate IDs considered relevant
            k_values: K values to compute metrics at
        """
        k_values = k_values or [1, 3, 5, 10]
        result = EvaluationResult(
            total_candidates=len(ranked_candidates),
            total_relevant=len(relevant_ids),
        )

        ranked_ids = [c.get("candidate_id", "") for c in ranked_candidates]

        for k in k_values:
            result.precision_at_k[k] = self._precision_at_k(ranked_ids, relevant_ids, k)
            result.recall_at_k[k] = self._recall_at_k(ranked_ids, relevant_ids, k)
            result.ndcg_at_k[k] = self._ndcg_at_k(ranked_ids, relevant_ids, k)

        result.mean_reciprocal_rank = self._mrr(ranked_ids, relevant_ids)

        return result

    def benchmark_against_keyword_baseline(self, candidates: list[dict],
                                            jd_text: str,
                                            relevant_ids: set[str],
                                            system_ranked: list[dict],
                                            k_values: list[int] = None) -> dict:
        """
        Compare the system's ranking against a keyword-based TF-IDF baseline.
        
        This demonstrates the improvement of semantic ranking
        over traditional ATS keyword matching.
        """
        k_values = k_values or [3, 5, 10]

        # Keyword baseline: rank by TF-IDF cosine similarity
        baseline_ranking = self._keyword_baseline_rank(candidates, jd_text)
        baseline_ids = [c["candidate_id"] for c in baseline_ranking]

        # System ranking
        system_ids = [c.get("candidate_id", "") for c in system_ranked]

        comparison = {"k_values": {}}
        for k in k_values:
            baseline_p = self._precision_at_k(baseline_ids, relevant_ids, k)
            system_p = self._precision_at_k(system_ids, relevant_ids, k)
            baseline_ndcg = self._ndcg_at_k(baseline_ids, relevant_ids, k)
            system_ndcg = self._ndcg_at_k(system_ids, relevant_ids, k)

            improvement_p = ((system_p - baseline_p) / max(baseline_p, 0.01)) * 100
            improvement_ndcg = ((system_ndcg - baseline_ndcg) / max(baseline_ndcg, 0.01)) * 100

            comparison["k_values"][k] = {
                "baseline_precision": round(baseline_p, 4),
                "system_precision": round(system_p, 4),
                "precision_improvement_pct": round(improvement_p, 1),
                "baseline_ndcg": round(baseline_ndcg, 4),
                "system_ndcg": round(system_ndcg, 4),
                "ndcg_improvement_pct": round(improvement_ndcg, 1),
            }

        # Overall summary
        avg_p_improvement = np.mean([
            v["precision_improvement_pct"] for v in comparison["k_values"].values()
        ])
        comparison["summary"] = {
            "avg_precision_improvement_pct": round(avg_p_improvement, 1),
            "verdict": (
                "Significant improvement over keyword baseline"
                if avg_p_improvement > 20
                else "Moderate improvement over keyword baseline"
                if avg_p_improvement > 5
                else "Comparable to keyword baseline"
            ),
        }

        return comparison

    def measure_latency(self, pipeline_fn, jd_text: str,
                        candidates: list[dict], n_runs: int = 5) -> dict:
        """
        Measure ranking latency over multiple runs.
        
        Args:
            pipeline_fn: Function that takes (jd_text, candidates) and returns ranked results
            jd_text: Job description text
            candidates: Candidate pool
            n_runs: Number of runs for averaging
        """
        latencies = []
        for _ in range(n_runs):
            start = time.time()
            pipeline_fn(jd_text, candidates)
            elapsed_ms = (time.time() - start) * 1000
            latencies.append(elapsed_ms)

        return {
            "avg_ms": round(np.mean(latencies), 2),
            "p50_ms": round(np.percentile(latencies, 50), 2),
            "p95_ms": round(np.percentile(latencies, 95), 2),
            "p99_ms": round(np.percentile(latencies, 99), 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "n_runs": n_runs,
            "n_candidates": len(candidates),
        }

    def _keyword_baseline_rank(self, candidates: list[dict], jd_text: str) -> list[dict]:
        """Rank candidates using TF-IDF keyword matching (traditional ATS)."""
        if not candidates:
            return []

        texts = []
        for c in candidates:
            parts = [c.get("summary", "")]
            parts.append(" ".join(c.get("skills", [])))
            for exp in c.get("experience", []):
                parts.extend(exp.get("responsibilities", []))
            texts.append(" ".join(filter(None, parts)))

        try:
            tfidf = TfidfVectorizer(max_features=3000, stop_words="english")
            all_texts = texts + [jd_text]
            tfidf_matrix = tfidf.fit_transform(all_texts)

            jd_vec = tfidf_matrix[-1:]
            candidate_vecs = tfidf_matrix[:-1]
            similarities = cosine_similarity(jd_vec, candidate_vecs)[0]

            indexed = list(enumerate(similarities))
            indexed.sort(key=lambda x: x[1], reverse=True)

            return [candidates[i] for i, _ in indexed]
        except Exception as e:
            logger.warning(f"Baseline ranking failed: {e}")
            return candidates

    # --- IR Metrics ---

    def _precision_at_k(self, ranked_ids: list[str], relevant: set[str], k: int) -> float:
        """Precision@K: fraction of top-K results that are relevant."""
        top_k = ranked_ids[:k]
        if not top_k:
            return 0.0
        relevant_in_k = sum(1 for cid in top_k if cid in relevant)
        return relevant_in_k / k

    def _recall_at_k(self, ranked_ids: list[str], relevant: set[str], k: int) -> float:
        """Recall@K: fraction of relevant items found in top-K."""
        if not relevant:
            return 0.0
        top_k = ranked_ids[:k]
        relevant_in_k = sum(1 for cid in top_k if cid in relevant)
        return relevant_in_k / len(relevant)

    def _ndcg_at_k(self, ranked_ids: list[str], relevant: set[str], k: int) -> float:
        """NDCG@K: Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, cid in enumerate(ranked_ids[:k]):
            rel = 1.0 if cid in relevant else 0.0
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG
        ideal_rels = sorted([1.0 if cid in relevant else 0.0 for cid in ranked_ids[:k]], reverse=True)
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    def _mrr(self, ranked_ids: list[str], relevant: set[str]) -> float:
        """Mean Reciprocal Rank: 1/rank of first relevant result."""
        for i, cid in enumerate(ranked_ids):
            if cid in relevant:
                return 1.0 / (i + 1)
        return 0.0
