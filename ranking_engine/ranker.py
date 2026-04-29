"""
Candidate Ranking Engine - Hybrid scoring combining multiple signals.

Final Score = Embedding Similarity + Skill Match + Experience Score + Education Score + Certifications Bonus

Includes Explainable AI layer showing WHY candidates ranked high/low.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a candidate's ranking score."""
    embedding_similarity: float = 0.0
    skill_match_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    certification_bonus: float = 0.0
    final_score: float = 0.0

    # Explainability fields
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    experience_match: str = ""
    education_match: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "scores": {
                "embedding_similarity": round(self.embedding_similarity, 4),
                "skill_match_score": round(self.skill_match_score, 4),
                "experience_score": round(self.experience_score, 4),
                "education_score": round(self.education_score, 4),
                "certification_bonus": round(self.certification_bonus, 4),
                "final_score": round(self.final_score, 4),
            },
            "explanation": {
                "matched_skills": self.matched_skills,
                "missing_skills": self.missing_skills,
                "extra_skills": self.extra_skills,
                "experience_match": self.experience_match,
                "education_match": self.education_match,
                "strengths": self.strengths,
                "weaknesses": self.weaknesses,
                "recommendation": self.recommendation,
            },
        }


@dataclass
class RankedCandidate:
    """A candidate with their ranking information."""
    candidate_id: str = ""
    name: str = ""
    email: str = ""
    rank: int = 0
    score_breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    resume_data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "name": self.name,
            "email": self.email,
            "rank": self.rank,
            "final_score": round(self.score_breakdown.final_score, 4),
            "score_breakdown": self.score_breakdown.to_dict(),
        }


class CandidateRanker:
    """
    Hybrid ranking engine that combines multiple scoring signals
    to produce a comprehensive candidate ranking with explanations.
    
    Scoring Weights (configurable):
    - Embedding Similarity: 35% (semantic relevance)
    - Skill Match: 30% (explicit skill overlap)
    - Experience: 20% (years and relevance)
    - Education: 10% (degree match)
    - Certifications: 5% (bonus)
    """

    DEFAULT_WEIGHTS = {
        "embedding_similarity": 0.35,
        "skill_match": 0.30,
        "experience": 0.20,
        "education": 0.10,
        "certifications": 0.05,
    }

    EDUCATION_LEVELS = {
        "phd": 4, "doctorate": 4,
        "master": 3, "mba": 3, "m.s": 3, "m.tech": 3,
        "bachelor": 2, "b.s": 2, "b.tech": 2, "b.e": 2,
        "associate": 1, "diploma": 1,
    }

    def __init__(self, weights: dict = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

    def rank_candidates(self, candidates: list[dict], jd_data: dict,
                        similarity_scores: list[float] = None) -> list[RankedCandidate]:
        """
        Rank a list of candidates against a job description.
        
        Args:
            candidates: List of parsed resume dicts
            jd_data: Parsed job description dict
            similarity_scores: Pre-computed embedding similarity scores
            
        Returns:
            Sorted list of RankedCandidate objects (highest first)
        """
        ranked = []

        for i, candidate in enumerate(candidates):
            emb_sim = similarity_scores[i] if similarity_scores and i < len(similarity_scores) else 0.0
            breakdown = self._compute_score(candidate, jd_data, emb_sim)
            
            rc = RankedCandidate(
                candidate_id=candidate.get("candidate_id", f"candidate_{i+1}"),
                name=candidate.get("name", "Unknown"),
                email=candidate.get("email", ""),
                score_breakdown=breakdown,
                resume_data=candidate,
            )
            ranked.append(rc)

        # Sort by final score descending
        ranked.sort(key=lambda x: x.score_breakdown.final_score, reverse=True)

        # Assign ranks
        for i, r in enumerate(ranked):
            r.rank = i + 1

        return ranked

    def _compute_score(self, candidate: dict, jd: dict, embedding_similarity: float) -> ScoreBreakdown:
        """Compute comprehensive score with explainability."""
        breakdown = ScoreBreakdown()

        # 1. Embedding Similarity Score (pre-computed)
        breakdown.embedding_similarity = max(0, min(1, embedding_similarity))

        # 2. Skill Match Score
        self._compute_skill_score(breakdown, candidate, jd)

        # 3. Experience Score
        self._compute_experience_score(breakdown, candidate, jd)

        # 4. Education Score
        self._compute_education_score(breakdown, candidate, jd)

        # 5. Certification Bonus
        self._compute_certification_bonus(breakdown, candidate)

        # Compute weighted final score
        breakdown.final_score = (
            self.weights["embedding_similarity"] * breakdown.embedding_similarity +
            self.weights["skill_match"] * breakdown.skill_match_score +
            self.weights["experience"] * breakdown.experience_score +
            self.weights["education"] * breakdown.education_score +
            self.weights["certifications"] * breakdown.certification_bonus
        )

        # Generate recommendation
        self._generate_explanation(breakdown)

        return breakdown

    def _compute_skill_score(self, breakdown: ScoreBreakdown, candidate: dict, jd: dict):
        """Compute skill match score and identify matched/missing skills."""
        candidate_skills = set(s.lower() for s in candidate.get("skills", []))
        required_skills = set(s.lower() for s in jd.get("required_skills", []))
        preferred_skills = set(s.lower() for s in jd.get("preferred_skills", []))
        all_jd_skills = required_skills | preferred_skills

        matched = candidate_skills & all_jd_skills
        missing = all_jd_skills - candidate_skills
        extra = candidate_skills - all_jd_skills

        breakdown.matched_skills = sorted([s.title() for s in matched])
        breakdown.missing_skills = sorted([s.title() for s in missing])
        breakdown.extra_skills = sorted([s.title() for s in extra])[:10]

        if all_jd_skills:
            # Weighted: required skills count more
            req_matched = len(candidate_skills & required_skills)
            pref_matched = len(candidate_skills & preferred_skills)
            req_total = max(len(required_skills), 1)
            pref_total = max(len(preferred_skills), 1)

            score = 0.7 * (req_matched / req_total) + 0.3 * (pref_matched / pref_total)
            breakdown.skill_match_score = min(1.0, score)
        else:
            breakdown.skill_match_score = 0.5  # Neutral if no JD skills specified

    def _compute_experience_score(self, breakdown: ScoreBreakdown, candidate: dict, jd: dict):
        """Compute experience match score."""
        candidate_years = candidate.get("total_experience_years", 0)
        min_years = jd.get("min_experience_years", 0)
        max_years = jd.get("max_experience_years", 0)

        if min_years == 0 and max_years == 0:
            breakdown.experience_score = 0.5
            breakdown.experience_match = "No experience requirement specified"
            return

        if candidate_years >= min_years:
            if max_years > 0 and candidate_years <= max_years:
                breakdown.experience_score = 1.0
            elif max_years > 0 and candidate_years > max_years:
                # Slightly over-qualified
                over = candidate_years - max_years
                breakdown.experience_score = max(0.6, 1.0 - over * 0.05)
            else:
                breakdown.experience_score = 1.0
        else:
            # Under-qualified
            ratio = candidate_years / max(min_years, 1)
            breakdown.experience_score = min(0.8, ratio)

        breakdown.experience_match = f"{candidate_years:.1f}/{min_years:.0f}-{max_years:.0f} years"

    def _compute_education_score(self, breakdown: ScoreBreakdown, candidate: dict, jd: dict):
        """Compute education match score."""
        candidate_edu = candidate.get("education", [])
        jd_edu_reqs = jd.get("education_requirements", [])

        if not jd_edu_reqs:
            breakdown.education_score = 0.5
            breakdown.education_match = "No specific education requirement"
            return

        # Find candidate's highest education level
        max_level = 0
        for edu in candidate_edu:
            degree = edu.get("degree", "").lower()
            for keyword, level in self.EDUCATION_LEVELS.items():
                if keyword in degree:
                    max_level = max(max_level, level)

        # Find required education level
        req_level = 0
        for req in jd_edu_reqs:
            req_lower = req.lower()
            for keyword, level in self.EDUCATION_LEVELS.items():
                if keyword in req_lower:
                    req_level = max(req_level, level)

        if req_level == 0:
            breakdown.education_score = 0.5
        elif max_level >= req_level:
            breakdown.education_score = 1.0
        else:
            breakdown.education_score = max(0.3, max_level / req_level)

        level_names = {0: "None", 1: "Associate/Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}
        breakdown.education_match = f"Candidate: {level_names.get(max_level, 'Unknown')} | Required: {level_names.get(req_level, 'Unknown')}"

    def _compute_certification_bonus(self, breakdown: ScoreBreakdown, candidate: dict):
        """Compute certification bonus score."""
        certs = candidate.get("certifications", [])
        if not certs:
            breakdown.certification_bonus = 0.0
        elif len(certs) >= 3:
            breakdown.certification_bonus = 1.0
        elif len(certs) == 2:
            breakdown.certification_bonus = 0.7
        else:
            breakdown.certification_bonus = 0.4

    def _generate_explanation(self, breakdown: ScoreBreakdown):
        """Generate human-readable explanation of the ranking."""
        # Strengths
        if breakdown.embedding_similarity > 0.7:
            breakdown.strengths.append("Strong semantic match with job description")
        if breakdown.skill_match_score > 0.7:
            breakdown.strengths.append(f"Excellent skill coverage ({len(breakdown.matched_skills)} skills matched)")
        if breakdown.experience_score > 0.8:
            breakdown.strengths.append("Experience level well-matched")
        if breakdown.education_score >= 1.0:
            breakdown.strengths.append("Education meets/exceeds requirements")
        if breakdown.certification_bonus > 0.5:
            breakdown.strengths.append("Has relevant certifications")
        if breakdown.extra_skills:
            breakdown.strengths.append(f"Additional skills: {', '.join(breakdown.extra_skills[:5])}")

        # Weaknesses
        if breakdown.embedding_similarity < 0.4:
            breakdown.weaknesses.append("Low semantic relevance to job description")
        if breakdown.missing_skills:
            breakdown.weaknesses.append(f"Missing skills: {', '.join(breakdown.missing_skills[:5])}")
        if breakdown.experience_score < 0.5:
            breakdown.weaknesses.append("Insufficient experience")
        if breakdown.education_score < 0.5:
            breakdown.weaknesses.append("Education below requirements")

        # Recommendation
        score = breakdown.final_score
        if score >= 0.8:
            breakdown.recommendation = "STRONG MATCH - Highly recommended for interview"
        elif score >= 0.6:
            breakdown.recommendation = "GOOD MATCH - Recommended for consideration"
        elif score >= 0.4:
            breakdown.recommendation = "MODERATE MATCH - May be suitable with additional screening"
        else:
            breakdown.recommendation = "WEAK MATCH - Does not meet key requirements"
