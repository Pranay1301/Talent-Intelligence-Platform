"""
Interview Recommendation Engine

Generates intelligent hiring recommendations:
- Strong Hire / Interview Recommended / Hold for Review / Weak Match / Reject
- Suggested interview focus areas based on skill gaps
- Role fit analysis and promotion potential assessment

Produces actionable recruiter-ready insights, not just scores.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class InterviewRecommendation:
    """Structured interview recommendation for a candidate."""
    decision: str = ""  # STRONG_HIRE, INTERVIEW, HOLD, WEAK, REJECT
    confidence: float = 0.0
    interview_focus_areas: list[str] = field(default_factory=list)
    role_fit: str = ""  # HIGH, MEDIUM, LOW
    promotion_potential: str = ""  # HIGH, MEDIUM, LOW
    suggested_round: str = ""  # TECHNICAL, SYSTEM_DESIGN, BEHAVIORAL, CULTURE_FIT
    risk_factors: list[str] = field(default_factory=list)
    hiring_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "interview_focus_areas": self.interview_focus_areas,
            "role_fit": self.role_fit,
            "promotion_potential": self.promotion_potential,
            "suggested_round": self.suggested_round,
            "risk_factors": self.risk_factors,
            "hiring_notes": self.hiring_notes,
        }


class InterviewRecommender:
    """
    Generates hiring recommendations and interview strategy
    based on candidate-job fit analysis.
    
    Goes beyond simple score thresholds — analyzes the nature of
    skill gaps and experience patterns to recommend specific
    interview focus areas.
    """

    # Decision thresholds
    THRESHOLDS = {
        "strong_hire": 0.80,
        "interview": 0.60,
        "hold": 0.40,
        "weak": 0.25,
    }

    # Skill categories for interview focus
    SKILL_CATEGORIES = {
        "core_ml": {"machine learning", "deep learning", "nlp", "computer vision",
                     "tensorflow", "pytorch", "scikit-learn", "transformers"},
        "system_design": {"docker", "kubernetes", "microservices", "aws", "gcp",
                          "azure", "distributed systems", "system design", "scalable"},
        "data_engineering": {"sql", "spark", "airflow", "kafka", "etl",
                             "data pipeline", "hadoop", "snowflake"},
        "mlops": {"mlops", "ci/cd", "model deployment", "monitoring",
                  "experiment tracking", "mlflow", "feature store"},
        "programming": {"python", "java", "c++", "golang", "rust",
                        "javascript", "typescript", "algorithms"},
        "leadership": {"leadership", "management", "mentoring", "cross-functional",
                       "stakeholder", "strategy", "roadmap"},
    }

    def recommend(self, score_breakdown: dict, candidate_data: dict,
                  jd_data: dict) -> InterviewRecommendation:
        """
        Generate interview recommendation for a candidate.
        
        Args:
            score_breakdown: Score breakdown dict with scores and explanation
            candidate_data: Parsed candidate data
            jd_data: Parsed JD data
            
        Returns:
            InterviewRecommendation with decision and focus areas
        """
        rec = InterviewRecommendation()
        scores = score_breakdown.get("scores", {})
        explanation = score_breakdown.get("explanation", {})
        final_score = scores.get("final_score", 0)

        # 1. Determine decision
        rec.decision, rec.confidence = self._determine_decision(final_score, scores)

        # 2. Analyze role fit
        rec.role_fit = self._assess_role_fit(scores, explanation)

        # 3. Determine interview focus areas
        rec.interview_focus_areas = self._identify_focus_areas(
            explanation, candidate_data, jd_data
        )

        # 4. Assess promotion potential
        rec.promotion_potential = self._assess_promotion_potential(
            candidate_data, jd_data
        )

        # 5. Suggest interview round type
        rec.suggested_round = self._suggest_round(explanation, rec.interview_focus_areas)

        # 6. Identify risk factors
        rec.risk_factors = self._identify_risks(scores, explanation, candidate_data)

        # 7. Generate hiring notes
        rec.hiring_notes = self._generate_notes(rec, explanation, candidate_data)

        return rec

    def _determine_decision(self, final_score: float, scores: dict) -> tuple[str, float]:
        """Determine hiring decision with confidence level."""
        if final_score >= self.THRESHOLDS["strong_hire"]:
            # High confidence if all sub-scores are balanced
            sub_scores = [
                scores.get("embedding_similarity", 0),
                scores.get("skill_match_score", 0),
                scores.get("experience_score", 0),
            ]
            balance = 1.0 - (max(sub_scores) - min(sub_scores)) if sub_scores else 0.5
            confidence = min(0.95, final_score * (0.7 + 0.3 * balance))
            return "STRONG_HIRE", confidence

        elif final_score >= self.THRESHOLDS["interview"]:
            return "INTERVIEW_RECOMMENDED", min(0.85, final_score * 1.1)

        elif final_score >= self.THRESHOLDS["hold"]:
            return "HOLD_FOR_REVIEW", 0.6

        elif final_score >= self.THRESHOLDS["weak"]:
            return "WEAK_MATCH", 0.7

        else:
            return "REJECT", 0.8

    def _assess_role_fit(self, scores: dict, explanation: dict) -> str:
        """Assess overall role fit."""
        semantic = scores.get("embedding_similarity", 0)
        skills = scores.get("skill_match_score", 0)
        experience = scores.get("experience_score", 0)

        avg_fit = (semantic * 0.4 + skills * 0.35 + experience * 0.25)

        if avg_fit >= 0.75:
            return "HIGH"
        elif avg_fit >= 0.50:
            return "MEDIUM"
        return "LOW"

    def _identify_focus_areas(self, explanation: dict, candidate: dict,
                               jd: dict) -> list[str]:
        """Identify specific areas the interview should focus on."""
        focus_areas = []
        missing_skills = set(s.lower() for s in explanation.get("missing_skills", []))

        # Check which categories have gaps
        for category, skills in self.SKILL_CATEGORIES.items():
            overlap = missing_skills & skills
            if overlap:
                category_name = category.replace("_", " ").title()
                focus_areas.append(
                    f"{category_name}: Assess proficiency in {', '.join(sorted(overlap)[:3])}"
                )

        # Experience gap focus
        exp_match = explanation.get("experience_match", "")
        if "insufficient" in exp_match.lower() or explanation.get("experience_score", 1) < 0.5:
            focus_areas.append(
                "Experience Depth: Probe for hands-on production experience and project complexity"
            )

        # If strong semantic but weak skills, focus on adaptability
        semantic = explanation.get("embedding_similarity", 0) if isinstance(explanation.get("embedding_similarity"), (int, float)) else 0
        if semantic > 0.7 and len(missing_skills) > 3:
            focus_areas.append(
                "Learning Agility: Candidate has relevant context but gaps in specific tools — assess adaptability"
            )

        # If no specific gaps found, recommend general assessment
        if not focus_areas:
            focus_areas.append("General Technical Assessment: Well-rounded candidate")
            focus_areas.append("Culture Fit & Motivation: Assess alignment with team values")

        return focus_areas

    def _assess_promotion_potential(self, candidate: dict, jd: dict) -> str:
        """Assess if candidate has growth potential beyond the role."""
        exp_years = candidate.get("total_experience_years", 0)
        skills_count = len(candidate.get("skills", []))
        has_leadership = any(
            "lead" in exp.get("title", "").lower() or
            "senior" in exp.get("title", "").lower() or
            "manager" in exp.get("title", "").lower()
            for exp in candidate.get("experience", [])
        )
        has_publications = len(candidate.get("publications", [])) > 0
        has_certs = len(candidate.get("certifications", [])) >= 2

        score = 0
        if exp_years >= 5:
            score += 2
        elif exp_years >= 3:
            score += 1
        if skills_count >= 15:
            score += 1
        if has_leadership:
            score += 2
        if has_publications:
            score += 1
        if has_certs:
            score += 1

        if score >= 5:
            return "HIGH"
        elif score >= 3:
            return "MEDIUM"
        return "LOW"

    def _suggest_round(self, explanation: dict, focus_areas: list[str]) -> str:
        """Suggest the most appropriate interview round type."""
        focus_text = " ".join(focus_areas).lower()

        if "system design" in focus_text or "scalable" in focus_text:
            return "SYSTEM_DESIGN"
        elif "leadership" in focus_text or "management" in focus_text:
            return "BEHAVIORAL_LEADERSHIP"
        elif "culture" in focus_text:
            return "CULTURE_FIT"
        else:
            return "TECHNICAL_DEEP_DIVE"

    def _identify_risks(self, scores: dict, explanation: dict,
                        candidate: dict) -> list[str]:
        """Identify hiring risk factors."""
        risks = []

        # Overqualification risk
        exp_score = scores.get("experience_score", 0)
        if exp_score < 0.7 and candidate.get("total_experience_years", 0) > 10:
            risks.append("Potential overqualification — may seek higher role quickly")

        # Skill gap risk
        missing = explanation.get("missing_skills", [])
        if len(missing) > 5:
            risks.append(f"Significant skill gaps ({len(missing)} missing) — ramp-up time needed")

        # Job hopping risk (short tenures)
        experience = candidate.get("experience", [])
        if len(experience) >= 3:
            short_stints = sum(1 for exp in experience
                              if "duration" in exp and "1" in str(exp.get("duration", "")))
            if short_stints >= 2:
                risks.append("Multiple short tenures — assess commitment and retention risk")

        # No project impact evidence
        projects = candidate.get("projects", [])
        if not projects:
            risks.append("No project portfolio — harder to assess practical impact")

        return risks

    def _generate_notes(self, rec: InterviewRecommendation,
                        explanation: dict, candidate: dict) -> str:
        """Generate concise hiring notes for the recruiter."""
        matched = explanation.get("matched_skills", [])
        missing = explanation.get("missing_skills", [])
        exp_years = candidate.get("total_experience_years", 0)

        parts = []

        if rec.decision == "STRONG_HIRE":
            parts.append(f"Strong profile with {len(matched)} matched skills and {exp_years:.0f} years experience.")
            if rec.interview_focus_areas:
                parts.append(f"Interview focus: {rec.interview_focus_areas[0]}")

        elif rec.decision == "INTERVIEW_RECOMMENDED":
            parts.append(f"Good candidate with solid skill coverage ({len(matched)} matches).")
            if missing:
                parts.append(f"Gaps in: {', '.join(missing[:3])}.")
            parts.append("Worth interviewing to assess depth.")

        elif rec.decision == "HOLD_FOR_REVIEW":
            parts.append(f"Moderate fit. {len(matched)} skills match but {len(missing)} gaps identified.")
            parts.append("Consider if gaps are trainable before proceeding.")

        elif rec.decision == "WEAK_MATCH":
            parts.append(f"Below threshold. Only {len(matched)} skill matches.")
            parts.append("Significant ramp-up would be needed.")

        else:
            parts.append("Does not meet minimum requirements for this role.")

        return " ".join(parts)
