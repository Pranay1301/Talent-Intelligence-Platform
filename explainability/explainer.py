"""
Enhanced Explainable AI Layer

Generates recruiter-friendly explanations for every ranking decision:
- Score contribution breakdown
- Feature importance analysis
- Comparative candidate analysis
- Natural language explanation generation
- Visual explanation data for dashboard

Goes beyond "matched/missing skills" to provide actionable hiring intelligence.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DetailedExplanation:
    """Comprehensive explanation of a ranking decision."""
    # Skill Analysis
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    partial_matches: list[str] = field(default_factory=list)
    skill_coverage_pct: float = 0.0

    # Experience Analysis
    experience_match: str = ""
    experience_gap_years: float = 0.0
    relevant_experience_highlights: list[str] = field(default_factory=list)

    # Education Analysis
    education_match: str = ""
    education_level_met: bool = False

    # Project Impact
    project_relevance: str = ""  # HIGH, MEDIUM, LOW, NONE
    relevant_projects: list[str] = field(default_factory=list)

    # Role Fit
    role_fit_level: str = ""  # HIGH, MEDIUM, LOW
    domain_alignment: str = ""
    seniority_assessment: str = ""

    # Score Attribution (which factors drove the score most)
    top_positive_factors: list[str] = field(default_factory=list)
    top_negative_factors: list[str] = field(default_factory=list)

    # Summary
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    recommendation: str = ""
    one_line_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "skill_analysis": {
                "matched": self.matched_skills,
                "missing": self.missing_skills,
                "extra": self.extra_skills,
                "partial_matches": self.partial_matches,
                "coverage_pct": round(self.skill_coverage_pct, 1),
            },
            "experience_analysis": {
                "match": self.experience_match,
                "gap_years": self.experience_gap_years,
                "highlights": self.relevant_experience_highlights,
            },
            "education_analysis": {
                "match": self.education_match,
                "level_met": self.education_level_met,
            },
            "project_impact": {
                "relevance": self.project_relevance,
                "relevant_projects": self.relevant_projects,
            },
            "role_fit": {
                "level": self.role_fit_level,
                "domain_alignment": self.domain_alignment,
                "seniority": self.seniority_assessment,
            },
            "score_attribution": {
                "positive_factors": self.top_positive_factors,
                "negative_factors": self.top_negative_factors,
            },
            "summary": {
                "strengths": self.strengths,
                "weaknesses": self.weaknesses,
                "recommendation": self.recommendation,
                "one_liner": self.one_line_summary,
            },
        }


class RankingExplainer:
    """
    Generates detailed, recruiter-friendly explanations for ranking decisions.
    
    Analyzes multiple dimensions of candidate-job fit and produces
    structured explanations with score attribution.
    """

    def explain(self, scores: dict, candidate: dict, jd: dict,
                skill_overlap: dict = None) -> DetailedExplanation:
        """
        Generate comprehensive explanation for a candidate's ranking.
        
        Args:
            scores: Score breakdown dictionary
            candidate: Parsed candidate data
            jd: Parsed JD data
            skill_overlap: Pre-computed skill overlap from ontology
        """
        exp = DetailedExplanation()

        # Skill analysis
        self._explain_skills(exp, candidate, jd, skill_overlap)

        # Experience analysis
        self._explain_experience(exp, scores, candidate, jd)

        # Education analysis
        self._explain_education(exp, scores, candidate, jd)

        # Project impact
        self._explain_projects(exp, candidate, jd)

        # Role fit
        self._explain_role_fit(exp, scores, candidate, jd)

        # Score attribution
        self._compute_attribution(exp, scores)

        # Generate summary
        self._generate_summary(exp, scores)

        return exp

    def _explain_skills(self, exp: DetailedExplanation, candidate: dict,
                        jd: dict, skill_overlap: dict):
        """Analyze skill match in detail."""
        if skill_overlap:
            exp.matched_skills = skill_overlap.get("matched", [])
            exp.missing_skills = skill_overlap.get("missing", [])
            exp.extra_skills = skill_overlap.get("extra", [])[:10]
            exp.partial_matches = skill_overlap.get("partial_matches", [])
            exp.skill_coverage_pct = skill_overlap.get("match_ratio", 0) * 100
        else:
            cand_skills = set(s.lower() for s in candidate.get("skills", []))
            jd_skills = set(s.lower() for s in
                           jd.get("required_skills", []) + jd.get("preferred_skills", []))
            exp.matched_skills = sorted(cand_skills & jd_skills)
            exp.missing_skills = sorted(jd_skills - cand_skills)
            exp.extra_skills = sorted(list(cand_skills - jd_skills)[:10])
            exp.skill_coverage_pct = (
                len(exp.matched_skills) / max(len(jd_skills), 1) * 100
            )

    def _explain_experience(self, exp: DetailedExplanation, scores: dict,
                            candidate: dict, jd: dict):
        """Analyze experience match."""
        cand_years = candidate.get("total_experience_years", 0)
        min_years = jd.get("min_experience_years", 0)
        max_years = jd.get("max_experience_years", 0)

        if min_years > 0:
            exp.experience_match = f"{cand_years:.1f} years (required: {min_years:.0f}-{max_years:.0f})"
            if cand_years < min_years:
                exp.experience_gap_years = min_years - cand_years
            else:
                exp.experience_gap_years = 0
        else:
            exp.experience_match = f"{cand_years:.1f} years (no specific requirement)"

        # Extract relevant experience highlights
        jd_text_lower = jd.get("raw_text", "").lower()
        for entry in candidate.get("experience", []):
            title = entry.get("title", "")
            for resp in entry.get("responsibilities", [])[:3]:
                # Check if responsibility is relevant to JD
                words = set(resp.lower().split())
                jd_words = set(jd_text_lower.split())
                overlap = len(words & jd_words) / max(len(words), 1)
                if overlap > 0.15:
                    exp.relevant_experience_highlights.append(
                        f"{title}: {resp[:100]}..."
                        if len(resp) > 100 else f"{title}: {resp}"
                    )
            if len(exp.relevant_experience_highlights) >= 3:
                break

    def _explain_education(self, exp: DetailedExplanation, scores: dict,
                           candidate: dict, jd: dict):
        """Analyze education match."""
        edu_score = scores.get("education_score", 0)
        edu = candidate.get("education", [])

        if edu:
            highest = edu[0].get("degree", "Unknown")
            exp.education_match = f"{highest}"
        else:
            exp.education_match = "No education data"

        exp.education_level_met = edu_score >= 0.8

    def _explain_projects(self, exp: DetailedExplanation, candidate: dict, jd: dict):
        """Analyze project relevance."""
        projects = candidate.get("projects", [])
        if not projects:
            exp.project_relevance = "NONE"
            return

        jd_text = jd.get("raw_text", "").lower()
        relevant = []

        for proj in projects:
            name = proj.get("name", "")
            details = " ".join(proj.get("details", []))
            combined = f"{name} {details}".lower()

            # Simple relevance check via word overlap
            proj_words = set(combined.split())
            jd_words = set(jd_text.split())
            overlap = len(proj_words & jd_words) / max(len(proj_words), 1)

            if overlap > 0.1:
                relevant.append(name)

        exp.relevant_projects = relevant

        if len(relevant) >= 2:
            exp.project_relevance = "HIGH"
        elif len(relevant) == 1:
            exp.project_relevance = "MEDIUM"
        else:
            exp.project_relevance = "LOW"

    def _explain_role_fit(self, exp: DetailedExplanation, scores: dict,
                          candidate: dict, jd: dict):
        """Assess overall role fit."""
        emb = scores.get("embedding_similarity", 0)
        skill = scores.get("skill_match_score", 0)

        avg = (emb + skill) / 2
        if avg >= 0.7:
            exp.role_fit_level = "HIGH"
        elif avg >= 0.45:
            exp.role_fit_level = "MEDIUM"
        else:
            exp.role_fit_level = "LOW"

        exp.domain_alignment = "Aligned" if emb > 0.6 else "Partial" if emb > 0.4 else "Weak"
        exp.seniority_assessment = self._assess_seniority(candidate, jd)

    def _assess_seniority(self, candidate: dict, jd: dict) -> str:
        """Assess seniority level match."""
        years = candidate.get("total_experience_years", 0)
        jd_title = jd.get("title", "").lower()

        if "senior" in jd_title or "lead" in jd_title or "staff" in jd_title:
            if years >= 5:
                return "Appropriate for senior role"
            elif years >= 3:
                return "Borderline — may need strong justification"
            return "Under-leveled for senior position"
        elif "junior" in jd_title or "entry" in jd_title:
            if years <= 3:
                return "Appropriate for junior/entry role"
            return "May be overqualified — assess motivation"
        return f"{years:.0f} years — assess against role expectations"

    def _compute_attribution(self, exp: DetailedExplanation, scores: dict):
        """Identify which factors most influenced the score."""
        score_items = [
            ("Semantic relevance", scores.get("embedding_similarity", 0)),
            ("Skill match", scores.get("skill_match_score", 0)),
            ("Experience fit", scores.get("experience_score", 0)),
            ("Education", scores.get("education_score", 0)),
            ("Certifications", scores.get("certification_bonus", 0)),
            ("Project impact", scores.get("project_impact_score", 0)),
        ]

        # Sort by score
        sorted_items = sorted(score_items, key=lambda x: x[1], reverse=True)

        exp.top_positive_factors = [
            f"{name}: {val:.0%}" for name, val in sorted_items[:3] if val >= 0.5
        ]
        exp.top_negative_factors = [
            f"{name}: {val:.0%}" for name, val in sorted_items if val < 0.4
        ]

    def _generate_summary(self, exp: DetailedExplanation, scores: dict):
        """Generate natural language summary."""
        final = scores.get("final_score", 0)

        # Strengths
        if exp.skill_coverage_pct >= 70:
            exp.strengths.append(f"Strong skill coverage ({exp.skill_coverage_pct:.0f}%)")
        if exp.project_relevance in ("HIGH", "MEDIUM"):
            exp.strengths.append("Relevant project experience")
        if not exp.experience_gap_years:
            exp.strengths.append("Experience meets requirements")
        if exp.education_level_met:
            exp.strengths.append("Education qualifications met")
        if exp.partial_matches:
            exp.strengths.append(f"Partial skill matches: {', '.join(exp.partial_matches[:3])}")
        if exp.extra_skills:
            exp.strengths.append(f"Additional skills: {', '.join(exp.extra_skills[:3])}")

        # Weaknesses
        if exp.missing_skills:
            exp.weaknesses.append(f"Missing {len(exp.missing_skills)} required skill(s)")
        if exp.experience_gap_years > 0:
            exp.weaknesses.append(f"Experience gap: {exp.experience_gap_years:.1f} years short")
        if not exp.education_level_met:
            exp.weaknesses.append("Education below stated requirements")
        if exp.project_relevance == "NONE":
            exp.weaknesses.append("No project portfolio available")

        # Recommendation
        if final >= 0.80:
            exp.recommendation = "STRONG MATCH — Highly recommended for interview"
        elif final >= 0.60:
            exp.recommendation = "GOOD MATCH — Recommended for consideration"
        elif final >= 0.40:
            exp.recommendation = "MODERATE MATCH — Additional screening needed"
        elif final >= 0.25:
            exp.recommendation = "WEAK MATCH — Significant gaps identified"
        else:
            exp.recommendation = "NOT RECOMMENDED — Does not meet requirements"

        # One-liner
        matched_count = len(exp.matched_skills)
        exp.one_line_summary = (
            f"{'Strong' if final >= 0.7 else 'Moderate' if final >= 0.4 else 'Weak'} "
            f"fit with {matched_count} skill matches, "
            f"{exp.skill_coverage_pct:.0f}% coverage, "
            f"{'meets' if not exp.experience_gap_years else 'falls short of'} experience requirements."
        )
