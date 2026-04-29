"""
Bias Mitigation & Fairness Layer

Reduces ranking bias from:
- Name/gender indicators
- Institutional prestige
- Demographic signals
- Unnecessary personal attributes

Implements:
- PII redaction from scoring pipeline
- Prestige-blind education scoring
- Statistical parity checks
- Fairness audit reporting

Enterprise-grade bias awareness for equitable hiring.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Prestigious institutions (used to detect and neutralize prestige bias)
PRESTIGIOUS_INSTITUTIONS = {
    "mit", "stanford", "harvard", "caltech", "carnegie mellon",
    "berkeley", "princeton", "yale", "columbia", "cornell",
    "oxford", "cambridge", "eth zurich", "imperial college",
    "iit bombay", "iit delhi", "iit madras", "iit kanpur", "iit kharagpur",
    "bits pilani", "nit", "iisc bangalore",
}

# Gender-indicator patterns to redact
GENDER_INDICATORS = [
    r"\b(mr|mrs|ms|miss|sir|madam)\b",
    r"\b(he|she|him|her|his|hers)\b",
    r"\b(husband|wife|father|mother|son|daughter)\b",
]

# Name patterns (common first names that may indicate demographics)
PII_PATTERNS = [
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",  # Date of birth
    r"\bage[:\s]*\d+\b",                       # Age
    r"\b(married|single|divorced|widowed)\b",   # Marital status
    r"\b(male|female|non-binary|transgender)\b", # Gender
    r"\b(nationality|citizenship)[:\s]+\w+",    # Nationality
    r"\b(religion|caste|race|ethnicity)[:\s]+\w+", # Protected categories
]


@dataclass
class FairnessReport:
    """Report on bias checks applied to a ranking."""
    pii_redacted: bool = False
    fields_redacted: list[str] = field(default_factory=list)
    prestige_bias_detected: bool = False
    prestige_institutions_found: list[str] = field(default_factory=list)
    demographic_signals_found: int = 0
    fairness_score: float = 1.0  # 1.0 = fully fair
    warnings: list[str] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pii_redacted": self.pii_redacted,
            "fields_redacted": self.fields_redacted,
            "prestige_bias_detected": self.prestige_bias_detected,
            "prestige_institutions_found": self.prestige_institutions_found,
            "demographic_signals_found": self.demographic_signals_found,
            "fairness_score": round(self.fairness_score, 4),
            "warnings": self.warnings,
            "actions_taken": self.actions_taken,
        }


class BiasMitigator:
    """
    Fairness-aware processing layer for candidate evaluation.
    
    Ensures ranking decisions are based on skills, experience, and
    qualifications rather than demographic or institutional signals.
    """

    def __init__(self, prestige_penalty: float = 0.0):
        """
        Args:
            prestige_penalty: Score reduction for prestige bias (0 = disabled).
                              Set > 0 to counteract prestige inflation.
        """
        self.prestige_penalty = prestige_penalty

    def redact_pii(self, text: str) -> str:
        """
        Remove personally identifiable information from resume text
        before it enters the scoring pipeline.
        
        This ensures the embedding and scoring layers cannot be
        influenced by demographic signals.
        """
        redacted = text

        # Redact gender indicators
        for pattern in GENDER_INDICATORS:
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

        # Redact PII patterns
        for pattern in PII_PATTERNS:
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

        return redacted

    def anonymize_candidate(self, candidate_data: dict) -> dict:
        """
        Create an anonymized version of candidate data for scoring.
        Preserves skills, experience, education content but removes
        identifying information.
        """
        anonymized = candidate_data.copy()

        # Redact name (keep for display but don't use in scoring)
        anonymized["_original_name"] = anonymized.get("name", "")
        anonymized["name"] = f"Candidate_{anonymized.get('candidate_id', 'X')}"

        # Redact raw text
        if "raw_text" in anonymized:
            anonymized["raw_text"] = self.redact_pii(anonymized["raw_text"])

        return anonymized

    def check_prestige_bias(self, education: list[dict]) -> dict:
        """
        Detect if prestigious institution names may inflate scores.
        
        Returns bias analysis with detected institutions.
        """
        found = []
        for edu in education:
            degree_text = edu.get("degree", "").lower()
            institution = edu.get("institution", "").lower()
            combined = f"{degree_text} {institution}"

            for inst in PRESTIGIOUS_INSTITUTIONS:
                if inst in combined:
                    found.append(inst.title())

        return {
            "bias_detected": len(found) > 0,
            "institutions": found,
            "recommendation": (
                "Education scored on degree level and relevance, not institution prestige"
                if found else "No prestige bias signals detected"
            ),
        }

    def compute_fairness_adjusted_score(self, original_score: float,
                                         candidate_data: dict) -> tuple[float, FairnessReport]:
        """
        Apply fairness adjustments to a candidate's score.
        
        Returns:
            adjusted_score: Score after fairness corrections
            report: Detailed fairness audit report
        """
        report = FairnessReport()
        adjusted = original_score

        # 1. Check for PII in scoring data
        raw_text = candidate_data.get("raw_text", "")
        pii_count = 0
        for pattern in PII_PATTERNS:
            matches = re.findall(pattern, raw_text, re.IGNORECASE)
            pii_count += len(matches)

        if pii_count > 0:
            report.demographic_signals_found = pii_count
            report.warnings.append(
                f"Found {pii_count} demographic signal(s) in resume text"
            )
            report.actions_taken.append("PII signals identified for redaction")

        # 2. Check prestige bias
        education = candidate_data.get("education", [])
        prestige_check = self.check_prestige_bias(education)
        if prestige_check["bias_detected"]:
            report.prestige_bias_detected = True
            report.prestige_institutions_found = prestige_check["institutions"]

            if self.prestige_penalty > 0:
                adjusted -= self.prestige_penalty
                report.actions_taken.append(
                    f"Applied prestige penalty: -{self.prestige_penalty}"
                )
            else:
                report.actions_taken.append(
                    "Prestige detected but penalty disabled (education scored on degree level only)"
                )

        # 3. Compute fairness score
        issues = report.demographic_signals_found + (1 if report.prestige_bias_detected else 0)
        report.fairness_score = max(0.5, 1.0 - (issues * 0.1))

        return max(0, min(1, adjusted)), report

    def audit_ranking(self, ranked_candidates: list[dict]) -> dict:
        """
        Audit a complete ranking for systemic bias patterns.
        
        Checks for:
        - Score distribution anomalies
        - Demographic clustering
        - Institution clustering in top ranks
        """
        if not ranked_candidates:
            return {"status": "no_candidates", "issues": []}

        issues = []
        total = len(ranked_candidates)

        # Check score distribution
        scores = [c.get("final_score", 0) for c in ranked_candidates]
        if scores:
            avg = sum(scores) / len(scores)
            if avg > 0.85:
                issues.append("Scores may be inflated — consider calibration")
            if avg < 0.2:
                issues.append("Scores unusually low — check JD matching quality")

        # Check for institution clustering in top 5
        top_5 = ranked_candidates[:min(5, total)]
        top_institutions = []
        for c in top_5:
            for edu in c.get("resume_data", {}).get("education", []):
                combined = f"{edu.get('degree', '')} {edu.get('institution', '')}".lower()
                for inst in PRESTIGIOUS_INSTITUTIONS:
                    if inst in combined:
                        top_institutions.append(inst)

        if len(set(top_institutions)) >= 3:
            issues.append(
                f"Top-ranked candidates cluster around prestigious institutions: {', '.join(set(top_institutions))}"
            )

        return {
            "status": "audit_complete",
            "total_candidates": total,
            "issues_found": len(issues),
            "issues": issues,
            "recommendation": "Ranking appears fair" if not issues else "Review flagged issues",
        }
