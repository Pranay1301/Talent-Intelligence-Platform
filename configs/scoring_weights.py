"""
Configurable scoring weights and thresholds for the ranking engine.
Supports different hiring priorities and role types.
"""

# --- Default Scoring Weights ---
DEFAULT_WEIGHTS = {
    "embedding_similarity": 0.25,
    "skill_match": 0.25,
    "experience_relevance": 0.15,
    "education_relevance": 0.08,
    "certification_bonus": 0.04,
    "project_impact": 0.10,
    "leadership_score": 0.05,
    "domain_alignment": 0.05,
    "seniority_match": 0.03,
}

# --- Role-Specific Weight Presets ---
WEIGHT_PRESETS = {
    "ml_engineer": {
        "embedding_similarity": 0.25,
        "skill_match": 0.25,
        "experience_relevance": 0.15,
        "education_relevance": 0.05,
        "certification_bonus": 0.03,
        "project_impact": 0.15,
        "leadership_score": 0.02,
        "domain_alignment": 0.07,
        "seniority_match": 0.03,
    },
    "senior_engineer": {
        "embedding_similarity": 0.20,
        "skill_match": 0.20,
        "experience_relevance": 0.20,
        "education_relevance": 0.05,
        "certification_bonus": 0.03,
        "project_impact": 0.10,
        "leadership_score": 0.10,
        "domain_alignment": 0.07,
        "seniority_match": 0.05,
    },
    "data_scientist": {
        "embedding_similarity": 0.25,
        "skill_match": 0.20,
        "experience_relevance": 0.15,
        "education_relevance": 0.12,
        "certification_bonus": 0.03,
        "project_impact": 0.15,
        "leadership_score": 0.02,
        "domain_alignment": 0.05,
        "seniority_match": 0.03,
    },
    "engineering_manager": {
        "embedding_similarity": 0.15,
        "skill_match": 0.15,
        "experience_relevance": 0.20,
        "education_relevance": 0.05,
        "certification_bonus": 0.02,
        "project_impact": 0.10,
        "leadership_score": 0.20,
        "domain_alignment": 0.08,
        "seniority_match": 0.05,
    },
}

# --- Interview Recommendation Thresholds ---
RECOMMENDATION_THRESHOLDS = {
    "strong_hire": 0.80,
    "interview_recommended": 0.60,
    "hold_for_review": 0.40,
    "weak_match": 0.25,
    # Below 0.25 = reject
}

# --- Bias Mitigation Flags ---
BIAS_FIELDS_TO_REDACT = [
    "name", "gender", "age", "nationality", "ethnicity",
    "marital_status", "photo", "religion",
]

PRESTIGIOUS_INSTITUTION_PENALTY = 0.0  # Set > 0 to penalize prestige bias
