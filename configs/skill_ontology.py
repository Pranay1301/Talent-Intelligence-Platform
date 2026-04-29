"""
Skill Ontology & Synonym Mapping

Semantic skill graph that handles:
- Synonym mapping (ML = Machine Learning)
- Abbreviation expansion (NLP = Natural Language Processing)
- Role taxonomy (ML Engineer ≈ Machine Learning Developer ≈ Predictive Modeling Engineer)
- Domain normalization
- Hierarchical skill relationships

This eliminates weak keyword matching and enables intelligent skill comparison.
"""

# --- Skill Synonym Map ---
# Maps canonical skill names to all known variations
SKILL_SYNONYMS: dict[str, list[str]] = {
    # Programming Languages
    "python": ["python3", "python 3", "py", "cpython"],
    "javascript": ["js", "es6", "es2015", "ecmascript"],
    "typescript": ["ts"],
    "c++": ["cpp", "c plus plus", "cplusplus"],
    "c#": ["csharp", "c sharp", "dotnet", ".net"],
    "golang": ["go", "go lang"],
    "rust": ["rustlang"],

    # ML/AI Core
    "machine learning": ["ml", "machine-learning", "statistical learning"],
    "deep learning": ["dl", "deep-learning", "neural network", "neural networks"],
    "natural language processing": ["nlp", "text mining", "text analytics", "computational linguistics"],
    "computer vision": ["cv", "image recognition", "image processing", "visual computing"],
    "reinforcement learning": ["rl", "reward learning"],
    "large language models": ["llm", "llms", "foundation models", "generative ai", "gen ai"],
    "artificial intelligence": ["ai", "a.i."],

    # ML Frameworks
    "tensorflow": ["tf", "tensorflow 2", "tf2"],
    "pytorch": ["torch", "py torch"],
    "scikit-learn": ["sklearn", "scikit learn", "sk-learn"],
    "hugging face": ["huggingface", "hf", "transformers library"],
    "xgboost": ["xg boost", "extreme gradient boosting"],
    "lightgbm": ["light gbm", "lgbm"],

    # Cloud Platforms
    "amazon web services": ["aws", "amazon cloud"],
    "google cloud platform": ["gcp", "google cloud"],
    "microsoft azure": ["azure", "az"],

    # DevOps & Infrastructure
    "kubernetes": ["k8s", "kube"],
    "docker": ["containerization", "docker containers"],
    "continuous integration": ["ci", "ci/cd", "cicd", "continuous delivery", "continuous deployment"],
    "infrastructure as code": ["iac", "terraform", "cloudformation"],
    "github actions": ["gh actions", "github ci"],

    # Databases
    "postgresql": ["postgres", "psql", "pg"],
    "mongodb": ["mongo"],
    "elasticsearch": ["elastic", "es"],
    "amazon dynamodb": ["dynamodb", "dynamo"],

    # Web Frameworks
    "fastapi": ["fast api"],
    "next.js": ["nextjs", "next"],
    "node.js": ["nodejs", "node"],
    "react.js": ["reactjs", "react"],
    "vue.js": ["vuejs", "vue"],

    # Data Engineering
    "apache spark": ["spark", "pyspark"],
    "apache kafka": ["kafka"],
    "apache airflow": ["airflow"],
    "apache hadoop": ["hadoop", "hdfs", "mapreduce"],

    # Data Science
    "data visualization": ["data viz", "dataviz"],
    "exploratory data analysis": ["eda"],
    "feature engineering": ["feature extraction", "feature selection"],
    "a/b testing": ["ab testing", "split testing", "experimentation"],
    "statistical analysis": ["statistics", "statistical modeling", "inferential statistics"],

    # MLOps
    "mlops": ["ml ops", "ml operations", "machine learning operations"],
    "model deployment": ["model serving", "model inference"],
    "experiment tracking": ["mlflow", "wandb", "weights and biases"],

    # Soft Skills / Methodologies
    "agile": ["agile methodology", "scrum", "kanban", "sprint planning"],
    "project management": ["program management", "pm"],
    "cross-functional collaboration": ["cross functional", "stakeholder management"],
}

# --- Role Taxonomy ---
# Maps role families to equivalent job titles
ROLE_TAXONOMY: dict[str, list[str]] = {
    "machine_learning_engineer": [
        "ml engineer", "machine learning engineer", "machine learning developer",
        "predictive modeling engineer", "ai engineer", "applied ml engineer",
        "ml software engineer", "deep learning engineer",
    ],
    "data_scientist": [
        "data scientist", "research scientist", "applied scientist",
        "quantitative analyst", "ml researcher", "ai researcher",
        "statistical modeler",
    ],
    "data_engineer": [
        "data engineer", "analytics engineer", "etl developer",
        "data platform engineer", "big data engineer", "data infrastructure engineer",
    ],
    "software_engineer": [
        "software engineer", "software developer", "sde", "swe",
        "backend engineer", "backend developer", "full stack engineer",
        "full stack developer", "frontend engineer",
    ],
    "devops_engineer": [
        "devops engineer", "site reliability engineer", "sre",
        "platform engineer", "infrastructure engineer", "cloud engineer",
    ],
    "nlp_engineer": [
        "nlp engineer", "natural language processing engineer",
        "conversational ai engineer", "text mining engineer",
        "computational linguist",
    ],
    "mlops_engineer": [
        "mlops engineer", "ml platform engineer", "ml infrastructure engineer",
        "machine learning operations engineer",
    ],
}

# --- Skill Hierarchy ---
# Parent-child relationships for skill categorization
SKILL_HIERARCHY: dict[str, list[str]] = {
    "artificial_intelligence": ["machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning"],
    "machine_learning": ["supervised learning", "unsupervised learning", "semi-supervised learning", "transfer learning"],
    "deep_learning": ["cnn", "rnn", "lstm", "transformer", "gan", "autoencoder", "attention mechanism"],
    "nlp": ["text classification", "named entity recognition", "sentiment analysis", "machine translation", "question answering", "summarization"],
    "data_engineering": ["etl", "data pipeline", "data warehouse", "data lake", "stream processing"],
    "cloud_computing": ["aws", "gcp", "azure", "serverless", "cloud functions"],
    "devops": ["docker", "kubernetes", "ci/cd", "terraform", "monitoring"],
    "databases": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra"],
}

# --- Domain Keywords ---
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "fintech": ["fintech", "financial technology", "banking", "payments", "trading", "risk management", "fraud detection"],
    "healthcare": ["healthcare", "health tech", "medical", "clinical", "pharma", "biotech", "health informatics"],
    "ecommerce": ["e-commerce", "ecommerce", "retail", "marketplace", "shopping", "inventory"],
    "adtech": ["advertising", "ad tech", "programmatic", "rtb", "dsp", "ssp", "attribution"],
    "autonomous": ["autonomous vehicles", "self-driving", "robotics", "drone", "adas"],
    "cybersecurity": ["security", "cybersecurity", "infosec", "threat detection", "vulnerability"],
    "edtech": ["education technology", "edtech", "e-learning", "lms", "learning platform"],
}

# --- Seniority Levels ---
SENIORITY_LEVELS = {
    "intern": 0,
    "junior": 1,
    "associate": 1,
    "mid": 2,
    "mid-level": 2,
    "senior": 3,
    "staff": 4,
    "principal": 5,
    "lead": 4,
    "manager": 4,
    "director": 5,
    "vp": 6,
    "head": 5,
    "chief": 6,
    "fellow": 6,
    "distinguished": 6,
}


class SkillOntology:
    """
    Skill ontology engine for intelligent skill matching.
    
    Handles synonym resolution, abbreviation expansion, and
    hierarchical skill comparison to avoid naive keyword matching.
    """

    def __init__(self):
        # Build reverse lookup: variation -> canonical name
        self._synonym_lookup: dict[str, str] = {}
        for canonical, variations in SKILL_SYNONYMS.items():
            self._synonym_lookup[canonical.lower()] = canonical.lower()
            for v in variations:
                self._synonym_lookup[v.lower()] = canonical.lower()

        # Build role reverse lookup
        self._role_lookup: dict[str, str] = {}
        for family, titles in ROLE_TAXONOMY.items():
            for title in titles:
                self._role_lookup[title.lower()] = family

    def normalize_skill(self, skill: str) -> str:
        """Normalize a skill to its canonical form."""
        return self._synonym_lookup.get(skill.lower().strip(), skill.lower().strip())

    def normalize_skills(self, skills: list[str]) -> list[str]:
        """Normalize a list of skills, removing duplicates."""
        seen = set()
        normalized = []
        for s in skills:
            canonical = self.normalize_skill(s)
            if canonical not in seen:
                seen.add(canonical)
                normalized.append(canonical)
        return normalized

    def compute_skill_overlap(self, skills_a: list[str], skills_b: list[str]) -> dict:
        """
        Compute intelligent skill overlap using ontology.
        
        Returns:
            matched: skills present in both (after normalization)
            missing: skills in B but not A
            extra: skills in A but not B
            match_ratio: percentage of B's skills covered
        """
        norm_a = set(self.normalize_skill(s) for s in skills_a)
        norm_b = set(self.normalize_skill(s) for s in skills_b)

        matched = norm_a & norm_b
        missing = norm_b - norm_a
        extra = norm_a - norm_b

        # Check hierarchical matches (partial credit)
        partial_matches = set()
        for skill_b in missing.copy():
            for parent, children in SKILL_HIERARCHY.items():
                children_lower = [c.lower() for c in children]
                if skill_b in children_lower:
                    # If candidate has parent skill, partial match
                    if parent.replace("_", " ") in norm_a:
                        partial_matches.add(skill_b)

        total_required = len(norm_b) if norm_b else 1
        match_ratio = (len(matched) + 0.5 * len(partial_matches)) / total_required

        return {
            "matched": sorted(matched),
            "missing": sorted(missing - partial_matches),
            "partial_matches": sorted(partial_matches),
            "extra": sorted(extra),
            "match_ratio": round(min(1.0, match_ratio), 4),
        }

    def get_role_family(self, title: str) -> str:
        """Get the role family for a job title."""
        return self._role_lookup.get(title.lower().strip(), "unknown")

    def are_roles_similar(self, title_a: str, title_b: str) -> bool:
        """Check if two job titles belong to the same role family."""
        family_a = self.get_role_family(title_a)
        family_b = self.get_role_family(title_b)
        return family_a == family_b and family_a != "unknown"

    def get_seniority_level(self, title: str) -> int:
        """Extract seniority level from a job title."""
        title_lower = title.lower()
        for keyword, level in SENIORITY_LEVELS.items():
            if keyword in title_lower:
                return level
        return 2  # Default to mid-level

    def get_domain(self, text: str) -> list[str]:
        """Identify domains from text."""
        text_lower = text.lower()
        domains = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(k in text_lower for k in keywords):
                domains.append(domain)
        return domains
