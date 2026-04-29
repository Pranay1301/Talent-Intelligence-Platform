"""
Job Description Parser - Extracts structured requirements from JD text.
Analyzes required/preferred skills, experience, education, and domain keywords.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedJD:
    """Structured representation of a parsed job description."""
    title: str = ""
    company: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    max_experience_years: float = 0.0
    education_requirements: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    domain_keywords: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title, "company": self.company,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "min_experience_years": self.min_experience_years,
            "max_experience_years": self.max_experience_years,
            "education_requirements": self.education_requirements,
            "responsibilities": self.responsibilities,
            "domain_keywords": self.domain_keywords,
            "raw_text": self.raw_text,
        }

    @property
    def all_skills(self) -> list[str]:
        return list(set(self.required_skills + self.preferred_skills))


class JDParser:
    """
    Job Description parser that extracts structured requirements.
    
    Uses regex patterns and keyword analysis to identify:
    - Required vs preferred skills
    - Experience requirements
    - Education requirements
    - Domain-specific keywords
    """

    KNOWN_SKILLS = {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "sql", "html", "css", "r", "scala", "kotlin", "swift",
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "scikit-learn", "transformers",
        "bert", "gpt", "llm", "langchain", "opencv", "spacy",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "github actions", "jenkins",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "dynamodb", "snowflake", "bigquery",
        "fastapi", "flask", "django", "spring boot", "express",
        "react", "angular", "vue", "next.js", "node.js",
        "git", "rest api", "graphql", "microservices", "kafka",
        "airflow", "spark", "hadoop", "tableau", "power bi",
        "agile", "scrum", "jira", "linux",
    }

    SECTION_PATTERNS = {
        "requirements": r"(?i)(requirements?|qualifications?|what\s+you.+need|must\s+have|minimum\s+qualifications?)",
        "preferred": r"(?i)(preferred|nice\s+to\s+have|bonus|desired|plus|additional)",
        "responsibilities": r"(?i)(responsibilities|what\s+you.+do|role|duties|key\s+responsibilities)",
        "about": r"(?i)(about|overview|description|who\s+we\s+are|company)",
    }

    def parse(self, jd_text: str) -> ParsedJD:
        """Parse job description text into structured format."""
        jd = ParsedJD(raw_text=jd_text)
        jd.title = self._extract_title(jd_text)
        jd.company = self._extract_company(jd_text)
        
        sections = self._split_sections(jd_text)
        
        req_text = sections.get("requirements", "")
        pref_text = sections.get("preferred", "")
        resp_text = sections.get("responsibilities", "")
        
        jd.required_skills = self._extract_skills(req_text if req_text else jd_text, required=True)
        jd.preferred_skills = self._extract_skills(pref_text, required=False)
        
        exp = self._extract_experience_requirement(jd_text)
        jd.min_experience_years = exp[0]
        jd.max_experience_years = exp[1]
        
        jd.education_requirements = self._extract_education_req(jd_text)
        jd.responsibilities = self._extract_bullets(resp_text)
        jd.domain_keywords = self._extract_domain_keywords(jd_text)
        
        return jd

    def _extract_title(self, text: str) -> str:
        lines = text.strip().split("\n")
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100 and not re.search(r"@|http|about|overview", line, re.I):
                title_patterns = [
                    r"(?i)(senior|junior|lead|staff|principal)?\s*(software|data|ml|ai|machine\s*learning|backend|frontend|full.?stack|devops|cloud|platform)\s*(engineer|developer|scientist|analyst|architect)",
                    r"(?i)(product|project|program)\s*(manager|lead|owner)",
                ]
                for p in title_patterns:
                    m = re.search(p, line)
                    if m:
                        return line
                if len(line.split()) <= 8 and line[0].isupper():
                    return line
        return ""

    def _extract_company(self, text: str) -> str:
        patterns = [
            r"(?i)(?:at|@|company[:\s])\s*([A-Z][\w\s&.]+)",
            r"(?i)about\s+([A-Z][\w\s&.]{2,30})",
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    def _split_sections(self, text: str) -> dict:
        sections = {}
        lines = text.split("\n")
        current = "general"
        content = []
        for line in lines:
            matched = None
            for name, pattern in self.SECTION_PATTERNS.items():
                if re.match(pattern, line.strip()):
                    matched = name
                    break
            if matched:
                if content:
                    sections[current] = "\n".join(content)
                current = matched
                content = []
            else:
                content.append(line)
        if content:
            sections[current] = "\n".join(content)
        return sections

    def _extract_skills(self, text: str, required: bool = True) -> list[str]:
        found = set()
        text_lower = text.lower()
        for skill in self.KNOWN_SKILLS:
            if skill in text_lower:
                found.add(skill.title() if len(skill) > 3 else skill.upper())
        return sorted(found)

    def _extract_experience_requirement(self, text: str) -> tuple[float, float]:
        patterns = [
            r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
            r"(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)",
            r"(?:minimum|at\s*least)\s*(\d+)\s*(?:years?|yrs?)",
        ]
        for p in patterns:
            m = re.search(p, text, re.I)
            if m:
                groups = m.groups()
                if len(groups) == 2 and groups[1]:
                    return (float(groups[0]), float(groups[1]))
                return (float(groups[0]), float(groups[0]) + 2)
        return (0.0, 0.0)

    def _extract_education_req(self, text: str) -> list[str]:
        reqs = []
        patterns = [
            r"(?i)(bachelor'?s?|b\.?s\.?|b\.?tech)\s*(?:degree)?\s*(?:in\s+[\w\s,]+)?",
            r"(?i)(master'?s?|m\.?s\.?|m\.?tech|mba)\s*(?:degree)?\s*(?:in\s+[\w\s,]+)?",
            r"(?i)(ph\.?d\.?|doctorate)\s*(?:in\s+[\w\s,]+)?",
        ]
        for p in patterns:
            for m in re.finditer(p, text):
                req = m.group().strip()
                if req and req not in reqs:
                    reqs.append(req)
        return reqs

    def _extract_bullets(self, text: str) -> list[str]:
        if not text:
            return []
        items = []
        for line in text.split("\n"):
            line = line.strip().lstrip("•-●○▪·*0123456789.) ")
            if line and len(line) > 10:
                items.append(line)
        return items

    def _extract_domain_keywords(self, text: str) -> list[str]:
        domains = [
            "fintech", "healthcare", "e-commerce", "saas", "edtech",
            "cybersecurity", "blockchain", "iot", "autonomous", "robotics",
            "natural language", "recommendation", "search", "advertising",
            "real-time", "distributed systems", "high-availability",
            "scalable", "production", "enterprise", "startup",
        ]
        found = []
        text_lower = text.lower()
        for d in domains:
            if d in text_lower:
                found.append(d.title())
        return found
