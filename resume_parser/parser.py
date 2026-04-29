"""
Resume Parser - Extracts structured information from PDF/DOCX resumes.
Uses PyPDF2, pdfplumber, python-docx for extraction and spaCy for NER.
"""

import re
import os
import logging
from dataclasses import dataclass, field

import spacy
import pdfplumber
from PyPDF2 import PdfReader
from docx import Document

logger = logging.getLogger(__name__)


@dataclass
class ParsedResume:
    """Structured representation of a parsed resume."""
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    summary: str = ""
    skills: list[str] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    experience: list[dict] = field(default_factory=list)
    projects: list[dict] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    total_experience_years: float = 0.0
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "email": self.email, "phone": self.phone,
            "linkedin": self.linkedin, "github": self.github,
            "summary": self.summary, "skills": self.skills,
            "education": self.education, "experience": self.experience,
            "projects": self.projects, "certifications": self.certifications,
            "total_experience_years": self.total_experience_years,
            "raw_text": self.raw_text,
        }


class ResumeParser:
    """Intelligent resume parser using spaCy NER + regex patterns."""

    TECH_SKILLS = {
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "html", "css",
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "transformers", "bert", "gpt", "llm", "langchain", "opencv", "spacy",
        "pandas", "numpy", "scipy", "matplotlib", "data science", "data engineering",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "snowflake",
        "fastapi", "flask", "django", "react", "angular", "vue", "next.js", "node.js",
        "git", "rest api", "graphql", "microservices", "agile", "scrum",
        "airflow", "spark", "hadoop", "kafka", "tableau", "power bi",
    }

    SECTION_PATTERNS = {
        "experience": r"(?i)(work\s*experience|professional\s*experience|experience|employment)",
        "education": r"(?i)(education|academic|qualification|degree)",
        "skills": r"(?i)(skills|technical\s*skills|core\s*competencies|technologies)",
        "projects": r"(?i)(projects|personal\s*projects|key\s*projects)",
        "certifications": r"(?i)(certifications?|certificates?|licenses?)",
        "summary": r"(?i)(summary|objective|profile|about\s*me)",
    }

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"Downloading spaCy model '{spacy_model}'...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)

    def parse(self, file_path: str) -> ParsedResume:
        """Parse a resume file (PDF/DOCX) and extract structured data."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            raw_text = self._extract_pdf_text(file_path)
        elif ext == ".docx":
            raw_text = self._extract_docx_text(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use PDF or DOCX.")
        return self._parse_text(raw_text)

    def parse_from_text(self, text: str) -> ParsedResume:
        """Parse resume from raw text."""
        return self._parse_text(text)

    def _extract_pdf_text(self, file_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2")
        if not text.strip():
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()

    def _extract_docx_text(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _parse_text(self, raw_text: str) -> ParsedResume:
        resume = ParsedResume(raw_text=raw_text)
        doc = self.nlp(raw_text[:50000])
        resume.name = self._extract_name(doc, raw_text)
        resume.email = self._extract_email(raw_text)
        resume.phone = self._extract_phone(raw_text)
        resume.linkedin = self._extract_linkedin(raw_text)
        resume.github = self._extract_github(raw_text)
        sections = self._split_sections(raw_text)
        resume.summary = sections.get("summary", "").strip()
        resume.skills = self._extract_skills(raw_text, sections)
        resume.education = self._extract_education(sections)
        resume.experience = self._extract_experience(sections)
        resume.projects = self._extract_projects(sections)
        resume.certifications = self._extract_certifications(sections)
        resume.total_experience_years = self._calc_years(resume.experience)
        return resume

    def _extract_name(self, doc, raw_text: str) -> str:
        for ent in doc.ents:
            if ent.label_ == "PERSON" and raw_text.index(ent.text) < 200:
                return ent.text.strip()
        lines = raw_text.strip().split("\n")
        for line in lines[:3]:
            line = line.strip()
            if line and not re.search(r"@|http|www|\d{3}|resume|curriculum", line, re.I):
                words = line.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
                    return line
        return ""

    def _extract_email(self, text: str) -> str:
        m = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        return m.group() if m else ""

    def _extract_phone(self, text: str) -> str:
        for p in [r"\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", r"\+?\d{10,13}"]:
            m = re.search(p, text)
            if m:
                return m.group()
        return ""

    def _extract_linkedin(self, text: str) -> str:
        m = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+", text, re.I)
        return m.group() if m else ""

    def _extract_github(self, text: str) -> str:
        m = re.search(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+", text, re.I)
        return m.group() if m else ""

    def _split_sections(self, text: str) -> dict:
        sections = {}
        lines = text.split("\n")
        current_section = "header"
        current_content = []
        for line in lines:
            matched = None
            for name, pattern in self.SECTION_PATTERNS.items():
                if re.match(pattern, line.strip()):
                    matched = name
                    break
            if matched:
                if current_content:
                    sections[current_section] = "\n".join(current_content)
                current_section = matched
                current_content = []
            else:
                current_content.append(line)
        if current_content:
            sections[current_section] = "\n".join(current_content)
        return sections

    def _extract_skills(self, full_text: str, sections: dict) -> list[str]:
        found = set()
        skills_section = sections.get("skills", "")
        if skills_section:
            for item in re.split(r"[,|•·\n;]", skills_section):
                item = item.strip().strip("-●")
                if item and len(item) < 50:
                    found.add(item)
        text_lower = full_text.lower()
        for skill in self.TECH_SKILLS:
            if skill in text_lower:
                found.add(skill.title() if len(skill) > 3 else skill.upper())
        return sorted(found)

    def _extract_education(self, sections: dict) -> list[dict]:
        edu = []
        text = sections.get("education", "")
        if not text:
            return edu
        degree_re = r"(?i)(B\.?S|Bachelor|B\.?Tech|M\.?S|Master|M\.?Tech|MBA|Ph\.?D|Diploma)"
        lines = text.split("\n")
        entry = {}
        for line in lines:
            line = line.strip()
            if not line:
                if entry:
                    edu.append(entry)
                    entry = {}
                continue
            if re.search(degree_re, line):
                if entry:
                    edu.append(entry)
                entry = {"degree": line}
            year = re.search(r"(19|20)\d{2}", line)
            if year and entry:
                entry["year"] = year.group()
            if "degree" in entry and "institution" not in entry and line != entry.get("degree"):
                entry["institution"] = line
        if entry:
            edu.append(entry)
        return edu

    def _extract_experience(self, sections: dict) -> list[dict]:
        exp = []
        text = sections.get("experience", "")
        if not text:
            return exp
        lines = text.split("\n")
        entry = {}
        bullets = []
        date_re = r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s*\d{4})\s*[-–to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s*\d{4}|Present|Current)"
        for line in lines:
            line = line.strip()
            if not line:
                if entry:
                    entry["responsibilities"] = bullets
                    exp.append(entry)
                    entry = {}
                    bullets = []
                continue
            dm = re.search(date_re, line, re.I)
            yr = re.search(r"(20\d{2})\s*[-–]\s*(20\d{2}|Present|Current)", line, re.I)
            if dm or yr:
                if entry:
                    entry["responsibilities"] = bullets
                    exp.append(entry)
                    bullets = []
                m = dm or yr
                entry = {"title": line[:m.start()].strip().strip("-–|"), "duration": m.group()}
            elif line.startswith(("•", "-", "●", "○", "▪", "·", "*")):
                bullets.append(line.lstrip("•-●○▪·* "))
            elif not entry and line:
                entry = {"title": line}
        if entry:
            entry["responsibilities"] = bullets
            exp.append(entry)
        return exp

    def _extract_projects(self, sections: dict) -> list[dict]:
        projects = []
        text = sections.get("projects", "")
        if not text:
            return projects
        lines = text.split("\n")
        proj = {}
        bullets = []
        for line in lines:
            line = line.strip()
            if not line:
                if proj:
                    proj["details"] = bullets
                    projects.append(proj)
                    proj = {}
                    bullets = []
                continue
            if line.startswith(("•", "-", "●", "○", "▪", "·", "*")):
                bullets.append(line.lstrip("•-●○▪·* "))
            elif not proj.get("name"):
                proj = {"name": line}
            else:
                bullets.append(line)
        if proj:
            proj["details"] = bullets
            projects.append(proj)
        return projects

    def _extract_certifications(self, sections: dict) -> list[str]:
        text = sections.get("certifications", "")
        if not text:
            return []
        return [l.strip().lstrip("•-●○▪·* ") for l in text.split("\n") if l.strip() and len(l.strip()) > 3]

    def _calc_years(self, experience: list[dict]) -> float:
        months = 0
        month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                      "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        for entry in experience:
            dur = entry.get("duration", "")
            if not dur:
                continue
            dates = re.findall(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s*(\d{4})", dur, re.I)
            if len(dates) >= 2:
                sm = month_map.get(dates[0][0][:3].lower(), 1)
                em = month_map.get(dates[1][0][:3].lower(), 12)
                months += max(0, (int(dates[1][1]) - int(dates[0][1])) * 12 + (em - sm))
            elif "present" in dur.lower() or "current" in dur.lower():
                yr = re.search(r"(20\d{2})", dur)
                if yr:
                    from datetime import datetime
                    months += (datetime.now().year - int(yr.group())) * 12
            else:
                yrs = re.findall(r"(20\d{2})", dur)
                if len(yrs) >= 2:
                    months += (int(yrs[1]) - int(yrs[0])) * 12
        return round(months / 12, 1)
