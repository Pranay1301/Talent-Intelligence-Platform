"""SQLAlchemy database models for persistent storage."""

from sqlalchemy import Column, String, Float, Integer, Text, DateTime, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class CandidateModel(Base):
    """Database model for stored candidates."""
    __tablename__ = "candidates"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, default="")
    email = Column(String, default="")
    phone = Column(String, default="")
    linkedin = Column(String, default="")
    github = Column(String, default="")
    summary = Column(Text, default="")
    skills = Column(JSON, default=list)
    education = Column(JSON, default=list)
    experience = Column(JSON, default=list)
    projects = Column(JSON, default=list)
    certifications = Column(JSON, default=list)
    total_experience_years = Column(Float, default=0.0)
    raw_text = Column(Text, default="")
    file_path = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "linkedin": self.linkedin,
            "github": self.github,
            "summary": self.summary,
            "skills": self.skills or [],
            "education": self.education or [],
            "experience": self.experience or [],
            "projects": self.projects or [],
            "certifications": self.certifications or [],
            "total_experience_years": self.total_experience_years,
            "raw_text": self.raw_text,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class RankingHistoryModel(Base):
    """Database model for ranking history."""
    __tablename__ = "ranking_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    jd_text = Column(Text, nullable=False)
    jd_title = Column(String, default="")
    results = Column(JSON, default=list)
    total_candidates = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_engine(database_url: str):
    """Create database engine."""
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, connect_args=connect_args)


def get_session(database_url: str):
    """Create database session."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
