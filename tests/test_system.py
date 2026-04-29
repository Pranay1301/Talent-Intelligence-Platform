"""
Unit tests for the Resume Screening System.

Tests cover:
- Resume parsing
- JD parsing
- Embedding engine
- Ranking engine
- API endpoints
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResumeParser:
    """Tests for resume parsing functionality."""

    def test_parse_from_text(self):
        from resume_parser.parser import ResumeParser
        parser = ResumeParser()
        
        sample_resume = """
John Doe
john.doe@email.com
+1-555-123-4567
linkedin.com/in/johndoe
github.com/johndoe

Summary
Experienced Machine Learning Engineer with 5 years of experience building production ML systems.

Skills
Python, TensorFlow, PyTorch, AWS, Docker, Kubernetes, NLP, Computer Vision, SQL, FastAPI

Experience
Senior ML Engineer at TechCorp          Jan 2022 - Present
• Built recommendation systems serving 10M+ users
• Deployed NLP models using FastAPI and Docker
• Reduced model inference latency by 40%

ML Engineer at DataInc          Jun 2019 - Dec 2021
• Developed computer vision pipeline for quality inspection
• Implemented A/B testing framework for ML experiments

Education
M.S. Computer Science, Stanford University, 2019
B.Tech Computer Science, IIT Delhi, 2017

Projects
Sentiment Analysis Engine
• Built real-time sentiment classifier using BERT
• Achieved 94% accuracy on benchmark dataset

Certifications
AWS Certified Machine Learning Specialty
Google Cloud Professional Data Engineer
"""
        result = parser.parse_from_text(sample_resume)
        
        assert result.email == "john.doe@email.com"
        assert len(result.skills) > 0
        assert result.total_experience_years > 0
        assert "Python" in result.skills or "python" in [s.lower() for s in result.skills]

    def test_extract_email(self):
        from resume_parser.parser import ResumeParser
        parser = ResumeParser()
        assert parser._extract_email("Contact: test@example.com") == "test@example.com"
        assert parser._extract_email("No email here") == ""

    def test_extract_phone(self):
        from resume_parser.parser import ResumeParser
        parser = ResumeParser()
        assert parser._extract_phone("+1-555-123-4567") != ""
        assert parser._extract_phone("No phone") == ""


class TestJDParser:
    """Tests for JD parsing functionality."""

    def test_parse_jd(self):
        from jd_parser.parser import JDParser
        parser = JDParser()
        
        sample_jd = """
Senior Machine Learning Engineer

About the Role
We are looking for a Senior ML Engineer to join our AI team.

Requirements
- 5+ years of experience in machine learning
- Strong proficiency in Python, TensorFlow or PyTorch
- Experience with NLP and computer vision
- Familiarity with AWS cloud services
- Bachelor's degree in Computer Science or related field

Preferred Qualifications
- Experience with Kubernetes and Docker
- Published research in ML
- Master's degree preferred

Responsibilities
- Design and implement ML models for production
- Lead model optimization and deployment
- Mentor junior engineers
"""
        result = parser.parse(sample_jd)
        
        assert len(result.required_skills) > 0
        assert result.min_experience_years > 0
        assert len(result.responsibilities) > 0

    def test_extract_experience(self):
        from jd_parser.parser import JDParser
        parser = JDParser()
        result = parser._extract_experience_requirement("5+ years of experience required")
        assert result[0] == 5.0


class TestRankingEngine:
    """Tests for the hybrid ranking engine."""

    def test_rank_candidates(self):
        from ranking_engine.ranker import CandidateRanker
        ranker = CandidateRanker()
        
        candidates = [
            {
                "name": "Alice",
                "skills": ["Python", "TensorFlow", "NLP", "AWS"],
                "total_experience_years": 5,
                "education": [{"degree": "M.S. Computer Science"}],
                "certifications": ["AWS ML Specialty"],
                "experience": [],
            },
            {
                "name": "Bob",
                "skills": ["Java", "Spring Boot"],
                "total_experience_years": 2,
                "education": [{"degree": "B.S. Computer Science"}],
                "certifications": [],
                "experience": [],
            },
        ]
        
        jd = {
            "required_skills": ["Python", "TensorFlow", "NLP"],
            "preferred_skills": ["AWS", "Docker"],
            "min_experience_years": 3,
            "max_experience_years": 7,
            "education_requirements": ["Master's degree"],
        }
        
        ranked = ranker.rank_candidates(candidates, jd, [0.85, 0.3])
        
        assert len(ranked) == 2
        assert ranked[0].name == "Alice"
        assert ranked[0].rank == 1
        assert ranked[0].score_breakdown.final_score > ranked[1].score_breakdown.final_score
        assert len(ranked[0].score_breakdown.matched_skills) > 0

    def test_score_breakdown_has_explanation(self):
        from ranking_engine.ranker import CandidateRanker
        ranker = CandidateRanker()
        
        candidates = [{
            "name": "Test",
            "skills": ["Python"],
            "total_experience_years": 2,
            "education": [],
            "certifications": [],
            "experience": [],
        }]
        jd = {"required_skills": ["Python", "Java"], "preferred_skills": [],
              "min_experience_years": 3, "max_experience_years": 5,
              "education_requirements": []}
        
        ranked = ranker.rank_candidates(candidates, jd, [0.5])
        breakdown = ranked[0].score_breakdown
        
        assert breakdown.recommendation != ""
        assert isinstance(breakdown.matched_skills, list)
        assert isinstance(breakdown.missing_skills, list)


class TestEmbeddingEngine:
    """Tests for the embedding engine (requires model download)."""

    @pytest.mark.slow
    def test_encode_and_similarity(self):
        from embedding_engine.engine import EmbeddingEngine
        engine = EmbeddingEngine()
        
        emb1 = engine.encode_single("Machine learning engineer with NLP experience")
        emb2 = engine.encode_single("Built AI systems for natural language processing")
        emb3 = engine.encode_single("Professional chef with culinary arts degree")
        
        sim_related = engine.compute_similarity(emb1, emb2)
        sim_unrelated = engine.compute_similarity(emb1, emb3)
        
        assert sim_related > sim_unrelated
        assert 0 <= sim_related <= 1

    @pytest.mark.slow
    def test_hybrid_similarity(self):
        from embedding_engine.engine import EmbeddingEngine
        engine = EmbeddingEngine()
        
        result = engine.hybrid_similarity(
            "Python developer with ML experience",
            "Software engineer specializing in machine learning with Python",
        )
        
        assert "dense_score" in result
        assert "sparse_score" in result
        assert "combined_score" in result
        assert result["combined_score"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
