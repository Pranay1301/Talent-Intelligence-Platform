# 🎯 AI-Powered Resume Screening & Candidate Ranking System

An intelligent, production-grade system that automatically screens resumes, matches them against job descriptions using **semantic similarity**, and ranks candidates with **explainable AI** — going far beyond traditional keyword-based filtering.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🧠 Problem Statement

Traditional resume screening is **slow, manual, and inaccurate** — keyword-based filtering misses strong candidates who describe their experience differently.

This system solves that by:
- **Reading resumes** automatically from PDF/DOCX files
- **Comparing** them semantically with job descriptions
- **Ranking candidates** by multi-dimensional relevance
- **Explaining** why each candidate is a strong or weak match

> *"Machine Learning Engineer" matches with "Built predictive AI systems" — even without exact keyword overlap.*

---

## ✨ Key Features

### 1. 📄 Intelligent Resume Parsing
- Supports **PDF** and **DOCX** formats
- Extracts: skills, education, experience, projects, certifications
- Uses **spaCy NER** + regex patterns for robust entity extraction
- Dual extraction engine (pdfplumber + PyPDF2 fallback)

### 2. 📋 Job Description Analysis
- Parses required vs. preferred skills
- Extracts experience requirements, education needs, domain keywords
- Section-aware parsing (requirements, responsibilities, preferred)

### 3. 🧬 Semantic Matching (Sentence-BERT)
- **Dense embeddings** via Sentence-BERT (`all-MiniLM-L6-v2`)
- **Sparse features** via TF-IDF with n-grams
- **Hybrid similarity** combining both signals for robust matching
- Captures contextual meaning beyond keyword overlap

### 4. 📊 Hybrid Scoring System
```
Final Score = 0.35 × Semantic Similarity
            + 0.30 × Skill Match Score
            + 0.20 × Experience Score
            + 0.10 × Education Score
            + 0.05 × Certification Bonus
```
Configurable weights for different hiring priorities.

### 5. ⚡ FAISS Vector Search
- **Facebook AI Similarity Search** for production-scale retrieval
- Low-latency nearest-neighbor search across large resume datasets
- Persistent index with save/load support
- Normalized embeddings with inner-product similarity

### 6. 🔍 Explainable AI Layer
Every ranking includes:
```
✅ Matched Skills: Python, NLP, FastAPI
❌ Missing Skills: AWS, Docker
💼 Experience Match: 3/4 years
🎓 Education: Master's (meets requirement)
🏆 Recommendation: STRONG MATCH - Highly recommended for interview
```

### 7. 📈 Recruiter Dashboard (Streamlit)
- Upload resumes (drag & drop)
- Paste job descriptions
- Interactive ranked candidate table
- Score breakdown radar charts
- Downloadable JSON/CSV reports

### 8. 🚀 Production API (FastAPI)
Full REST API with auto-generated docs:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload_resume` | POST | Upload & parse resume file |
| `/api/upload_resume_text` | POST | Submit resume as text |
| `/api/rank_candidates` | POST | Rank candidates against JD |
| `/api/candidates` | GET | List all candidates |
| `/api/candidates/{id}` | GET | Get candidate details |
| `/api/get_candidate_score/{id}` | GET | Score candidate for a JD |
| `/api/export_report` | POST | Export ranking report |
| `/api/health` | GET | System health check |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Resume     │────▶│  Resume      │────▶│   Embedding     │
│   (PDF/DOCX) │     │  Parser      │     │   Engine        │
└─────────────┘     │  (spaCy+NER) │     │  (SBERT+TF-IDF) │
                    └──────────────┘     └────────┬────────┘
                                                  │
┌─────────────┐     ┌──────────────┐              ▼
│   Job        │────▶│  JD Parser   │     ┌─────────────────┐
│   Description│     │              │     │   FAISS Vector   │
└─────────────┘     └──────┬───────┘     │   Store          │
                           │              └────────┬────────┘
                           ▼                       │
                    ┌──────────────┐               │
                    │  Ranking     │◀──────────────┘
                    │  Engine      │
                    │  (Hybrid)    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ FastAPI   │ │Streamlit │ │ Reports  │
       │ REST API  │ │Dashboard │ │ Export   │
       └──────────┘ └──────────┘ └──────────┘
```

---

## 📁 Project Structure

```
resume-screening-system/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # FastAPI route definitions
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   └── screening_service.py   # Main orchestration service
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_utils.py          # File handling utilities
│   │   └── text_utils.py          # NLP text preprocessing
│   ├── __init__.py
│   ├── config.py                  # Application configuration
│   └── main.py                    # FastAPI entry point
│
├── resume_parser/
│   ├── __init__.py
│   └── parser.py                  # PDF/DOCX resume extraction
│
├── jd_parser/
│   ├── __init__.py
│   └── parser.py                  # Job description analysis
│
├── embedding_engine/
│   ├── __init__.py
│   ├── engine.py                  # Sentence-BERT + TF-IDF embeddings
│   └── vector_store.py            # FAISS index management
│
├── ranking_engine/
│   ├── __init__.py
│   └── ranker.py                  # Hybrid scoring + explainability
│
├── dashboard/
│   ├── __init__.py
│   └── streamlit_app.py           # Recruiter dashboard UI
│
├── database/
│   ├── __init__.py
│   └── models.py                  # SQLAlchemy ORM models
│
├── faiss_index/                   # FAISS index storage
├── tests/
│   ├── __init__.py
│   └── test_system.py             # Unit & integration tests
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **NLP** | spaCy, NLTK, Sentence Transformers, BERT |
| **ML** | Scikit-learn, TF-IDF, Cosine Similarity |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | Streamlit, Plotly |
| **Database** | SQLAlchemy, SQLite (dev) / PostgreSQL (prod) |
| **Deployment** | Docker, Docker Compose |
| **Testing** | Pytest, HTTPx |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/resume-screening-system.git
cd resume-screening-system
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings (defaults work for local development)
```

### 5. Start the API Server
```bash
uvicorn app.main:app --reload --port 8000
```
API docs available at: `http://localhost:8000/docs`

### 6. Start the Dashboard (in another terminal)
```bash
streamlit run dashboard/streamlit_app.py
```
Dashboard at: `http://localhost:8501`

### Docker Deployment
```bash
docker-compose up --build
```
- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`

---

## 📖 Usage Guide

### Upload Resumes via API
```bash
# Upload a PDF resume
curl -X POST http://localhost:8000/api/upload_resume \
  -F "file=@resume.pdf"

# Upload resume as text
curl -X POST http://localhost:8000/api/upload_resume_text \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "resume_text": "Experienced ML Engineer..."}'
```

### Rank Candidates
```bash
curl -X POST http://localhost:8000/api/rank_candidates \
  -H "Content-Type: application/json" \
  -d '{
    "jd_text": "Looking for a Senior ML Engineer with 5+ years...",
    "top_k": 10
  }'
```

### Response Example
```json
{
  "job_title": "Senior Machine Learning Engineer",
  "total_candidates": 25,
  "ranked_candidates": [
    {
      "rank": 1,
      "name": "Alice Chen",
      "final_score": 0.87,
      "score_breakdown": {
        "scores": {
          "embedding_similarity": 0.92,
          "skill_match_score": 0.85,
          "experience_score": 1.0,
          "education_score": 1.0,
          "certification_bonus": 0.7
        },
        "explanation": {
          "matched_skills": ["Python", "TensorFlow", "NLP", "AWS"],
          "missing_skills": ["Kubernetes"],
          "experience_match": "6.0/5-8 years",
          "recommendation": "STRONG MATCH - Highly recommended for interview"
        }
      }
    }
  ]
}
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_system.py::TestRankingEngine -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 🔧 Configuration

Key configuration via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./resume_screening.db` | Database connection string |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-BERT model |
| `SPACY_MODEL` | `en_core_web_sm` | spaCy NLP model |
| `API_PORT` | `8000` | FastAPI server port |
| `FAISS_INDEX_PATH` | `./faiss_index/index.faiss` | FAISS index file location |

### Scoring Weights
Customize in `ranking_engine/ranker.py`:
```python
DEFAULT_WEIGHTS = {
    "embedding_similarity": 0.35,  # Semantic relevance
    "skill_match": 0.30,           # Explicit skill overlap
    "experience": 0.20,            # Years and relevance
    "education": 0.10,             # Degree match
    "certifications": 0.05,        # Bonus
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for state-of-the-art semantic embeddings
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI for efficient similarity search
- [spaCy](https://spacy.io/) for industrial-strength NLP
- [FastAPI](https://fastapi.tiangolo.com/) for high-performance API development
- [Streamlit](https://streamlit.io/) for rapid dashboard prototyping
