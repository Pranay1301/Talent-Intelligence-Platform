# 🎯 AI-Powered Intelligent Talent Screening, Ranking & Hiring Recommendation Platform

### Production-Grade ML + NLP + Vector Search + Explainable AI + Multi-Stage Ranking System

An enterprise-level machine learning platform that automates resume screening, semantically matches candidates to job descriptions, ranks them using a multi-stage hybrid pipeline, and delivers explainable hiring recommendations — built for recruiters and hiring teams at scale.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Sentence--BERT](https://img.shields.io/badge/Sentence--BERT-Embeddings-FF6F00?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-4285F4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 🧠 Problem Statement

Traditional ATS (Applicant Tracking Systems) rely on **keyword matching**, causing strong candidates to be missed because they describe experience differently. A candidate who "Built predictive AI systems" gets rejected by a filter looking for "Machine Learning Engineer."

**This platform solves that problem** using contextual semantic understanding, multi-stage ranking, and explainable AI — reducing recruiter effort while improving hiring quality and fairness.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    MULTI-STAGE RANKING PIPELINE                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Resume   │───▶│ Resume       │───▶│ Embedding Engine      │  │
│  │ PDF/DOCX │    │ Parser       │    │ (Sentence-BERT +      │  │
│  └──────────┘    │ (spaCy+NER)  │    │  TF-IDF Hybrid)       │  │
│                  └──────────────┘    └───────────┬───────────┘  │
│                                                  │               │
│  ┌──────────┐    ┌──────────────┐                ▼               │
│  │ Job      │───▶│ JD Parser    │    ┌───────────────────────┐  │
│  │ Desc.    │    │ + Importance  │    │ FAISS Vector Store    │  │
│  └──────────┘    │   Weighting  │    │ (Low-Latency Search)  │  │
│                  └──────┬───────┘    └───────────┬───────────┘  │
│                         │                        │               │
│  ┌──────────────────────┼────────────────────────┘               │
│  │                      ▼                                        │
│  │  ╔══════════════════════════════════════════════════════════╗ │
│  │  ║  STAGE 1: FAISS Retrieval (Fast Candidate Shortlist)    ║ │
│  │  ╠══════════════════════════════════════════════════════════╣ │
│  │  ║  STAGE 2: Hybrid Multi-Signal Scoring                   ║ │
│  │  ║  • Embedding Similarity  • Skill Ontology Match         ║ │
│  │  ║  • Experience Relevance  • Project Impact               ║ │
│  │  ║  • Education Match       • Leadership Score             ║ │
│  │  ║  • Domain Alignment      • Seniority Match              ║ │
│  │  ╠══════════════════════════════════════════════════════════╣ │
│  │  ║  STAGE 3: Re-Ranking + Intelligence Layer               ║ │
│  │  ║  • Explainable AI        • Interview Recommendations    ║ │
│  │  ║  • Bias Audit            • Score Attribution            ║ │
│  │  ╚══════════════════════════════════════════════════════════╝ │
│  │                      │                                        │
│  └──────────────────────┼────────────────────────────────────────┘
│                         │                                        │
│         ┌───────────────┼───────────────┐                        │
│         ▼               ▼               ▼                        │
│  ┌──────────┐   ┌────────────┐   ┌──────────┐                  │
│  │ FastAPI  │   │ Streamlit  │   │ Reports  │                  │
│  │ REST API │   │ Dashboard  │   │ CSV/JSON │                  │
│  └──────────┘   └────────────┘   └──────────┘                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. 📄 Intelligent Resume Parsing
- **PDF & DOCX** support with dual extraction (pdfplumber + PyPDF2 fallback)
- Extracts: skills, experience, education, projects, certifications, achievements, leadership roles
- **spaCy NER** + regex pipelines for robust entity extraction
- Handles noisy formatting and real-world resumes

### 2. 📋 Job Description Intelligence Engine
- Classifies requirements: **critical / preferred / bonus**
- Extracts role seniority, domain relevance, tech stack, and experience requirements
- JD importance weighting for smarter matching

### 3. 🧬 Skill Ontology + Synonym Mapping
```
"ML Engineer" ≈ "Machine Learning Developer" ≈ "Predictive Modeling Engineer"
"NLP" = "Natural Language Processing"
"K8s" = "Kubernetes"
```
- **200+ skill synonyms** with abbreviation expansion
- Role taxonomy for intelligent title matching
- Hierarchical skill graph (partial credit for parent skills)
- Domain normalization across industries

### 4. ⚡ Multi-Stage Hybrid Ranking Pipeline

| Stage | Description | Speed |
|-------|-------------|-------|
| **Stage 1: Retrieval** | FAISS nearest-neighbor search | ~5ms |
| **Stage 2: Scoring** | 9-dimensional hybrid scoring with ontology | ~50ms |
| **Stage 3: Re-ranking** | Explanations + interview recs + bias audit | ~100ms |

**Hybrid Scoring Formula:**
```
Final Score = 0.25 × Semantic Similarity
            + 0.25 × Skill Match (Ontology-Aware)
            + 0.15 × Experience Relevance
            + 0.10 × Project Impact
            + 0.08 × Education Match
            + 0.05 × Leadership Score
            + 0.05 × Domain Alignment
            + 0.04 × Certification Bonus
            + 0.03 × Seniority Match
```
Configurable weights with role-specific presets (ML Engineer, Data Scientist, Engineering Manager).

### 5. 🔍 Explainable AI Layer
Every ranking decision includes transparent reasoning:
```
✅ Matched Skills: Python, NLP, FastAPI (75% coverage)
❌ Missing Skills: AWS, Docker
💼 Experience: 5.0/3-7 years (meets requirement)
📂 Project Relevance: HIGH (2 relevant projects)
🌐 Domain Alignment: Aligned
🎓 Education: Master's (meets requirement)

Top Positive Factors: Semantic relevance: 89%, Skill match: 75%
Top Negative Factors: Domain alignment: 45%

📝 One-liner: Strong fit with 6 skill matches, 75% coverage, meets experience requirements.
```

### 6. 🎯 Interview Recommendation Engine
```
Decision:        STRONG_HIRE (92% confidence)
Role Fit:        HIGH
Growth Potential: MEDIUM
Suggested Round: TECHNICAL_DEEP_DIVE

Interview Focus Areas:
- System Design: Assess proficiency in Docker, Kubernetes
- MLOps: Probe model deployment experience

Risk Factors:
- Multiple short tenures — assess commitment

Hiring Notes:
Strong profile with 6 matched skills and 5 years experience.
Interview focus: System Design depth assessment.
```

### 7. ✅ Bias Mitigation Layer
- **PII redaction** from scoring pipeline (name, gender, age, nationality)
- **Prestige-blind** education scoring (degree level, not institution name)
- **Fairness audit** on complete rankings (detects institutional clustering)
- **Statistical parity** checks with configurable thresholds
- Enterprise-grade compliance for equitable hiring

### 8. 📊 Model Evaluation Framework
Measurable KPIs that interviewers love:

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K results that are relevant |
| **Recall@K** | Fraction of relevant items found in top-K |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **MRR** | Mean Reciprocal Rank |
| **Latency** | P50/P95/P99 ranking latency |
| **Baseline Comparison** | Semantic system vs. keyword-based ATS |

### 9. 📈 Recruiter Analytics Dashboard
- Resume upload with drag & drop
- JD input with intelligent parsing
- Interactive ranked candidate table with decisions
- Radar chart score breakdowns
- Interview recommendation panels
- Hiring funnel analytics (skill distribution, experience ranges)
- Bias/fairness audit display
- Downloadable CSV/JSON reports

### 10. 🚀 Production API (FastAPI)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload_resume` | POST | Upload & parse resume file |
| `/api/upload_resume_text` | POST | Submit resume as text |
| `/api/analyze_resume/{id}` | GET | Detailed resume analysis |
| `/api/parse_jd` | POST | Parse & classify JD requirements |
| `/api/rank_candidates` | POST | Multi-stage candidate ranking |
| `/api/get_candidate_score/{id}` | GET | Score candidate against JD |
| `/api/get_explanations` | POST | Detailed ranking explanations |
| `/api/interview_recommendation/{id}` | GET | Interview strategy & focus areas |
| `/api/export_report` | POST | Export ranking report |
| `/api/dashboard_metrics` | GET | Recruiter analytics data |
| `/api/health` | GET | System health check |

---

## 📁 Project Structure

```
talent-intelligence-platform/
│
├── app/
│   ├── api/
│   │   └── routes.py                  # FastAPI endpoints (14 routes)
│   ├── models/
│   │   └── schemas.py                 # Pydantic request/response models
│   ├── services/
│   │   ├── screening_service.py       # Main orchestration service
│   │   └── evaluation.py             # Ranking evaluation framework
│   ├── pipelines/
│   │   └── screening_pipeline.py      # Multi-stage ranking pipeline
│   ├── utils/
│   │   ├── file_utils.py             # File handling
│   │   └── text_utils.py             # NLP text preprocessing
│   ├── config.py                      # Application configuration
│   └── main.py                        # FastAPI entry point
│
├── resume_parser/
│   └── parser.py                      # PDF/DOCX resume extraction
│
├── jd_parser/
│   └── parser.py                      # JD analysis & requirement classification
│
├── embedding_engine/
│   ├── engine.py                      # Sentence-BERT + TF-IDF hybrid
│   └── vector_store.py               # FAISS index management
│
├── ranking_engine/
│   └── ranker.py                      # Hybrid scoring engine
│
├── explainability/
│   └── explainer.py                   # Score attribution & explanation generation
│
├── interview_engine/
│   └── recommender.py                # Interview strategy & recommendations
│
├── bias_mitigation/
│   └── fairness.py                   # PII redaction, prestige detection, fairness audit
│
├── configs/
│   ├── skill_ontology.py             # 200+ skill synonyms, role taxonomy, skill hierarchy
│   └── scoring_weights.py            # Configurable weights & role presets
│
├── dashboard/
│   └── streamlit_app.py              # Recruiter intelligence dashboard
│
├── database/
│   └── models.py                      # SQLAlchemy ORM models
│
├── faiss_index/                       # FAISS index persistence
├── tests/
│   └── test_system.py                # Unit & integration tests
│
├── .github/workflows/ci.yml          # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **NLP** | spaCy, NLTK, Sentence Transformers (BERT), Transformers |
| **ML** | Scikit-learn, XGBoost, TF-IDF, Cosine Similarity |
| **Vector Search** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | Sentence-BERT (`all-MiniLM-L6-v2`) |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | Streamlit, Plotly |
| **Database** | SQLAlchemy, SQLite (dev) / PostgreSQL (prod) |
| **Deployment** | Docker, Docker Compose, GitHub Actions |
| **Testing** | Pytest, HTTPx |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip / venv

### Installation
```bash
git clone https://github.com/yourusername/talent-intelligence-platform.git
cd talent-intelligence-platform

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env
```

### Run
```bash
# Terminal 1: API Server
uvicorn app.main:app --reload --port 8000

# Terminal 2: Dashboard
streamlit run dashboard/streamlit_app.py
```
- API docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

### Docker
```bash
docker-compose up --build
```

### ☁️ Cloud Deployment (Render.com Free Tier)

This project is configured for 1-click cloud deployment. The provided `Dockerfile` and `start.sh` automatically spin up both the FastAPI backend and Streamlit dashboard in a single free container.

1. Go to [Render Dashboard](https://dashboard.render.com/) and log in with GitHub.
2. Click **New** → **Web Service**.
3. Select this repository.
4. Set **Runtime** to `Docker` and **Instance Type** to `Free`.
5. Click **Create Web Service**. Your live URL will be ready in 3-5 minutes!

---

## 📖 Usage Example

### Rank Candidates
```bash
curl -X POST http://localhost:8000/api/rank_candidates \
  -H "Content-Type: application/json" \
  -d '{
    "jd_text": "Looking for a Senior ML Engineer with 5+ years experience in Python, TensorFlow, NLP, and AWS. Must have production deployment experience.",
    "top_k": 5
  }'
```

### Response (Abbreviated)
```json
{
  "job_title": "Senior ML Engineer",
  "total_candidates_in_pool": 50,
  "ranked_candidates": [
    {
      "rank": 1,
      "name": "Alice Chen",
      "final_score": 0.87,
      "score_breakdown": {
        "scores": {
          "embedding_similarity": 0.92,
          "skill_match": 0.85,
          "experience_relevance": 1.0,
          "project_impact": 0.8,
          "domain_alignment": 0.7
        },
        "explanation": {
          "skill_analysis": { "matched": ["Python", "TensorFlow", "NLP"], "coverage_pct": 75.0 },
          "summary": { "recommendation": "STRONG MATCH — Highly recommended for interview" }
        }
      },
      "interview_recommendation": {
        "decision": "STRONG_HIRE",
        "confidence": 0.92,
        "interview_focus_areas": ["System Design: Assess Docker, Kubernetes proficiency"],
        "role_fit": "HIGH",
        "hiring_notes": "Strong profile with 6 matched skills and 5 years experience."
      },
      "fairness_score": 1.0
    }
  ],
  "pipeline_metrics": {
    "stage1_retrieval_ms": 4.5,
    "stage2_scoring_ms": 52.3,
    "stage3_reranking_ms": 98.1,
    "total_ms": 155.2
  },
  "fairness_audit": { "status": "audit_complete", "issues_found": 0 }
}
```

---

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=html
```

---

## 🔧 Configuration

### Scoring Weight Presets
```python
# configs/scoring_weights.py
WEIGHT_PRESETS = {
    "ml_engineer":       { "project_impact": 0.15, "leadership_score": 0.02, ... },
    "senior_engineer":   { "experience_relevance": 0.20, "leadership_score": 0.10, ... },
    "data_scientist":    { "education_relevance": 0.12, "project_impact": 0.15, ... },
    "engineering_manager": { "leadership_score": 0.20, "experience_relevance": 0.20, ... },
}
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./resume_screening.db` | Database connection |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-BERT model |
| `FAISS_INDEX_PATH` | `./faiss_index/index.faiss` | FAISS index location |
| `API_PORT` | `8000` | API server port |

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) — State-of-the-art semantic embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Meta AI similarity search
- [spaCy](https://spacy.io/) — Industrial-strength NLP
- [FastAPI](https://fastapi.tiangolo.com/) — High-performance async API
- [Streamlit](https://streamlit.io/) — Rapid dashboard development
