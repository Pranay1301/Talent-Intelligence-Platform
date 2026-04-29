"""
Streamlit Recruiter Dashboard

Professional dashboard for:
- Uploading resumes (PDF/DOCX)
- Pasting job descriptions
- Viewing ranked candidate tables
- Score breakdown visualization
- Downloadable reports
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Page config
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API base URL
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .score-high { color: #00c853; font-weight: bold; }
    .score-mid { color: #ff9100; font-weight: bold; }
    .score-low { color: #ff1744; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


def api_request(method, endpoint, **kwargs):
    """Make API request with error handling."""
    try:
        url = f"{API_URL}{endpoint}"
        resp = getattr(requests, method)(url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to API. Make sure the FastAPI server is running on " + API_URL)
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def render_sidebar():
    """Render sidebar with system info."""
    with st.sidebar:
        st.markdown("## 🎯 AI Resume Screener")
        st.markdown("---")
        
        # System stats
        stats = api_request("get", "/api/health")
        if stats:
            st.metric("Total Candidates", stats.get("total_candidates", 0))
            st.metric("FAISS Vectors", stats.get("faiss_vectors", 0))
            st.metric("API Status", "🟢 Online")
        else:
            st.metric("API Status", "🔴 Offline")
        
        st.markdown("---")
        st.markdown("### How it works")
        st.markdown("""
        1. **Upload** resumes (PDF/DOCX)
        2. **Paste** job description
        3. **Rank** candidates instantly
        4. **Review** explainable scores
        5. **Export** reports
        """)
        
        st.markdown("---")
        st.markdown("*Built with Sentence-BERT, FAISS, FastAPI*")


def render_upload_tab():
    """Resume upload interface."""
    st.markdown("### 📄 Upload Resumes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Upload Files")
        uploaded_files = st.file_uploader(
            "Drop resume files here",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            help="Upload PDF or DOCX resume files",
        )
        
        if uploaded_files and st.button("🚀 Process Resumes", type="primary"):
            progress = st.progress(0)
            results = []
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    resp = api_request(
                        "post", "/api/upload_resume",
                        files={"file": (file.name, file.getvalue(), file.type)}
                    )
                    if resp:
                        results.append(resp)
                progress.progress((i + 1) / len(uploaded_files))
            
            if results:
                st.success(f"✅ Successfully processed {len(results)} resume(s)")
                for r in results:
                    st.info(f"**{r['name']}** — {r['skills_count']} skills, {r['experience_years']} years exp")
    
    with col2:
        st.markdown("#### Paste Resume Text")
        name = st.text_input("Candidate Name", placeholder="John Doe")
        resume_text = st.text_area(
            "Resume Text",
            height=250,
            placeholder="Paste the full resume text here...",
        )
        
        if resume_text and st.button("📝 Submit Text Resume"):
            resp = api_request(
                "post", "/api/upload_resume_text",
                json={"name": name, "resume_text": resume_text},
            )
            if resp:
                st.success(f"✅ Processed: **{resp['name']}** — {resp['skills_count']} skills")


def render_ranking_tab():
    """Candidate ranking interface."""
    st.markdown("### 🏆 Rank Candidates")
    
    jd_text = st.text_area(
        "Paste Job Description",
        height=200,
        placeholder="Paste the full job description here to rank candidates against...",
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        top_k = st.slider("Top candidates to show", 1, 50, 10)
    with col2:
        export_format = st.selectbox("Export format", ["json", "csv"])
    
    if jd_text and st.button("🔍 Rank Candidates", type="primary"):
        with st.spinner("Analyzing candidates with AI..."):
            result = api_request(
                "post", "/api/rank_candidates",
                json={"jd_text": jd_text, "top_k": top_k},
            )
        
        if result and result.get("ranked_candidates"):
            st.markdown(f"### Results: {result.get('job_title', 'Job Analysis')}")
            st.markdown(f"*Analyzed {result['total_candidates']} candidates*")
            
            # Store results in session state
            st.session_state["last_ranking"] = result
            
            # Summary metrics
            candidates = result["ranked_candidates"]
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total Ranked", len(candidates))
            with cols[1]:
                if candidates:
                    st.metric("Top Score", f"{candidates[0]['final_score']:.1%}")
            with cols[2]:
                strong = sum(1 for c in candidates if c["final_score"] >= 0.6)
                st.metric("Strong Matches", strong)
            with cols[3]:
                avg = sum(c["final_score"] for c in candidates) / len(candidates) if candidates else 0
                st.metric("Avg Score", f"{avg:.1%}")
            
            st.markdown("---")
            
            # Ranked table
            df = pd.DataFrame([{
                "Rank": c["rank"],
                "Name": c["name"],
                "Score": f"{c['final_score']:.1%}",
                "Recommendation": c.get("score_breakdown", {}).get("explanation", {}).get("recommendation", ""),
            } for c in candidates])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Score breakdown chart
            if candidates:
                render_score_charts(candidates)
            
            # Detailed breakdown for each candidate
            st.markdown("### 📋 Detailed Breakdown")
            for c in candidates:
                render_candidate_detail(c)
            
            # Export button
            if st.button("📥 Export Report"):
                resp = api_request(
                    "post", "/api/export_report",
                    json={"jd_text": jd_text, "top_k": top_k},
                    params={"format": export_format},
                )
                if resp:
                    st.success(f"Report exported: {resp.get('file_path', '')}")
        
        elif result:
            st.warning("No candidates found. Upload some resumes first!")


def render_score_charts(candidates):
    """Render score visualization charts."""
    st.markdown("### 📊 Score Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart of final scores
        fig = px.bar(
            x=[c["name"] for c in candidates],
            y=[c["final_score"] for c in candidates],
            title="Candidate Scores",
            labels={"x": "Candidate", "y": "Score"},
            color=[c["final_score"] for c in candidates],
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart for top candidate
        top = candidates[0]
        scores = top.get("score_breakdown", {}).get("scores", {})
        
        categories = ["Semantic Match", "Skill Match", "Experience", "Education", "Certifications"]
        values = [
            scores.get("embedding_similarity", 0),
            scores.get("skill_match_score", 0),
            scores.get("experience_score", 0),
            scores.get("education_score", 0),
            scores.get("certification_bonus", 0),
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(102, 126, 234, 0.3)",
            line_color="#667eea",
        ))
        fig.update_layout(
            title=f"Score Breakdown: {top['name']}",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_candidate_detail(candidate):
    """Render detailed breakdown for a single candidate."""
    score = candidate["final_score"]
    color = "score-high" if score >= 0.6 else "score-mid" if score >= 0.4 else "score-low"
    
    with st.expander(f"**#{candidate['rank']} {candidate['name']}** — Score: {score:.1%}", expanded=False):
        breakdown = candidate.get("score_breakdown", {})
        scores = breakdown.get("scores", {})
        explanation = breakdown.get("explanation", {})
        
        # Score cards
        cols = st.columns(5)
        score_items = [
            ("🧠 Semantic", scores.get("embedding_similarity", 0)),
            ("🎯 Skills", scores.get("skill_match_score", 0)),
            ("💼 Experience", scores.get("experience_score", 0)),
            ("🎓 Education", scores.get("education_score", 0)),
            ("📜 Certs", scores.get("certification_bonus", 0)),
        ]
        for col, (label, val) in zip(cols, score_items):
            with col:
                st.metric(label, f"{val:.0%}")
        
        # Explanation
        col1, col2 = st.columns(2)
        with col1:
            matched = explanation.get("matched_skills", [])
            if matched:
                st.markdown("**✅ Matched Skills:**")
                st.markdown(", ".join(f"`{s}`" for s in matched))
            
            strengths = explanation.get("strengths", [])
            if strengths:
                st.markdown("**💪 Strengths:**")
                for s in strengths:
                    st.markdown(f"- {s}")
        
        with col2:
            missing = explanation.get("missing_skills", [])
            if missing:
                st.markdown("**❌ Missing Skills:**")
                st.markdown(", ".join(f"`{s}`" for s in missing))
            
            weaknesses = explanation.get("weaknesses", [])
            if weaknesses:
                st.markdown("**⚠️ Weaknesses:**")
                for w in weaknesses:
                    st.markdown(f"- {w}")
        
        # Experience & Education
        exp_match = explanation.get("experience_match", "")
        edu_match = explanation.get("education_match", "")
        if exp_match:
            st.info(f"**Experience:** {exp_match}")
        if edu_match:
            st.info(f"**Education:** {edu_match}")
        
        # Recommendation
        rec = explanation.get("recommendation", "")
        if rec:
            if "STRONG" in rec:
                st.success(f"🏆 {rec}")
            elif "GOOD" in rec:
                st.info(f"👍 {rec}")
            elif "MODERATE" in rec:
                st.warning(f"🤔 {rec}")
            else:
                st.error(f"⚠️ {rec}")


def render_candidates_tab():
    """Browse all stored candidates."""
    st.markdown("### 👥 Candidate Database")
    
    resp = api_request("get", "/api/candidates")
    if not resp:
        return
    
    candidates = resp.get("candidates", [])
    if not candidates:
        st.info("No candidates uploaded yet. Go to the Upload tab to add resumes.")
        return
    
    st.metric("Total Candidates", len(candidates))
    
    df = pd.DataFrame([{
        "ID": c["candidate_id"],
        "Name": c["name"],
        "Email": c["email"],
        "Skills": len(c.get("skills", [])),
        "Experience (yrs)": c.get("total_experience_years", 0),
        "Education": len(c.get("education", [])),
    } for c in candidates])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Delete candidate
    st.markdown("---")
    del_id = st.text_input("Candidate ID to delete", placeholder="e.g. abc123")
    if del_id and st.button("🗑️ Delete Candidate", type="secondary"):
        resp = api_request("delete", f"/api/candidates/{del_id}")
        if resp:
            st.success(resp.get("message", "Deleted"))
            st.rerun()


# Main app
def main():
    render_sidebar()
    
    st.markdown('<p class="main-header">AI Resume Screener</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent candidate ranking powered by NLP & semantic embeddings</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Resumes", "🏆 Rank Candidates", "👥 Candidates"])
    
    with tab1:
        render_upload_tab()
    with tab2:
        render_ranking_tab()
    with tab3:
        render_candidates_tab()


if __name__ == "__main__":
    main()
