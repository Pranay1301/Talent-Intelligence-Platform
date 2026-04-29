"""
Recruiter Intelligence Dashboard

Enterprise-grade Streamlit dashboard for talent screening:
- Resume upload (PDF/DOCX drag & drop)
- JD input with intelligent parsing
- Multi-stage ranked candidate table
- Score breakdown with radar charts
- Interview recommendation panel
- Candidate explanation cards
- Missing skills analysis
- Hiring funnel analytics
- Bias/fairness audit display
- Downloadable PDF/CSV reports
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
    page_title="AI Talent Intelligence Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #8892b0;
        margin-bottom: 2rem;
        letter-spacing: 0.02em;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px;
        color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .decision-strong { background: linear-gradient(135deg, #00c853, #00e676); padding: 0.8rem 1.2rem; border-radius: 8px; color: white; font-weight: 600; }
    .decision-interview { background: linear-gradient(135deg, #2196f3, #42a5f5); padding: 0.8rem 1.2rem; border-radius: 8px; color: white; font-weight: 600; }
    .decision-hold { background: linear-gradient(135deg, #ff9100, #ffab40); padding: 0.8rem 1.2rem; border-radius: 8px; color: white; font-weight: 600; }
    .decision-weak { background: linear-gradient(135deg, #ff1744, #ff5252); padding: 0.8rem 1.2rem; border-radius: 8px; color: white; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; font-family: 'Inter', sans-serif; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 12px; margin-bottom: 0.5rem; }
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
        st.error(f"⚠️ Cannot connect to API at {API_URL}. Start the FastAPI server first.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🎯 Talent Intelligence")
        st.markdown("---")

        stats = api_request("get", "/api/health")
        if stats:
            st.metric("Candidates", stats.get("total_candidates", 0))
            st.metric("FAISS Vectors", stats.get("faiss_vectors", 0))
            st.markdown("🟢 **API Online**")
        else:
            st.markdown("🔴 **API Offline**")

        st.markdown("---")
        st.markdown("### Pipeline")
        st.markdown("""
        1. 📄 Upload resumes
        2. 📋 Input job description
        3. ⚡ FAISS retrieval
        4. 🧠 Hybrid scoring
        5. 🔍 Re-ranking + explanations
        6. 🎯 Interview recommendations
        7. ✅ Bias audit
        """)
        st.markdown("---")
        st.caption("Sentence-BERT · FAISS · FastAPI · Explainable AI")


def render_upload_tab():
    st.markdown("### 📄 Resume Ingestion")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### File Upload")
        uploaded_files = st.file_uploader(
            "Drop resume files", type=["pdf", "docx"],
            accept_multiple_files=True, help="PDF or DOCX resumes",
        )
        if uploaded_files and st.button("🚀 Process Resumes", type="primary"):
            progress = st.progress(0)
            results = []
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    resp = api_request("post", "/api/upload_resume",
                        files={"file": (file.name, file.getvalue(), file.type)})
                    if resp:
                        results.append(resp)
                progress.progress((i + 1) / len(uploaded_files))
            if results:
                st.success(f"✅ Processed {len(results)} resume(s)")
                for r in results:
                    st.info(f"**{r['name']}** — {r['skills_count']} skills, {r['experience_years']} yrs exp")

    with col2:
        st.markdown("#### Text Input")
        name = st.text_input("Candidate Name", placeholder="Jane Smith")
        resume_text = st.text_area("Resume Text", height=250, placeholder="Paste full resume text...")
        if resume_text and st.button("📝 Submit Text Resume"):
            resp = api_request("post", "/api/upload_resume_text",
                json={"name": name, "resume_text": resume_text})
            if resp:
                st.success(f"✅ **{resp['name']}** — {resp['skills_count']} skills detected")


def render_ranking_tab():
    st.markdown("### 🏆 Multi-Stage Candidate Ranking")

    jd_text = st.text_area("Job Description", height=200,
        placeholder="Paste the full job description to rank candidates against...")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Top candidates", 1, 50, 10)
    with col2:
        export_format = st.selectbox("Export format", ["json", "csv"])
    with col3:
        bias_check = st.checkbox("Enable bias audit", value=True)

    if jd_text and st.button("🔍 Rank Candidates", type="primary"):
        with st.spinner("Running multi-stage AI pipeline..."):
            result = api_request("post", "/api/rank_candidates",
                json={"jd_text": jd_text, "top_k": top_k})

        if result and result.get("ranked_candidates"):
            st.session_state["last_ranking"] = result
            candidates = result["ranked_candidates"]

            # Pipeline metrics
            metrics = result.get("pipeline_metrics", {})
            st.markdown(f"### Results: {result.get('job_title', 'Analysis')}")

            mcols = st.columns(5)
            with mcols[0]:
                st.metric("Ranked", len(candidates))
            with mcols[1]:
                st.metric("Top Score", f"{candidates[0]['final_score']:.1%}" if candidates else "—")
            with mcols[2]:
                strong = sum(1 for c in candidates
                    if c.get("interview_recommendation", {}).get("decision") in ("STRONG_HIRE", "INTERVIEW_RECOMMENDED"))
                st.metric("Interview-Ready", strong)
            with mcols[3]:
                st.metric("Total Pool", result.get("total_candidates_in_pool", "—"))
            with mcols[4]:
                st.metric("Pipeline Time", f"{metrics.get('total_ms', 0):.0f}ms")

            st.markdown("---")

            # Ranked table with decisions
            df = pd.DataFrame([{
                "Rank": c["rank"],
                "Name": c["name"],
                "Score": f"{c['final_score']:.1%}",
                "Decision": c.get("interview_recommendation", {}).get("decision", ""),
                "Role Fit": c.get("interview_recommendation", {}).get("role_fit", ""),
                "Fairness": f"{c.get('fairness_score', 1):.0%}",
            } for c in candidates])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Charts
            render_score_charts(candidates)

            # Detailed breakdowns
            st.markdown("### 📋 Candidate Intelligence")
            for c in candidates:
                render_candidate_detail(c)

            # Fairness audit
            audit = result.get("fairness_audit", {})
            if audit:
                render_fairness_audit(audit)

            # Export
            if st.button("📥 Export Report"):
                resp = api_request("post", "/api/export_report",
                    json={"jd_text": jd_text, "top_k": top_k},
                    params={"format": export_format})
                if resp:
                    st.success(f"Report exported: {resp.get('file_path', '')}")

        elif result:
            st.warning("No candidates found. Upload some resumes first!")


def render_score_charts(candidates):
    st.markdown("### 📊 Score Distribution")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=[c["name"] for c in candidates],
            y=[c["final_score"] for c in candidates],
            title="Candidate Scores",
            labels={"x": "Candidate", "y": "Score"},
            color=[c["final_score"] for c in candidates],
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False, height=400,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top = candidates[0]
        scores = top.get("score_breakdown", {}).get("scores", {})
        categories = ["Semantic", "Skills", "Experience", "Education", "Projects", "Domain"]
        values = [
            scores.get("embedding_similarity", 0),
            scores.get("skill_match", scores.get("skill_match_score", 0)),
            scores.get("experience_relevance", scores.get("experience_score", 0)),
            scores.get("education_relevance", scores.get("education_score", 0)),
            scores.get("project_impact", 0),
            scores.get("domain_alignment", 0),
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
            height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_candidate_detail(candidate):
    score = candidate["final_score"]
    decision = candidate.get("interview_recommendation", {}).get("decision", "")
    decision_emoji = {"STRONG_HIRE": "🟢", "INTERVIEW_RECOMMENDED": "🔵",
                       "HOLD_FOR_REVIEW": "🟡", "WEAK_MATCH": "🟠", "REJECT": "🔴"}.get(decision, "⚪")

    with st.expander(f"{decision_emoji} **#{candidate['rank']} {candidate['name']}** — {score:.1%} | {decision}", expanded=False):
        breakdown = candidate.get("score_breakdown", {})
        scores = breakdown.get("scores", {})
        explanation = breakdown.get("explanation", {})
        interview = candidate.get("interview_recommendation", {})

        # Score metrics
        cols = st.columns(6)
        items = [
            ("🧠 Semantic", scores.get("embedding_similarity", 0)),
            ("🎯 Skills", scores.get("skill_match", scores.get("skill_match_score", 0))),
            ("💼 Experience", scores.get("experience_relevance", scores.get("experience_score", 0))),
            ("🎓 Education", scores.get("education_relevance", scores.get("education_score", 0))),
            ("📂 Projects", scores.get("project_impact", 0)),
            ("🌐 Domain", scores.get("domain_alignment", 0)),
        ]
        for col, (label, val) in zip(cols, items):
            with col:
                st.metric(label, f"{val:.0%}")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            # Skills
            skill_analysis = explanation.get("skill_analysis", {})
            matched = skill_analysis.get("matched", explanation.get("summary", {}).get("strengths", []))
            missing = skill_analysis.get("missing", [])
            coverage = skill_analysis.get("coverage_pct", 0)

            if isinstance(matched, list) and matched:
                st.markdown(f"**✅ Matched Skills** ({coverage:.0f}% coverage):")
                st.markdown(", ".join(f"`{s}`" for s in matched[:10]))

            strengths = explanation.get("summary", {}).get("strengths", [])
            if strengths:
                st.markdown("**💪 Strengths:**")
                for s in strengths[:4]:
                    st.markdown(f"- {s}")

        with col2:
            if isinstance(missing, list) and missing:
                st.markdown("**❌ Missing Skills:**")
                st.markdown(", ".join(f"`{s}`" for s in missing[:8]))

            weaknesses = explanation.get("summary", {}).get("weaknesses", [])
            if weaknesses:
                st.markdown("**⚠️ Gaps:**")
                for w in weaknesses[:4]:
                    st.markdown(f"- {w}")

        # Interview Recommendation
        st.markdown("---")
        st.markdown("#### 🎯 Interview Recommendation")

        icols = st.columns(4)
        with icols[0]:
            st.metric("Decision", interview.get("decision", "—").replace("_", " "))
        with icols[1]:
            st.metric("Confidence", f"{interview.get('confidence', 0):.0%}")
        with icols[2]:
            st.metric("Role Fit", interview.get("role_fit", "—"))
        with icols[3]:
            st.metric("Growth Potential", interview.get("promotion_potential", "—"))

        focus = interview.get("interview_focus_areas", [])
        if focus:
            st.markdown("**📌 Interview Focus Areas:**")
            for f in focus:
                st.markdown(f"- {f}")

        risks = interview.get("risk_factors", [])
        if risks:
            st.warning("**Risk Factors:** " + " | ".join(risks))

        notes = interview.get("hiring_notes", "")
        if notes:
            st.info(f"**📝 Hiring Notes:** {notes}")


def render_fairness_audit(audit):
    st.markdown("### ✅ Fairness Audit")
    issues = audit.get("issues", [])
    if not issues:
        st.success("✅ Ranking passed fairness audit — no systemic bias detected")
    else:
        for issue in issues:
            st.warning(f"⚠️ {issue}")
    st.caption(f"Recommendation: {audit.get('recommendation', '')}")


def render_analytics_tab():
    st.markdown("### 📈 Recruiter Analytics")
    metrics = api_request("get", "/api/dashboard_metrics")
    if not metrics:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", metrics.get("total_candidates", 0))
    with col2:
        st.metric("Rankings Performed", metrics.get("total_rankings_performed", 0))
    with col3:
        st.metric("FAISS Vectors", metrics.get("faiss_vectors", 0))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        top_skills = metrics.get("top_skills", {})
        if top_skills:
            fig = px.bar(
                x=list(top_skills.values()),
                y=list(top_skills.keys()),
                orientation="h",
                title="Top Skills in Candidate Pool",
                labels={"x": "Count", "y": "Skill"},
                color=list(top_skills.values()),
                color_continuous_scale="Viridis",
            )
            fig.update_layout(height=500, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        exp_dist = metrics.get("experience_distribution", {})
        if exp_dist:
            fig = px.pie(
                names=list(exp_dist.keys()),
                values=list(exp_dist.values()),
                title="Experience Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=500,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)


def render_candidates_tab():
    st.markdown("### 👥 Candidate Database")
    resp = api_request("get", "/api/candidates")
    if not resp:
        return

    candidates = resp.get("candidates", [])
    if not candidates:
        st.info("No candidates uploaded yet.")
        return

    st.metric("Total Candidates", len(candidates))
    df = pd.DataFrame([{
        "ID": c["candidate_id"],
        "Name": c["name"],
        "Email": c["email"],
        "Skills": len(c.get("skills", [])),
        "Experience": f"{c.get('total_experience_years', 0):.1f} yrs",
        "Education": len(c.get("education", [])),
        "Certs": len(c.get("certifications", [])),
    } for c in candidates])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    del_id = st.text_input("Candidate ID to delete", placeholder="e.g. abc123")
    if del_id and st.button("🗑️ Delete", type="secondary"):
        resp = api_request("delete", f"/api/candidates/{del_id}")
        if resp:
            st.success(resp.get("message", "Deleted"))
            st.rerun()


# Main app
def main():
    render_sidebar()

    st.markdown('<p class="main-header">AI Talent Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise-grade candidate screening powered by NLP, semantic embeddings & explainable AI</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload Resumes", "🏆 Rank Candidates", "📈 Analytics", "👥 Candidates"
    ])
    with tab1:
        render_upload_tab()
    with tab2:
        render_ranking_tab()
    with tab3:
        render_analytics_tab()
    with tab4:
        render_candidates_tab()


if __name__ == "__main__":
    main()
