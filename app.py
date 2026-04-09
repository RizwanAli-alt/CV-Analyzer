"""
AI CV Analyzer — Streamlit Application

Changes from previous version:

  Issue #1 (HF NER):
    - Imported is_hf_available() from analyzer.skills
    - Added HF NER status row in sidebar Module Status section

  Issue #2 (scorer weights):
    - Score breakdown table now shows the 4 SRS dimension scores as the primary
      columns, with the UI detail scores shown in a secondary expander.

  Issue #5 (filename typos):
    - Import changed from 'analyzer.utilities' (was 'analyzer.utilites')
    - Import changed from 'analyzer.suggestions' (was 'analyzer.suggerstions')
    - Both files are now correctly named on disk.

  Issue #9 (similarity re import):
    - No change needed in app.py; fix is in similarity.py itself.
"""

import json
import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path (works from any working directory)
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analyzer.parser import extract_text_from_bytes
from analyzer.skills import extract_skills, _SPACY_AVAILABLE, is_hf_available   # ← added is_hf_available
from analyzer.scorer import calculate_score
from analyzer.gap import detect_skill_gaps
from analyzer.suggestions import generate_suggestions          # ← fixed: was suggerstions
from analyzer.similarity import calculate_similarity, is_similarity_available
from analyzer.utilities import validate_file, get_file_size_mb, check_section_completeness  # ← fixed: was utilites

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI CV Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "cv_text": None,
        "analysis": None,
        "job_description": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _sidebar():
    with st.sidebar:
        st.title("⚙️ Settings & Info")
        st.divider()

        st.markdown("### 🔌 Module Status")

        # spaCy status
        st.markdown(
            f"{'✅' if _SPACY_AVAILABLE else '⚠️'} **spaCy NLP** — "
            f"{'active' if _SPACY_AVAILABLE else 'not installed (keyword-only mode)'}"
        )

        # Hugging Face NER status  (Issue #1 — new row)
        _hf = is_hf_available()
        st.markdown(
            f"{'✅' if _hf else '⚠️'} **HF Transformer NER** (SRS CV-FR-01) — "
            f"{'active (dslim/bert-base-NER)' if _hf else 'not installed (SRS non-compliant)'}"
        )

        # Sentence Transformers status
        st.markdown(
            f"{'✅' if is_similarity_available() else '⚠️'} **Sentence Transformers** — "
            f"{'active' if is_similarity_available() else 'not installed (similarity = 0)'}"
        )

        missing_any = not _SPACY_AVAILABLE or not _hf or not is_similarity_available()
        if missing_any:
            with st.expander("📦 Install missing dependencies"):
                st.code(
                    "pip install spacy sentence-transformers scikit-learn transformers torch\n"
                    "python -m spacy download en_core_web_sm",
                    language="bash",
                )

        st.divider()
        st.markdown("### 📋 Job Description")
        st.markdown("Paste a job description for a more accurate similarity match:")
        jd = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            height=200,
            placeholder="Paste the job description here...",
            label_visibility="collapsed",
        )
        st.session_state.job_description = jd

        st.divider()
        st.markdown("**Tip:** The more complete your CV, the better the analysis.")
        st.markdown("Supported formats: **PDF**, **DOCX** (max 5 MB)")

# ---------------------------------------------------------------------------
# File upload & text extraction
# ---------------------------------------------------------------------------

def _upload_section():
    st.subheader("📤 Upload Your Resume")

    uploaded = st.file_uploader(
        "Select a PDF or DOCX file (max 5 MB)",
        type=["pdf", "docx"],
        help="Supported: PDF, DOCX",
    )

    if uploaded is None:
        return

    size_mb = get_file_size_mb(uploaded)
    valid, err_msg = validate_file(uploaded.name, size_mb)

    if not valid:
        st.error(f"❌ {err_msg}")
        return

    try:
        with st.spinner("🔄 Extracting text from file…"):
            file_bytes = uploaded.read()
            cv_text = extract_text_from_bytes(file_bytes, uploaded.name)

        if not cv_text or len(cv_text.strip()) < 50:
            st.error(
                "❌ Could not extract meaningful text. "
                "Make sure the file isn't scanned/image-only."
            )
            return

        if cv_text != st.session_state.cv_text:
            st.session_state.cv_text = cv_text
            st.session_state.analysis = None

        st.success(f"✅ File processed — {len(cv_text.split())} words extracted.")

    except Exception as e:
        logger.exception("File processing failed")
        st.error(f"❌ Error processing file: {e}")

# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------

def _run_analysis() -> dict:
    """Run all analysis modules and return results dict."""
    text = st.session_state.cv_text
    jd = st.session_state.job_description or None

    with st.spinner("🧠 Analysing CV… (HF NER may take a few seconds on first run)"):
        skills = extract_skills(text)
        skills.setdefault("technical", [])
        skills.setdefault("soft", [])

        score_result = calculate_score(text, skills)
        gaps = detect_skill_gaps(skills)
        suggestions = generate_suggestions(score_result, skills, gaps)
        similarity = calculate_similarity(text, jd)
        sections = check_section_completeness(text)

    return {
        "skills": skills,
        "score": score_result,
        "gaps": gaps,
        "suggestions": suggestions,
        "similarity": similarity,
        "sections": sections,
    }

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_score(score_result: dict):
    st.subheader("📊 CV Quality Score")

    score = score_result["score"]
    bd = score_result["breakdown"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Score", f"{score} / 100")
        st.progress(min(score / 100, 1.0))
        if score >= 80:
            st.caption("🟢 Excellent")
        elif score >= 60:
            st.caption("🟡 Good")
        else:
            st.caption("🔴 Needs Improvement")

    with col2:
        st.metric("Word Count", bd["word_count"])

    with col3:
        reading_time = max(1, bd["word_count"] // 200)
        st.metric("Est. Reading Time", f"~{reading_time} min")

    # ── Primary: 4 SRS dimension scores ─────────────────────────────────────
    st.markdown("#### Score Breakdown (SRS CV-FR-02 — 4 Dimensions)")

    import pandas as pd

    srs_dims = {
        "keywords_score":      ("Keywords & ATS",   0.30),
        "completeness_score":  ("Section Coverage", 0.25),
        "skill_density_score": ("Skill Density",    0.25),
        "formatting_score":    ("Formatting",        0.20),
    }

    rows = []
    for key, (label, weight) in srs_dims.items():
        val = bd.get(key, 0)
        rows.append({
            "Dimension (SRS)": label,
            "Score": val,
            "Weight": f"{int(weight*100)}%",
            "Weighted": round(val * weight, 1),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            go.Bar(
                x=[r["Dimension (SRS)"] for r in rows],
                y=[r["Score"] for r in rows],
                marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                text=[f"{r['Score']}" for r in rows],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="SRS Dimension Scores",
            yaxis=dict(range=[0, 110], title="Score (0–100)"),
            xaxis_tickangle=-15,
            height=380,
            margin=dict(t=50, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for chart visualization: pip install plotly")

    # ── Secondary: UI detail sub-scores in expander ──────────────────────────
    with st.expander("🔍 Detailed sub-scores (display only)"):
        detail_keys = {
            "experience_score":      "Experience",
            "education_score":       "Education",
            "keyword_density_score": "Keyword Density",
            "projects_score":        "Projects",
        }
        detail_rows = [
            {"Component": label, "Score": bd.get(key, 0)}
            for key, label in detail_keys.items()
        ]
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
        st.caption(
            "These sub-scores are for information only and do not affect the overall "
            "weighted total, which is computed from the 4 SRS dimensions above."
        )


def _display_sections(sections_result: dict):
    st.subheader("📋 CV Section Completeness")
    secs = sections_result["sections"]
    pct = sections_result["completeness_percentage"]

    st.progress(pct / 100)
    st.caption(
        f"{sections_result['completed_count']} / {sections_result['total_expected']} "
        f"sections detected ({pct}%)"
    )

    cols = st.columns(4)
    for i, (name, present) in enumerate(secs.items()):
        with cols[i % 4]:
            icon = "✅" if present else "❌"
            st.markdown(f"{icon} **{name.capitalize()}**")


def _display_skills(extracted_skills: dict):
    st.subheader("🎯 Extracted Skills")
    _hf_active = is_hf_available()
    st.caption(
        f"Extraction pipeline: keyword matching"
        f"{' + spaCy NER' if _SPACY_AVAILABLE else ''}"
        f"{' + HF BERT NER ✅' if _hf_active else ' (HF NER ⚠️ not installed)'}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔧 Technical Skills")
        tech = extracted_skills.get("technical", [])
        if tech:
            for i in range(0, len(tech), 3):
                row = tech[i:i+3]
                cols = st.columns(len(row))
                for j, skill in enumerate(row):
                    with cols[j]:
                        st.markdown(
                            f'<span style="background:#1f77b420;border:1px solid #1f77b4;'
                            f'border-radius:12px;padding:2px 10px;font-size:0.85em">{skill}</span>',
                            unsafe_allow_html=True,
                        )
                st.write("")
        else:
            st.info("No technical skills detected")

    with col2:
        st.markdown("#### 🤝 Soft Skills")
        soft = extracted_skills.get("soft", [])
        if soft:
            for i in range(0, len(soft), 3):
                row = soft[i:i+3]
                cols = st.columns(len(row))
                for j, skill in enumerate(row):
                    with cols[j]:
                        st.markdown(
                            f'<span style="background:#2ca02c20;border:1px solid #2ca02c;'
                            f'border-radius:12px;padding:2px 10px;font-size:0.85em">{skill}</span>',
                            unsafe_allow_html=True,
                        )
                st.write("")
        else:
            st.info("No soft skills detected")


def _display_gaps(gaps: dict):
    st.subheader("⚠️ Skill Gap Analysis")

    coverage = gaps["coverage_percentage"]
    st.markdown(
        f"**High-demand skill coverage:** {gaps['user_has_high_demand_count']} / "
        f"{gaps['total_high_demand']} ({coverage}%)"
    )
    st.progress(coverage / 100)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔴 High-Priority Missing Skills")
        hp = gaps.get("high_priority_skills", [])
        if hp:
            for skill in hp[:8]:
                st.error(f"🔴 {skill}")
        else:
            st.success("✅ All high-demand skills covered!")

    with col2:
        st.markdown("#### 🟡 Emerging Skills to Consider")
        em = gaps.get("emerging_missing_skills", [])
        if em:
            for skill in em[:8]:
                st.warning(f"🟡 {skill}")
        else:
            st.success("✅ Good emerging skills coverage!")

    try:
        import plotly.graph_objects as go

        have = gaps["user_has_high_demand_count"]
        missing = gaps["total_high_demand"] - have

        fig = go.Figure(
            go.Pie(
                labels=["Skills You Have", "Skills Missing"],
                values=[have, missing],
                hole=0.55,
                marker_colors=["#2ca02c", "#d62728"],
                textinfo="label+percent",
            )
        )
        fig.update_layout(
            title="High-Demand Skill Coverage",
            height=350,
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass


def _display_suggestions(suggestions: list):
    st.subheader("💡 Improvement Suggestions")

    high   = [s for s in suggestions if s["priority"] == "High"]
    medium = [s for s in suggestions if s["priority"] == "Medium"]
    low    = [s for s in suggestions if s["priority"] == "Low"]

    if high:
        st.markdown("#### 🔴 High Priority")
        for s in high:
            st.error(f"**[{s['category']}]** {s['message']}")

    if medium:
        st.markdown("#### 🟡 Medium Priority")
        for s in medium:
            st.warning(f"**[{s['category']}]** {s['message']}")

    if low:
        st.markdown("#### 🟢 Low Priority")
        for s in low:
            st.info(f"**[{s['category']}]** {s['message']}")

    if not suggestions:
        st.success("🎉 No major issues found — your CV looks great!")


def _display_similarity(similarity_score: float):
    st.subheader("🔍 CV–Job Match (Semantic Similarity)")

    using_sample = not (
        st.session_state.job_description
        and st.session_state.job_description.strip()
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Match Score", f"{similarity_score}%")
        if not is_similarity_available():
            st.warning("⚠️ sentence-transformers not installed — score is 0.")

    with col2:
        if not is_similarity_available():
            st.info("Install sentence-transformers to enable semantic matching.")
        elif similarity_score >= 70:
            st.success(f"✅ Strong match ({similarity_score}%)")
        elif similarity_score >= 50:
            st.warning(f"⚠️ Moderate match — add more industry keywords ({similarity_score}%)")
        else:
            st.error(f"❌ Low match — tailor CV to the job description ({similarity_score}%)")

        if using_sample:
            st.caption(
                "ℹ️ Using the built-in sample job description. "
                "Paste a real JD in the sidebar for a more accurate score."
            )


def _export_section(analysis: dict):
    st.markdown("---")
    st.subheader("📥 Export Analysis Report")

    report = {
        "overall_score": analysis["score"]["score"],
        "score_breakdown": analysis["score"]["breakdown"],
        "extracted_skills": analysis["skills"],
        "section_completeness": analysis["sections"],
        "skill_gaps": {
            "high_priority_missing": analysis["gaps"]["high_priority_skills"],
            "emerging_missing": analysis["gaps"]["emerging_missing_skills"],
            "coverage_percentage": analysis["gaps"]["coverage_percentage"],
        },
        "suggestions": analysis["suggestions"],
        "similarity_score": analysis["similarity"],
    }

    st.download_button(
        label="⬇️ Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name="cv_analysis_report.json",
        mime="application/json",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _sidebar()

    st.title("📄 AI-Powered CV Analyzer")
    st.markdown(
        "Upload your resume to get an **AI-driven quality score**, "
        "**skill gap analysis**, and **actionable improvement suggestions**."
    )
    st.divider()

    _upload_section()

    if st.session_state.cv_text is None:
        st.info("👆 Upload your CV above to get started.")
        return

    st.divider()

    if st.button("🚀 Analyse CV", type="primary", use_container_width=True):
        try:
            st.session_state.analysis = _run_analysis()
        except Exception as e:
            logger.exception("Analysis pipeline failed")
            st.error(f"❌ Analysis failed: {e}")
            return

    if st.session_state.analysis is None:
        st.info("Click **Analyse CV** to begin.")
        return

    analysis = st.session_state.analysis

    st.divider()
    _display_score(analysis["score"])

    st.divider()
    _display_sections(analysis["sections"])

    st.divider()
    _display_skills(analysis["skills"])

    st.divider()
    _display_gaps(analysis["gaps"])

    st.divider()
    _display_suggestions(analysis["suggestions"])

    st.divider()
    _display_similarity(analysis["similarity"])

    _export_section(analysis)


if __name__ == "__main__":
    main()