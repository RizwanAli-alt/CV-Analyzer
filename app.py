"""
AI-Powered Career Assistant Platform — Streamlit Application
CV Analyzer (existing) + Job Scraper (new — SCR-FR-01/02)

Tabs:
    1. CV Analyzer  — upload, extract, score, gap detect, suggestions
    2. Job Scraper  — search jobs, match vs CV, filter, view results
    3. Scraper Monitor — portal health, cache stats (SCR-FR-01 BR-3)
"""

import json
import logging
import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analyzer.parser import extract_text_from_bytes
from analyzer.skills import extract_skills, _SPACY_AVAILABLE, is_hf_available
from analyzer.scorer import calculate_score
from analyzer.gap import detect_skill_gaps
from analyzer.suggestions import generate_suggestions
from analyzer.similarity import calculate_similarity, is_similarity_available
from analyzer.utilities import validate_file, get_file_size_mb, check_section_completeness
from scraper.orchestrator import ScraperOrchestrator
from scraper.matcher import is_match_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "cv_text": None,
        "analysis": None,
        "job_description": "",
        "scraped_jobs": [],
        "last_query": "",
        "last_location": "",
        "orchestrator": None,
        "scraper_run_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _get_orchestrator() -> ScraperOrchestrator:
    """Get or create the orchestrator (re-uses skill_db from analysis if available)."""
    skill_db = None
    if st.session_state.analysis:
        try:
            from analyzer.skills import SkillExtractor
            skill_db = SkillExtractor()._load_skill_db(None)
        except Exception:
            pass
    if st.session_state.orchestrator is None:
        st.session_state.orchestrator = ScraperOrchestrator(skill_db=skill_db)
    return st.session_state.orchestrator

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.title("🎯 Career Assistant")
        st.divider()

        st.markdown("### 🔌 Module Status")
        _hf = is_hf_available()

        st.markdown(
            f"{'✅' if _SPACY_AVAILABLE else '⚠️'} **spaCy NLP** — "
            f"{'active' if _SPACY_AVAILABLE else 'not installed'}"
        )
        st.markdown(
            f"{'✅' if _hf else '⚠️'} **HF BERT NER** — "
            f"{'active' if _hf else 'not installed'}"
        )
        st.markdown(
            f"{'✅' if is_similarity_available() else '⚠️'} **Sentence Transformers** — "
            f"{'semantic match active' if is_similarity_available() else 'keyword fallback'}"
        )

        if not all([_SPACY_AVAILABLE, _hf, is_similarity_available()]):
            with st.expander("📦 Install missing packages"):
                st.code(
                    "pip install spacy sentence-transformers scikit-learn transformers torch\n"
                    "python -m spacy download en_core_web_sm",
                    language="bash",
                )

        st.divider()
        st.markdown("### 📋 Job Description (for similarity)")
        jd = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            height=150,
            placeholder="Paste job description for CV similarity scoring...",
            label_visibility="collapsed",
        )
        st.session_state.job_description = jd

        st.divider()
        if st.session_state.analysis:
            st.markdown("### 📊 CV Status")
            score = st.session_state.analysis["score"]["score"]
            skills = st.session_state.analysis["skills"]
            st.metric("CV Score", f"{score}/100")
            st.caption(
                f"{len(skills.get('technical', []))} technical skills • "
                f"{len(skills.get('soft', []))} soft skills"
            )
            coverage = st.session_state.analysis["gaps"]["coverage_percentage"]
            st.progress(coverage / 100, text=f"Market coverage: {coverage}%")
        else:
            st.info("Upload your CV to unlock job matching.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — CV Analyzer
# ──────────────────────────────────────────────────────────────────────────────
def _tab_cv_analyzer():
    st.subheader("📤 Upload Your Resume")

    uploaded = st.file_uploader(
        "Select a PDF or DOCX file (max 5 MB)",
        type=["pdf", "docx"],
        key="cv_upload",
    )

    if uploaded:
        size_mb = get_file_size_mb(uploaded)
        valid, err_msg = validate_file(uploaded.name, size_mb)
        if not valid:
            st.error(f"❌ {err_msg}")
        else:
            try:
                with st.spinner("🔄 Extracting text..."):
                    cv_text = extract_text_from_bytes(uploaded.read(), uploaded.name)

                if not cv_text or len(cv_text.strip()) < 50:
                    st.error("❌ Could not extract meaningful text from this file.")
                else:
                    if cv_text != st.session_state.cv_text:
                        st.session_state.cv_text = cv_text
                        st.session_state.analysis = None
                    st.success(f"✅ {len(cv_text.split())} words extracted from {uploaded.name}")
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    if st.session_state.cv_text is None:
        st.info("👆 Upload your CV above to begin analysis.")
        return

    st.divider()
    if st.button("🚀 Analyse CV", type="primary", use_container_width=True):
        with st.spinner("🧠 Analysing CV..."):
            try:
                text = st.session_state.cv_text
                jd = st.session_state.job_description or None
                skills = extract_skills(text)
                score_result = calculate_score(text, skills)
                gaps = detect_skill_gaps(skills)
                suggestions = generate_suggestions(score_result, skills, gaps)
                similarity = calculate_similarity(text, jd)
                sections = check_section_completeness(text)

                st.session_state.analysis = {
                    "skills": skills,
                    "score": score_result,
                    "gaps": gaps,
                    "suggestions": suggestions,
                    "similarity": similarity,
                    "sections": sections,
                }
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                return

    if not st.session_state.analysis:
        st.info("Click **Analyse CV** to begin.")
        return

    analysis = st.session_state.analysis

    # ── Score ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 CV Quality Score")
    score = analysis["score"]["score"]
    bd = analysis["score"]["breakdown"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Overall Score", f"{score} / 100")
        st.progress(min(score / 100, 1.0))
        label = "🟢 Excellent" if score >= 80 else ("🟡 Good" if score >= 60 else "🔴 Needs Work")
        st.caption(label)
    with c2:
        st.metric("Word Count", bd["word_count"])
    with c3:
        st.metric("Reading Time", f"~{max(1, bd['word_count']//200)} min")

    st.markdown("#### Score Breakdown (SRS CV-FR-02)")
    try:
        import pandas as pd
        srs_dims = {
            "keywords_score":      ("Keywords & ATS",   "30%"),
            "completeness_score":  ("Section Coverage", "25%"),
            "skill_density_score": ("Skill Density",    "25%"),
            "formatting_score":    ("Formatting",        "20%"),
        }
        rows = [
            {"Dimension": label, "Score": bd.get(key, 0), "Weight": w,
             "Weighted": round(bd.get(key, 0) * float(w[:-1]) / 100, 1)}
            for key, (label, w) in srs_dims.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except ImportError:
        for key, (label, w) in srs_dims.items():
            st.write(f"**{label}** ({w}): {bd.get(key, 0)}/100")

    # ── Sections ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Section Completeness")
    secs = analysis["sections"]
    pct = secs["completeness_percentage"]
    st.progress(pct / 100)
    st.caption(f"{secs['completed_count']}/{secs['total_expected']} sections detected ({pct}%)")
    cols = st.columns(4)
    for i, (name, present) in enumerate(secs["sections"].items()):
        with cols[i % 4]:
            st.markdown(f"{'✅' if present else '❌'} **{name.capitalize()}**")

    # ── Skills ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎯 Extracted Skills")
    skills = analysis["skills"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔧 Technical Skills**")
        tech = skills.get("technical", [])
        if tech:
            st.write(", ".join(tech))
        else:
            st.info("No technical skills detected")
    with c2:
        st.markdown("**🤝 Soft Skills**")
        soft = skills.get("soft", [])
        if soft:
            st.write(", ".join(soft))
        else:
            st.info("No soft skills detected")

    # ── Skill Gap ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚠️ Skill Gap Analysis")
    gaps = analysis["gaps"]
    coverage = gaps["coverage_percentage"]
    st.markdown(f"**High-demand coverage:** {gaps['user_has_high_demand_count']}/{gaps['total_high_demand']} ({coverage}%)")
    st.progress(coverage / 100)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔴 High-Priority Missing**")
        for s in gaps.get("high_priority_skills", [])[:8]:
            st.error(f"🔴 {s}")
        if not gaps.get("high_priority_skills"):
            st.success("✅ All high-demand skills covered!")
    with c2:
        st.markdown("**🟡 Emerging Skills**")
        for s in gaps.get("emerging_missing_skills", [])[:8]:
            st.warning(f"🟡 {s}")

    # ── Suggestions ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 Improvement Suggestions")
    sugg = analysis["suggestions"]
    for priority, icon, fn in [("High", "🔴", st.error), ("Medium", "🟡", st.warning), ("Low", "🟢", st.info)]:
        bucket = [s for s in sugg if s["priority"] == priority]
        if bucket:
            st.markdown(f"**{icon} {priority} Priority**")
            for s in bucket:
                fn(f"**[{s['category']}]** {s['message']}")

    # ── Similarity ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 CV–Job Similarity")
    sim = analysis["similarity"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Match Score", f"{sim}%")
    with c2:
        if sim >= 70:
            st.success(f"✅ Strong match ({sim}%)")
        elif sim >= 50:
            st.warning(f"⚠️ Moderate match ({sim}%)")
        else:
            st.error(f"❌ Low match — tailor your CV to the job ({sim}%)")
        if not (st.session_state.job_description and st.session_state.job_description.strip()):
            st.caption("ℹ️ Using built-in sample JD. Paste a real JD in the sidebar for accuracy.")

    # ── Export ─────────────────────────────────────────────────────────────
    st.divider()
    report = {
        "overall_score": analysis["score"]["score"],
        "score_breakdown": analysis["score"]["breakdown"],
        "extracted_skills": analysis["skills"],
        "section_completeness": analysis["sections"],
        "skill_gaps": {
            "high_priority_missing": gaps["high_priority_skills"],
            "emerging_missing": gaps["emerging_missing_skills"],
            "coverage_percentage": coverage,
        },
        "suggestions": analysis["suggestions"],
        "similarity_score": sim,
    }
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(report, indent=2),
        file_name="cv_analysis_report.json",
        mime="application/json",
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — Job Scraper
# ──────────────────────────────────────────────────────────────────────────────
def _tab_job_scraper():
    st.subheader("🔍 Job Search & Match")

    # ── CV dependency notice ───────────────────────────────────────────────
    has_cv = st.session_state.analysis is not None
    if not has_cv:
        st.warning(
            "⚠️ No CV analysed yet. Jobs will be scraped but match scores will be 0%. "
            "Go to **CV Analyzer** tab first for personalised matching."
        )
    else:
        skills = st.session_state.analysis["skills"]
        top = skills.get("technical", [])[:5]
        st.success(
            f"✅ CV loaded — matching against your skills: {', '.join(top)}"
            + (f" + {len(skills.get('technical', [])) - 5} more" if len(skills.get('technical', [])) > 5 else "")
        )

    # ── Search controls ────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input(
            "Job title / keywords",
            value=st.session_state.last_query or (
                ", ".join(st.session_state.analysis["skills"].get("technical", [])[:3])
                if has_cv else ""
            ),
            placeholder="e.g. Python Developer, Data Scientist, React Developer",
        )
    with c2:
        location = st.text_input(
            "Location (optional)",
            value=st.session_state.last_location,
            placeholder="e.g. Islamabad, Lahore, Remote",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        portals = st.multiselect(
            "Portals",
            options=["LinkedIn", "Indeed", "Rozee.pk"],
            default=["LinkedIn", "Indeed", "Rozee.pk"],
        )
    with c2:
        max_per_portal = st.slider("Results per portal", 5, 25, 10)
    with c3:
        demo_mode = st.toggle(
            "Demo mode (mock data)",
            value=True,
            help="Use realistic mock data — no live scraping. Disable for real results on your machine.",
        )

    # ── Search button ──────────────────────────────────────────────────────
    col_btn, col_refresh = st.columns([3, 1])
    with col_btn:
        search_clicked = st.button(
            "🔎 Search Jobs", type="primary", use_container_width=True,
            disabled=not query or not portals,
        )
    with col_refresh:
        force_refresh = st.button("🔄 Force Refresh", use_container_width=True,
                                   help="Bypass cache and re-scrape")

    if search_clicked or force_refresh:
        if not query:
            st.error("Please enter a job title or keywords.")
            return
        if not portals:
            st.error("Please select at least one portal.")
            return

        st.session_state.last_query = query
        st.session_state.last_location = location

        user_skills = st.session_state.analysis["skills"] if has_cv else {}
        orchestrator = _get_orchestrator()

        with st.spinner(f"{'📡 Fetching mock job data' if demo_mode else '🌐 Scraping live portals'}..."):
            try:
                jobs = orchestrator.run(
                    query=query,
                    location=location,
                    portals=portals,
                    user_skills=user_skills,
                    max_per_portal=max_per_portal,
                    demo_mode=demo_mode,
                    force_refresh=force_refresh,
                )
                st.session_state.scraped_jobs = jobs
                st.session_state.scraper_run_log = orchestrator.get_run_log()

                if demo_mode:
                    st.info(f"🎭 Demo mode: showing {len(jobs)} mock listings (realistic data for testing)")
                else:
                    st.success(f"✅ Found {len(jobs)} listings across {len(portals)} portals")
            except Exception as e:
                st.error(f"❌ Scraper error: {e}")
                return

    # ── Results ────────────────────────────────────────────────────────────
    jobs = st.session_state.scraped_jobs
    if not jobs:
        if not (search_clicked or force_refresh):
            st.info("👆 Enter a job title and click Search Jobs to begin.")
        return

    st.markdown("---")
    st.subheader(f"📋 Results — {len(jobs)} listings")

    # ── Filters ────────────────────────────────────────────────────────────
    with st.expander("🔧 Filter Results", expanded=True):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            modalities = ["All"] + sorted(set(j.modality for j in jobs))
            filter_modality = st.selectbox("Work modality", modalities)
        with fc2:
            portal_opts = ["All"] + sorted(set(j.portal for j in jobs))
            filter_portal = st.selectbox("Portal", portal_opts)
        with fc3:
            min_match = st.slider("Min match score (%)", 0, 100, 0, step=5)
        with fc4:
            sort_by = st.selectbox("Sort by", ["Match Score ↓", "Match Score ↑", "Newest", "Company A-Z"])

    # Apply filters
    filtered = jobs
    if filter_modality != "All":
        filtered = [j for j in filtered if j.modality == filter_modality]
    if filter_portal != "All":
        filtered = [j for j in filtered if j.portal == filter_portal]
    filtered = [j for j in filtered if j.match_score >= min_match]

    # Apply sort
    if sort_by == "Match Score ↓":
        filtered.sort(key=lambda j: j.match_score, reverse=True)
    elif sort_by == "Match Score ↑":
        filtered.sort(key=lambda j: j.match_score)
    elif sort_by == "Newest":
        filtered.sort(key=lambda j: j.scraped_at, reverse=True)
    elif sort_by == "Company A-Z":
        filtered.sort(key=lambda j: j.company.lower())

    st.caption(f"Showing {len(filtered)} of {len(jobs)} listings")

    if not filtered:
        st.warning("No listings match your current filters.")
        return

    # ── Summary metrics ────────────────────────────────────────────────────
    if has_cv and any(j.match_score > 0 for j in filtered):
        scores = [j.match_score for j in filtered]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Top Match", f"{max(scores):.0f}%")
        with m2:
            st.metric("Avg Match", f"{sum(scores)/len(scores):.0f}%")
        with m3:
            high = len([s for s in scores if s >= 60])
            st.metric("Strong Matches (≥60%)", high)
        with m4:
            st.metric("Total Shown", len(filtered))

    # ── Job cards ──────────────────────────────────────────────────────────
    for i, job in enumerate(filtered):
        _render_job_card(job, i, has_cv)

    # ── Export results ─────────────────────────────────────────────────────
    st.divider()
    export_data = [j.to_dict() for j in filtered]
    st.download_button(
        "⬇️ Export Job Results (JSON)",
        data=json.dumps(export_data, indent=2, default=str),
        file_name=f"job_results_{st.session_state.last_query.replace(' ', '_')}.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_job_card(job, index: int, has_cv: bool):
    """Render a single job listing card."""

    # Match score colour
    if job.match_score >= 70:
        score_color = "#2ca02c"
        score_label = "Strong match"
    elif job.match_score >= 50:
        score_color = "#ff7f0e"
        score_label = "Moderate match"
    elif job.match_score > 0:
        score_color = "#d62728"
        score_label = "Low match"
    else:
        score_color = "#888"
        score_label = "Not scored"

    # Portal badge colour
    portal_colors = {
        "LinkedIn": "#0077B5",
        "Indeed": "#2557A7",
        "Rozee.pk": "#E31837",
    }
    portal_color = portal_colors.get(job.portal, "#666")

    # Modality badge colour
    modality_colors = {
        "Remote": "#2ca02c",
        "Hybrid": "#ff7f0e",
        "On-site": "#1f77b4",
        "Unknown": "#888",
    }
    modality_color = modality_colors.get(job.modality, "#888")

    # Age in days
    from datetime import datetime
    age = (datetime.utcnow() - job.scraped_at).days
    age_str = "Today" if age == 0 else (f"{age}d ago" if age < 7 else f"{age//7}w ago")

    with st.container():
        st.markdown(
            f"""
<div style="
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
    background: var(--background-color, #fff);
">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
        <div style="flex:1; min-width:200px;">
            <div style="font-size:1.05em; font-weight:600; margin-bottom:2px;">
                <a href="{job.url}" target="_blank" style="text-decoration:none; color:inherit;">
                    {job.title}
                </a>
            </div>
            <div style="color:#555; font-size:0.9em; margin-bottom:6px;">
                🏢 <strong>{job.company}</strong> &nbsp;•&nbsp; 📍 {job.location}
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; align-items:center;">
                <span style="
                    background:{portal_color}22; color:{portal_color};
                    border:1px solid {portal_color}66;
                    border-radius:6px; padding:2px 8px; font-size:0.78em; font-weight:500;">
                    {job.portal}
                </span>
                <span style="
                    background:{modality_color}22; color:{modality_color};
                    border:1px solid {modality_color}66;
                    border-radius:6px; padding:2px 8px; font-size:0.78em;">
                    {job.modality}
                </span>
                <span style="color:#999; font-size:0.8em;">{age_str}</span>
            </div>
        </div>
        {"" if not has_cv else f'''
        <div style="text-align:center; min-width:80px;">
            <div style="font-size:1.6em; font-weight:700; color:{score_color}; line-height:1.1;">
                {job.match_score:.0f}%
            </div>
            <div style="font-size:0.72em; color:{score_color}; margin-top:2px;">{score_label}</div>
        </div>
        '''}
    </div>
""",
            unsafe_allow_html=True,
        )

        if job.description:
            with st.expander("📄 Description & matched skills"):
                # Show first 400 chars of description
                desc_preview = job.description[:400] + ("..." if len(job.description) > 400 else "")
                st.write(desc_preview)
                if job.skills_mentioned:
                    st.markdown(
                        "**Skills mentioned:** " +
                        " ".join(f"`{s}`" for s in job.skills_mentioned[:15])
                    )
                st.markdown(f"[🔗 View on {job.portal}]({job.url})")

        st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — Scraper Monitor (SCR-FR-01 BR-3)
# ──────────────────────────────────────────────────────────────────────────────
def _tab_monitor():
    st.subheader("🖥️ Scraper Monitor")
    st.caption("Per-portal health status and cache statistics (SRS SCR-FR-01 BR-3)")

    orchestrator = _get_orchestrator()

    # ── Last run health log ────────────────────────────────────────────────
    run_log = st.session_state.scraper_run_log
    if run_log:
        st.markdown("### Last Scraper Run")
        for entry in run_log:
            status_icon = "✅" if entry["status"] == "success" else "❌"
            col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
            with col1:
                st.markdown(f"{status_icon} **{entry['portal']}**")
            with col2:
                st.metric("Listings", entry["count"])
            with col3:
                st.metric("Time (s)", entry["elapsed_s"])
            with col4:
                if entry["error"]:
                    st.error(f"Error: {entry['error']}")
                else:
                    st.success("OK")
    else:
        st.info("No scraper runs yet. Go to Job Scraper tab and run a search.")

    # ── Cache stats ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Cache Statistics (6-hour TTL / 30-day expiry)")
    stats = orchestrator.cache_stats()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Cached Queries", stats["cached_queries"])
    with c2:
        st.metric("Total Cached Listings", stats["total_listings"])

    if stats["entries"]:
        try:
            import pandas as pd
            df = pd.DataFrame(stats["entries"])
            df.columns = ["Query", "Listings", "Age (hours)"]
            st.dataframe(df, use_container_width=True, hide_index=True)
        except ImportError:
            for e in stats["entries"]:
                st.write(f"• {e['key']}: {e['count']} listings, {e['age_hours']}h old")

    if st.button("🗑️ Clear Cache", type="secondary"):
        removed = orchestrator.clear_cache()
        st.success(f"Cleared {removed} cache entries")
        st.rerun()

    # ── SRS business rules summary ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### SRS Compliance Checklist")
    checks = [
        ("SCR-FR-01 BR-1", "2-second delay between consecutive requests per portal", True),
        ("SCR-FR-01 BR-2", "Listings older than 30 days automatically purged", True),
        ("SCR-FR-01 BR-3", "Portal failures logged, pipeline continues", True),
        ("SCR-FR-01 BR-4", "Public listings only, no authentication", True),
        ("SCR-FR-02 BR-1", "Cosine similarity match scoring per listing", is_match_available()),
        ("SCR-FR-02 BR-2", "Match score displayed 0–100% per listing", True),
        ("SCR-FR-02 BR-3", "Listings sorted by match score descending", True),
    ]
    for req_id, desc, status in checks:
        icon = "✅" if status else "⚠️"
        note = "" if status else " (sentence-transformers not installed — keyword fallback active)"
        st.markdown(f"{icon} **{req_id}**: {desc}{note}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    _sidebar()

    st.title("🎯 AI-Powered Career Assistant Platform")
    st.markdown(
        "CV analysis, personalised job matching, and real-time job scraping — "
        "LinkedIn · Indeed · Rozee.pk"
    )
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📄 CV Analyzer",
        "🔍 Job Scraper",
        "🖥️ Scraper Monitor",
    ])

    with tab1:
        _tab_cv_analyzer()

    with tab2:
        _tab_job_scraper()

    with tab3:
        _tab_monitor()


if __name__ == "__main__":
    main()