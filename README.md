# AI-Powered Career Assistant Platform

Streamlit app that combines:

- CV analysis and quality scoring
- Skill extraction and market gap detection
- CV-to-job similarity scoring
- Multi-portal job scraping and match ranking
- Scraper monitoring and cache insights

The project currently runs as a local Streamlit prototype with modular business logic in `analyzer/` and `scraper/`.

## Features

### 1) CV Analyzer Tab

- Upload CV in PDF or DOCX (max 5 MB)
- Extract text safely from file bytes
- Extract skills using:
    - keyword matching (`models/skill_db.json`)
    - spaCy NER (if installed)
    - Hugging Face BERT NER (if installed)
- Score CV quality (0-100) using 4 weighted dimensions:
    - Keywords & ATS (30%)
    - Section Coverage (25%)
    - Skill Density (25%)
    - Formatting (20%)
- Detect section completeness (contact, summary, experience, education, skills, projects, certifications)
- Run skill-gap analysis against market demand (`models/market_skills.json`)
- Generate prioritized suggestions (High/Medium/Low)
- Compute CV-Job similarity (semantic model when available)
- Export analysis report as JSON

### 2) Job Scraper Tab

- Search jobs by query and optional location
- Scrape from selected portals:
    - LinkedIn
    - Indeed
    - Rozee.pk
- Demo mode enabled by default (uses realistic mock listings)
- Optional live scraping mode for real portal requests
- Match-score jobs against analyzed CV skills (0-100)
- Filter by modality, portal, and minimum score
- Sort by score, recency, or company
- Export filtered results as JSON

### 3) Scraper Monitor Tab

- Last run health per portal (success/error, count, elapsed time)
- Cache statistics
- One-click cache clear
- SRS rule checklist for scraper and matcher behavior

## Architecture

```
cv_analyzer/
├── app.py                     # Streamlit entry point and UI tabs
├── requirements.txt
├── analyzer/
│   ├── __init__.py
│   ├── parser.py              # PDF/DOCX extraction, text cleaning, text stats
│   ├── skills.py              # Keyword + spaCy + HF NER extraction
│   ├── scorer.py              # 4-dimension weighted CV scoring engine
│   ├── gap.py                 # Market skill gap detection
│   ├── suggestions.py         # Priority suggestion generation
│   ├── similarity.py          # CV-JD semantic similarity
│   └── utilities.py           # File validation, section/contact helpers
├── scraper/
│   ├── base.py                # Base scraper, JobListing, throttling, retries
│   ├── linkedin.py            # LinkedIn scraper
│   ├── indeed.py              # Indeed scraper
│   ├── rozee.py               # Rozee.pk scraper
│   ├── matcher.py             # Job-to-skill match scoring and sorting
│   ├── cache.py               # In-memory cache (6h TTL, 30-day purge)
│   ├── mock_data.py           # Demo-mode mock listings
│   └── orchestrator.py        # Unified scrape -> score -> sort pipeline
├── models/
│   ├── skill_db.json
│   └── market_skills.json
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10+ (3.11 recommended)
- Internet connection for:
    - first-time model downloads (optional components)
    - live scraping mode

## Setup

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) (Optional) Ensure spaCy model is available

```powershell
python -m spacy download en_core_web_sm
```

Note: `requirements.txt` already includes the `en_core_web_sm` wheel, but this command is still safe if you need to repair the model install.

## Run

```powershell
streamlit run app.py
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Typical Workflow

1. Open the app and upload a CV (PDF/DOCX).
2. Optionally paste a target job description in the sidebar.
3. Click **Analyse CV**.
4. Review score, gaps, and suggestions.
5. Go to **Job Scraper** and search jobs.
6. Keep Demo mode on for offline testing, or disable it for live scraping.
7. Apply filters/sorting and export results if needed.
8. Check **Scraper Monitor** for portal health and cache details.

## Optional Components and Fallbacks

- If spaCy is unavailable, extraction still works using keyword matching and HF (if available).
- If Hugging Face transformers are unavailable, extraction still works using keyword + spaCy.
- If sentence-transformers are unavailable:
    - CV-JD semantic similarity returns 0 in `analyzer/similarity.py`.
    - Job matching in `scraper/matcher.py` falls back to TF-IDF/keyword overlap scoring.
- Sidebar module status in the app shows what is active at runtime.

## Scraping and Caching Behavior

- Per-portal request throttling: 2 seconds between requests
- Retry strategy for transient HTTP failures
- Portal failures are isolated (one portal can fail without stopping others)
- Cache TTL: 6 hours
- Listing freshness window: 30 days (older entries purged)

## Known Notes

- Live scraping reliability depends on target site HTML changes and network conditions.
- Demo mode is the fastest/stablest way to validate UI and matching behavior.
- First run of transformer models can take extra time due to model download.

## Quick Troubleshooting

- `ModuleNotFoundError`: activate the same virtual environment used for installation.
- Empty/low-quality CV extraction: try another PDF or a DOCX export of the same resume.
- Low match scores: ensure CV was analyzed first, then rerun job search.
- No jobs in live mode: retry with a broader query/location or switch to Demo mode to verify the pipeline.

## Entry Points

- Main app: `app.py`
- CV analysis pipeline:
    - `analyzer/parser.py`
    - `analyzer/skills.py`
    - `analyzer/scorer.py`
    - `analyzer/gap.py`
    - `analyzer/suggestions.py`
    - `analyzer/similarity.py`
- Scraper pipeline:
    - `scraper/orchestrator.py`
    - `scraper/base.py`
    - `scraper/linkedin.py`
    - `scraper/indeed.py`
    - `scraper/rozee.py`
    - `scraper/matcher.py`
    - `scraper/cache.py`
