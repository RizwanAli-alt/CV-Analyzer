# AI CV Analyzer — Setup & Run Guide
## Windows 11 | Python 3.11

---

## Project Structure

```
cv_analyzer/
├── app.py                    ← Streamlit entry point
├── requirements.txt
├── analyzer/
│   ├── __init__.py           ← (was _init_.py — fixed)
│   ├── parser.py
│   ├── skills.py
│   ├── scorer.py
│   ├── gap.py
│   ├── suggestions.py        ← (was suggerstions.py — fixed)
│   ├── similarity.py
│   └── utilities.py          ← (was utilites.py — fixed)
└── models/
    ├── skill_db.json
    └── market_skills.json
```

---

## 1. Create virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Download spaCy language model

```powershell
python -m spacy download en_core_web_sm
```

## 4. Run the app

```powershell
streamlit run app.py
```

---

## Notes

- **spaCy** is optional but recommended — without it the app falls back to keyword-only skill extraction.
- **sentence-transformers** is optional — without it the similarity score shows 0% and a warning is displayed.
- Both missing dependencies are shown clearly in the sidebar so you always know the module status.
- The first run of sentence-transformers will download the `all-MiniLM-L6-v2` model (~90 MB) automatically.

---

## Fixes applied vs original code

| # | Problem | Fixed in |
|---|---------|---------|
| 1 | `_init_.py` → `__init__.py` | `analyzer/__init__.py` |
| 2 | `suggerstions.py` typo | renamed to `suggestions.py` |
| 3 | `utilites.py` typo + wrong import in app.py | renamed to `utilities.py` |
| 4 | Score breakdown keys mismatched between scorer/suggestions/app | `scorer.py`, `suggestions.py`, `app.py` |
| 5 | Double analysis run on every page reload | `app.py` (cached in session_state) |
| 6 | Unsafe temp files (not cleaned on exception) | `parser.py` → `extract_text_from_bytes()` |
| 7 | `clean_text()` destroyed newlines, breaking section detection | `parser.py` |
| 8 | `sentence-transformers` missing from requirements | `requirements.txt` |
| 9 | spaCy NLP not integrated (SRS requirement) | `skills.py` |
| 10 | Hardcoded year 2026 in experience calculator | `utilities.py` |
| 11 | Duplicate entries in skill_db.json | `models/skill_db.json` |
| 12 | No job description input for user (SRS requirement) | sidebar in `app.py` |
| 13 | No gap visualization chart (SRS requirement) | `app.py` donut chart |
| 14 | `st.pyplot()` deprecation | replaced with `st.plotly_chart()` |
| 15 | `validate_file()` didn't return error message | `utilities.py` returns `(bool, str)` |