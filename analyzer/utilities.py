"""
Utility functions for CV analysis.

FIX (Issue #5 — filename typo):
  File was named 'utilites.py' (missing 't'). app.py imports from
  'analyzer.utilities' which caused a ModuleNotFoundError. Renamed to
  utilities.py. This is the correct canonical file going forward.

Other fixes retained from previous version:
  - calculate_experience_years() uses datetime.date.today().year (not hardcoded)
  - get_file_size_mb() handles Streamlit UploadedFile objects correctly
  - NullHandler added for Django logging safety
"""

import logging
import re
import datetime
from typing import Optional

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for library / Django use

MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = {"pdf", "docx"}


def validate_file(filename: str, file_size_mb: float) -> tuple:
    """
    Validate an uploaded file.

    Args:
        filename: Name of the uploaded file
        file_size_mb: Size in megabytes

    Returns:
        (is_valid: bool, error_message: str)
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        msg = f"Unsupported file type '.{ext}'. Please upload a PDF or DOCX file."
        logger.warning(msg)
        return False, msg

    if file_size_mb > MAX_FILE_SIZE_MB:
        msg = (
            f"File is too large ({file_size_mb:.2f} MB). "
            f"Maximum allowed is {MAX_FILE_SIZE_MB} MB."
        )
        logger.warning(msg)
        return False, msg

    return True, ""


def get_file_size_mb(file_object) -> float:
    """
    Get file size in megabytes from a Streamlit UploadedFile or file-like object.

    Args:
        file_object: Streamlit UploadedFile or any file-like object

    Returns:
        Size in MB
    """
    try:
        if hasattr(file_object, "size") and file_object.size:
            return file_object.size / (1024 * 1024)
        if hasattr(file_object, "getbuffer"):
            return len(file_object.getbuffer()) / (1024 * 1024)
        if hasattr(file_object, "read"):
            data = file_object.read()
            file_object.seek(0)
            return len(data) / (1024 * 1024)
    except Exception as e:
        logger.error(f"Could not determine file size: {e}")
    return 0.0


def normalize_skill_name(skill: str) -> str:
    """Lowercase and strip a skill name for comparison."""
    return skill.lower().strip()


def calculate_experience_years(text: str) -> int:
    """
    Estimate total years of experience from date ranges in CV text.

    Args:
        text: CV text

    Returns:
        Estimated years of experience (0 if none detected)
    """
    current_year = datetime.date.today().year
    pattern = r"(\d{4})\s*[-–]\s*(?:(\d{4})|(present|current|now))"
    matches = re.findall(pattern, text, re.IGNORECASE)

    total = 0
    for start_str, end_str, present in matches:
        try:
            start = int(start_str)
            end = current_year if present else int(end_str)
            if 1970 <= start <= current_year and start <= end:
                total += end - start
        except ValueError:
            continue

    return max(total, 0)


def extract_contact_info(text: str) -> dict:
    """
    Extract contact information from CV text.

    Args:
        text: CV text

    Returns:
        {"email", "phone", "linkedin", "github"} — None for missing fields
    """
    contact = {"email": None, "phone": None, "linkedin": None, "github": None}

    email_m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    if email_m:
        contact["email"] = email_m.group()

    phone_m = re.search(r"(\+?\d[\d\s\-().]{7,}\d)", text)
    if phone_m:
        contact["phone"] = phone_m.group().strip()

    linkedin_m = re.search(r"linkedin\.com/in/[a-zA-Z0-9\-]+", text, re.I)
    if linkedin_m:
        contact["linkedin"] = linkedin_m.group()

    github_m = re.search(r"github\.com/[a-zA-Z0-9\-]+", text, re.I)
    if github_m:
        contact["github"] = github_m.group()

    return contact


def check_section_completeness(text: str) -> dict:
    """
    Check which standard CV sections are present.

    Args:
        text: CV text

    Returns:
        {"sections": {...}, "completed_count": int, "completeness_percentage": float}
    """
    sections = {
        "contact": bool(re.search(r"(email|phone|linkedin|@)", text, re.I)),
        "summary": bool(re.search(r"\b(summary|objective|profile|about)\b", text, re.I)),
        "experience": bool(re.search(r"\b(experience|employment|work history)\b", text, re.I)),
        "education": bool(re.search(r"\b(education|academic|university|college)\b", text, re.I)),
        "skills": bool(re.search(r"\b(skills?|competencies|technical skills)\b", text, re.I)),
        "projects": bool(re.search(r"\b(projects?|portfolio)\b", text, re.I)),
        "certifications": bool(re.search(r"\b(certifications?|awards|licenses)\b", text, re.I)),
    }

    completed = sum(1 for v in sections.values() if v)
    total = len(sections)

    return {
        "sections": sections,
        "completed_count": completed,
        "total_expected": total,
        "completeness_percentage": round(completed / total * 100, 1),
    }