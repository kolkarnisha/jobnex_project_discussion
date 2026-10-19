"""
analyzer.py - JOBNEX MVP-2 resume analyzer
==========================================

Compares a resume against a job description (JD), scores the match, and
enriches the result with suggestions, interview prep, learning resources and
a dashboard status. The returned dict maps 1:1 to the ``resume_analyzer``
MySQL table (minus ``analysis_id`` and ``created_at``, which MySQL generates).

Quick start
-----------
    from analyzer import analyze_resume, to_db_row, INSERT_SQL, InvalidInputError

    result = analyze_resume(resume, jd, certifications=["aws"], experience_years=2,
                            region_filter="india")      # JSON-serializable dict
    cursor.execute(INSERT_SQL, to_db_row(result))       # JSON columns -> strings

Design notes
------------
* Pure Python, no third-party dependencies. FastAPI is not imported here, so
  the module is easy to unit-test and reuse from workers or scripts.
* Skill extraction sits behind a small ``SkillExtractor`` interface, and
  ``register_enricher`` lets you bolt on LLM-generated content later without
  touching the core scoring logic.
* Resume/JD text may contain personal data, so it is never written to logs.
* The column name ``sugesstions`` (sic) is kept to match the DB schema.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Configuration
# =============================================================================

MAX_TEXT_LENGTH = 100_000          # characters per resume / JD (abuse guard)
MIN_EXPERIENCE_YEARS = 0           # matches <input min="0" max="40"> in index.html
MAX_EXPERIENCE_YEARS = 40
MAX_CERTIFICATIONS = 20

# Values mirror the controls in index.html.
ALLOWED_REGIONS = ("us", "global", "india", "bengaluru")
DEFAULT_REGION = "global"
ALLOWED_SOCIAL_LOGINS = ("google", "email")
ALLOWED_PRIVACY_SETTINGS = ("private", "public")
DEFAULT_PRIVACY_SETTING = "private"

# match_score thresholds for dashboard_status (analysis state, not application state).
READY_THRESHOLD = 75.0
IMPROVE_THRESHOLD = 40.0

# Max questions returned per interview category.
QUESTION_LIMITS = {"technical": 5, "behavioral": 3, "hr": 3, "situational": 3}


# =============================================================================
# 2. Exceptions  (map InvalidInputError -> HTTP 422, AnalysisError -> HTTP 500)
# =============================================================================

class AnalyzerError(Exception):
    """Base class for all analyzer errors."""


class InvalidInputError(AnalyzerError, ValueError):
    """The caller supplied bad input (safe to show the message to the user)."""


class AnalysisError(AnalyzerError):
    """Unexpected failure while analysing (do not expose internals to the user)."""


# =============================================================================
# 3. Static data: skills, certifications, interview bank, resources
# =============================================================================

# canonical key -> display label + aliases. Aliases are normalised at load time,
# so "React.js" and "react js" are equivalent. Add new skills here only.
SKILL_CATALOG: Dict[str, Dict[str, Any]] = {
    "python": {"label": "Python", "aliases": ["python", "python3"]},
    "sql": {"label": "SQL", "aliases": ["sql", "mysql", "postgresql", "postgres", "sqlite"]},
    "react": {"label": "React", "aliases": ["react", "reactjs", "react.js"]},
    "fastapi": {"label": "FastAPI", "aliases": ["fastapi", "fast api"]},
    "docker": {"label": "Docker", "aliases": ["docker", "dockerfile", "docker compose"]},
}

# Certifications the JD might ask for (keys match the checkboxes in index.html).
CERT_CATALOG: Dict[str, Dict[str, Any]] = {
    "aws": {"label": "AWS", "phrases": ["aws certified", "aws certification", "aws solutions architect"]},
    "azure": {"label": "Microsoft Azure", "phrases": ["azure certified", "azure certification", "microsoft certified"]},
    "gcp": {"label": "Google Cloud", "phrases": ["google cloud certified", "google cloud certification",
                                                  "gcp certified", "gcp certification"]},
    "pmp": {"label": "PMP", "phrases": ["pmp", "project management professional"]},
    "scrum": {"label": "Scrum Master", "phrases": ["scrum master", "certified scrum master", "csm", "psm"]},
}

# Built-in interview bank (same content as QUESTION_BANK in script.js).
QUESTION_BANK: Dict[str, List[Dict[str, str]]] = {
    "technical": [
        {"question": "Explain a technical project you built and the decisions you made.",
         "tip": "Explain the problem, your role, the tools you chose, the trade-offs, and the result. Be specific about what you personally implemented."},
        {"question": "How would you investigate an API request that is returning an error?",
         "tip": "Check the status code and response body, reproduce the request, inspect server logs, verify the payload and authentication, then isolate the cause."},
        {"question": "What is the difference between client-side and server-side validation?",
         "tip": "Client-side validation improves feedback. Server-side validation is essential because clients can be bypassed and requests can be crafted manually."},
        {"question": "How do you prevent duplicate form submissions?",
         "tip": "Disable the submit action while a request is in progress, guard against repeated handlers, and use backend idempotency where required."},
    ],
    "behavioral": [
        {"question": "Tell me about a time you had to learn a new technology quickly.",
         "tip": "Describe the goal, how you broke the topic into manageable parts, how you practised, and what you delivered or learned."},
        {"question": "Describe a mistake you made and how you handled it.",
         "tip": "Own the mistake, explain its impact without blaming others, describe the fix, and state the process change that reduced the chance of recurrence."},
        {"question": "Tell me about a time you received difficult feedback.",
         "tip": "Explain what the feedback was, how you clarified it, what you changed, and how you assessed whether the change worked."},
        {"question": "How do you prioritize when several tasks are urgent?",
         "tip": "Compare impact, deadline, dependencies, and risk. Communicate trade-offs early and confirm priorities with the stakeholder."},
    ],
    "hr": [
        {"question": "Tell me about yourself and the experience most relevant to this role.",
         "tip": "Use a brief present-past-future structure and connect your skills and recent work to this specific role."},
        {"question": "Why are you interested in this role and company?",
         "tip": "Connect the actual responsibilities to your strengths and goals. Research the company before a real interview."},
        {"question": "What are your strengths, and how have you demonstrated them?",
         "tip": "Choose two strengths relevant to the role and support each with a concrete example rather than adjectives alone."},
        {"question": "What questions would you like to ask the interviewer?",
         "tip": "Ask about success measures, team workflows, onboarding, technical challenges, or how feedback is given."},
    ],
    "situational": [
        {"question": "A production issue appears shortly before a release. What would you do?",
         "tip": "Assess impact and urgency, communicate quickly, follow rollback or mitigation procedures, and verify the fix before closing the incident."},
        {"question": "A teammate disagrees with your proposed implementation. How would you respond?",
         "tip": "Ask about their concerns, compare options against agreed requirements, use evidence or a small test, and document the decision."},
        {"question": "A stakeholder requests a feature but gives incomplete requirements. What is your next step?",
         "tip": "Clarify the user problem, expected outcome, edge cases, constraints, and acceptance criteria before committing."},
        {"question": "You discover that a task will miss its deadline. What would you do?",
         "tip": "Raise the risk early, explain cause and impact, suggest a smaller safe scope or revised plan, and agree on next steps."},
    ],
}

# One skill-specific technical question per catalog skill.
SKILL_QUESTIONS: Dict[str, Dict[str, str]] = {
    "python": {"question": "When would you choose a list, tuple, set or dict in Python, and why?",
               "tip": "Compare mutability, ordering, lookup cost and uniqueness, with a real example from your own code."},
    "sql": {"question": "How does an INNER JOIN differ from a LEFT JOIN, and how would you speed up a slow query?",
            "tip": "Explain the row-matching difference, then cover indexes, reading the query plan, and avoiding unnecessary columns or subqueries."},
    "react": {"question": "How do useState and useEffect differ, and what causes unnecessary re-renders?",
              "tip": "Distinguish state from side effects, mention dependency arrays, and explain how you would find and fix a re-render problem."},
    "fastapi": {"question": "How does FastAPI use type hints and Pydantic models to validate requests?",
                "tip": "Cover request/response models, automatic 422 errors, dependency injection and generated OpenAPI docs."},
    "docker": {"question": "What is the difference between a Docker image and a container?",
               "tip": "Explain layers, immutability of images, container lifecycle, and when you would use multi-stage builds or volumes."},
}

# Starter learning links. Replace/verify before launch (index.html warns that
# placeholder course links must point to verified destinations).
RESOURCE_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "python": [
        {"title": "The Python Tutorial", "provider": "Python.org", "type": "Official docs",
         "level": "Beginner", "url": "https://docs.python.org/3/tutorial/"},
        {"title": "Getting Started with Python", "provider": "Python.org", "type": "Free resource",
         "level": "Beginner", "url": "https://www.python.org/about/gettingstarted/"},
    ],
    "sql": [
        {"title": "SQL Tutorial", "provider": "W3Schools", "type": "Free tutorial",
         "level": "Beginner", "url": "https://www.w3schools.com/sql/"},
        {"title": "SQLBolt: Learn SQL with Simple Exercises", "provider": "SQLBolt", "type": "Interactive lessons",
         "level": "Beginner", "url": "https://sqlbolt.com/"},
    ],
    "react": [
        {"title": "Learn React", "provider": "react.dev", "type": "Official docs",
         "level": "Beginner", "url": "https://react.dev/learn"},
        {"title": "Tutorial: Tic-Tac-Toe", "provider": "react.dev", "type": "Hands-on tutorial",
         "level": "Beginner", "url": "https://react.dev/learn/tutorial-tic-tac-toe"},
    ],
    "fastapi": [
        {"title": "FastAPI Tutorial - User Guide", "provider": "FastAPI", "type": "Official docs",
         "level": "Beginner", "url": "https://fastapi.tiangolo.com/tutorial/"},
        {"title": "Learn FastAPI", "provider": "FastAPI", "type": "Free resource",
         "level": "Beginner", "url": "https://fastapi.tiangolo.com/learn/"},
    ],
    "docker": [
        {"title": "Get Started with Docker", "provider": "Docker", "type": "Official docs",
         "level": "Beginner", "url": "https://docs.docker.com/get-started/"},
        {"title": "Docker Workshop", "provider": "Docker", "type": "Hands-on tutorial",
         "level": "Beginner", "url": "https://docs.docker.com/get-started/workshop/"},
    ],
}


# =============================================================================
# 4. Text normalisation
# =============================================================================

_PUNCT_RE = re.compile(r"[^a-z0-9+#\s]")   # keep + and # so "c++" / "c#" survive
# Matches "3 years", "3+ years", "3-5 yrs"; group 1 is the lower bound.
_YEARS_RE = re.compile(r"(\d{1,2})\s*(?:\+|(?:-|\u2013|to)\s*\d{1,2})?\s*(?:years?|yrs?)\b")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def _contains_phrase(padded_text: str, phrase: str) -> bool:
    """Whole-word match: 'sql' matches 'sql server' but not 'nosqlx'."""
    return f" {normalize_text(phrase)} " in padded_text


# =============================================================================
# 5. Skill extraction (swap-in point for future AI integration)
# =============================================================================

class SkillExtractor(Protocol):
    """Anything with ``extract(raw_text) -> set of canonical skill keys``."""

    def extract(self, text: str) -> Set[str]:
        ...


class KeywordSkillExtractor:
    """Default extractor: whole-word alias matching against SKILL_CATALOG."""

    def __init__(self, catalog: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        catalog = catalog or SKILL_CATALOG
        self._aliases = {
            skill: tuple(normalize_text(a) for a in meta["aliases"])
            for skill, meta in catalog.items()
        }

    def extract(self, text: str) -> Set[str]:
        padded = f" {normalize_text(text)} "
        return {
            skill for skill, aliases in self._aliases.items()
            if any(f" {alias} " in padded for alias in aliases)
        }


_skill_extractor: SkillExtractor = KeywordSkillExtractor()
_enrichers: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []


def set_skill_extractor(extractor: SkillExtractor) -> None:
    """Replace the default extractor (e.g. with an LLM / embedding based one)."""
    global _skill_extractor
    if not callable(getattr(extractor, "extract", None)):
        raise TypeError("extractor must provide an extract(text) method")
    _skill_extractor = extractor


def register_enricher(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
    """
    Register a post-processing hook, e.g. to replace ``interview_prep`` or
    ``sugesstions`` with LLM output. It receives the finished result dict and
    must return a dict with the same keys. Hook failures are logged and
    ignored, so a flaky AI call can never break the core analysis.
    """
    _enrichers.append(fn)


def clear_enrichers() -> None:
    _enrichers.clear()


# =============================================================================
# 6. Input validation
# =============================================================================

def _validate_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidInputError(f"{name} must be a string.")
    value = value.strip()
    if not value:
        raise InvalidInputError(f"{name} must not be empty.")
    if len(value) > MAX_TEXT_LENGTH:
        raise InvalidInputError(f"{name} is too long (maximum {MAX_TEXT_LENGTH} characters).")
    return value


def _validate_experience_years(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise InvalidInputError("experience_years must be a whole number.")
    if isinstance(value, str):
        value = value.strip()
        if not value:                      # the HTML form sends "" when left empty
            return None
        if not value.isdigit():
            raise InvalidInputError("experience_years must be a whole number.")
        value = int(value)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise InvalidInputError("experience_years must be a whole number.")
    if not MIN_EXPERIENCE_YEARS <= value <= MAX_EXPERIENCE_YEARS:
        raise InvalidInputError(
            f"experience_years must be between {MIN_EXPERIENCE_YEARS} and {MAX_EXPERIENCE_YEARS}."
        )
    return value


def _validate_certifications(value: Any) -> List[str]:
    """Accept None, 'aws, gcp', or ['aws', 'gcp']; return a deduplicated lowercase list."""
    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:
        raise InvalidInputError("certifications must be a string or a list of strings.")

    cleaned: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise InvalidInputError("certifications must contain only strings.")
        item = " ".join(item.lower().split())
        if item and item not in cleaned:
            cleaned.append(item)
    if len(cleaned) > MAX_CERTIFICATIONS:
        raise InvalidInputError(f"Too many certifications (maximum {MAX_CERTIFICATIONS}).")
    return cleaned


def _validate_choice(name: str, value: Any, allowed: Iterable[str], default: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if not isinstance(value, str):
        raise InvalidInputError(f"{name} must be a string.")
    value = value.strip().lower()
    if value not in allowed:
        raise InvalidInputError(f"{name} must be one of: {', '.join(allowed)}.")
    return value


# =============================================================================
# 7. Analysis building blocks
# =============================================================================

def _ordered(skills: Iterable[str]) -> List[str]:
    """Deterministic order: catalog order first, then anything else alphabetically."""
    order = {key: i for i, key in enumerate(SKILL_CATALOG)}
    return sorted(skills, key=lambda s: (order.get(s, len(order)), s))


def _label(skill: str) -> str:
    return SKILL_CATALOG.get(skill, {}).get("label", skill)


def _required_years(jd_text: str) -> Optional[int]:
    """Heuristic: highest 'N years' lower bound in the JD (e.g. '3-5 years' -> 3)."""
    years = [int(m.group(1)) for m in _YEARS_RE.finditer(jd_text.lower())]
    years = [y for y in years if 0 < y <= MAX_EXPERIENCE_YEARS]
    return max(years) if years else None


def _certification_gaps(resume_text: str, jd_text: str, held: List[str]) -> List[str]:
    """Labels of certifications the JD mentions that the candidate doesn't list."""
    jd_padded = f" {normalize_text(jd_text)} "
    resume_padded = f" {normalize_text(resume_text)} "
    gaps = []
    for key, meta in CERT_CATALOG.items():
        in_jd = any(_contains_phrase(jd_padded, p) for p in meta["phrases"])
        has_it = key in held or any(_contains_phrase(resume_padded, p) for p in meta["phrases"])
        if in_jd and not has_it:
            gaps.append(meta["label"])
    return gaps


def _build_suggestions(missing: List[str], jd_skills: Set[str], cert_gaps: List[str],
                       jd_text: str, experience_years: Optional[int]) -> List[str]:
    suggestions = [f"Try learning {_label(s)} and add it to your resume." for s in missing]

    if not jd_skills:
        suggestions.append(
            "No recognised skills were found in the job description. "
            "Paste the full job description for a more accurate match."
        )

    for label in cert_gaps:
        suggestions.append(
            f"The job mentions {label} certification. Consider earning it and listing it on your resume."
        )

    required = _required_years(jd_text)
    if required is not None and experience_years is not None and experience_years < required:
        suggestions.append(
            f"The job asks for about {required}+ years of experience and you listed {experience_years}. "
            "Highlight relevant projects and internships to bridge the gap."
        )

    if not suggestions:
        suggestions.append(
            "Your resume covers every skill we detected in the job description. "
            "Tailor your summary and achievements to this role."
        )
    return suggestions


def _build_interview_prep(matched: List[str], missing: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Technical questions are skill-specific first, then topped up from the bank."""
    skill_questions = [SKILL_QUESTIONS[s] for s in matched + missing if s in SKILL_QUESTIONS]
    prep: Dict[str, List[Dict[str, str]]] = {}
    for category, limit in QUESTION_LIMITS.items():
        pool = (skill_questions if category == "technical" else []) + QUESTION_BANK[category]
        prep[category] = [dict(q) for q in pool[:limit]]   # copies: never leak shared constants
    return prep


def _build_resources(missing: List[str]) -> List[Dict[str, str]]:
    resources = []
    for skill in missing:
        for item in RESOURCE_CATALOG.get(skill, []):
            resources.append({"skill": skill, **item})
    return resources


def _dashboard_status(score: float, jd_skills: Set[str]) -> str:
    """
    Status of the *analysis*. Application statuses (Applied / Interviewing /
    Offer / Rejected / Withdrawn) stay user-controlled in the dashboard table.
    """
    if not jd_skills:
        return "Insufficient Data"
    if score >= READY_THRESHOLD:
        return "Ready to Apply"
    if score >= IMPROVE_THRESHOLD:
        return "Needs Improvement"
    return "Low Match"


# =============================================================================
# 8. Public API
# =============================================================================

RESULT_KEYS = (
    "resume_text", "jd_text", "match_score", "matched_skills", "missing_skills",
    "sugesstions", "certifications", "experience_years", "region_filter",
    "interview_prep", "suggested_resources", "dashboard_status",
    "auto_apply", "auto_refusal", "social_login", "privacy_setting",
)


def analyze_resume(
    resume_text: str,
    jd_text: str,
    certifications: Optional[Any] = None,
    experience_years: Optional[Any] = None,
    region_filter: Optional[str] = None,
    social_login: Optional[str] = None,
    privacy_setting: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse ``resume_text`` against ``jd_text``.

    Scoring: ``match_score = matched / required * 100`` where *required* is the
    set of catalog skills the JD mentions and *matched* is the subset that also
    appears in the resume. (Skills the JD never asks for are not counted as
    missing.) Certifications and experience only add suggestions; they do not
    change the score.

    Returns a JSON-serializable dict whose keys equal the ``resume_analyzer``
    columns (see ``RESULT_KEYS``). Use ``to_db_row`` before inserting.

    Raises:
        InvalidInputError: bad caller input (map to HTTP 422).
        AnalysisError: unexpected internal failure (map to HTTP 500).
    """
    # --- validate (everything here is the caller's fault) --------------------
    resume = _validate_text("resume_text", resume_text)
    jd = _validate_text("jd_text", jd_text)
    certs = _validate_certifications(certifications)
    years = _validate_experience_years(experience_years)
    region = _validate_choice("region_filter", region_filter, ALLOWED_REGIONS, DEFAULT_REGION)
    social = _validate_choice("social_login", social_login, ALLOWED_SOCIAL_LOGINS, None)
    privacy = _validate_choice("privacy_setting", privacy_setting,
                               ALLOWED_PRIVACY_SETTINGS, DEFAULT_PRIVACY_SETTING)

    # --- analyse (anything raised here is ours) ------------------------------
    try:
        resume_skills = _skill_extractor.extract(resume)
        jd_skills = _skill_extractor.extract(jd)

        matched = _ordered(jd_skills & resume_skills)
        missing = _ordered(jd_skills - resume_skills)
        score = round(len(matched) / len(jd_skills) * 100, 2) if jd_skills else 0.0

        result: Dict[str, Any] = {
            "resume_text": resume,
            "jd_text": jd,
            "match_score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "sugesstions": _build_suggestions(
                missing, jd_skills, _certification_gaps(resume, jd, certs), jd, years),
            "certifications": certs,
            "experience_years": years,
            "region_filter": region,
            "interview_prep": _build_interview_prep(matched, missing),
            "suggested_resources": _build_resources(missing),
            "dashboard_status": _dashboard_status(score, jd_skills),
            "auto_apply": 0,       # automation is not active in MVP-2
            "auto_refusal": 0,
            "social_login": social,
            "privacy_setting": privacy,
        }
    except AnalyzerError:
        raise
    except Exception as exc:       # noqa: BLE001 - deliberate catch-all boundary
        logger.exception("Resume analysis failed")   # traceback only; no resume/JD text
        raise AnalysisError("Resume analysis failed unexpectedly.") from exc

    return _apply_enrichers(result)


def _apply_enrichers(result: Dict[str, Any]) -> Dict[str, Any]:
    """Run optional hooks; a failing or malformed hook is skipped, not fatal."""
    for hook in _enrichers:
        try:
            enriched = hook(dict(result))
        except Exception:          # noqa: BLE001
            logger.exception("Enricher %r failed; keeping previous result", hook)
            continue
        if isinstance(enriched, dict) and all(k in enriched for k in RESULT_KEYS):
            result = enriched
        else:
            logger.warning("Enricher %r returned an invalid result; ignored", hook)
    return result


# =============================================================================
# 9. Database helpers (MySQL: json / text columns need strings)
# =============================================================================

DB_COLUMNS = RESULT_KEYS
_JSON_COLUMNS = ("matched_skills", "missing_skills", "sugesstions",
                 "interview_prep", "suggested_resources")

# Named placeholders work with PyMySQL / mysql-connector. Column names are
# constants defined above, never user input.
INSERT_SQL = (
    "INSERT INTO resume_analyzer (" + ", ".join(DB_COLUMNS) + ") "
    "VALUES (" + ", ".join(f"%({c})s" for c in DB_COLUMNS) + ")"
)


def to_db_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an ``analyze_resume`` result into values ready for ``INSERT_SQL``."""
    row = {c: result[c] for c in DB_COLUMNS}
    for column in _JSON_COLUMNS:
        row[column] = json.dumps(row[column], ensure_ascii=False)
    row["certifications"] = ",".join(row["certifications"]) if row["certifications"] else None
    return row
