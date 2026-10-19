"""
insert_test.py - verify that analyzer.py results insert correctly into MySQL
============================================================================

What it does
------------
1. Runs ``analyze_resume()`` on a sample resume / job description plus the
   optional filters (certifications, experience, region, social login, privacy).
2. Converts the result with ``to_db_row()`` and inserts it with ``INSERT_SQL``.
3. Commits, prints the new ``analysis_id``, then reads the row back and checks
   that what MySQL stored matches what the analyzer produced.

Usage
-----
    python insert_test.py              # insert one row and verify it
    python insert_test.py --cleanup    # ...then delete the test row again

Database connection
-------------------
* If your project has ``database.py`` with ``get_connection()`` (as in the
  MVP-1 insert_test.py), that function is used, so credentials stay in one place.
* Otherwise it falls back to ``mysql.connector`` and these environment
  variables:  JOBNEX_DB_HOST (localhost), JOBNEX_DB_PORT (3306),
  JOBNEX_DB_USER (root), JOBNEX_DB_PASSWORD (empty), JOBNEX_DB_NAME (required).

Install the driver once with:  pip install mysql-connector-python

Standalone: no FastAPI imports. Exit code is 0 on success, 1 on any failure.
"""

import argparse
import json
import os
import sys

from analyzer import (
    INSERT_SQL,
    AnalyzerError,
    analyze_resume,
    to_db_row,
)


# -----------------------------------------------------------------------------
# Sample input (edit freely)
# -----------------------------------------------------------------------------

SAMPLE_RESUME = """
Priya Sharma - Backend Developer
Built REST APIs with Python and FastAPI, deployed with Docker.
Worked with MySQL databases and wrote complex SQL queries.
Completed an AWS Certified Cloud Practitioner course.
""".strip()

SAMPLE_JD = """
We are hiring a Backend Engineer with 3+ years of experience.
Required: Python, SQL, React and Docker. FastAPI is a plus.
Google Cloud certified candidates are preferred.
""".strip()

SAMPLE_FILTERS = {
    "certifications": ["aws"],
    "experience_years": 2,
    "region_filter": "india",
    "social_login": "google",
    "privacy_setting": "private",
}


# -----------------------------------------------------------------------------
# Database connection
# -----------------------------------------------------------------------------

def open_connection():
    """Return an open MySQL connection (project helper first, env vars second)."""
    try:
        # Same helper the MVP-1 insert_test.py used.
        from database import get_connection
        return get_connection()
    except ImportError:
        pass  # no database.py in this project -> use the environment fallback

    import mysql.connector  # imported lazily so a missing driver gives a clear message

    database = os.getenv("JOBNEX_DB_NAME")
    if not database:
        raise RuntimeError(
            "No database.py found and JOBNEX_DB_NAME is not set. "
            "Provide database.get_connection() or set the JOBNEX_DB_* variables."
        )

    return mysql.connector.connect(
        host=os.getenv("JOBNEX_DB_HOST", "localhost"),
        port=int(os.getenv("JOBNEX_DB_PORT", "3306")),
        user=os.getenv("JOBNEX_DB_USER", "root"),
        password=os.getenv("JOBNEX_DB_PASSWORD", ""),
        database=database,
    )


# -----------------------------------------------------------------------------
# Read-back verification
# -----------------------------------------------------------------------------

def _loads(value):
    """JSON/TEXT columns may come back as str or bytes depending on the driver."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    return json.loads(value)


def fetch_row(cursor, analysis_id):
    """Fetch one row as a dict (works with mysql.connector and PyMySQL cursors)."""
    cursor.execute("SELECT * FROM resume_analyzer WHERE analysis_id = %s", (analysis_id,))
    record = cursor.fetchone()
    if record is None:
        return None
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, record))


def compare(result, stored):
    """Return a list of human-readable mismatches between analyzer output and the DB row."""
    problems = []

    def check(column, expected, actual):
        if expected != actual:
            problems.append(f"{column}: expected {expected!r}, stored {actual!r}")

    check("resume_text", result["resume_text"], stored["resume_text"])
    check("jd_text", result["jd_text"], stored["jd_text"])
    check("match_score", round(result["match_score"], 2), round(float(stored["match_score"]), 2))
    check("experience_years", result["experience_years"], stored["experience_years"])
    check("region_filter", result["region_filter"], stored["region_filter"])
    check("dashboard_status", result["dashboard_status"], stored["dashboard_status"])
    check("auto_apply", result["auto_apply"], int(stored["auto_apply"]))
    check("auto_refusal", result["auto_refusal"], int(stored["auto_refusal"]))
    check("social_login", result["social_login"], stored["social_login"])
    check("privacy_setting", result["privacy_setting"], stored["privacy_setting"])

    # certifications is stored as comma-separated text (NULL when empty).
    expected_certs = ",".join(result["certifications"]) or None
    check("certifications", expected_certs, stored["certifications"])

    # JSON / serialized columns: compare parsed values, not raw strings
    # (MySQL's JSON type may re-order keys or change whitespace).
    for column in ("matched_skills", "missing_skills", "sugesstions",
                   "interview_prep", "suggested_resources"):
        check(column, result[column], _loads(stored[column]))

    return problems


# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------

def run(cleanup):
    # Step 1 - analyse. Bad input raises InvalidInputError (an AnalyzerError).
    result = analyze_resume(SAMPLE_RESUME, SAMPLE_JD, **SAMPLE_FILTERS)
    print(f"Analysis done: score={result['match_score']}, "
          f"matched={result['matched_skills']}, missing={result['missing_skills']}, "
          f"status={result['dashboard_status']!r}")

    # Step 2 - convert JSON/list fields into strings MySQL can store.
    row = to_db_row(result)

    connection = None
    cursor = None
    try:
        # Step 3 - connect.
        connection = open_connection()
        cursor = connection.cursor()

        # Step 4 - insert (INSERT_SQL uses %(name)s placeholders, so pass the dict).
        cursor.execute(INSERT_SQL, row)
        analysis_id = cursor.lastrowid

        # Step 5 - commit, then report the new primary key.
        connection.commit()
        print(f"Analysis saved successfully! Record ID: {analysis_id}")

        # Step 6 - read the row back and compare it with the analyzer output.
        stored = fetch_row(cursor, analysis_id)
        if stored is None:
            print("VERIFY FAILED: inserted row could not be read back.")
            return False

        problems = compare(result, stored)
        if problems:
            print("VERIFY FAILED: stored data differs from analyzer output:")
            for problem in problems:
                print("  -", problem)
            return False
        print(f"VERIFY OK: all {len(row)} columns round-tripped correctly "
              f"(created_at={stored['created_at']}).")

        # Optional - remove the test row so the table stays clean.
        if cleanup:
            cursor.execute("DELETE FROM resume_analyzer WHERE analysis_id = %s", (analysis_id,))
            connection.commit()
            print(f"Test row {analysis_id} deleted (--cleanup).")

        return True

    except Exception:
        # Undo a half-finished transaction, then let main() report the error.
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass  # connection may already be dead; the original error matters more
        raise

    finally:
        # Always release DB resources, even after an error.
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description="Insert an analyzer.py result into MySQL and verify it.")
    parser.add_argument("--cleanup", action="store_true",
                        help="delete the inserted test row after verifying it")
    args = parser.parse_args()

    try:
        ok = run(cleanup=args.cleanup)
    except AnalyzerError as error:
        print(f"Analyzer error: {error}")
        return 1
    except ImportError as error:
        print(f"Missing dependency: {error}. Try: pip install mysql-connector-python")
        return 1
    except Exception as error:  # DB connection, SQL, or unexpected errors
        print(f"Error: {type(error).__name__}: {error}")
        return 1

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
