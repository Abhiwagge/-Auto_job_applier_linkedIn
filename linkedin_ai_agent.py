# ║  linkedin_ai_agent.py — Production-Grade AI Job Search Agent                 ║
# ║  Candidate : Abhishek Wagge | Data Analyst / MIS Executive / BI Fresher      ║
# ║                                                                              ║
# ║  Architecture:                                                               ║
# ║    1.  Config & Dataclasses                                                  ║
# ║    2.  Database Layer          (SQLite, 6 tables)                            ║
# ║    3.  AI Engine               (Google Gemini via API)                       ║
# ║    4.  JD Analyzer             (rule-based ATS scoring)                      ║
# ║    5.  Recruiter Extractor     (Playwright DOM scraping)                     ║
# ║    6.  Message Generator       (3 variants + follow-ups)                     ║
# ║    7.  Cover Letter Generator  (personalised, 250-word cap)                  ║
# ║    8.  Resume Tailor           (keyword gap analysis)                        ║
# ║    9.  Form Fill Engine        (LinkedIn Easy Apply modal)                   ║
# ║   10.  Outreach Workflow       (send message → schedule follow-ups)          ║
# ║   11.  Analytics Dashboard     (console report)                              ║
# ║   12.  LinkedInAgent           (orchestrates everything)                     ║
# ║                                                                              ║
# ║  Install:                                                                    ║
# ║    pip install playwright google-generativeai                                ║
# ║    playwright install chromium                                               ║
# ║  Run:                                                                        ║
# ║    python linkedin_ai_agent.py                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import json
import logging
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from playwright.sync_api import sync_playwright, Page, Locator

# ── Optional Google Gemini SDK (used for AI-powered generation) ────────────────
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — LOGGING                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("LinkedInAIAgent")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — DATACLASSES                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass
class Profile:
    """Candidate profile — edit these fields before running."""
    name:             str = "Abhishek Wagge"
    email:            str = "abhishekwagge6363@gmail.com"
    phone:            str = "+91 6363686757"
    location:         str = "Bangalore, Karnataka, India"
    linkedin_url:     str = "https://linkedin.com/in/abhishekwagge"
    experience_years: str = "0"
    notice_period:    str = "Immediately"
    current_ctc:      str = "0"
    expected_ctc:     str = "Negotiable"
    skills:           str = (
        "Microsoft Excel (Advanced), SQL, Power BI, MIS Reporting, "
        "Pivot Tables, VLOOKUP, Data Cleaning, KPI Analysis, "
        "Google Sheets, Report Automation, Power Query, Data Visualization"
    )
    education:        str = "B.Com, Gulbarga University, 70%, 2024"
    certifications:   str = (
        "Google Data Analytics Certificate (In Progress), "
        "Excel Skills for Business (Coursera), "
        "SQL for Data Science (Great Learning), "
        "Power BI Fundamentals (Microsoft Learn)"
    )
    projects:         str = (
        "Sales MIS Dashboard (Excel) — automated 10+ sheets into unified view; "
        "Customer Data Analysis (SQL) — segmentation, revenue trends, CLV; "
        "Business KPI Report (Power BI) — executive dashboard, YoY growth."
    )
    summary:          str = (
        "Results-driven fresher specialising in Data Analytics, MIS Reporting, "
        "and Business Intelligence. Proficient in Advanced Excel, SQL, and "
        "Power BI. Immediate joiner pursuing Google Data Analytics certification."
    )


@dataclass
class JobInfo:
    """All metadata extracted from a LinkedIn job card and job page."""
    title:          str = ""
    company:        str = ""
    url:            str = ""
    location:       str = ""
    industry:       str = ""
    jd_text:        str = ""
    recruiter_name: str = ""
    recruiter_url:  str = ""
    hiring_manager: str = ""


@dataclass
class ATSResult:
    """Result of the AI/rule-based JD analysis."""
    ats_score:       int          = 0
    match_score:     int          = 0
    decision:        str          = "SKIP"   # "APPLY" or "SKIP"
    reason:          str          = ""
    required_skills: list[str]    = field(default_factory=list)
    preferred_skills: list[str]   = field(default_factory=list)
    matching_skills:  list[str]   = field(default_factory=list)
    missing_skills:   list[str]   = field(default_factory=list)
    ats_keywords:     list[str]   = field(default_factory=list)
    responsibilities: list[str]   = field(default_factory=list)
    tools_mentioned:  list[str]   = field(default_factory=list)
    soft_skills:      list[str]   = field(default_factory=list)
    job_priority:     str          = "LOW"   # HIGH / MEDIUM / LOW
    custom_pitch:     str          = ""
    resume_summary:   str          = ""
    suggested_skills: str          = ""


@dataclass
class RecruiterMessages:
    """Three message variants + three follow-up messages."""
    version_a:   str = ""   # ≤ 300 chars — ultra-short
    version_b:   str = ""   # ≤ 500 chars — professional
    version_c:   str = ""   # ≤ 700 chars — networking
    followup_3:  str = ""   # Day 3 follow-up
    followup_7:  str = ""   # Day 7 follow-up
    followup_14: str = ""   # Day 14 follow-up


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — GLOBAL CONFIG                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
CONFIG = {
    "targets": {
        "keywords": [
            "Data Analyst Fresher",
            "MIS Executive Fresher",
            "Business Intelligence Analyst Fresher",
            "Junior Data Analyst",
            "MIS Analyst",
            "Reporting Analyst Fresher",
            "Excel MIS Executive",
            "Power BI Developer Fresher",
            "SQL Data Analyst",
            "Data Analyst Trainee",
            "Business Analyst Fresher",
            "Operations Analyst Fresher",
        ],
        "locations":   ["Bengaluru, Karnataka, India", "Hyderabad, Telangana, India"],
        "date_posted": "r604800",   # past 7 days
        "exp_level":   "1",          # 1 = Entry level
        "easy_apply":  True,
    },
    "limits": {
        "linkedin_per_day":         30,
        "recruiters_per_day":       20,
        "messages_per_hour":         5,
    },
    "delays": {
        "between_jobs":     [10, 20],
        "between_searches": [8,  15],
        "page_load":        [4,   7],
        "typing":           [40, 100],
        "message_send":     [5,  12],
    },
    "filters": {
        "min_match_score": 40,
        "skip_keywords": [
            "Senior", "Lead", "Manager", "Director", "Head of", "Principal",
            "3+ years", "5+ years", "8+ years", "Machine Learning Engineer",
            "Data Scientist", "Frontend Developer", "Backend Developer",
            "DevOps", "QA Engineer", "Software Engineer", "Java Developer",
        ],
    },
    "db_path":  "data/linkedin_agent.db",
    "log_path": "data/agent.log",
}

PROFILE = Profile()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — DATABASE LAYER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Database:
    """
    SQLite wrapper with 6 tables:
        jobs, applications, recruiters, messages, followups, ats_scores
    """

    DDL = """
    -- Jobs master table
    CREATE TABLE IF NOT EXISTS jobs (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        title        TEXT NOT NULL,
        company      TEXT NOT NULL,
        url          TEXT UNIQUE,
        location     TEXT,
        industry     TEXT,
        jd_text      TEXT,
        created_at   TEXT DEFAULT (date('now'))
    );

    -- Application records
    CREATE TABLE IF NOT EXISTS applications (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id       INTEGER REFERENCES jobs(id),
        platform     TEXT DEFAULT 'LinkedIn',
        status       TEXT,          -- Applied | Skipped | Error | Interview
        cover_letter TEXT,
        applied_on   TEXT DEFAULT (date('now'))
    );

    -- Recruiter / hiring-manager contacts
    CREATE TABLE IF NOT EXISTS recruiters (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id          INTEGER REFERENCES jobs(id),
        name            TEXT,
        linkedin_url    TEXT,
        company         TEXT,
        is_hiring_mgr   INTEGER DEFAULT 0,
        discovered_on   TEXT DEFAULT (date('now'))
    );

    -- Outreach messages sent
    CREATE TABLE IF NOT EXISTS messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        recruiter_id    INTEGER REFERENCES recruiters(id),
        job_id          INTEGER REFERENCES jobs(id),
        variant         TEXT,       -- A | B | C
        message_text    TEXT,
        sent_at         TEXT,
        status          TEXT,       -- Sent | Failed | Skipped
        response_rcvd   INTEGER DEFAULT 0,
        response_text   TEXT
    );

    -- Follow-up schedule
    CREATE TABLE IF NOT EXISTS followups (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id      INTEGER REFERENCES messages(id),
        recruiter_id    INTEGER REFERENCES recruiters(id),
        job_id          INTEGER REFERENCES jobs(id),
        followup_day    INTEGER,    -- 3, 7, or 14
        scheduled_date  TEXT,
        message_text    TEXT,
        status          TEXT DEFAULT 'Pending'  -- Pending | Sent | Skipped
    );

    -- ATS / match scores per application
    CREATE TABLE IF NOT EXISTS ats_scores (
         id               INTEGER PRIMARY KEY AUTOINCREMENT,
         job_id           INTEGER REFERENCES jobs(id),
         ats_score        INTEGER,
         match_score      INTEGER,
         job_priority     TEXT,
         matching_skills  TEXT,
         missing_skills   TEXT,
         ats_keywords     TEXT,
         resume_summary   TEXT,
         suggested_skills TEXT,
         scored_on        TEXT DEFAULT (date('now'))
    );
    """

    def __init__(self, db_path: str = CONFIG["db_path"]):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        for stmt in self.DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)
        self.conn.commit()
        log.info(f"✅ Database ready: {db_path}")

    def upsert_job(self, job: JobInfo) -> int:
        cur = self.conn.execute(
            "SELECT id FROM jobs WHERE url=?", (job.url,))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO jobs (title,company,url,location,industry,jd_text) "
            "VALUES (?,?,?,?,?,?)",
            (job.title, job.company, job.url,
             job.location, job.industry, job.jd_text[:4000]))
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def already_applied(self, company: str, title: str) -> bool:
        cur = self.conn.execute(
            "SELECT 1 FROM jobs j JOIN applications a ON a.job_id=j.id "
            "WHERE j.company=? AND j.title=? AND a.status='Applied' LIMIT 1",
            (company.strip(), title.strip()))
        return cur.fetchone() is not None

    def log_application(self, job_id: int, status: str, cover_letter: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO applications (job_id,status,cover_letter) VALUES (?,?,?)",
            (job_id, status, cover_letter))
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def today_count(self, platform: str = "LinkedIn") -> int:
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM applications "
            "WHERE applied_on=? AND platform=? AND status='Applied'",
            (str(date.today()), platform))
        return cur.fetchone()[0]

    def upsert_recruiter(self, job_id: int, name: str, linkedin_url: str, company: str, is_hiring_mgr: bool = False) -> int:
        cur = self.conn.execute(
            "SELECT id FROM recruiters WHERE linkedin_url=? AND job_id=?",
            (linkedin_url, job_id))
        row = cur.fetchone()
        if row:
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO recruiters "
            "(job_id,name,linkedin_url,company,is_hiring_mgr) "
            "VALUES (?,?,?,?,?)",
            (job_id, name, linkedin_url, company, int(is_hiring_mgr)))
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def log_message(self, recruiter_id: int, job_id: int, variant: str, text: str, status: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages "
            "(recruiter_id,job_id,variant,message_text,sent_at,status) "
            "VALUES (?,?,?,?,?,?)",
            (recruiter_id, job_id, variant,
             text, datetime.now().isoformat(), status))
        self.conn.commit()
        return cur.lastrowid  # type: ignore

    def update_message_response(self, message_id: int, response: str) -> None:
        self.conn.execute(
            "UPDATE messages SET response_rcvd=1, response_text=? WHERE id=?",
            (response, message_id))
        self.conn.commit()

    def schedule_followup(self, message_id: int, recruiter_id: int, job_id: int, day: int, text: str) -> None:
        scheduled = (date.today() + timedelta(days=day)).isoformat()
        self.conn.execute(
            "INSERT INTO followups "
            "(message_id,recruiter_id,job_id,followup_day,scheduled_date,message_text) "
            "VALUES (?,?,?,?,?,?)",
            (message_id, recruiter_id, job_id, day, scheduled, text))
        self.conn.commit()

    def due_followups(self) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT f.*, r.linkedin_url, r.name AS recruiter_name "
            "FROM followups f JOIN recruiters r ON r.id=f.recruiter_id "
            "WHERE f.status='Pending' AND f.scheduled_date<=? "
            "ORDER BY f.scheduled_date",
            (str(date.today()),))
        return cur.fetchall()

    def mark_followup_sent(self, followup_id: int) -> None:
        self.conn.execute(
            "UPDATE followups SET status='Sent' WHERE id=?", (followup_id,))
        self.conn.commit()

    def log_ats(self, job_id: int, result: ATSResult) -> None:
        self.conn.execute(
            "INSERT INTO ats_scores "
            "(job_id,ats_score,match_score,job_priority,"
            " matching_skills,missing_skills,ats_keywords,"
            " resume_summary,suggested_skills) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, result.ats_score, result.match_score,
             result.job_priority,
             json.dumps(result.matching_skills),
             json.dumps(result.missing_skills),
             json.dumps(result.ats_keywords),
             result.resume_summary,
             result.suggested_skills))
        self.conn.commit()

    def analytics(self) -> dict:
        def q(sql: str):
            return self.conn.execute(sql).fetchall()

        apps          = q("SELECT status, COUNT(*) n FROM applications GROUP BY status")
        recruiters    = q("SELECT COUNT(*) n FROM recruiters")[0]["n"]
        msgs_sent     = q("SELECT COUNT(*) n FROM messages WHERE status='Sent'")[0]["n"]
        replies       = q("SELECT COUNT(*) n FROM messages WHERE response_rcvd=1")[0]["n"]
        avg_ats       = q("SELECT AVG(ats_score) a FROM ats_scores")[0]["a"] or 0
        avg_match     = q("SELECT AVG(match_score) a FROM ats_scores")[0]["a"] or 0
        top_companies = q(
            "SELECT j.company, COUNT(*) n FROM applications a "
            "JOIN jobs j ON j.id=a.job_id "
            "WHERE a.status='Applied' GROUP BY j.company "
            "ORDER BY n DESC LIMIT 5")

        return {
            "applications":   {r["status"]: r["n"] for r in apps},
            "recruiters":     recruiters,
            "messages_sent":  msgs_sent,
            "replies":        replies,
            "avg_ats_score":  round(avg_ats, 1),
            "avg_match_score": round(avg_match, 1),
            "top_companies":  [(r["company"], r["n"]) for r in top_companies],
        }

    def export_json(self, path: str = "data/applications.json") -> None:
        rows = self.conn.execute(
            "SELECT j.company, j.title, a.platform, j.url, a.status, "
            "       s.match_score, a.applied_on "
            "FROM applications a "
            "JOIN jobs j ON j.id=a.job_id "
            "LEFT JOIN ats_scores s ON s.job_id=j.id "
            "ORDER BY a.id DESC LIMIT 500").fetchall()
        data = [dict(r) for r in rows]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(path, "w"), indent=2)
        log.info(f"📊 Exported {len(data)} records → {path}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — AI ENGINE (Google Gemini)                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class AIEngine:
    """
    Wraps the Google Gemini API.
    Set GOOGLE_API_KEY env-var before running.
    If the SDK is missing or API key is absent, every method falls back to rules.
    """

    MODEL   = "gemini-1.5-flash"

    def __init__(self) -> None:
        self._client_available = False
        if _GEMINI_AVAILABLE:
            api_key = os.environ.get("AQ.Ab8RN6KVPgTddDwONZDSjejO4TryzAa9WzaPf0c9nrxLvz7Qmg")
            if api_key:
                genai.configure(api_key=api_key)
                self._client_available = True
                log.info("🤖 Google Gemini AI Engine ready.")
            else:
                log.warning("GOOGLE_API_KEY not found in env. Using rule-based fallback.")
        else:
            log.warning("google-generativeai not installed. Using rule-based fallback.")

    def _call(self, system: str, user: str, json_mode: bool = False) -> str:
        """Call Gemini and return text. Returns '' on any error."""
        if not self._client_available:
            return ""
        try:
            model = genai.GenerativeModel(
                model_name=self.MODEL,
                system_instruction=system,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json" if json_mode else "text/plain",
                    temperature=0.2
                )
            )
            resp = model.generate_content(user)
            return resp.text.strip()
        except Exception as e:
            log.warning(f"AI call error: {e}")
            return ""

    def analyze_jd(self, jd_text: str, profile: Profile) -> dict:
        system = (
            "You are an expert ATS and recruitment analyst. "
            "Respond ONLY with valid JSON — no markdown, no preamble."
        )
        user = f"""
Analyze this job description for the candidate profile below and return JSON:

JD:
{jd_text[:3000]}

CANDIDATE PROFILE:
Skills: {profile.skills}
Experience: {profile.experience_years} years
Education: {profile.education}
Certifications: {profile.certifications}
Projects: {profile.projects}

Return JSON with EXACTLY these keys:
{{
  "ats_score": 0,
  "match_score": 0,
  "decision": "APPLY",
  "reason": "brief reason",
  "required_skills": ["..."],
  "preferred_skills": ["..."],
  "matching_skills": ["..."],
  "missing_skills": ["..."],
  "ats_keywords": ["..."],
  "responsibilities": ["..."],
  "tools_mentioned": ["..."],
  "soft_skills": ["..."],
  "job_priority": "HIGH",
  "custom_pitch": "one-sentence pitch matching candidate to role",
  "resume_summary": "suggested 2-sentence resume summary for this role",
  "suggested_skills": "comma-separated skills to add/highlight"
}}
"""
        raw = self._call(system, user, json_mode=True)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return {}  

    def generate_messages(self, job: JobInfo, ats: ATSResult, profile: Profile) -> dict:
        system = (
            "You are a professional LinkedIn outreach copywriter. "
            "Respond ONLY with valid JSON — no markdown, no preamble."
        )
        skills_str = ", ".join(ats.matching_skills[:4]) or profile.skills[:80]
        user = f"""
Write 6 LinkedIn outreach messages for:
Candidate: {profile.name}
Role applied: {job.title}
Company: {job.company}
Recruiter: {job.recruiter_name or 'Hiring Manager'}
Candidate's matching skills: {skills_str}
Availability: Immediate joiner

Return JSON with EXACTLY these keys:
{{
  "version_a": "ultra-short ≤300 chars",
  "version_b": "professional ≤500 chars",
  "version_c": "networking ≤700 chars",
  "followup_3": "day-3 follow-up ≤300 chars",
  "followup_7": "day-7 follow-up ≤300 chars",
  "followup_14": "day-14 follow-up ≤300 chars"
}}

Rules:
- Address recruiter by first name.
- Mention {job.company} and the job title.
- Mention strongest 2-3 matching skills.
- Mention application already submitted.
- End with a polite call to action.
- Keep follow-ups brief and non-pushy.
"""
        raw = self._call(system, user, json_mode=True)
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                pass
        return {}

    def generate_cover_letter(self, job: JobInfo, ats: ATSResult, profile: Profile) -> str:
        system = "You are an expert cover-letter writer."
        skills_str = ", ".join(ats.matching_skills[:5]) or profile.skills[:100]
        user = f"""
Write a professional cover letter (max 250 words) for:
Candidate: {profile.name}
Role: {job.title}
Company: {job.company}
Matching skills: {skills_str}
Projects: {profile.projects}
ATS keywords to include: {', '.join(ats.ats_keywords[:8])}
Availability: Immediate joiner

Format:
Dear Hiring Manager,
[Body — 3 short paragraphs]
[Sign-off]
{profile.name}
"""
        result = self._call(system, user, json_mode=False)
        return result or ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — RULE-BASED JD ANALYZER (ATS scoring without AI)                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class JDAnalyzer:
    """
    Rule-based JD analysis — always runs first.
    If AIEngine is available it enriches / overrides these results.
    """

    MUST_SKILLS = [
        (r"\bexcel\b",           "Microsoft Excel"),
        (r"\bsql\b",             "SQL"),
        (r"\bpower.?bi\b",       "Power BI"),
        (r"\bmis\b",             "MIS Reporting"),
        (r"\bdata.?anal",        "Data Analytics"),
        (r"\breporting\b",       "Reporting"),
        (r"\bdashboard\b",       "Dashboard Design"),
        (r"\bbusiness.?intel",   "Business Intelligence"),
        (r"\bkpi\b",             "KPI Analysis"),
        (r"\bpivot\b",           "Pivot Tables"),
    ]
    BONUS_SKILLS = [
        (r"\bvlookup\b",         "VLOOKUP"),
        (r"\bpower.?query\b",    "Power Query"),
        (r"\bgoogle.?sheet",     "Google Sheets"),
        (r"\bdata.?clean",       "Data Cleaning"),
        (r"\bdata.?visual",      "Data Visualization"),
        (r"\btableau\b",         "Tableau"),
        (r"\bdata.?model",       "Data Modeling"),
        (r"\bconditional.?form", "Conditional Formatting"),
    ]
    STOP_PATTERNS = [
        r"senior\s+data", r"lead\s+analyst", r"data\s+scientist",
        r"machine\s+learning\s+engineer", r"\bdevops\b",
        r"call.?center", r"\bbpo\b", r"voice\s+process",
        r"frontend\s+developer", r"backend\s+developer",
        r"software\s+engineer\b", r"\bjava\s+developer\b",
    ]
    LOCATION_PATTERNS = [
        r"bengaluru", r"bangalore", r"hyderabad", r"remote",
        r"work\s+from\s+home", r"\bwfh\b",
    ]
    SOFT_SKILL_PATTERNS = [
        (r"commun",     "Communication"),
        (r"team",       "Teamwork"),
        (r"analytic",   "Analytical Thinking"),
        (r"problem",    "Problem Solving"),
        (r"detail",     "Attention to Detail"),
        (r"proactive",  "Proactive"),
        (r"organ",      "Organisation"),
    ]

    def analyze(self, jd_text: str, title: str = "") -> ATSResult:
        txt = (jd_text + " " + title).lower()

        # Hard-stop checks
        for pat in self.STOP_PATTERNS:
            if re.search(pat, txt, re.I):
                return ATSResult(decision="SKIP", reason=f"Hard-stop: '{pat}'")

        # Experience years check
        yrs = re.findall(r"(\d+)\s*[\+\-]?\s*years?", txt)
        if yrs and min(int(y) for y in yrs) >= 3 \
                and "fresher" not in txt and "0 year" not in txt:
            return ATSResult(decision="SKIP", reason="Requires 3+ years exp.")

        score = 0

        # Location bonus
        if any(re.search(p, txt) for p in self.LOCATION_PATTERNS):
            score += 15

        # Must-have skill matching
        matching, missing = [], []
        for pat, skill in self.MUST_SKILLS:
            if re.search(pat, txt, re.I):
                matching.append(skill)
                score += 8
            else:
                missing.append(skill)
        score = min(score, 55)

        # Bonus skills
        bonus_found = []
        for pat, skill in self.BONUS_SKILLS:
            if re.search(pat, txt, re.I):
                bonus_found.append(skill)
                score += 5
        score = min(score, 80)

        # Fresher / entry-level bonus
        if re.search(r"fresher|entry.?level|trainee|intern|0.?1\s*year", txt, re.I):
            score += 20
        score = min(score, 100)

        soft = [s for p, s in self.SOFT_SKILL_PATTERNS if re.search(p, txt, re.I)]
        ats_kw = matching + bonus_found

        tools = []
        for tool_pat in [r"excel", r"sql", r"power.?bi", r"tableau",
                         r"google.?sheet", r"python", r"r\b"]:
            if re.search(tool_pat, txt, re.I):
                tools.append(re.findall(tool_pat, txt, re.I)[0].title())

        priority = "HIGH" if score >= 75 else ("MEDIUM" if score >= 50 else "LOW")

        skills_list = matching[:3] or ["Advanced Excel", "SQL", "Power BI"]
        pitch = (
            f"Fresher Data Analyst skilled in {', '.join(skills_list)}. "
            f"Projects: Sales MIS Dashboard, Customer SQL Analysis, Power BI KPI. "
            f"Immediate joiner."
        )

        decision = "APPLY" if score >= CONFIG["filters"]["min_match_score"] else "SKIP"

        return ATSResult(
            ats_score=score,
            match_score=score,
            decision=decision,
            reason=f"Rule-based score {score}/100",
            required_skills=matching,
            preferred_skills=bonus_found,
            matching_skills=matching + bonus_found,
            missing_skills=missing[:5],
            ats_keywords=ats_kw,
            tools_mentioned=tools,
            soft_skills=soft,
            job_priority=priority,
            custom_pitch=pitch,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — RECRUITER EXTRACTOR                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class RecruiterExtractor:

    def extract_recruiter(self, page: Page) -> tuple[str, str]:
        selectors = [
            ".hirer-card__hirer-information a.app-aware-link",
            ".jobs-poster__name",
            ".recruitment-insights__recruiter-name",
            "a[data-control-name='recruiter_card']",
            ".jobs-company__box a.ember-view",
        ]
        for sel in selectors:
            try:
                el = page.query_selector(sel)
                if el:
                    name = el.inner_text().strip()
                    href = el.get_attribute("href") or ""
                    if name and "/in/" in href:
                        full_url = (
                            href if href.startswith("http")
                            else f"https://www.linkedin.com{href}"
                        )
                        log.info(f"  👤 Recruiter: {name} — {full_url}")
                        return name, full_url
            except Exception:
                pass
        return "", ""

    def extract_hiring_manager(self, page: Page) -> str:
        try:
            el = page.query_selector(".hirer-card__hirer-information h3")
            if el:
                return el.inner_text().strip()
        except Exception:
            pass
        return ""

    def extract_industry(self, page: Page) -> str:
        for sel in [
            ".jobs-unified-top-card__job-insight span",
            ".jobs-company__industry",
            "[data-test-id='company-industry']",
        ]:
            try:
                el = page.query_selector(sel)
                if el:
                    return el.inner_text().strip()
            except Exception:
                pass
        return ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — MESSAGE GENERATOR (rule-based fallback)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def generate_recruiter_message(job: JobInfo, ats: ATSResult, profile: Profile) -> RecruiterMessages:
    rname = job.recruiter_name.split()[0] if job.recruiter_name else "there"
    skills_str = (
        ", ".join(ats.matching_skills[:3])
        if ats.matching_skills
        else "Advanced Excel, SQL, Power BI"
    )

    va = (
        f"Hi {rname}, I applied for {job.title} at {job.company}. "
        f"Strong in {skills_str}. Immediate joiner — would love to connect!"
    )[:300]

    vb = (
        f"Hi {rname},\n\n"
        f"I recently submitted my application for the {job.title} role at "
        f"{job.company}. My background in {skills_str} aligns closely with "
        f"the requirements I saw in the JD.\n\n"
        f"I am a 2024 B.Com graduate, an immediate joiner, and excited about "
        f"contributing to data-driven decisions at {job.company}.\n\n"
        f"Would you be open to a brief chat?\n\n"
        f"Thanks,\n{profile.name}"
    )[:500]

    vc = (
        f"Hi {rname},\n\n"
        f"I hope you're doing well! I recently applied for the {job.title} "
        f"position at {job.company} and wanted to reach out personally.\n\n"
        f"As a fresher specialising in {skills_str}, I have built hands-on "
        f"projects: a Sales MIS Dashboard in Excel, a Customer Segmentation "
        f"analysis in SQL, and an executive KPI Report in Power BI. I'm also "
        f"pursuing the Google Data Analytics Certificate.\n\n"
        f"I believe my profile is a strong fit for this role and I am an "
        f"immediate joiner. I would genuinely appreciate 10 minutes of your "
        f"time to discuss how I can add value to {job.company}.\n\n"
        f"Thank you for your time,\n{profile.name}\n"
        f"{profile.phone} | {profile.email}"
    )[:700]

    f3 = (
        f"Hi {rname}, just a quick follow-up on my {job.title} application "
        f"at {job.company}. Still very interested — please let me know if "
        f"you need any additional information. Thank you!"
    )[:300]

    f7 = (
        f"Hi {rname}, following up again on the {job.title} role at "
        f"{job.company}. I remain enthusiastic and available immediately. "
        f"Happy to share my portfolio projects if helpful. Thank you!"
    )[:300]

    f14 = (
        f"Hi {rname}, I understand you're busy. I'm still very interested in "
        f"the {job.title} position at {job.company} and would welcome any "
        f"update when convenient. Thanks for your time!"
    )[:300]

    return RecruiterMessages(
        version_a=va, version_b=vb, version_c=vc,
        followup_3=f3, followup_7=f7, followup_14=f14
    )


def generate_followup_message(recruiter_name: str, job_title: str, company: str, day: int) -> str:
    rname = recruiter_name.split()[0] if recruiter_name else "there"
    templates = {
        3: (
            f"Hi {rname}, just a quick follow-up on my {job_title} application "
            f"at {company}. Still very excited about the opportunity — please let "
            f"me know if you need anything. Thank you!"
        ),
        7: (
            f"Hi {rname}, following up again on my {job_title} application at "
            f"{company}. I remain very interested and am available to start "
            f"immediately. Happy to provide any further information. Thank you!"
        ),
        14: (
            f"Hi {rname}, I hope the hiring process is going well. I'm still "
            f"keenly interested in the {job_title} role at {company} and would "
            f"welcome any update at your convenience. Thank you for your time!"
        ),
    }
    return templates.get(day, templates[14])


def generate_cover_letter(job: JobInfo, ats: ATSResult, profile: Profile) -> str:
    skills_str = (
        ", ".join(ats.matching_skills[:4])
        if ats.matching_skills
        else "Advanced Excel, SQL, Power BI, MIS Reporting"
    )
    kw_str = ", ".join(ats.ats_keywords[:6]) if ats.ats_keywords else skills_str

    letter = (
        f"Dear Hiring Manager,\n\n"
        f"I am {profile.name}, a 2024 B.Com graduate from Gulbarga University "
        f"with a strong foundation in Data Analytics, MIS Reporting, and "
        f"Business Intelligence. I am writing to express my interest in the "
        f"{job.title} role at {job.company}.\n\n"
        f"My core competencies include {skills_str}. These align directly with "
        f"the key requirements I identified in your job description, particularly "
        f"around {kw_str}.\n\n"
        f"Highlights from my project work:\n"
        f"• Sales MIS Dashboard (Excel) — consolidated 10+ data sheets into a "
        f"unified automated dashboard.\n"
        f"• Customer Data Analysis (SQL) — segmentation, revenue trends, and "
        f"lifetime-value insights on a retail dataset.\n"
        f"• Business KPI Report (Power BI) — executive-level dashboard tracking "
        f"revenue, opex, and YoY growth.\n\n"
        f"I am an immediate joiner and enthusiastic about contributing to "
        f"data-driven decision-making at {job.company}. I would welcome the "
        f"opportunity to discuss my profile further.\n\n"
        f"Thank you,\n{profile.name}\n"
        f"{profile.phone} | {profile.email}"
    )
    words = letter.split()
    if len(words) > 250:
        letter = " ".join(words[:250]) + "…"
    return letter


def tailor_resume(ats: ATSResult, profile: Profile) -> dict:
    candidate_skills_lower = profile.skills.lower()

    if ats.ats_keywords:
        hits = sum(
            1 for kw in ats.ats_keywords
            if kw.lower() in candidate_skills_lower
        )
        resume_ats_score = round((hits / len(ats.ats_keywords)) * 100)
    else:
        resume_ats_score = 50

    missing_keywords = [
        kw for kw in ats.missing_skills
        if kw.lower() not in candidate_skills_lower
    ]

    suggested_summary = ats.resume_summary or (
        f"Results-driven Data Analytics fresher with hands-on experience in "
        f"{', '.join(ats.matching_skills[:3] or ['Excel', 'SQL', 'Power BI'])}. "
        f"Immediate joiner pursuing Google Data Analytics certification."
    )

    suggested_skills = ats.suggested_skills or (
        profile.skills + (
            (", " + ", ".join(ats.preferred_skills[:3]))
            if ats.preferred_skills else ""
        )
    )

    return {
        "ats_resume_score":    resume_ats_score,
        "missing_keywords":    missing_keywords,
        "suggested_summary":   suggested_summary,
        "suggested_skills":    suggested_skills,
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — FORM ANSWER ENGINE                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
_QA_MAP: list[tuple[str, str]] = [
    (r"years.?of.?experience|how.?many.?years|experience.?in",  "0"),
    (r"current.?ctc|current.?salary",                           "0"),
    (r"expected.?ctc|expected.?salary",                         "Negotiable"),
    (r"notice.?period|when.?can.?you.?join|earliest.?start",   "Immediately"),
    (r"city|location|where.?are.?you.?based",                   "Bangalore"),
    (r"phone|mobile|contact.?number",                           PROFILE.phone),
    (r"linkedin",                                               PROFILE.linkedin_url),
    (r"cover.?letter|why.?apply|why.?this.?role|additional",   PROFILE.summary),
    (r"excel|spreadsheet|pivot",
     "Advanced Excel — Pivot Tables, VLOOKUP, Power Query, MIS dashboards."),
    (r"\bsql\b|database|query",
     "SQL — JOINs, GROUP BY, subqueries. Built customer segmentation project."),
    (r"power.?bi|bi.?tool|dashboard|tableau",
     "Power BI — slicers, drill-through, Power Query, KPI dashboards."),
    (r"mis|report.?generat",
     "Automated MIS reporting in Excel, consolidating 10+ sheets."),
    (r"certification|google.?data|coursera",
     "Google Data Analytics (In Progress), Excel Skills for Business, SQL for DS."),
    (r"strength|tell.*about.*yourself|introduce",
     "Detail-oriented Data Analytics fresher. Excel, SQL, Power BI. Immediate joiner."),
    (r"relocation",  "Yes, open to relocation to Bengaluru or Hyderabad."),
    (r"remote|hybrid|wfh", "Comfortable with remote, hybrid, or in-office."),
    (r"gender",      "Male"),
    (r"disability",  "No"),
    (r"veteran",     "No"),
    (r"sponsorship|visa", "Indian citizen, no sponsorship required."),
    (r"salary|compensation", "Negotiable"),
]

_KEY_MAP: dict[str, str] = {
    "name":    PROFILE.name,
    "email":   PROFILE.email,
    "first":   PROFILE.name.split()[0],
    "last":    PROFILE.name.split()[-1],
    "phone":   PROFILE.phone,
    "mobile":  PROFILE.phone,
    "city":    "Bangalore",
    "address": PROFILE.location,
    "linkedin": PROFILE.linkedin_url,
}

def linkedin_answer(label: str) -> str:
    label_l = label.lower().strip()
    for pattern, ans in _QA_MAP:
        if re.search(pattern, label_l, re.I):
            return ans
    for key, val in _KEY_MAP.items():
        if key in label_l:
            return val
    return ""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — HUMAN BROWSER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class HumanBrowser:
    def __init__(self, headless: bool = False) -> None:
        self.headless    = headless
        self._playwright = None
        self._browser    = None
        self.page: Optional[Page] = None

    def start(self) -> Page:
        self._playwright = sync_playwright().start()
        self._browser    = self._playwright.chromium.launch(
            headless=self.headless,
            args=["--disable-blink-features=AutomationControlled",
                  "--no-sandbox", "--disable-dev-shm-usage",
                  "--start-maximized"])
        ctx = self._browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"),
            locale="en-IN",
            timezone_id="Asia/Kolkata")
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        self.page = ctx.new_page()
        return self.page

    def stop(self) -> None:
        try:
            if self._browser:    self._browser.close()
            if self._playwright: self._playwright.stop()
        except Exception:
            pass

    def wait(self, lo: float = 1.0, hi: float = 3.0) -> None:
        time.sleep(random.uniform(lo, hi))

    def type_human(self, el: Locator, text: str) -> None:
        try:
            el.triple_click()
            self.page.keyboard.press("Control+a")  # type: ignore
            self.page.keyboard.press("Backspace")
            for ch in str(text):
                el.type(ch, delay=random.randint(
                    CONFIG["delays"]["typing"][0],
                    CONFIG["delays"]["typing"][1]))
            self.wait(0.2, 0.6)
        except Exception as e:
            log.debug(f"type_human: {e}")

    def scroll(self, times: int = 2) -> None:
        for _ in range(times):
            self.page.evaluate(  # type: ignore
                f"window.scrollBy(0,{random.randint(200, 500)})")
            self.wait(0.3, 0.9)

    def safe_click(self, selector: str, timeout: int = 5000) -> bool:
        try:
            el = self.page.wait_for_selector(selector, timeout=timeout)  # type: ignore
            if el and el.is_visible():
                el.scroll_into_view_if_needed()
                self.wait(0.3, 0.8)
                el.click()
                return True
        except Exception:
            pass
        return False


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11 — OUTREACH WORKFLOW                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def schedule_followups(db: Database, message_id: int, recruiter_id: int, job_id: int, messages: RecruiterMessages) -> None:
    for day, text in [(3, messages.followup_3),
                      (7, messages.followup_7),
                      (14, messages.followup_14)]:
        db.schedule_followup(message_id, recruiter_id, job_id, day, text)
        log.info(
            f"  📅 Follow-up scheduled: Day {day} "
            f"({(date.today() + timedelta(days=day)).isoformat()})")


def send_recruiter_message(page: Page, browser: HumanBrowser, db: Database, job: JobInfo, recruiter_id: int, job_id: int, messages: RecruiterMessages) -> bool:
    if not job.recruiter_url:
        log.info("  ℹ️  No recruiter URL — skipping outreach.")
        return False

    try:
        log.info(f"  📨 Opening recruiter profile: {job.recruiter_url}")
        page.goto(job.recruiter_url,
                  wait_until="domcontentloaded", timeout=30000)
        browser.wait(3, 5)

        msg_btn = None
        for sel in [
            "button.pvs-profile-actions__action[aria-label*='Message']",
            "button[aria-label*='Message']",
            "button:has-text('Message')",
            "a:has-text('Message')",
        ]:
            try:
                btn = page.wait_for_selector(sel, timeout=4000)
                if btn and btn.is_visible():
                    msg_btn = btn
                    break
            except Exception:
                pass

        if not msg_btn:
            log.info("  ❌ No Message button on recruiter profile.")
            db.log_message(recruiter_id, job_id, "B", messages.version_b, "Failed")
            return False

        msg_btn.scroll_into_view_if_needed()
        browser.wait(0.5, 1.5)
        msg_btn.click()
        browser.wait(2, 4)

        compose = None
        for sel in [
            "div.msg-form__contenteditable",
            "div[aria-label='Write a message…']",
            "div[data-placeholder='Write a message…']",
            "textarea.msg-form__textarea",
        ]:
            try:
                el = page.wait_for_selector(sel, timeout=5000)
                if el and el.is_visible():
                    compose = el
                    break
            except Exception:
                pass

        if not compose:
            log.info("  ❌ Could not locate message compose box.")
            db.log_message(recruiter_id, job_id, "B", messages.version_b, "Failed")
            return False

        browser.type_human(compose, messages.version_b)  # type: ignore
        browser.wait(1, 2)

        for sel in [
            "button.msg-form__send-button",
            "button[aria-label='Send']",
            "button:has-text('Send')",
        ]:
            try:
                send = page.query_selector(sel)
                if send and send.is_visible():
                    send.click()
                    browser.wait(2, 4)
                    msg_id = db.log_message(
                        recruiter_id, job_id, "B", messages.version_b, "Sent")
                    schedule_followups(db, msg_id, recruiter_id,
                                       job_id, messages)
                    log.info(
                        f"  ✅ Message sent to {job.recruiter_name} "
                        f"({job.company})")
                    return True
            except Exception:
                pass

        log.warning("  ⚠️ Send button not found — message not sent.")
        db.log_message(recruiter_id, job_id, "B", messages.version_b, "Failed")
        return False

    except Exception as e:
        log.error(f"  🚨 Outreach error: {e}")
        db.log_message(recruiter_id, job_id, "B", messages.version_b, "Failed")
        return False


def process_due_followups(page: Page, browser: HumanBrowser, db: Database) -> None:
    due = db.due_followups()
    if not due:
        log.info("📅 No follow-ups due today.")
        return

    log.info(f"📅 Processing {len(due)} due follow-up(s).")
    for row in due:
        recruiter_url = row["linkedin_url"]
        recruiter_name = row["recruiter_name"]
        text = row["message_text"]
        fu_id = row["id"]

        if not recruiter_url:
            db.mark_followup_sent(fu_id)
            continue

        try:
            page.goto(recruiter_url, wait_until="domcontentloaded", timeout=30000)
            browser.wait(3, 5)

            for sel in [
                "button[aria-label*='Message']",
                "button:has-text('Message')",
            ]:
                try:
                    btn = page.wait_for_selector(sel, timeout=4000)
                    if btn and btn.is_visible():
                        btn.click()
                        browser.wait(2, 3)
                        compose = page.wait_for_selector(
                            "div.msg-form__contenteditable, "
                            "textarea.msg-form__textarea",
                            timeout=5000)
                        if compose:
                            browser.type_human(compose, text)  # type: ignore
                            browser.wait(1, 2)
                            send = page.query_selector(
                                "button.msg-form__send-button, "
                                "button[aria-label='Send']")
                            if send and send.is_visible():
                                send.click()
                                browser.wait(2, 4)
                                db.mark_followup_sent(fu_id)
                                log.info(
                                    f"  ✅ Follow-up sent to {recruiter_name}")
                                break
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"  ⚠️ Follow-up error for {recruiter_name}: {e}")

        browser.wait(*CONFIG["delays"]["message_send"])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 12 — ANALYTICS DASHBOARD                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def print_analytics(db: Database) -> None:
    data = db.analytics()
    apps = data["applications"]
    total = sum(apps.values())

    print("\n" + "═" * 60)
    print("  📊  LINKEDIN AI AGENT — ANALYTICS DASHBOARD")
    print("═" * 60)
    print(f"  {'Date:':<30} {date.today()}")
    print("─" * 60)
    print(f"  {'Applications Sent:':<30} {apps.get('Applied', 0)}")
    print(f"  {'Skipped (JD mismatch):':<30} {apps.get('Skipped - JD Score', 0)}")
    print(f"  {'Errors:':<30} {apps.get('Error', 0)}")
    print(f"  {'Interviews Scheduled:':<30} {apps.get('Interview', 0)}")
    print(f"  {'Total Tracked:':<30} {total}")
    print("─" * 60)
    print(f"  {'Recruiters Contacted:':<30} {data['recruiters']}")
    print(f"  {'Messages Sent:':<30} {data['messages_sent']}")
    print(f"  {'Replies Received:':<30} {data['replies']}")
    if data['messages_sent']:
        rate = round(data['replies'] / data['messages_sent'] * 100, 1)
        print(f"  {'Reply Rate:':<30} {rate}%")
    print("─" * 60)
    print(f"  {'Avg ATS Score:':<30} {data['avg_ats_score']}/100")
    print(f"  {'Avg Match Score:':<30} {data['avg_match_score']}/100")
    print("─" * 60)
    if data["top_companies"]:
        print("  Top Companies Applied To:")
        for company, n in data["top_companies"]:
            print(f"    {company:<40} {n} application(s)")
    print("═" * 60 + "\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 13 — LINKEDIN AGENT (Orchestrator)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class LinkedInAgent:
    BASE_URL = "https://www.linkedin.com"

    def __init__(self) -> None:
        self.db           = Database()
        self.ai           = AIEngine()
        self.jd_analyzer  = JDAnalyzer()
        self.rec_extractor = RecruiterExtractor()
        self.browser      = HumanBrowser(headless=False)
        self.page: Optional[Page] = None
        self.limit        = CONFIG["limits"]["linkedin_per_day"]
        self.rec_limit    = CONFIG["limits"]["recruiters_per_day"]
        self.recruiters_today = 0
        self.report       = {
            "applied": 0, "skipped": 0, "errors": 0,
            "messages_sent": 0, "cover_letters": 0
        }

    def login(self) -> None:
        log.info("Opening LinkedIn login...")
        self.page = self.browser.start()
        self.page.goto("https://www.linkedin.com/login",
                       wait_until="domcontentloaded")
        print("\n" + "═" * 55)
        print("  Please log in to LinkedIn in the browser window.")
        print("  Then return here and press Enter to continue.")
        print("═" * 55)
        input("\n  ▶ Press Enter after logging in...\n")
        self.browser.wait(2, 4)
        log.info("✅ Logged in to LinkedIn.")

    def _search_url(self, keyword: str, location: str) -> str:
        kw  = keyword.replace(" ", "%20")
        loc = location.replace(" ", "%20").replace(",", "%2C")
        easy = "&f_LF=f_AL" if CONFIG["targets"]["easy_apply"] else ""
        return (
            f"{self.BASE_URL}/jobs/search/"
            f"?keywords={kw}&location={loc}"
            f"&f_TPR={CONFIG['targets']['date_posted']}"
            f"&f_E={CONFIG['targets']['exp_level']}{easy}"
            f"&sortBy=DD"
        )

    def search(self, keyword: str, location: str) -> list[dict]:
        url = self._search_url(keyword, location)
        log.info(f"🔍 {url}")
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)  # type: ignore
        except Exception as e:
            log.warning(f"Page load error: {e}")
            return []
        self.browser.wait(*CONFIG["delays"]["page_load"])

        for _ in range(5):
            self.browser.scroll(3)
            self.browser.wait(1.5, 2.5)

        try:
            self.page.wait_for_selector(  # type: ignore
                "ul.jobs-search__results-list li, "
                ".job-card-container, "
                ".jobs-search-results__list-item",
                timeout=10000)
        except Exception:
            log.warning("No job cards found.")
            return []

        cards = self.page.query_selector_all(  # type: ignore
            "ul.jobs-search__results-list li, "
            ".job-card-container, "
            ".jobs-search-results__list-item")
        log.info(f"  Found {len(cards)} cards.")

        skip_kw = [k.lower() for k in CONFIG["filters"]["skip_keywords"]]
        jobs: list[dict] = []
        for card in cards[:50]:
            try:
                title_el = card.query_selector(
                    "a.job-card-list__title, "
                    "a.job-card-container__link, "
                    ".job-card-list__title")
                co_el = card.query_selector(
                    ".job-card-container__primary-description, "
                    ".job-card-list__company-name, "
                    "span.job-card-container__company-name")
                if not title_el:
                    continue
                title   = title_el.inner_text().strip()
                company = co_el.inner_text().strip() if co_el else "Unknown"
                href    = title_el.get_attribute("href") or ""
                if not href:
                    continue
                if not href.startswith("http"):
                    href = self.BASE_URL + href
                if any(kw in title.lower() for kw in skip_kw):
                    log.debug(f"  Keyword skip: {title}")
                    continue
                jobs.append({"title": title, "company": company, "url": href})
            except Exception:
                continue

        log.info(f"  ✅ {len(jobs)} jobs after keyword filter.")
        return jobs

    def _jd_text(self) -> str:
        selectors = [
            ".jobs-description__content",
            ".job-view-layout .jobs-description",
            "#job-details",
            ".jobs-box__html-content",
        ]
        for sel in selectors:
            try:
                el = self.page.query_selector(sel)  # type: ignore
                if el:
                    return el.inner_text().strip()[:4000]
            except Exception:
                pass
        return self.page.evaluate("document.body.innerText").strip()[:4000]  # type: ignore

    def _fill_modal_page(self) -> None:
        modal = self.page.query_selector(  # type: ignore
            "div[data-test-modal-id='easy-apply-modal'], "
            ".jobs-easy-apply-modal, "
            "div.artdeco-modal") or self.page

        for inp in modal.query_selector_all(
                "input[type='text'], input[type='number'], "
                "input[type='tel'], input[type='email'], textarea"):
            try:
                label = (
                    inp.get_attribute("aria-label")
                    or inp.get_attribute("placeholder")
                    or inp.get_attribute("name")
                    or inp.get_attribute("id") or ""
                )
                if not label:
                    inp_id = inp.get_attribute("id")
                    if inp_id:
                        lbl = modal.query_selector(f"label[for='{inp_id}']")
                        if lbl:
                            label = lbl.inner_text().strip()
                if label:
                    ans = linkedin_answer(label)
                    if ans:
                        inp.scroll_into_view_if_needed()
                        self.browser.type_human(inp, ans)
                        self.browser.wait(0.2, 0.5)
            except Exception as e:
                log.debug(f"Input fill: {e}")

        for sel_el in modal.query_selector_all("select"):
            try:
                label = ""
                sel_id = sel_el.get_attribute("id")
                if sel_id:
                    lbl = modal.query_selector(f"label[for='{sel_id}']")
                    if lbl:
                        label = lbl.inner_text().strip()
                label = label or sel_el.get_attribute("aria-label") or ""
                ans = linkedin_answer(label)
                opts = sel_el.query_selector_all("option")
                matched = False
                if ans:
                    for opt in opts:
                        if ans.lower() in opt.inner_text().lower():
                            sel_el.select_option(value=opt.get_attribute("value"))
                            matched = True
                            break
                if not matched:
                    for opt in opts:
                        v = opt.get_attribute("value")
                        if v:
                            sel_el.select_option(value=v)
                            break
                self.browser.wait(0.2, 0.5)
            except Exception as e:
                log.debug(f"Select fill: {e}")

        for fieldset in modal.query_selector_all("fieldset"):
            try:
                legend = fieldset.query_selector("legend")
                if not legend:
                    continue
                question = legend.inner_text().strip()
                ans = linkedin_answer(question)
                if ans:
                    for lbl in fieldset.query_selector_all("label"):
                        if ans.lower()[:3] in lbl.inner_text().lower():
                            lbl.click()
                            break
            except Exception as e:
                log.debug(f"Radio fill: {e}")

        self.browser.wait(0.5, 1.0)

    def _easy_apply(self, cover_letter: str) -> bool:
        easy_btn = None
        for sel in [
            "button.jobs-apply-button",
            "button[aria-label*='Easy Apply']",
            "button:has-text('Easy Apply')",
        ]:
            try:
                btn = self.page.wait_for_selector(sel, timeout=5000)  # type: ignore
                if btn and btn.is_visible():
                    easy_btn = btn
                    break
            except Exception:
                pass

        if not easy_btn:
            return False

        easy_btn.scroll_into_view_if_needed()
        self.browser.wait(0.5, 1.5)
        easy_btn.click()
        self.browser.wait(2, 4)

        for step in range(12):
            modal_open = self.page.query_selector(  # type: ignore
                "div[data-test-modal-id='easy-apply-modal'], "
                ".jobs-easy-apply-modal, div.artdeco-modal")
            if not modal_open:
                break

            self._fill_modal_page()
            self.browser.wait(0.5, 1.5)

            for sel in [
                "textarea[aria-label*='cover']",
                "textarea[aria-label*='Cover']",
                "textarea[placeholder*='cover']",
            ]:
                try:
                    cl_el = self.page.query_selector(sel)  # type: ignore
                    if cl_el and cl_el.is_visible():
                        self.browser.type_human(cl_el, cover_letter)
                except Exception:
                    pass

            for sel in [
                "button[aria-label*='Submit application']",
                "button:has-text('Submit application')",
                "button:has-text('Submit')",
            ]:
                try:
                    btn = self.page.query_selector(sel)  # type: ignore
                    if btn and btn.is_visible():
                        btn.scroll_into_view_if_needed()
                        self.browser.wait(0.5, 1.0)
                        btn.click()
                        self.browser.wait(2, 5)
                        return True
                except Exception:
                    pass

            advanced = False
            for sel in [
                "button[aria-label*='Continue to next step']",
                "button[aria-label*='Review your application']",
                "button:has-text('Next')",
                "button:has-text('Review')",
                "button:has-text('Continue')",
            ]:
                try:
                    btn = self.page.query_selector(sel)  # type: ignore
                    if btn and btn.is_visible():
                        btn.scroll_into_view_if_needed()
                        self.browser.wait(0.5, 1.0)
                        btn.click()
                        self.browser.wait(1.5, 3.0)
                        advanced = True
                        break
                except Exception:
                    pass

            if not advanced:
                log.warning(f"  ⚠️ No Next/Submit at step {step + 1}.")
                self.browser.safe_click(
                    "button[aria-label='Dismiss'], button:has-text('Discard')")
                return False

        return False

    def apply(self, raw_job: dict) -> None:
        title, company, url = (
            raw_job["title"], raw_job["company"], raw_job["url"])

        if self.db.already_applied(company, title):
            log.info(f"  ↩️  Already applied: {title} @ {company}")
            return

        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=30000)  # type: ignore
            self.browser.wait(2, 5)

            jd_text = self._jd_text()
            rec_name, rec_url = self.rec_extractor.extract_recruiter(self.page)  # type: ignore
            hiring_mgr = self.rec_extractor.extract_hiring_manager(self.page)    # type: ignore
            industry   = self.rec_extractor.extract_industry(self.page)          # type: ignore

            job = JobInfo(
                title=title, company=company, url=url,
                jd_text=jd_text, industry=industry,
                recruiter_name=rec_name, recruiter_url=rec_url,
                hiring_manager=hiring_mgr,
            )
            job_id = self.db.upsert_job(job)

            ats = self.jd_analyzer.analyze(jd_text, title)
            log.info(
                f"  📋 [{ats.match_score}/100 | {ats.job_priority}] "
                f"{title} @ {company}")

            ai_data = self.ai.analyze_jd(jd_text, PROFILE)
            if ai_data:
                ats.ats_score       = ai_data.get("ats_score",       ats.ats_score)
                ats.match_score     = ai_data.get("match_score",     ats.match_score)
                ats.decision        = ai_data.get("decision",        ats.decision)
                ats.required_skills = ai_data.get("required_skills", ats.required_skills)
                ats.preferred_skills= ai_data.get("preferred_skills",ats.preferred_skills)
                ats.matching_skills = ai_data.get("matching_skills", ats.matching_skills)
                ats.missing_skills  = ai_data.get("missing_skills",  ats.missing_skills)
                ats.ats_keywords    = ai_data.get("ats_keywords",    ats.ats_keywords)
                ats.tools_mentioned = ai_data.get("tools_mentioned", ats.tools_mentioned)
                ats.soft_skills     = ai_data.get("soft_skills",     ats.soft_skills)
                ats.job_priority    = ai_data.get("job_priority",    ats.job_priority)
                ats.custom_pitch    = ai_data.get("custom_pitch",    ats.custom_pitch)
                ats.resume_summary  = ai_data.get("resume_summary",  ats.resume_summary)
                ats.suggested_skills= ai_data.get("suggested_skills",ats.suggested_skills)
                log.info(
                    f"  🤖 AI Score: {ats.match_score}/100 "
                    f"| Missing: {', '.join(ats.missing_skills[:3])}")

            self.db.log_ats(job_id, ats)

            if ats.decision == "SKIP":
                log.info(f"  🚫 Skip: {ats.reason}")
                self.db.log_application(job_id, "Skipped - JD Score")
                self.report["skipped"] += 1
                return

            resume_data = tailor_resume(ats, PROFILE)
            log.info(
                f"  📝 ATS Resume Score: {resume_data['ats_resume_score']}/100 "
                f"| Missing kw: {resume_data['missing_keywords'][:3]}")

            cover_letter = self.ai.generate_cover_letter(job, ats, PROFILE)
            if not cover_letter:
                cover_letter = generate_cover_letter(job, ats, PROFILE)
            self.report["cover_letters"] += 1

            success = self._easy_apply(cover_letter)

            if success:
                self.db.log_application(job_id, "Applied", cover_letter)
                self.report["applied"] += 1
                log.info(
                    f"  ✅ Applied [{ats.match_score}/100]: {title} @ {company}")
                log.info(f"  💬 {ats.custom_pitch}")

                if rec_url and self.recruiters_today < self.rec_limit:
                    recruiter_id = self.db.upsert_recruiter(
                        job_id, rec_name, rec_url, company)

                    msgs_data = self.ai.generate_messages(job, ats, PROFILE)
                    if msgs_data:
                        msgs = RecruiterMessages(**{
                            k: msgs_data.get(k, "")
                            for k in RecruiterMessages.__dataclass_fields__
                        })
                    else:
                        msgs = generate_recruiter_message(job, ats, PROFILE)

                    log.info(f"\n  ── Recruiter Messages ──────────────────────")
                    log.info(f"  [A] {msgs.version_a}")
                    log.info(f"  [B] {msgs.version_b[:100]}…")
                    log.info(f"  [C] {msgs.version_c[:100]}…")

                    sent = send_recruiter_message(
                        self.page, self.browser,  # type: ignore
                        self.db, job,
                        recruiter_id, job_id, msgs)
                    if sent:
                        self.report["messages_sent"] += 1
                        self.recruiters_today += 1
                        self.browser.wait(*CONFIG["delays"]["message_send"])
                else:
                    if not rec_url:
                        log.info("  ℹ️  No recruiter profile found — skipping outreach.")
                    else:
                        log.info("  ℹ️  Daily recruiter message limit reached.")
            else:
                self.db.log_application(job_id, "Easy Apply Failed")
                self.report["errors"] += 1

        except Exception as e:
            log.error(f"  🚨 Error [{title}]: {str(e)[:150]}")
            self.report["errors"] += 1

    def run(self) -> None:
        try:
            self.login()

            process_due_followups(self.page, self.browser, self.db)  # type: ignore

            count = self.db.today_count("LinkedIn")

            for keyword in CONFIG["targets"]["keywords"]:
                for location in CONFIG["targets"]["locations"]:
                    if count >= self.limit:
                        log.info("🛑 Daily application limit reached.")
                        return

                    log.info(f"\n📋 Keyword: {keyword} | Location: {location}")
                    jobs = self.search(keyword, location)

                    for raw_job in jobs:
                        if count >= self.limit:
                            log.info("🛑 Daily application limit reached.")
                            return
                        self.apply(raw_job)
                        count = self.db.today_count("LinkedIn")
                        time.sleep(
                            random.uniform(*CONFIG["delays"]["between_jobs"]))

                    time.sleep(
                        random.uniform(*CONFIG["delays"]["between_searches"]))

        except KeyboardInterrupt:
            log.info("⏹️ Stopped by user.")
        except Exception as e:
            log.critical(f"Fatal error: {e}", exc_info=True)
        finally:
            log.info(
                f"🏁 Applied today: {self.db.today_count('LinkedIn')}")
            print_analytics(self.db)
            self.db.export_json()
            self._print_report()
            self.browser.stop()

    def _print_report(self) -> None:
        r = self.report
        print("═" * 55)
        print("  📋 SESSION SUMMARY")
        print("═" * 55)
        print(f"  ✅ Applied          : {r['applied']}")
        print(f"  🚫 Skipped          : {r['skipped']}")
        print(f"  ❌ Errors           : {r['errors']}")
        print(f"  📨 Messages sent    : {r['messages_sent']}")
        print(f"  📄 Cover letters    : {r['cover_letters']}")
        print(f"  📅 Date             : {date.today()}")
        print("═" * 55 + "\n")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║   LinkedIn AI Job Search Agent — Abhishek Wagge          ║
║   Data Analyst | MIS Executive | BI Fresher              ║
║   Bengaluru / Hyderabad                                  ║
║                                                          ║
║   Modules active:                                        ║
║     ✔ JD Analyzer (rule-based + AI)                      ║
║     ✔ Cover Letter Generator                             ║
║     ✔ Resume Tailoring Engine                            ║
║     ✔ Recruiter Extractor                                ║
║     ✔ Message Generator (3 variants + 3 follow-ups)      ║
║     ✔ Auto Outreach Workflow                             ║
║     ✔ Follow-Up Scheduler                                ║
║     ✔ SQLite Database (6 tables)                         ║
║     ✔ Analytics Dashboard                                ║
╚══════════════════════════════════════════════════════════╝
""")
    agent = LinkedInAgent()
    agent.run()
    # export OPENAI_API_KEY="AQ.Ab8RN6KVPgTddDwONZDSjejO4TryzAa9WzaPf0c9nrxLvz7Qmg"
