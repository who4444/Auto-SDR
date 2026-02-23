
from __future__ import annotations
import argparse
import base64
import json
import os
import re
import sys
from typing import Optional
import httpx
import google.generativeai as genai
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from supabase import create_client, Client
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class Settings:
    GEMINI_API_KEY: str        = os.getenv("GEMINI_API_KEY", "")
    SCRAPINGDOG_API_KEY: str   = os.getenv("SCRAPINGDOG_API_KEY", "")
    SUPABASE_URL: str          = os.getenv("SUPABASE_URL", "")
    SUPABASE_SERVICE_KEY: str  = os.getenv("SUPABASE_SERVICE_KEY", "")
    GITHUB_TOKEN: str          = os.getenv("GITHUB_TOKEN", "")

    LLM_MODEL: str        = "gemini-2.0-flash"       # Free: 1000 RPD
    EMBEDDING_MODEL: str  = "gemini-embedding-001"    # Free: 1000 RPD
    EMBEDDING_DIM: int    = int(os.getenv("EMBEDDING_DIM", "768"))  # 768 | 1536 | 3072

    GITHUB_TOP_REPOS: int    = 3
    README_MAX_CHARS: int    = 6000
    MIN_COMPLETENESS: float  = 0.4

    MAX_RETRIES: int       = 3
    RETRY_WAIT_MIN: float  = 2.0
    RETRY_WAIT_MAX: float  = 10.0


cfg = Settings()


# ── Logger ────────────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level=os.getenv("LOG_LEVEL", "INFO"),
    colorize=True,
)
os.makedirs("logs", exist_ok=True)
logger.add(
    "logs/enrichment.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODELS (Pydantic DTOs)
# ══════════════════════════════════════════════════════════════════════════════

class GitHubRepo(BaseModel):
    name: str
    url: str
    description: Optional[str] = None
    stars: int = 0
    language: Optional[str] = None
    readme: Optional[str] = None
    readme_truncated: bool = False


class GitHubProfile(BaseModel):
    username: str
    bio: Optional[str] = None
    repos: list[GitHubRepo] = []
    fetch_error: Optional[str] = None


class LinkedInProfile(BaseModel):
    name: Optional[str] = None
    headline: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    job_history: list[dict] = []
    skills: list[str] = []
    fetch_error: Optional[str] = None


class RepoInsight(BaseModel):
    repo_name: str
    technologies: list[str] = []
    actual_role: str = "unknown"   # architect | lead | contributor | solo | unknown
    difficulty_score: int = 1      # 1–5
    key_insight: str = ""
    reasoning_error: Optional[str] = None


class EnrichedCandidate(BaseModel):
    linkedin_url: str
    github_url: Optional[str] = None
    linkedin: LinkedInProfile = LinkedInProfile()
    github: GitHubProfile = GitHubProfile(username="")
    repo_insights: list[RepoInsight] = []
    enriched_summary: str = ""
    embedding: Optional[list[float]] = None
    profile_completeness: float = 0.0
    completeness_breakdown: dict = {}
    warnings: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LINKEDIN SCRAPER (ScrapingDog)
# ══════════════════════════════════════════════════════════════════════════════

_SCRAPINGDOG_URL = "https://api.scrapingdog.com/profile"


@retry(
    stop=stop_after_attempt(cfg.MAX_RETRIES),
    wait=wait_exponential(min=cfg.RETRY_WAIT_MIN, max=cfg.RETRY_WAIT_MAX),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True,
)
def _call_scrapingdog(linkedin_url: str) -> dict:
    with httpx.Client(timeout=30.0) as client:
        r = client.get(_SCRAPINGDOG_URL, params={
            "api_key": cfg.SCRAPINGDOG_API_KEY,
            "type": "profile",
            "linkd_url": linkedin_url,
        })
    if r.status_code == 404:
        raise ValueError("LinkedIn profile không tồn tại hoặc bị khóa")
    if r.status_code == 403:
        raise ValueError("LinkedIn profile private — không thể scrape")
    r.raise_for_status()
    return r.json()


def _parse_job_history(experience_list: list) -> list[dict]:
    jobs = []
    for exp in (experience_list or []):
        job = {
            "title":    exp.get("title", ""),
            "company":  exp.get("company", exp.get("companyName", "")),
            "duration": exp.get("duration", exp.get("date1", "")),
            "description": exp.get("description", "")[:500],
        }
        if job["title"] or job["company"]:
            jobs.append(job)
    return jobs[:8]


def scrape_linkedin(linkedin_url: str) -> LinkedInProfile:
    """Scrape LinkedIn qua ScrapingDog. Không raise — lỗi ghi vào fetch_error."""
    logger.info(f"[LinkedIn] Scraping: {linkedin_url}")

    if not cfg.SCRAPINGDOG_API_KEY:
        logger.warning("[LinkedIn] SCRAPINGDOG_API_KEY chưa cấu hình — bỏ qua")
        return LinkedInProfile(fetch_error="SCRAPINGDOG_API_KEY missing")

    try:
        raw  = _call_scrapingdog(linkedin_url)
        data = raw.get("profile", raw)

        experience_raw = data.get("experience", data.get("positions", []))
        job_history    = _parse_job_history(experience_raw)

        skills_raw = data.get("skills", [])
        skills = (
            [s.get("name", s) if isinstance(s, dict) else str(s) for s in skills_raw]
            if isinstance(skills_raw, list) else []
        )

        profile = LinkedInProfile(
            name      = data.get("fullName", data.get("name", "")),
            headline  = data.get("headline", data.get("title", "")),
            bio       = data.get("summary",  data.get("about", "")),
            location  = data.get("location", ""),
            job_history = job_history,
            skills      = skills[:20],
        )
        logger.success(f"[LinkedIn] OK: {profile.name} | {len(profile.job_history)} jobs | {len(profile.skills)} skills")
        return profile

    except ValueError as e:
        logger.warning(f"[LinkedIn] {e}")
        return LinkedInProfile(fetch_error=str(e))
    except httpx.HTTPStatusError as e:
        logger.error(f"[LinkedIn] HTTP {e.response.status_code}")
        return LinkedInProfile(fetch_error=f"HTTP {e.response.status_code}")
    except Exception as e:
        logger.error(f"[LinkedIn] {e}")
        return LinkedInProfile(fetch_error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GITHUB FETCHER
# ══════════════════════════════════════════════════════════════════════════════

_GITHUB_BASE = "https://api.github.com"


def _github_headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if cfg.GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {cfg.GITHUB_TOKEN}"
    return h


@retry(
    stop=stop_after_attempt(cfg.MAX_RETRIES),
    wait=wait_exponential(min=cfg.RETRY_WAIT_MIN, max=cfg.RETRY_WAIT_MAX),
    retry=retry_if_exception_type(httpx.HTTPStatusError),
    reraise=True,
)
def _gh_get(client: httpx.Client, path: str) -> dict | list:
    r = client.get(f"{_GITHUB_BASE}{path}", headers=_github_headers())
    if r.status_code == 404:
        raise ValueError(f"Not found: {path}")
    r.raise_for_status()
    return r.json()


def _fetch_readme(client: httpx.Client, owner: str, repo: str) -> tuple[str, bool]:
    try:
        data    = _gh_get(client, f"/repos/{owner}/{repo}/readme")
        content = base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")

        # Strip heavy markdown — giữ text thuần để embedding hiệu quả hơn
        content = re.sub(r"!\[.*?\]\(.*?\)", "", content)          # images
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)  # links
        content = re.sub(r"```[\s\S]*?```", "[code block]", content) # fenced code
        content = re.sub(r"`[^`]+`", "", content)                   # inline code
        content = re.sub(r"#{1,6}\s+", "", content)                 # headings
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = content.strip()

        if len(content) > cfg.README_MAX_CHARS:
            return content[:cfg.README_MAX_CHARS] + "...[truncated]", True
        return content, False

    except Exception as e:
        logger.debug(f"[GitHub] README not found for {owner}/{repo}: {e}")
        return "", False


def fetch_github(github_url: str) -> GitHubProfile:
    """Fetch GitHub profile + top repos + READMEs. Không raise — lỗi ghi vào fetch_error."""
    logger.info(f"[GitHub] Fetching: {github_url}")

    # Extract username
    match = re.search(r"github\.com/([^/?#]+)", github_url)
    if not match:
        return GitHubProfile(username="", fetch_error=f"Invalid GitHub URL: {github_url}")
    username = match.group(1).rstrip("/").split("/")[0]

    try:
        with httpx.Client(timeout=20.0) as client:
            user_data  = _gh_get(client, f"/users/{username}")
            repos_data = _gh_get(client, f"/users/{username}/repos?sort=stargazers&per_page=30&type=public")

            if not isinstance(repos_data, list):
                repos_data = []

            # Ưu tiên repos không phải fork, có description hoặc stars
            meaningful = [
                r for r in repos_data
                if not r.get("fork") and (r.get("stargazers_count", 0) > 0 or r.get("description"))
            ] or repos_data

            repos = []
            for rd in meaningful[:cfg.GITHUB_TOP_REPOS]:
                readme, truncated = _fetch_readme(client, username, rd["name"])
                repos.append(GitHubRepo(
                    name        = rd["name"],
                    url         = rd.get("html_url", f"https://github.com/{username}/{rd['name']}"),
                    description = rd.get("description"),
                    stars       = rd.get("stargazers_count", 0),
                    language    = rd.get("language"),
                    readme      = readme or None,
                    readme_truncated = truncated,
                ))
                logger.debug(f"  {rd['name']} ({repos[-1].stars}⭐) README={'yes' if readme else 'no'}")

        profile = GitHubProfile(username=username, bio=user_data.get("bio") or "", repos=repos)
        logger.success(f"[GitHub] OK: @{username} | {len(repos)} repos | {sum(1 for r in repos if r.readme)} with README")
        return profile

    except ValueError as e:
        logger.warning(f"[GitHub] {e}")
        return GitHubProfile(username=username, fetch_error=str(e))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logger.warning("[GitHub] Rate limit — set GITHUB_TOKEN để tăng giới hạn")
            return GitHubProfile(username=username, fetch_error="GitHub rate limit — set GITHUB_TOKEN")
        return GitHubProfile(username=username, fetch_error=f"HTTP {e.response.status_code}")
    except Exception as e:
        logger.error(f"[GitHub] {e}")
        return GitHubProfile(username=username, fetch_error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LLM REASONING (Gemini Flash)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """Bạn là kỹ sư senior phân tích năng lực kỹ thuật.
Đọc README GitHub repo và trích xuất thông tin năng lực thực sự của developer.
Chỉ trả về JSON thuần, không markdown, không giải thích thêm.
actual_role phải là: architect | lead | contributor | solo | unknown"""

_REPO_PROMPT = """
Phân tích repo sau, trả về JSON với format chính xác:

=== THÔNG TIN REPO ===
Tên: {repo_name}
Ngôn ngữ: {language}
Stars: {stars}
Mô tả: {description}

=== NỘI DUNG README ===
{readme}

=== OUTPUT (JSON thuần, không có text khác) ===
{{
  "technologies": ["tech thực sự dùng, không phải chỉ đề cập"],
  "actual_role": "architect | lead | contributor | solo | unknown",
  "difficulty_score": 3,
  "key_insight": "1 câu mô tả điểm nổi bật về năng lực developer"
}}

Quy tắc:
- technologies: tech core thực sự build (bỏ qua CI/CD tools phụ)
- difficulty_score: 1 (tutorial) → 5 (production-scale phức tạp)
- key_insight ví dụ: "Xây dựng distributed caching layer xử lý 10k req/s trên K8s"
"""

genai.configure(api_key=cfg.GEMINI_API_KEY)
_llm = genai.GenerativeModel(cfg.LLM_MODEL)


@retry(stop=stop_after_attempt(cfg.MAX_RETRIES), wait=wait_exponential(min=2, max=15), reraise=False)
def _call_gemini(prompt: str) -> str:
    resp = _llm.generate_content(
        f"{_SYSTEM_PROMPT}\n\n{prompt}",
        generation_config=genai.types.GenerationConfig(temperature=0.2, max_output_tokens=512),
    )
    return resp.text


def _parse_json(text: str) -> dict:
    text = re.sub(r"```json\s*|```\s*", "", text).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"Không tìm thấy JSON: {text[:200]}")
    return json.loads(m.group())


def analyze_repo(repo: GitHubRepo) -> RepoInsight:
    """Phân tích 1 repo bằng Gemini. Không raise — lỗi ghi vào reasoning_error."""
    if not repo.readme:
        return RepoInsight(
            repo_name       = repo.name,
            technologies    = [repo.language] if repo.language else [],
            key_insight     = repo.description or "",
            reasoning_error = "No README",
        )

    if not cfg.GEMINI_API_KEY:
        return RepoInsight(repo_name=repo.name, reasoning_error="GEMINI_API_KEY missing")

    logger.info(f"[LLM] Analyzing: {repo.name} ({len(repo.readme)} chars)")

    try:
        raw  = _call_gemini(_REPO_PROMPT.format(
            repo_name   = repo.name,
            language    = repo.language or "unknown",
            stars       = repo.stars,
            description = repo.description or "N/A",
            readme      = repo.readme,
        ))
        data = _parse_json(raw)

        actual_role = data.get("actual_role", "unknown")
        if actual_role not in ("architect", "lead", "contributor", "solo", "unknown"):
            actual_role = "unknown"

        insight = RepoInsight(
            repo_name        = repo.name,
            technologies     = [str(t) for t in data.get("technologies", []) if t][:10],
            actual_role      = actual_role,
            difficulty_score = max(1, min(5, int(data.get("difficulty_score", 1)))),
            key_insight      = str(data.get("key_insight", ""))[:300],
        )
        logger.success(f"[LLM] {repo.name}: role={insight.actual_role}, diff={insight.difficulty_score}, tech={insight.technologies[:2]}")
        return insight

    except json.JSONDecodeError as e:
        logger.error(f"[LLM] JSON parse error for {repo.name}: {e}")
        return RepoInsight(repo_name=repo.name, reasoning_error=f"JSON error: {e}")
    except Exception as e:
        logger.error(f"[LLM] {repo.name}: {e}")
        return RepoInsight(repo_name=repo.name, reasoning_error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CONTEXT MERGER
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_jobs(jobs: list[dict]) -> str:
    lines = []
    for job in jobs[:5]:
        line = f"- {job.get('title', '')} tại {job.get('company', '')}"
        if job.get("duration"):
            line += f" ({job['duration']})"
        if job.get("description"):
            line += f": {job['description'][:200]}"
        lines.append(line)
    return "\n".join(lines)


def _fmt_insights(insights: list[RepoInsight]) -> str:
    lines = []
    for i in insights:
        if i.reasoning_error and not i.key_insight:
            continue
        parts = [f"Project: {i.repo_name}"]
        if i.technologies:
            parts.append(f"Tech: {', '.join(i.technologies)}")
        if i.actual_role != "unknown":
            parts.append(f"Vai trò: {i.actual_role}")
        if i.difficulty_score > 1:
            parts.append(f"Độ phức tạp: {i.difficulty_score}/5")
        if i.key_insight:
            parts.append(f"Nổi bật: {i.key_insight}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def build_enriched_summary(
    linkedin: LinkedInProfile,
    github: GitHubProfile,
    insights: list[RepoInsight],
    name: str = "",
) -> str:
    """
    Hợp nhất LinkedIn + GitHub → 1 document thuần văn bản.
    Document này sẽ được embedding model encode thành vector
    đại diện cho "toàn bộ chân dung năng lực" của ứng viên.
    """
    sections = []

    # 1. Identity
    identity = []
    if name or linkedin.name:
        identity.append(f"Tên: {name or linkedin.name}")
    if linkedin.headline:
        identity.append(f"Vị trí hiện tại: {linkedin.headline}")
    if linkedin.location:
        identity.append(f"Địa điểm: {linkedin.location}")
    if identity:
        sections.append("\n".join(identity))

    # 2. LinkedIn bio
    if linkedin.bio and len(linkedin.bio) > 20:
        sections.append(f"Tóm tắt nghề nghiệp:\n{linkedin.bio[:800]}")

    # 3. Skills
    if linkedin.skills:
        sections.append(f"Kỹ năng: {', '.join(linkedin.skills)}")

    # 4. Job history
    job_str = _fmt_jobs(linkedin.job_history)
    if job_str:
        sections.append(f"Kinh nghiệm làm việc:\n{job_str}")

    # 5. GitHub bio
    if github.bio and len(github.bio) > 10:
        sections.append(f"GitHub bio: {github.bio}")

    # 6. GitHub projects (từ LLM insights, fallback sang repo list)
    insights_str = _fmt_insights(insights)
    if insights_str:
        sections.append(f"Dự án GitHub nổi bật:\n{insights_str}")
    elif github.repos:
        fallback = "\n".join(
            f"- {r.name}" + (f" ({r.language})" if r.language else "") + (f": {r.description}" if r.description else "")
            for r in github.repos[:3]
        )
        sections.append(f"GitHub repos:\n{fallback}")

    summary = "\n\n".join(s for s in sections if s.strip())
    if not summary.strip():
        logger.warning("[Merger] enriched_summary rỗng — thiếu dữ liệu đầu vào")
    else:
        logger.debug(f"[Merger] enriched_summary: {len(summary)} chars")
    return summary


def compute_completeness(
    linkedin: LinkedInProfile,
    github: GitHubProfile,
    insights: list[RepoInsight],
    embedding_ok: bool = False,
) -> tuple[float, dict]:
    bd = {}

    # LinkedIn bio + job history: 0.30
    li = 0.0
    if not linkedin.fetch_error:
        if linkedin.bio and len(linkedin.bio) > 50:
            li += 0.15
        if linkedin.job_history:
            li += 0.15
    bd["linkedin"] = round(li, 2)

    # GitHub exists: 0.15
    gh = 0.15 if (not github.fetch_error and github.username) else 0.0
    bd["github_exists"] = round(gh, 2)

    # README found: 0.25
    rm = 0.25 if any(r.readme for r in github.repos) else 0.0
    bd["github_readme"] = round(rm, 2)

    # LLM reasoning: 0.20
    llm = 0.20 if any(not i.reasoning_error or i.key_insight for i in insights) else 0.0
    bd["llm_reasoning"] = round(llm, 2)

    # Embedding: 0.10
    em = 0.10 if embedding_ok else 0.0
    bd["embedding"] = round(em, 2)

    total = round(min(1.0, li + gh + rm + llm + em), 2)
    logger.debug(f"[Completeness] {total} | {bd}")
    return total, bd


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EMBEDDING (Gemini Embedding-001)
# ══════════════════════════════════════════════════════════════════════════════

@retry(stop=stop_after_attempt(cfg.MAX_RETRIES), wait=wait_exponential(min=3, max=20), reraise=True)
def _call_embed(text: str, task_type: str) -> list[float]:
    result = genai.embed_content(
        model                = cfg.EMBEDDING_MODEL,
        content              = text,
        task_type            = task_type,
        output_dimensionality = cfg.EMBEDDING_DIM,
    )
    return result["embedding"]


def embed_candidate(summary: str) -> list[float] | None:
    """Embed candidate profile (RETRIEVAL_DOCUMENT). Returns None nếu thất bại."""
    if not summary or len(summary.strip()) < 50:
        logger.warning("[Embed] Summary quá ngắn")
        return None
    if not cfg.GEMINI_API_KEY:
        logger.error("[Embed] GEMINI_API_KEY missing")
        return None
    try:
        text = summary[:8000]
        logger.info(f"[Embed] Creating {cfg.EMBEDDING_DIM}D vector ({len(text)} chars)...")
        vec = _call_embed(text, "RETRIEVAL_DOCUMENT")
        logger.success(f"[Embed] OK: {len(vec)} dimensions")
        return vec
    except Exception as e:
        logger.error(f"[Embed] {e}")
        return None


def embed_jd(jd_text: str) -> list[float] | None:
    """Embed Job Description (RETRIEVAL_QUERY). Returns None nếu thất bại."""
    if not jd_text or len(jd_text.strip()) < 20:
        return None
    try:
        vec = _call_embed(jd_text[:8000], "RETRIEVAL_QUERY")
        logger.success(f"[Embed] JD OK: {len(vec)}D")
        return vec
    except Exception as e:
        logger.error(f"[Embed] JD failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUPABASE DB
# ══════════════════════════════════════════════════════════════════════════════

_supabase_client: Client | None = None


def _get_db() -> Client:
    global _supabase_client
    if _supabase_client is None:
        if not cfg.SUPABASE_URL or not cfg.SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_URL / SUPABASE_SERVICE_KEY chưa cấu hình")
        _supabase_client = create_client(cfg.SUPABASE_URL, cfg.SUPABASE_SERVICE_KEY)
    return _supabase_client


def upsert_candidate(candidate: EnrichedCandidate) -> dict | None:
    """Upsert vào Talent_Candidates, conflict trên linkedin_url."""
    try:
        db = _get_db()
        payload: dict = {
            "linkedin_url":          candidate.linkedin_url,
            "enriched_summary":      candidate.enriched_summary,
            "profile_completeness":  candidate.profile_completeness,
            "tags":                  [],
        }
        if candidate.github_url:
            payload["github_url"] = candidate.github_url
        if candidate.linkedin.name:
            payload["name"] = candidate.linkedin.name
        if candidate.linkedin.bio:
            payload["raw_bio"] = candidate.linkedin.bio
        if candidate.embedding:
            payload["embedding"] = candidate.embedding

        result = db.table("Talent_Candidates").upsert(payload, on_conflict="linkedin_url").execute()

        if result.data:
            rec = result.data[0]
            logger.success(f"[DB] Upsert OK: {rec.get('name')} (id={rec.get('id')}, completeness={candidate.profile_completeness})")
            return rec
        logger.error(f"[DB] Upsert returned no data")
        return None

    except Exception as e:
        logger.error(f"[DB] Upsert failed: {e}")
        return None


def match_candidates(jd_embedding: list[float], threshold: float = 0.75, limit: int = 20) -> list[dict]:
    """Gọi RPC match_candidates (cần tạo function trong Supabase SQL Editor trước)."""
    try:
        result = _get_db().rpc("match_candidates", {
            "query_embedding": jd_embedding,
            "match_threshold": threshold,
            "match_count":     limit,
        }).execute()
        candidates = result.data or []
        logger.info(f"[DB] match_candidates: {len(candidates)} results (threshold={threshold})")
        return candidates
    except Exception as e:
        logger.error(f"[DB] match_candidates RPC failed: {e}")
        return []


def upsert_jd_embedding(requisition_id: str, embedding: list[float]) -> bool:
    """Update requirement_embedding cho 1 JD."""
    try:
        _get_db().table("Open_Requisitions").update(
            {"requirement_embedding": embedding}
        ).eq("id", requisition_id).execute()
        logger.success(f"[DB] JD embedding updated: {requisition_id}")
        return True
    except Exception as e:
        logger.error(f"[DB] JD embedding update failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_enrichment(
    linkedin_url: str,
    github_url: str | None = None,
    save_to_db: bool = True,
) -> EnrichedCandidate:
    """
    Pipeline chính:
      LinkedIn + GitHub → LLM reasoning → enriched_summary → embedding → Supabase

    Args:
        linkedin_url: URL LinkedIn (bắt buộc)
        github_url:   URL GitHub  (optional)
        save_to_db:   Lưu vào Supabase nếu True (default)

    Returns:
        EnrichedCandidate với đầy đủ thông tin và completeness score
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info("ENRICHMENT PIPELINE START")
    logger.info(f"  LinkedIn: {linkedin_url}")
    logger.info(f"  GitHub:   {github_url or 'N/A'}")
    logger.info(sep)

    candidate = EnrichedCandidate(linkedin_url=linkedin_url, github_url=github_url)

    # ── Step 1: LinkedIn ───────────────────────────────────────
    logger.info("[1/5] Scraping LinkedIn...")
    candidate.linkedin = scrape_linkedin(linkedin_url)
    if candidate.linkedin.fetch_error:
        candidate.warnings.append(f"LinkedIn: {candidate.linkedin.fetch_error}")

    # ── Step 2: GitHub ─────────────────────────────────────────
    if github_url:
        logger.info("[2/5] Fetching GitHub...")
        candidate.github = fetch_github(github_url)
        if candidate.github.fetch_error:
            candidate.warnings.append(f"GitHub: {candidate.github.fetch_error}")
    else:
        logger.info("[2/5] GitHub URL not provided — skipping")
        candidate.warnings.append("GitHub URL không được cung cấp")

    # ── Step 3: LLM Reasoning ──────────────────────────────────
    repos_with_readme = [r for r in candidate.github.repos if r.readme]
    if repos_with_readme:
        logger.info(f"[3/5] LLM reasoning {len(repos_with_readme)} repos...")
        candidate.repo_insights = [analyze_repo(r) for r in repos_with_readme]
    else:
        logger.info("[3/5] No READMEs — skipping LLM")

    # ── Step 4: Context Merge ──────────────────────────────────
    logger.info("[4/5] Building enriched summary...")
    candidate.enriched_summary = build_enriched_summary(
        linkedin = candidate.linkedin,
        github   = candidate.github,
        insights = candidate.repo_insights,
        name     = candidate.linkedin.name or "",
    )

    if not candidate.enriched_summary:
        candidate.warnings.append("Không đủ dữ liệu để tạo enriched summary")
        candidate.profile_completeness, candidate.completeness_breakdown = compute_completeness(
            candidate.linkedin, candidate.github, candidate.repo_insights
        )
        _log_result(candidate)
        return candidate

    # ── Step 5: Embedding ──────────────────────────────────────
    logger.info("[5/5] Creating vector embedding...")
    candidate.embedding = embed_candidate(candidate.enriched_summary)
    if not candidate.embedding:
        candidate.warnings.append("Embedding thất bại — profile không xuất hiện trong kết quả match")

    # ── Completeness ───────────────────────────────────────────
    candidate.profile_completeness, candidate.completeness_breakdown = compute_completeness(
        candidate.linkedin, candidate.github, candidate.repo_insights,
        embedding_ok = candidate.embedding is not None,
    )

    # ── Save DB ────────────────────────────────────────────────
    if save_to_db:
        if candidate.profile_completeness >= cfg.MIN_COMPLETENESS:
            upsert_candidate(candidate)
        else:
            logger.warning(f"[DB] Completeness {candidate.profile_completeness} < {cfg.MIN_COMPLETENESS} — skip DB, cần review thủ công")
            candidate.warnings.append(f"Completeness quá thấp ({candidate.profile_completeness}) — không lưu DB")

    _log_result(candidate)
    return candidate


def _completeness_label(score: float) -> str:
    return (
        "Excellent" if score >= 0.8
        else "Good" if score >= 0.6
        else "Fair" if score >= 0.4
        else "Poor"
    )


def _log_result(c: EnrichedCandidate) -> None:
    logger.info("=" * 60)
    logger.info("ENRICHMENT COMPLETE")
    logger.info(f"  Name:         {c.linkedin.name or 'Unknown'}")
    logger.info(f"  Completeness: {c.profile_completeness} ({_completeness_label(c.profile_completeness)})")
    logger.info(f"  Summary:      {len(c.enriched_summary)} chars")
    logger.info(f"  Embedding:    {'✓' if c.embedding else '✗'}")
    logger.info(f"  Repos OK:     {len([i for i in c.repo_insights if not i.reasoning_error])}/{len(c.repo_insights)}")
    for w in c.warnings:
        logger.warning(f"  ⚠ {w}")
    logger.info("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CLI & WEBHOOK SERVER
# ══════════════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser(
        description="AI Talent Enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python enrichment.py --linkedin https://linkedin.com/in/username
  python enrichment.py --linkedin https://linkedin.com/in/username --github https://github.com/username
  python enrichment.py --linkedin https://linkedin.com/in/username --no-db --output result.json
  python enrichment.py --server
        """,
    )
    parser.add_argument("--linkedin", required=False, default=None)
    parser.add_argument("--github",   default=None)
    parser.add_argument("--no-db",    action="store_true")
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()
    if not args.linkedin:
        parser.print_help()
        sys.exit(1)
    c = run_enrichment(args.linkedin, args.github, save_to_db=not args.no_db)

    print(f"\n{'='*60}\nKẾT QUẢ ENRICHMENT\n{'='*60}")
    print(f"Tên:           {c.linkedin.name or 'N/A'}")
    print(f"Completeness:  {c.profile_completeness*100:.0f}% ({_completeness_label(c.profile_completeness)})")
    print(f"Embedding:     {'✓' if c.embedding else '✗'}")
    print(f"\nBreakdown:")
    for k, v in c.completeness_breakdown.items():
        print(f"  {k}: {v}")
    if c.warnings:
        print(f"\nWarnings:")
        for w in c.warnings:
            print(f"  ⚠ {w}")
    print(f"\nEnriched Summary Preview:\n{'-'*40}")
    print(c.enriched_summary[:500] + ("..." if len(c.enriched_summary) > 500 else ""))
    if c.repo_insights:
        print(f"\nGitHub Insights:")
        for i in c.repo_insights:
            if not i.reasoning_error:
                print(f"  • {i.repo_name}: {i.actual_role}, diff={i.difficulty_score}/5")
                if i.key_insight:
                    print(f"    → {i.key_insight}")

    if args.output:
        out = {
            "linkedin_url":           c.linkedin_url,
            "github_url":             c.github_url,
            "name":                   c.linkedin.name,
            "profile_completeness":   c.profile_completeness,
            "completeness_breakdown": c.completeness_breakdown,
            "enriched_summary":       c.enriched_summary,
            "embedding_length":       len(c.embedding) if c.embedding else 0,
            "repo_insights": [
                {"repo": i.repo_name, "technologies": i.technologies,
                 "actual_role": i.actual_role, "difficulty_score": i.difficulty_score,
                 "key_insight": i.key_insight}
                for i in c.repo_insights if not i.reasoning_error
            ],
            "warnings": c.warnings,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Kết quả lưu vào: {args.output}")


def _create_server():
    """FastAPI webhook server cho n8n."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel as PM
    except ImportError:
        logger.error("Chạy: pip install fastapi uvicorn")
        return None

    app = FastAPI(title="AI Talent Enrichment", version="1.0.0")

    # Enable CORS for n8n
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class Req(PM):
        linkedin_url: str
        github_url: Optional[str] = None
        save_to_db: bool = True

    @app.post("/enrich")
    async def enrich(req: Req):
        """
        n8n HTTP Request node:
          POST http://localhost:8000/enrich
          Body: {"linkedin_url": "...", "github_url": "...", "save_to_db": true}
        Response: Trả về full candidate data hoặc error
        """
        logger.info(f"[API] POST /enrich - LinkedIn: {req.linkedin_url}")
        if not req.linkedin_url:
            logger.warning("[API] Missing linkedin_url")
            raise HTTPException(400, "linkedin_url is required")
        try:
            c = run_enrichment(req.linkedin_url, req.github_url, req.save_to_db)
            logger.info(f"[API] Enrichment complete for {c.linkedin.name}")
            return {
                "success":                  True,
                "name":                     c.linkedin.name,
                "linkedin_url":             c.linkedin_url,
                "github_url":               c.github_url,
                "profile_completeness":     c.profile_completeness,
                "completeness_label":       _completeness_label(c.profile_completeness),
                "completeness_breakdown":   c.completeness_breakdown,
                "enriched_summary":         c.enriched_summary,
                "enriched_summary_preview": c.enriched_summary[:500],
                "embedding_created":        c.embedding is not None,
                "embedding_length":         len(c.embedding) if c.embedding else 0,
                "warnings":                 c.warnings,
                "repo_insights_count":      len([i for i in c.repo_insights if not i.reasoning_error]),
                "repo_insights": [
                    {
                        "repo_name": i.repo_name,
                        "technologies": i.technologies,
                        "actual_role": i.actual_role,
                        "difficulty_score": i.difficulty_score,
                        "key_insight": i.key_insight,
                    }
                    for i in c.repo_insights if not i.reasoning_error
                ],
            }
        except Exception as e:
            logger.error(f"[API] Enrichment failed: {e}")
            raise HTTPException(500, f"Enrichment failed: {str(e)}")

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "ok",
            "version": "1.0.0",
            "service": "AI Talent Enrichment",
        }

    @app.get("/config")
    async def config():
        """Show API configuration (không có sensitive keys)"""
        return {
            "embedding_model": cfg.EMBEDDING_MODEL,
            "embedding_dim": cfg.EMBEDDING_DIM,
            "llm_model": cfg.LLM_MODEL,
            "github_top_repos": cfg.GITHUB_TOP_REPOS,
            "min_completeness": cfg.MIN_COMPLETENESS,
        }

    return app


if __name__ == "__main__":
    if "--server" in sys.argv:
        sys.argv.remove("--server")
        try:
            import uvicorn
        except ImportError:
            logger.error("Chạy: pip install fastapi uvicorn")
            sys.exit(1)
        
        logger.info("="*60)
        logger.info("Starting n8n Webhook Server")
        logger.info("="*60)
        logger.info("Available endpoints:")
        logger.info("  POST   http://0.0.0.0:8000/enrich    - Enrich candidate profile")
        logger.info("  GET    http://0.0.0.0:8000/health    - Health check")
        logger.info("  GET    http://0.0.0.0:8000/config    - Show configuration")
        logger.info("="*60)
        
        app = _create_server()
        if app:
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        else:
            logger.error("Failed to create FastAPI app")
            sys.exit(1)
    else:
        _cli()
