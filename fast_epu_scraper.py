# -*- coding: utf-8 -*-
"""
Fast Irish EPU Scraper - Optimized version with concurrent requests.
Uses IDENTICAL Boolean search criteria as the original script.
"""

import datetime as dt
import hashlib
import logging
import os
import re
import sqlite3
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x, **kwargs):
        return x


# =========================
# Configuration
# =========================

# Date range to scrape
START_DATE = "2025-10-01"
END_DATE = "2025-12-31"

# Output directory (new, separate from original)
OUTDIR = "out_fast_epu_scrape"
STATE_DB_NAME = "fast_epu_state.sqlite"

# Run all sources at once
RUN_SOURCES = ["IrishTimes", "IrishExaminer", "IrishIndependent"]

# Network settings - optimized for speed
USE_CACHE = True
TIMEOUT_SECONDS = 15
MAX_WORKERS = 10  # Concurrent requests
USER_AGENT = "Mozilla/5.0 (compatible; IrishEPUResearch/1.0)"

# Processing
PROCESS_LIMIT_PER_RUN: Optional[int] = None
MAX_ATTEMPTS_FETCH_FAIL = 2
PROCESS_COMMIT_EVERY = 100

VERBOSE = True


# =========================
# Boolean query terms - IDENTICAL to original
# =========================

BOOLEAN_QUERY_TEXT = (
    '(economy OR economic) AND (uncertainty OR uncertain) AND '
    '(regulation OR legislation OR deficit OR surplus OR "House of the Oireachtas" OR Oireachtas OR '
    '"Seanad Éireann" OR Seanad OR "Dáil Éireann" OR Dáil OR Government OR "Áras an Uachtaráin" OR '
    '"Central Bank of Ireland" OR "Irish central bank" OR Policy OR Taoiseach OR Tánaiste OR TD OR '
    '"Teachta Dála" OR President OR "Local authority" OR "Local authorities" OR council OR budget OR '
    '"department of finance" OR referendum OR referenda OR constitution OR "constitutional amendment" OR '
    'minister OR "Fianna Fáil" OR "Fine Gael")'
)

GROUP_ECONOMY = ["economy", "economic"]
GROUP_UNCERTAINTY = ["uncertainty", "uncertain"]
GROUP_POLICY = [
    "regulation",
    "legislation",
    "deficit",
    "surplus",
    "House of the Oireachtas",
    "Oireachtas",
    "Seanad Éireann",
    "Seanad",
    "Dáil Éireann",
    "Dáil",
    "Government",
    "Áras an Uachtaráin",
    "Central Bank of Ireland",
    "Irish central bank",
    "Policy",
    "Taoiseach",
    "Tánaiste",
    "TD",
    "Teachta Dála",
    "President",
    "Local authority",
    "Local authorities",
    "council",
    "budget",
    "department of finance",
    "referendum",
    "referenda",
    "constitution",
    "constitutional amendment",
    "minister",
    "Fianna Fáil",
    "Fine Gael",
]


# =========================
# Logging
# =========================

LOGGER = logging.getLogger("fast_epu_scraper")


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)


# =========================
# Helpers - IDENTICAL text processing
# =========================

def parse_yyyy_mm_dd(x: str) -> dt.date:
    return dt.date.fromisoformat(x)


def month_start(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)


def date_range_inclusive(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def normalise_url(url: str) -> str:
    p = urlparse(url.strip())
    if not p.scheme:
        return url.strip()
    p2 = p._replace(fragment="", query="")
    netloc = p2.netloc.lower().replace(":80", "").replace(":443", "")
    p2 = p2._replace(netloc=netloc)
    return urlunparse(p2)


def normalise_text(text: str) -> str:
    """IDENTICAL to original - critical for matching consistency"""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compile_term_pattern(term: str) -> re.Pattern:
    """IDENTICAL to original"""
    term_n = normalise_text(term)
    parts = term_n.split()
    if not parts:
        return re.compile(r"(?!)")
    core = r"\s+".join(re.escape(p) for p in parts)
    pat = r"(?<![A-Za-z0-9])" + core + r"(?![A-Za-z0-9])"
    return re.compile(pat, flags=re.IGNORECASE)


ECON_PATTERNS = [compile_term_pattern(t) for t in GROUP_ECONOMY]
UNC_PATTERNS = [compile_term_pattern(t) for t in GROUP_UNCERTAINTY]
POL_PATTERNS = [compile_term_pattern(t) for t in GROUP_POLICY]


def any_match(patterns: Sequence[re.Pattern], text_norm: str) -> bool:
    return any(p.search(text_norm) for p in patterns)


def evaluate_boolean_query(title: str, body: str) -> Tuple[bool, bool, bool, bool]:
    """IDENTICAL to original - evaluates the Boolean search criteria"""
    t = normalise_text(f"{title}\n{body}")
    econ = any_match(ECON_PATTERNS, t)
    unc = any_match(UNC_PATTERNS, t)
    pol = any_match(POL_PATTERNS, t)
    return econ, unc, pol, (econ and unc and pol)


def soup_html(html: str) -> BeautifulSoup:
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except Exception:
            continue
    return BeautifulSoup(html, "html.parser")


# =========================
# HTTP Session
# =========================

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IE,en;q=0.9",
        "Connection": "keep-alive",
    })
    return s


def cached_get(session: requests.Session, url: str, cache_path: Path, is_xml: bool) -> Optional[str]:
    if USE_CACHE and cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    try:
        r = session.get(url, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        LOGGER.debug("Request failed %s (%s)", url, exc)
        return None

    text = r.text or ""

    if is_xml:
        ok = 200 <= r.status_code < 300
        looks_xml = ("<urlset" in text) or ("<sitemapindex" in text) or text.strip().startswith("<?xml")
        if not ok and not looks_xml:
            return None
    else:
        if not (200 <= r.status_code < 300):
            return None

    if USE_CACHE:
        try:
            ensure_dir(cache_path.parent)
            cache_path.write_text(text, encoding="utf-8", errors="ignore")
        except Exception:
            pass

    return text


# =========================
# XML helpers
# =========================

def xml_extract_locs(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    out: List[str] = []
    for el in root.iter():
        if isinstance(el.tag, str) and el.tag.endswith("loc") and el.text:
            out.append(el.text.strip())
    return out


def xml_extract_url_entries(xml_text: str) -> List[Tuple[str, Optional[str]]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    entries: List[Tuple[str, Optional[str]]] = []
    for url_node in root.findall(".//{*}url"):
        loc = None
        lastmod = None
        for child in list(url_node):
            if not isinstance(child.tag, str):
                continue
            if child.tag.endswith("loc") and child.text:
                loc = child.text.strip()
            elif child.tag.endswith("lastmod") and child.text:
                lastmod = child.text.strip()
        if loc:
            entries.append((loc, lastmod))
    return entries


def parse_lastmod_date(lastmod_text: Optional[str]) -> Optional[dt.date]:
    if not lastmod_text:
        return None
    ts = pd.to_datetime(lastmod_text, errors="coerce", utc=True)
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.to_pydatetime().date()


def parse_datetime_maybe(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.to_pydatetime()


def date_from_url_yyyy_mm_dd(url: str) -> Optional[dt.date]:
    m = re.search(r"/((?:19|20)\d{2})/(\d{2})/(\d{2})/", url)
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


# =========================
# SQLite state database
# =========================

def iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def db_connect(db_path: Path) -> sqlite3.Connection:
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            published_date TEXT,
            month TEXT,
            title TEXT,
            text_len INTEGER,
            paywalled INTEGER,
            econ_hit INTEGER,
            uncertainty_hit INTEGER,
            policy_hit INTEGER,
            match INTEGER,
            status TEXT NOT NULL DEFAULT 'discovered',
            attempts_fetch_fail INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            discovered_at TEXT,
            updated_at TEXT
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_source_month ON articles(source, month);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_month ON articles(month);")
    conn.commit()


def db_upsert_discovered(conn: sqlite3.Connection, source: str, url: str, date_hint: Optional[dt.date]) -> bool:
    url = normalise_url(url)
    pd_str = date_hint.isoformat() if date_hint else None
    m_str = month_start(date_hint).isoformat() if date_hint else None
    now = iso_now()

    cur = conn.execute("""
        INSERT OR IGNORE INTO articles
        (url, source, published_date, month, status, attempts_fetch_fail, last_error, discovered_at, updated_at)
        VALUES (?, ?, ?, ?, 'discovered', 0, NULL, ?, ?);
    """, (url, source, pd_str, m_str, now, now))
    inserted = cur.rowcount == 1

    if (not inserted) and date_hint:
        conn.execute("""
            UPDATE articles
            SET published_date = COALESCE(published_date, ?),
                month = COALESCE(month, ?),
                updated_at = ?
            WHERE url = ?;
        """, (pd_str, m_str, now, url))

    return inserted


def db_get_pending(conn: sqlite3.Connection, *, sources: Sequence[str], start: dt.date, end: dt.date,
                   limit: Optional[int]) -> List[sqlite3.Row]:
    if not sources:
        return []

    start_m = month_start(start).isoformat()
    end_m = month_start(end).isoformat()

    placeholders = ",".join(["?"] * len(sources))
    params: List[object] = list(sources)

    sql = f"""
        SELECT url, source
        FROM articles
        WHERE source IN ({placeholders})
          AND month IS NOT NULL
          AND month >= ?
          AND month <= ?
          AND (status = 'discovered' OR (status = 'fetch_failed' AND attempts_fetch_fail < ?))
        ORDER BY COALESCE(published_date, '9999-12-31'), url
    """

    params = params + [start_m, end_m, int(MAX_ATTEMPTS_FETCH_FAIL)]

    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    return list(conn.execute(sql, params).fetchall())


# =========================
# URL filters - IDENTICAL to original
# =========================

IT_EXCLUDES = (
    "/video/", "/podcast", "/podcasts/", "/sponsored/", "/advertising-feature",
    "/photography/", "/photos/", "/weather/", "/crossword", "/crosswords",
    "/games/", "/horoscopes", "/newsletter", "/sign-up", "/subscribe/",
    "/myaccount/", "/membership/", "/zephr/", "/captcha/", "/status/",
)
IE_EXCLUDES = ("/sponsored/", "/sponsored-content/", "/sponsoredshowcase/")


def looks_like_irish_times_article(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in {"http", "https"} or not p.netloc.endswith("irishtimes.com"):
        return False
    path = p.path.lower().rstrip("/")
    if not path or path == "/":
        return False
    if any(bad in path for bad in IT_EXCLUDES):
        return False
    if re.search(r"\.(jpg|jpeg|png|gif|webp|pdf|mp4|mp3)$", path):
        return False
    return bool(re.search(r"/(19|20)\d{2}/\d{2}/\d{2}/", path) or re.search(r"-1\.\d+$", path))


def looks_like_irish_examiner_article(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in {"http", "https"} or not p.netloc.endswith("irishexaminer.com"):
        return False
    path = p.path.lower()
    if any(bad in path for bad in IE_EXCLUDES):
        return False
    return bool(re.search(r"/arid-\d+", path))


def looks_like_irish_independent_article(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in {"http", "https"} or not p.netloc.endswith("independent.ie"):
        return False
    path = p.path.lower()
    if any(x in path for x in ("/robots.txt", "/sitemap", "/account", "/api", "/search")):
        return False
    if any(x in path for x in ("/video", "/videos", "/podcast", "/podcasts")):
        return False
    return path.endswith(".html") or bool(re.search(r"/a\d+\.html$", path))


# =========================
# Discovery functions
# =========================

def discover_irish_times(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date,
                         end: dt.date) -> int:
    templates = (
        "https://www.irishtimes.com/arc/outboundfeeds/sitemap3/{date}/",
        "https://www.irishtimes.com/arc/outboundfeeds/sitemap2/{date}/",
    )
    inserted = 0
    seen: set = set()

    for day in tqdm(list(date_range_inclusive(start, end)), desc="Discover Irish Times", unit="day"):
        ymd = day.isoformat()
        locs: List[str] = []

        for tmpl in templates:
            sm_url = tmpl.format(date=ymd)
            cache_path = cache_dir / "sitemaps" / f"it_{ymd}_{safe_sha1(sm_url)[:10]}.xml"
            xml_text = cached_get(session, sm_url, cache_path, is_xml=True)
            if not xml_text:
                continue
            locs = xml_extract_locs(xml_text)
            if locs:
                break

        for loc in locs:
            u = normalise_url(loc)
            if u in seen:
                continue
            seen.add(u)
            if not looks_like_irish_times_article(u):
                continue
            hint = date_from_url_yyyy_mm_dd(u) or day
            if db_upsert_discovered(conn, "IrishTimes", u, hint):
                inserted += 1

        conn.commit()

    return inserted


def discover_irish_examiner(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date,
                            end: dt.date) -> int:
    index_urls = [
        "https://www.irishexaminer.com/sitemap-index/44-google_sitemap.xml",
        "https://www.irishexaminer.com/sitemap-index/227-google_channel_sitemap.xml",
    ]

    start_m = month_start(start)
    end_m = month_start(end)

    def month_in_range(y: int, m: int) -> bool:
        return start_m <= dt.date(y, m, 1) <= end_m

    sitemap_urls: List[str] = []
    for idx_url in index_urls:
        cache_path = cache_dir / "sitemaps" / f"ie_idx_{safe_sha1(idx_url)[:10]}.xml"
        xml_text = cached_get(session, idx_url, cache_path, is_xml=True)
        if xml_text:
            sitemap_urls.extend(xml_extract_locs(xml_text))

    selected: List[str] = []
    for sm in sitemap_urls:
        sm = normalise_url(sm)
        m = re.search(r"/sitemap/(\d+)-(\d{4})-(\d{1,2})-(\d{4})\.xml$", sm)
        if m and month_in_range(int(m.group(2)), int(m.group(3))):
            selected.append(sm)

    if not selected:
        selected = sitemap_urls[:200]

    inserted = 0
    seen: set = set()

    for sm_url in tqdm(selected, desc="Discover Irish Examiner", unit="sitemap"):
        cache_path = cache_dir / "sitemaps" / f"ie_sm_{safe_sha1(sm_url)[:12]}.xml"
        xml_text = cached_get(session, sm_url, cache_path, is_xml=True)
        if not xml_text:
            continue

        for loc, lastmod in xml_extract_url_entries(xml_text):
            u = normalise_url(loc)
            if u in seen:
                continue
            seen.add(u)
            if not looks_like_irish_examiner_article(u):
                continue

            hint = parse_lastmod_date(lastmod)
            if hint and not (start <= hint <= end):
                continue

            if db_upsert_discovered(conn, "IrishExaminer", u, hint):
                inserted += 1

        conn.commit()

    return inserted


def discover_irish_independent(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date,
                               end: dt.date) -> int:
    inserted = 0
    seen: set = set()

    for day in tqdm(list(date_range_inclusive(start, end)), desc="Discover Irish Independent", unit="day"):
        sm_candidates = [
            f"https://www.independent.ie/sitemap/{day.year}/{day.month}/{day.day}.xml",
            f"https://www.independent.ie/sitemap/{day.year}/{day.month:02d}/{day.day:02d}.xml",
        ]
        xml_text = None
        for sm_url in sm_candidates:
            cache_path = cache_dir / "sitemaps" / f"ii_{day.isoformat()}_{safe_sha1(sm_url)[:10]}.xml"
            xml_text = cached_get(session, sm_url, cache_path, is_xml=True)
            if xml_text:
                break

        if not xml_text:
            continue

        for loc in xml_extract_locs(xml_text):
            u = normalise_url(loc)
            if u in seen:
                continue
            seen.add(u)
            if not looks_like_irish_independent_article(u):
                continue
            hint = date_from_url_yyyy_mm_dd(u) or day
            if db_upsert_discovered(conn, "IrishIndependent", u, hint):
                inserted += 1

        conn.commit()

    return inserted


# =========================
# Article extraction - IDENTICAL to original
# =========================

def extract_article(html: str, url: str, source: str) -> Tuple[str, str, Optional[dt.date], bool]:
    """IDENTICAL extraction logic to original"""
    soup = soup_html(html)

    title = ""
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        title = str(og.get("content")).strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)

    published_dt: Optional[dt.datetime] = None
    meta_candidates = [
        soup.find("meta", attrs={"property": "article:published_time"}),
        soup.find("meta", attrs={"property": "og:published_time"}),
        soup.find("meta", attrs={"name": "parsely-pub-date"}),
        soup.find("meta", attrs={"name": "publish-date"}),
        soup.find("meta", attrs={"name": "date"}),
    ]
    for m in meta_candidates:
        if m and m.get("content"):
            published_dt = parse_datetime_maybe(str(m.get("content")))
            if published_dt:
                break

    if not published_dt:
        time_tag = soup.find("time")
        if time_tag:
            published_dt = parse_datetime_maybe(time_tag.get("datetime"))

    published_date = published_dt.date() if published_dt else date_from_url_yyyy_mm_dd(url)

    if source == "IrishTimes":
        container = (
                soup.find("div", attrs={"data-testid": "article-body"})
                or soup.find(attrs={"itemprop": "articleBody"})
                or soup.find("article")
        )
    else:
        container = (
                soup.find(attrs={"itemprop": "articleBody"})
                or soup.find("div", class_=re.compile(r"(article|story).*body", re.I))
                or soup.find("article")
        )

    texts: List[str] = []
    if container:
        for node in container.find_all(["p", "li"], recursive=True):
            if node.find_parent(["aside", "nav", "footer"]):
                continue
            t = node.get_text(" ", strip=True)
            if t and len(t) >= 20:
                texts.append(t)

    body = "\n".join(texts).strip()
    if not body and container:
        body = container.get_text("\n", strip=True)

    page_text = soup.get_text(" ", strip=True).lower()
    paywalled = bool(
        len(body) < 300 and any(k in page_text for k in ("subscribe", "subscription", "sign in", "log in", "already a subscriber", "premium")))

    return title, body, published_date, paywalled


# =========================
# Concurrent processing
# =========================

def process_single_article(url: str, source: str, session: requests.Session, cache_dir: Path,
                           start: dt.date, end: dt.date) -> dict:
    """Process a single article and return results dict"""
    result = {
        "url": url,
        "source": source,
        "status": "fetch_failed",
        "published_date": None,
        "title": "",
        "text_len": 0,
        "paywalled": False,
        "econ_hit": False,
        "unc_hit": False,
        "pol_hit": False,
        "match": False,
        "error": None,
    }

    cache_path = cache_dir / "articles" / f"{safe_sha1(url)}.html"
    html = cached_get(session, url, cache_path, is_xml=False)

    if not html:
        result["error"] = "fetch_failed_or_non_200"
        return result

    try:
        title, body, published_date, paywalled = extract_article(html, url, source)

        result["published_date"] = published_date
        result["title"] = title
        result["text_len"] = len(body)
        result["paywalled"] = paywalled

        if published_date and not (start <= published_date <= end):
            result["status"] = "out_of_range"
            return result

        econ_hit, unc_hit, pol_hit, match = evaluate_boolean_query(title, body)
        result["econ_hit"] = econ_hit
        result["unc_hit"] = unc_hit
        result["pol_hit"] = pol_hit
        result["match"] = match

        if paywalled and len(body) < 300:
            result["status"] = "paywalled_or_empty"
        elif len(body) == 0:
            result["status"] = "empty_text"
        else:
            result["status"] = "ok"

    except Exception as exc:
        result["error"] = f"parse_error: {exc}"
        result["status"] = "fetch_failed"

    return result


def fetch_and_process_concurrent(conn: sqlite3.Connection, session: requests.Session, cache_dir: Path,
                                 start: dt.date, end: dt.date, sources: Sequence[str]) -> int:
    """Process articles using concurrent requests"""
    pending = db_get_pending(conn, sources=sources, start=start, end=end, limit=PROCESS_LIMIT_PER_RUN)
    if not pending:
        LOGGER.info("No pending URLs for selected sources.")
        return 0

    LOGGER.info("Pending URLs for %s: %d", ",".join(sources), len(pending))

    processed = 0
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_article, str(row["url"]), str(row["source"]),
                            session, cache_dir, start, end): row
            for row in pending
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing articles", unit="article"):
            result = future.result()
            results.append(result)
            processed += 1

            # Batch update to database
            if processed % PROCESS_COMMIT_EVERY == 0:
                update_db_batch(conn, results)
                results = []

    # Final batch
    if results:
        update_db_batch(conn, results)

    return processed


def update_db_batch(conn: sqlite3.Connection, results: List[dict]) -> None:
    """Batch update database with results"""
    now = iso_now()
    for r in results:
        pd_str = r["published_date"].isoformat() if r["published_date"] else None
        m_str = month_start(r["published_date"]).isoformat() if r["published_date"] else None

        if r["status"] == "fetch_failed":
            conn.execute("""
                UPDATE articles
                SET status = 'fetch_failed',
                    attempts_fetch_fail = attempts_fetch_fail + 1,
                    last_error = ?,
                    updated_at = ?
                WHERE url = ?;
            """, (r["error"][:500] if r["error"] else None, now, r["url"]))
        else:
            conn.execute("""
                UPDATE articles
                SET source = ?,
                    published_date = COALESCE(?, published_date),
                    month = COALESCE(?, month),
                    title = ?,
                    text_len = ?,
                    paywalled = ?,
                    econ_hit = ?,
                    uncertainty_hit = ?,
                    policy_hit = ?,
                    match = ?,
                    status = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE url = ?;
            """, (
                r["source"],
                pd_str,
                m_str,
                r["title"],
                int(r["text_len"]),
                int(bool(r["paywalled"])),
                int(bool(r["econ_hit"])),
                int(bool(r["unc_hit"])),
                int(bool(r["pol_hit"])),
                int(bool(r["match"])),
                r["status"],
                r["error"],
                now,
                r["url"],
            ))
    conn.commit()


# =========================
# Export functions
# =========================

def export_results(conn: sqlite3.Connection, outdir: Path, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Export results to Excel and return summary DataFrame"""
    start_m = month_start(start).isoformat()
    end_m = month_start(end).isoformat()

    # Get monthly summary by source
    df_monthly = pd.read_sql_query("""
        SELECT source,
               month,
               COUNT(*) AS total_articles,
               SUM(COALESCE(match, 0)) AS matched_articles,
               SUM(COALESCE(paywalled, 0)) AS paywalled_articles
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
          AND status NOT IN ('out_of_range', 'discovered', 'fetch_failed')
        GROUP BY source, month
        ORDER BY month, source;
    """, conn, params=(start_m, end_m))

    # Combined monthly
    df_combined = df_monthly.groupby("month", as_index=False).agg(
        total_articles=("total_articles", "sum"),
        matched_articles=("matched_articles", "sum"),
        paywalled_articles=("paywalled_articles", "sum"),
    )
    df_combined["share"] = df_combined["matched_articles"] / df_combined["total_articles"]

    # Get all matched articles
    df_matches = pd.read_sql_query("""
        SELECT source, published_date, url, title, text_len, status
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
          AND COALESCE(match, 0) = 1
        ORDER BY published_date, source, url;
    """, conn, params=(start_m, end_m))

    # Status summary
    df_status = pd.read_sql_query("""
        SELECT source, status, COUNT(*) AS n
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
        GROUP BY source, status
        ORDER BY source, status;
    """, conn, params=(start_m, end_m))

    # Write to Excel
    excel_path = outdir / "fast_epu_results.xlsx"
    try:
        import openpyxl
        with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
            df_combined.to_excel(w, index=False, sheet_name="monthly_combined")
            df_monthly.to_excel(w, index=False, sheet_name="monthly_by_source")
            df_matches.to_excel(w, index=False, sheet_name="matched_articles")
            df_status.to_excel(w, index=False, sheet_name="status_summary")

            # Add run info
            run_info = pd.DataFrame({
                "item": ["date_range", "boolean_query", "sources"],
                "value": [f"{START_DATE} to {END_DATE}", BOOLEAN_QUERY_TEXT, ", ".join(RUN_SOURCES)],
            })
            run_info.to_excel(w, index=False, sheet_name="run_info")

        LOGGER.info("Wrote Excel to %s", excel_path)
    except ImportError:
        LOGGER.warning("openpyxl not available, skipping Excel export")

    # Also save CSV
    df_combined.to_csv(outdir / "monthly_combined.csv", index=False)
    df_matches.to_csv(outdir / "matched_articles.csv", index=False)

    return df_combined


# =========================
# Main
# =========================

def main() -> None:
    configure_logging(VERBOSE)

    start = parse_yyyy_mm_dd(START_DATE)
    end = parse_yyyy_mm_dd(END_DATE)

    outdir = Path(OUTDIR)
    cache_dir = outdir / "cache"
    ensure_dir(outdir)
    ensure_dir(outdir / "state")
    ensure_dir(cache_dir / "sitemaps")
    ensure_dir(cache_dir / "articles")

    db_path = outdir / "state" / STATE_DB_NAME
    conn = db_connect(db_path)
    db_init(conn)

    session = build_session()

    LOGGER.info("Fast EPU Scraper - %s to %s", START_DATE, END_DATE)
    LOGGER.info("Running sources: %s", ", ".join(RUN_SOURCES))
    LOGGER.info("Max concurrent workers: %d", MAX_WORKERS)

    # Discovery phase
    total_new = 0
    if "IrishTimes" in RUN_SOURCES:
        n = discover_irish_times(session, conn, cache_dir, start, end)
        LOGGER.info("Irish Times new URLs: %d", n)
        total_new += n

    if "IrishExaminer" in RUN_SOURCES:
        n = discover_irish_examiner(session, conn, cache_dir, start, end)
        LOGGER.info("Irish Examiner new URLs: %d", n)
        total_new += n

    if "IrishIndependent" in RUN_SOURCES:
        n = discover_irish_independent(session, conn, cache_dir, start, end)
        LOGGER.info("Irish Independent new URLs: %d", n)
        total_new += n

    LOGGER.info("Total new URLs discovered: %d", total_new)

    # Processing phase with concurrent requests
    processed = fetch_and_process_concurrent(conn, session, cache_dir, start, end, RUN_SOURCES)
    LOGGER.info("Processed URLs: %d", processed)

    # Export results
    df_summary = export_results(conn, outdir, start, end)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print("\nOutput files written to:", outdir.resolve())

    conn.close()


if __name__ == "__main__":
    main()
