# -*- coding: utf-8 -*-
"""
Irish EPU, October 2025 only, staged by paper, restart safe, exports Excel after each run.

How to use
1) Run Irish Times first
   RUN_SOURCES = ["IrishTimes"]
   Then run the script. It writes OUTDIR/irish_epu_2025-10.xlsx

2) Run Irish Examiner next
   Change RUN_SOURCES = ["IrishExaminer"]
   Run again. It reuses the same SQLite state DB and updates the same Excel workbook.

3) Run Irish Independent last
   Change RUN_SOURCES = ["IrishIndependent"]
   Run again.

Restart safety
- Progress is stored in OUTDIR/state/irish_epu_state.sqlite
- If you interrupt or your laptop restarts, rerun the script and it continues.
- It commits progress every PROCESS_COMMIT_EVERY articles.

Excel output
- Requires openpyxl. If it is missing, the script will still save a CSV and tell you how to install openpyxl.
"""

import datetime as dt
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# tqdm is optional
try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:
    def tqdm(x, **kwargs):  # type: ignore
        return x


# =========================
# Configuration
# =========================

# October 2025 only
START_DATE = "2025-12-01"
END_DATE = "2025-12-31"

# Strongly recommended to use a dedicated folder for this October-only run
# You can make this absolute to avoid Spyder working directory confusion
# Example:
    
# OUTDIR = r"C:\Users\jonor\Documents\out_irish_epu_oct2025"
OUTDIR = "out_irish_epu_oct2025"
STATE_DB_NAME = "irish_epu_state.sqlite"

# Run one paper at a time (edit between runs)
#RUN_SOURCES = ["IrishTimes"]
RUN_SOURCES = ["IrishExaminer"]
#RUN_SOURCES = ["IrishIndependent"]

# Network behaviour (tuned for speed)
USE_CACHE = True
TIMEOUT_SECONDS = 25
SLEEP_MIN_SECONDS = 0.0
SLEEP_MAX_SECONDS = 0.05
USER_AGENT = "Mozilla/5.0 (compatible; IrishEPUResearch/1.0)"

# Proxy placeholders (only used if you set PROXY_URL)
USERNAME = "%username%"
PASSWORD = "%password%"
PROXY_URL: Optional[str] = None
USE_ENV_PROXIES = True

# TLS and CA bundles
VERIFY_SSL = True
CA_BUNDLE_PATH: Optional[str] = None

# Cookies JSON (browser export), only if paywalled
IRISH_TIMES_COOKIES_JSON: Optional[str] = None
IRISH_EXAMINER_COOKIES_JSON: Optional[str] = None
IRISH_INDEPENDENT_COOKIES_JSON: Optional[str] = None

# Irish Independent permission flag
I_HAVE_PERMISSION_FOR_IRISH_INDEPENDENT = True

# Independent discovery mode (fastest first)
IRISH_INDEPENDENT_DISCOVERY_MODE = "sitemap"  # "sitemap" | "robots"

# Restart and batching
PROCESS_LIMIT_PER_RUN: Optional[int] = None   # set e.g. 500 for a quick preview
MAX_ATTEMPTS_FETCH_FAIL = 2
PROCESS_COMMIT_EVERY = 50
RETRY_PAYWALLED_OR_EMPTY = False

# Outputs
EXPORT_CSV_TOO = True
EXCEL_FILENAME = "irish_epu_2025-10.xlsx"

VERBOSE = True


# =========================
# Boolean query terms
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

LOGGER = logging.getLogger("irish_epu_oct2025")


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
# Helpers
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


def polite_sleep() -> None:
    if SLEEP_MAX_SECONDS <= 0:
        return
    time.sleep(random.uniform(max(0.0, SLEEP_MIN_SECONDS), max(SLEEP_MIN_SECONDS, SLEEP_MAX_SECONDS)))


def normalise_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compile_term_pattern(term: str) -> re.Pattern:
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
# Requests sessions, proxies, cookies, caching
# =========================

def resolved_proxy_url() -> Optional[str]:
    if not PROXY_URL:
        return None
    return PROXY_URL.replace("%username%", USERNAME).replace("%password%", PASSWORD)


def resolve_verify_setting() -> object:
    if CA_BUNDLE_PATH:
        return CA_BUNDLE_PATH
    env_ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if env_ca and Path(env_ca).exists():
        return env_ca
    return bool(VERIFY_SSL)


def load_cookies_from_json(session: requests.Session, cookies_json_path: Path) -> None:
    data = json.loads(cookies_json_path.read_text(encoding="utf-8"))
    cookies = data["cookies"] if isinstance(data, dict) and "cookies" in data else data
    if not isinstance(cookies, list):
        raise ValueError("Cookies JSON should be a list of cookies or a dict with a 'cookies' list.")
    for c in cookies:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        value = c.get("value")
        domain = c.get("domain")
        path = c.get("path", "/")
        if not name or value is None:
            continue
        if isinstance(domain, str) and domain.startswith("."):
            domain = domain[1:]
        session.cookies.set(name, value, domain=domain, path=path)


def build_session(cookies_json: Optional[str]) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IE,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    s.verify = resolve_verify_setting()
    s.trust_env = bool(USE_ENV_PROXIES)
    purl = resolved_proxy_url()
    if purl:
        s.proxies.update({"http": purl, "https": purl})
        s.trust_env = False
    if cookies_json:
        load_cookies_from_json(s, Path(cookies_json))
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
        LOGGER.warning("Request failed %s (%s)", url, exc)
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
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
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
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_source_month ON articles(source, month);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_month ON articles(month);")
    conn.commit()


def db_upsert_discovered(conn: sqlite3.Connection, source: str, url: str, date_hint: Optional[dt.date]) -> bool:
    url = normalise_url(url)
    pd_str = date_hint.isoformat() if date_hint else None
    m_str = month_start(date_hint).isoformat() if date_hint else None
    now = iso_now()

    cur = conn.execute(
        """
        INSERT OR IGNORE INTO articles
        (url, source, published_date, month, status, attempts_fetch_fail, last_error, discovered_at, updated_at)
        VALUES (?, ?, ?, ?, 'discovered', 0, NULL, ?, ?);
        """,
        (url, source, pd_str, m_str, now, now),
    )
    inserted = cur.rowcount == 1

    if (not inserted) and date_hint:
        conn.execute(
            """
            UPDATE articles
            SET published_date = COALESCE(published_date, ?),
                month = COALESCE(month, ?),
                updated_at = ?
            WHERE url = ?;
            """,
            (pd_str, m_str, now, url),
        )

    return inserted


def db_mark_attempt_fetch_failed(conn: sqlite3.Connection, url: str, source: str, error: str) -> None:
    now = iso_now()
    conn.execute(
        """
        UPDATE articles
        SET source = ?,
            status = 'fetch_failed',
            attempts_fetch_fail = attempts_fetch_fail + 1,
            last_error = ?,
            updated_at = ?
        WHERE url = ?;
        """,
        (source, error[:500], now, url),
    )


def db_mark_processed(
    conn: sqlite3.Connection,
    url: str,
    *,
    source: str,
    published_date: Optional[dt.date],
    title: str,
    text_len: int,
    paywalled: bool,
    econ_hit: bool,
    unc_hit: bool,
    pol_hit: bool,
    match: bool,
    status: str,
    error: Optional[str],
) -> None:
    now = iso_now()
    pd_str = published_date.isoformat() if published_date else None
    m_str = month_start(published_date).isoformat() if published_date else None

    conn.execute(
        """
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
        """,
        (
            source,
            pd_str,
            m_str,
            title,
            int(text_len),
            int(bool(paywalled)),
            int(bool(econ_hit)),
            int(bool(unc_hit)),
            int(bool(pol_hit)),
            int(bool(match)),
            status,
            error,
            now,
            url,
        ),
    )


def db_get_pending(
    conn: sqlite3.Connection,
    *,
    sources: Sequence[str],
    start: dt.date,
    end: dt.date,
    limit: Optional[int],
) -> List[sqlite3.Row]:
    if not sources:
        return []

    start_m = month_start(start).isoformat()
    end_m = month_start(end).isoformat()

    placeholders = ",".join(["?"] * len(sources))
    params: List[object] = list(sources)

    where_status = "(status = 'discovered' OR (status = 'fetch_failed' AND attempts_fetch_fail < ?))"
    params_status: List[object] = [int(MAX_ATTEMPTS_FETCH_FAIL)]
    if RETRY_PAYWALLED_OR_EMPTY:
        where_status = (
            "(status = 'discovered' OR "
            "(status = 'fetch_failed' AND attempts_fetch_fail < ?) OR "
            "(status IN ('paywalled_or_empty', 'empty_text')))"
        )
        params_status = [int(MAX_ATTEMPTS_FETCH_FAIL)]

    sql = f"""
        SELECT url, source
        FROM articles
        WHERE source IN ({placeholders})
          AND month IS NOT NULL
          AND month >= ?
          AND month <= ?
          AND {where_status}
        ORDER BY COALESCE(published_date, '9999-12-31'), url
    """

    params = params + [start_m, end_m] + params_status

    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    return list(conn.execute(sql, params).fetchall())


# =========================
# URL filters and discovery
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


def discover_irish_times(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date, end: dt.date) -> int:
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
            polite_sleep()
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


def discover_irish_examiner(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date, end: dt.date) -> int:
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
        polite_sleep()
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
        polite_sleep()
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


def parse_robots_sitemaps(robots_text: str) -> List[str]:
    out: List[str] = []
    for line in robots_text.splitlines():
        if line.lower().startswith("sitemap:"):
            u = line.split(":", 1)[1].strip()
            if u:
                out.append(u)
    return out


def discover_irish_independent_from_daily_sitemaps(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date, end: dt.date) -> int:
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
            polite_sleep()
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


def discover_irish_independent_from_robots(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date, end: dt.date) -> int:
    robots_url = "https://www.independent.ie/robots.txt"
    cache_path = cache_dir / "robots" / f"ii_robots_{safe_sha1(robots_url)[:10]}.txt"
    txt = cached_get(session, robots_url, cache_path, is_xml=False)
    polite_sleep()
    if not txt:
        return 0

    sitemap_urls = parse_robots_sitemaps(txt)
    if not sitemap_urls:
        return 0

    inserted = 0
    seen: set = set()

    for sm_url in tqdm(sitemap_urls[:50], desc="Independent robots sitemaps", unit="sitemap"):
        sm_url = normalise_url(sm_url)
        cache_path = cache_dir / "sitemaps" / f"ii_sm_{safe_sha1(sm_url)[:12]}.xml"
        xml_text = cached_get(session, sm_url, cache_path, is_xml=True)
        polite_sleep()
        if not xml_text:
            continue

        if "<sitemapindex" in xml_text:
            children = xml_extract_locs(xml_text)
            for child in children[:200]:
                child = normalise_url(child)
                cpath = cache_dir / "sitemaps" / f"ii_child_{safe_sha1(child)[:12]}.xml"
                child_xml = cached_get(session, child, cpath, is_xml=True)
                polite_sleep()
                if not child_xml:
                    continue
                for loc, lastmod in xml_extract_url_entries(child_xml):
                    u = normalise_url(loc)
                    if u in seen:
                        continue
                    seen.add(u)
                    if not looks_like_irish_independent_article(u):
                        continue
                    hint = parse_lastmod_date(lastmod) or date_from_url_yyyy_mm_dd(u)
                    if hint and not (start <= hint <= end):
                        continue
                    if db_upsert_discovered(conn, "IrishIndependent", u, hint):
                        inserted += 1
                conn.commit()
        else:
            for loc, lastmod in xml_extract_url_entries(xml_text):
                u = normalise_url(loc)
                if u in seen:
                    continue
                seen.add(u)
                if not looks_like_irish_independent_article(u):
                    continue
                hint = parse_lastmod_date(lastmod) or date_from_url_yyyy_mm_dd(u)
                if hint and not (start <= hint <= end):
                    continue
                if db_upsert_discovered(conn, "IrishIndependent", u, hint):
                    inserted += 1
            conn.commit()

    return inserted


def discover_irish_independent(session: requests.Session, conn: sqlite3.Connection, cache_dir: Path, start: dt.date, end: dt.date) -> int:
    mode = (IRISH_INDEPENDENT_DISCOVERY_MODE or "sitemap").lower().strip()
    if mode == "robots":
        return discover_irish_independent_from_robots(session, conn, cache_dir, start, end)
    return discover_irish_independent_from_daily_sitemaps(session, conn, cache_dir, start, end)


# =========================
# Processing
# =========================

def extract_article(html: str, url: str, source: str) -> Tuple[str, str, Optional[dt.date], bool]:
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
    paywalled = bool(len(body) < 300 and any(k in page_text for k in ("subscribe", "subscription", "sign in", "log in", "already a subscriber", "premium")))

    return title, body, published_date, paywalled


def fetch_and_process_pending(
    conn: sqlite3.Connection,
    sessions: dict,
    cache_dir: Path,
    start: dt.date,
    end: dt.date,
    sources: Sequence[str],
) -> int:
    pending = db_get_pending(conn, sources=sources, start=start, end=end, limit=PROCESS_LIMIT_PER_RUN)
    if not pending:
        LOGGER.info("No pending URLs for selected sources.")
        return 0

    LOGGER.info("Pending URLs for %s: %d", ",".join(sources), len(pending))

    processed = 0
    for row in tqdm(pending, desc="Fetch and classify", unit="article"):
        url = str(row["url"])
        source = str(row["source"])
        session = sessions.get(source)
        if session is None:
            continue

        cache_path = cache_dir / "articles" / f"{safe_sha1(url)}.html"
        html = cached_get(session, url, cache_path, is_xml=False)
        polite_sleep()

        if not html:
            db_mark_attempt_fetch_failed(conn, url, source, "fetch_failed_or_non_200")
        else:
            try:
                title, body, published_date, paywalled = extract_article(html, url, source)

                if published_date and not (start <= published_date <= end):
                    db_mark_processed(
                        conn,
                        url,
                        source=source,
                        published_date=published_date,
                        title=title,
                        text_len=len(body),
                        paywalled=paywalled,
                        econ_hit=False,
                        unc_hit=False,
                        pol_hit=False,
                        match=False,
                        status="out_of_range",
                        error=None,
                    )
                else:
                    econ_hit, unc_hit, pol_hit, match = evaluate_boolean_query(title, body)
                    status = "ok"
                    if paywalled and len(body) < 300:
                        status = "paywalled_or_empty"
                    elif len(body) == 0:
                        status = "empty_text"

                    db_mark_processed(
                        conn,
                        url,
                        source=source,
                        published_date=published_date,
                        title=title,
                        text_len=len(body),
                        paywalled=paywalled,
                        econ_hit=econ_hit,
                        unc_hit=unc_hit,
                        pol_hit=pol_hit,
                        match=match,
                        status=status,
                        error=None,
                    )
            except Exception as exc:
                db_mark_attempt_fetch_failed(conn, url, source, f"parse_error: {exc}")

        processed += 1
        if processed % int(PROCESS_COMMIT_EVERY) == 0:
            conn.commit()

    conn.commit()
    return processed


# =========================
# Exports
# =========================

def export_articles_csv(conn: sqlite3.Connection, out_csv: Path, start: dt.date, end: dt.date) -> None:
    start_m = month_start(start).isoformat()
    end_m = month_start(end).isoformat()
    df = pd.read_sql_query(
        """
        SELECT source, url, published_date, month, title, text_len, paywalled,
               econ_hit, uncertainty_hit, policy_hit, match, status,
               attempts_fetch_fail, last_error, discovered_at, updated_at
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?;
        """,
        conn,
        params=(start_m, end_m),
    )
    df.to_csv(out_csv, index=False)


def export_excel_snapshot(conn: sqlite3.Connection, out_xlsx: Path, start: dt.date, end: dt.date, run_sources: Sequence[str]) -> None:
    start_m = month_start(start).isoformat()
    end_m = month_start(end).isoformat()

    df_status = pd.read_sql_query(
        """
        SELECT source, status, COUNT(*) AS n
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
        GROUP BY source, status
        ORDER BY source, status;
        """,
        conn,
        params=(start_m, end_m),
    )

    df_pending = pd.read_sql_query(
        """
        SELECT
          source,
          SUM(CASE WHEN status='discovered' THEN 1 ELSE 0 END) AS pending_discovered,
          SUM(CASE WHEN status='fetch_failed' AND attempts_fetch_fail < ? THEN 1 ELSE 0 END) AS pending_retry_fetch_failed,
          SUM(CASE WHEN status IN ('paywalled_or_empty','empty_text') THEN 1 ELSE 0 END) AS paywalled_or_empty,
          SUM(CASE WHEN status='fetch_failed' AND attempts_fetch_fail >= ? THEN 1 ELSE 0 END) AS fetch_failed_exhausted,
          COUNT(*) AS total_urls
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
        GROUP BY source
        ORDER BY total_urls DESC;
        """,
        conn,
        params=(int(MAX_ATTEMPTS_FETCH_FAIL), int(MAX_ATTEMPTS_FETCH_FAIL), start_m, end_m),
    )
    if not df_pending.empty:
        df_pending["pending_total"] = df_pending["pending_discovered"] + df_pending["pending_retry_fetch_failed"]

    df_monthly_by_source = pd.read_sql_query(
        """
        SELECT source,
               month,
               COUNT(*) AS total_articles,
               SUM(COALESCE(match, 0)) AS matched_articles,
               SUM(COALESCE(paywalled, 0)) AS paywalled_articles
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
          AND status != 'out_of_range'
        GROUP BY source, month
        ORDER BY source, month;
        """,
        conn,
        params=(start_m, end_m),
    )

    df_monthly = pd.DataFrame()
    if not df_monthly_by_source.empty:
        df_monthly_by_source["share"] = df_monthly_by_source["matched_articles"] / df_monthly_by_source["total_articles"]
        df_monthly_by_source["paywall_share"] = df_monthly_by_source["paywalled_articles"] / df_monthly_by_source["total_articles"]
        df_monthly = df_monthly_by_source.groupby("month", as_index=False).agg(
            total_articles=("total_articles", "sum"),
            matched_articles=("matched_articles", "sum"),
            paywalled_articles=("paywalled_articles", "sum"),
        )
        df_monthly["share"] = df_monthly["matched_articles"] / df_monthly["total_articles"]
        df_monthly["paywall_share"] = df_monthly["paywalled_articles"] / df_monthly["total_articles"]

    df_matches = pd.read_sql_query(
        """
        SELECT source, published_date, url, title, paywalled, text_len, status
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
          AND COALESCE(match, 0) = 1
        ORDER BY published_date, source, url;
        """,
        conn,
        params=(start_m, end_m),
    )

    df_errors = pd.read_sql_query(
        """
        SELECT COALESCE(last_error, '(none)') AS last_error, COUNT(*) AS n
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
          AND status='fetch_failed'
        GROUP BY COALESCE(last_error, '(none)')
        ORDER BY n DESC
        LIMIT 25;
        """,
        conn,
        params=(start_m, end_m),
    )

    df_articles_small = pd.read_sql_query(
        """
        SELECT source, published_date, url, title, status, match, paywalled, text_len, attempts_fetch_fail
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
        ORDER BY published_date, source, url;
        """,
        conn,
        params=(start_m, end_m),
    )

    try:
        import openpyxl  # noqa: F401
    except Exception:
        LOGGER.error("openpyxl is not installed, cannot write Excel. Install with: pip install openpyxl")
        return

    run_info = pd.DataFrame(
        {
            "item": ["date_range", "run_sources_this_time", "boolean_query"],
            "value": [f"{START_DATE} to {END_DATE}", ", ".join(run_sources), BOOLEAN_QUERY_TEXT],
        }
    )

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        run_info.to_excel(w, index=False, sheet_name="run_info")
        df_status.to_excel(w, index=False, sheet_name="status_by_source")
        df_pending.to_excel(w, index=False, sheet_name="pending_by_source")
        df_monthly.to_excel(w, index=False, sheet_name="monthly_combined")
        df_monthly_by_source.to_excel(w, index=False, sheet_name="monthly_by_source")
        df_matches.to_excel(w, index=False, sheet_name="matches")
        df_errors.to_excel(w, index=False, sheet_name="top_errors")
        df_articles_small.to_excel(w, index=False, sheet_name="articles")

    LOGGER.info("Wrote Excel snapshot %s", str(out_xlsx))


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
    ensure_dir(cache_dir / "robots")

    db_path = outdir / "state" / STATE_DB_NAME
    conn = db_connect(db_path)
    db_init(conn)

    # Sessions only for sources you are running now
    sessions: dict = {}

    if "IrishTimes" in RUN_SOURCES:
        sessions["IrishTimes"] = build_session(IRISH_TIMES_COOKIES_JSON)
    if "IrishExaminer" in RUN_SOURCES:
        sessions["IrishExaminer"] = build_session(IRISH_EXAMINER_COOKIES_JSON)
    if "IrishIndependent" in RUN_SOURCES:
        if not I_HAVE_PERMISSION_FOR_IRISH_INDEPENDENT:
            raise RuntimeError("Irish Independent selected but I_HAVE_PERMISSION_FOR_IRISH_INDEPENDENT is False.")
        sessions["IrishIndependent"] = build_session(IRISH_INDEPENDENT_COOKIES_JSON)

    LOGGER.info("Running October 2025 only, %s to %s", START_DATE, END_DATE)
    LOGGER.info("Running sources this time: %s", ", ".join(RUN_SOURCES))

    # Discovery for selected sources only
    total_new = 0
    if "IrishTimes" in RUN_SOURCES:
        n = discover_irish_times(sessions["IrishTimes"], conn, cache_dir, start, end)
        LOGGER.info("Irish Times new URLs inserted %d", n)
        total_new += n

    if "IrishExaminer" in RUN_SOURCES:
        n = discover_irish_examiner(sessions["IrishExaminer"], conn, cache_dir, start, end)
        LOGGER.info("Irish Examiner new URLs inserted %d", n)
        total_new += n

    if "IrishIndependent" in RUN_SOURCES:
        n = discover_irish_independent(sessions["IrishIndependent"], conn, cache_dir, start, end)
        LOGGER.info("Irish Independent new URLs inserted %d", n)
        total_new += n

    LOGGER.info("Total new URLs inserted this run %d", total_new)

    # Process only selected sources
    processed = fetch_and_process_pending(conn, sessions, cache_dir, start, end, RUN_SOURCES)
    LOGGER.info("Processed URLs this run %d", processed)

    # Export
    excel_path = outdir / EXCEL_FILENAME
    export_excel_snapshot(conn, excel_path, start, end, RUN_SOURCES)

    if EXPORT_CSV_TOO:
        export_articles_csv(conn, outdir / "irish_epu_oct2025_articles.csv", start, end)

    print("\nExcel written to")
    print(str(excel_path.resolve()))

    # Quick pending summary
    start_m = month_start(start).isoformat()
    end_m = month_start(end).isoformat()
    df_pending = pd.read_sql_query(
        """
        SELECT
          source,
          SUM(CASE WHEN status='discovered' THEN 1 ELSE 0 END) AS pending_discovered,
          SUM(CASE WHEN status='fetch_failed' AND attempts_fetch_fail < ? THEN 1 ELSE 0 END) AS pending_retry_fetch_failed,
          COUNT(*) AS total_urls
        FROM articles
        WHERE month IS NOT NULL
          AND month >= ?
          AND month <= ?
        GROUP BY source
        ORDER BY total_urls DESC;
        """,
        conn,
        params=(int(MAX_ATTEMPTS_FETCH_FAIL), start_m, end_m),
    )
    print("\nPending by source in October 2025")
    print(df_pending.to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
