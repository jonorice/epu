# -*- coding: utf-8 -*-
"""
Fast Irish EPU Scraper - January 2026
"""

import datetime as dt
import hashlib
import logging
import os
import re
import sqlite3
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

START_DATE = "2026-01-01"
END_DATE = "2026-01-31"

OUTDIR = "out_fast_epu_scrape"
STATE_DB_NAME = "fast_epu_state.sqlite"
RUN_SOURCES = ["IrishTimes", "IrishExaminer"]

USE_CACHE = True
TIMEOUT_SECONDS = 20
MAX_WORKERS = 10
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

PROCESS_LIMIT_PER_RUN: Optional[int] = None
MAX_ATTEMPTS_FETCH_FAIL = 2
PROCESS_COMMIT_EVERY = 100
VERBOSE = True

# =========================
# Boolean query terms - IDENTICAL to original
# =========================

GROUP_ECONOMY = ["economy", "economic"]
GROUP_UNCERTAINTY = ["uncertainty", "uncertain"]
GROUP_POLICY = [
    "regulation", "legislation", "deficit", "surplus",
    "House of the Oireachtas", "Oireachtas", "Seanad Éireann", "Seanad",
    "Dáil Éireann", "Dáil", "Government", "Áras an Uachtaráin",
    "Central Bank of Ireland", "Irish central bank", "Policy",
    "Taoiseach", "Tánaiste", "TD", "Teachta Dála", "President",
    "Local authority", "Local authorities", "council", "budget",
    "department of finance", "referendum", "referenda", "constitution",
    "constitutional amendment", "minister", "Fianna Fáil", "Fine Gael",
]

LOGGER = logging.getLogger("fast_epu_scraper")

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)

def parse_yyyy_mm_dd(x: str) -> dt.date:
    return dt.date.fromisoformat(x)

def month_start(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, 1)

def date_range_inclusive(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)

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
    return urlunparse(p2._replace(netloc=netloc))

def normalise_text(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", t.lower()).strip()

def compile_term_pattern(term: str) -> re.Pattern:
    term_n = normalise_text(term)
    parts = term_n.split()
    if not parts:
        return re.compile(r"(?!)")
    core = r"\s+".join(re.escape(p) for p in parts)
    return re.compile(r"(?<![A-Za-z0-9])" + core + r"(?![A-Za-z0-9])", flags=re.IGNORECASE)

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

def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IE,en;q=0.9",
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
    except requests.RequestException:
        return None
    text = r.text or ""
    if is_xml:
        if not (200 <= r.status_code < 300) and not any(x in text for x in ("<urlset", "<sitemapindex", "<?xml")):
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

def xml_extract_locs(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    return [el.text.strip() for el in root.iter() if isinstance(el.tag, str) and el.tag.endswith("loc") and el.text]

def xml_extract_url_entries(xml_text: str) -> List[Tuple[str, Optional[str]]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    entries = []
    for url_node in root.findall(".//{*}url"):
        loc, lastmod = None, None
        for child in list(url_node):
            if isinstance(child.tag, str):
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
    return ts.to_pydatetime().date() if not pd.isna(ts) else None

def parse_datetime_maybe(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    return ts.to_pydatetime() if not pd.isna(ts) else None

def date_from_url_yyyy_mm_dd(url: str) -> Optional[dt.date]:
    m = re.search(r"/((?:19|20)\d{2})/(\d{2})/(\d{2})/", url)
    if not m:
        return None
    try:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None

def iso_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def db_connect(db_path: Path) -> sqlite3.Connection:
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def db_init(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY, source TEXT NOT NULL, published_date TEXT, month TEXT,
            title TEXT, text_len INTEGER, paywalled INTEGER, econ_hit INTEGER,
            uncertainty_hit INTEGER, policy_hit INTEGER, match INTEGER,
            status TEXT NOT NULL DEFAULT 'discovered', attempts_fetch_fail INTEGER NOT NULL DEFAULT 0,
            last_error TEXT, discovered_at TEXT, updated_at TEXT
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_month ON articles(month);")
    conn.commit()

def db_upsert_discovered(conn: sqlite3.Connection, source: str, url: str, date_hint: Optional[dt.date]) -> bool:
    url = normalise_url(url)
    pd_str = date_hint.isoformat() if date_hint else None
    m_str = month_start(date_hint).isoformat() if date_hint else None
    now = iso_now()
    cur = conn.execute("""
        INSERT OR IGNORE INTO articles (url, source, published_date, month, status, attempts_fetch_fail, discovered_at, updated_at)
        VALUES (?, ?, ?, ?, 'discovered', 0, ?, ?);
    """, (url, source, pd_str, m_str, now, now))
    return cur.rowcount == 1

def db_get_pending(conn: sqlite3.Connection, sources: Sequence[str], start: dt.date, end: dt.date, limit: Optional[int]) -> List[sqlite3.Row]:
    start_m, end_m = month_start(start).isoformat(), month_start(end).isoformat()
    placeholders = ",".join(["?"] * len(sources))
    params = list(sources) + [start_m, end_m, MAX_ATTEMPTS_FETCH_FAIL]
    sql = f"""SELECT url, source FROM articles WHERE source IN ({placeholders}) AND month >= ? AND month <= ?
              AND (status = 'discovered' OR (status = 'fetch_failed' AND attempts_fetch_fail < ?))"""
    if limit:
        sql += f" LIMIT {limit}"
    return list(conn.execute(sql, params).fetchall())

IT_EXCLUDES = ("/video/", "/podcast", "/sponsored/", "/advertising-feature", "/photography/", "/weather/", "/crossword", "/games/", "/horoscopes", "/newsletter", "/subscribe/", "/myaccount/")
IE_EXCLUDES = ("/sponsored/", "/sponsored-content/")

def looks_like_irish_times_article(url: str) -> bool:
    p = urlparse(url)
    if not p.netloc.endswith("irishtimes.com"):
        return False
    path = p.path.lower().rstrip("/")
    if any(bad in path for bad in IT_EXCLUDES) or re.search(r"\.(jpg|png|gif|pdf|mp4)$", path):
        return False
    return bool(re.search(r"/(19|20)\d{2}/\d{2}/\d{2}/", path) or re.search(r"-1\.\d+$", path))

def looks_like_irish_examiner_article(url: str) -> bool:
    p = urlparse(url)
    if not p.netloc.endswith("irishexaminer.com"):
        return False
    return bool(re.search(r"/arid-\d+", p.path.lower())) and not any(bad in p.path.lower() for bad in IE_EXCLUDES)

def discover_irish_times(session, conn, cache_dir, start, end):
    templates = ["https://www.irishtimes.com/arc/outboundfeeds/sitemap3/{date}/", "https://www.irishtimes.com/arc/outboundfeeds/sitemap2/{date}/"]
    inserted, seen = 0, set()
    for day in tqdm(list(date_range_inclusive(start, end)), desc="Discover Irish Times"):
        for tmpl in templates:
            sm_url = tmpl.format(date=day.isoformat())
            xml_text = cached_get(session, sm_url, cache_dir / "sitemaps" / f"it_{day.isoformat()}.xml", True)
            if xml_text:
                for loc in xml_extract_locs(xml_text):
                    u = normalise_url(loc)
                    if u not in seen and looks_like_irish_times_article(u):
                        seen.add(u)
                        if db_upsert_discovered(conn, "IrishTimes", u, date_from_url_yyyy_mm_dd(u) or day):
                            inserted += 1
                break
        conn.commit()
    return inserted

def discover_irish_examiner(session, conn, cache_dir, start, end):
    index_urls = ["https://www.irishexaminer.com/sitemap-index/44-google_sitemap.xml", "https://www.irishexaminer.com/sitemap-index/227-google_channel_sitemap.xml"]
    start_m, end_m = month_start(start), month_start(end)
    sitemap_urls = []
    for idx_url in index_urls:
        xml_text = cached_get(session, idx_url, cache_dir / "sitemaps" / f"ie_idx_{safe_sha1(idx_url)[:10]}.xml", True)
        if xml_text:
            sitemap_urls.extend(xml_extract_locs(xml_text))
    selected = [sm for sm in sitemap_urls if (m := re.search(r"/sitemap/(\d+)-(\d{4})-(\d{1,2})-", sm)) and start_m <= dt.date(int(m.group(2)), int(m.group(3)), 1) <= end_m]
    if not selected:
        selected = sitemap_urls[:200]
    inserted, seen = 0, set()
    for sm_url in tqdm(selected, desc="Discover Irish Examiner"):
        xml_text = cached_get(session, sm_url, cache_dir / "sitemaps" / f"ie_sm_{safe_sha1(sm_url)[:12]}.xml", True)
        if xml_text:
            for loc, lastmod in xml_extract_url_entries(xml_text):
                u = normalise_url(loc)
                hint = parse_lastmod_date(lastmod)
                if u not in seen and looks_like_irish_examiner_article(u) and (not hint or start <= hint <= end):
                    seen.add(u)
                    if db_upsert_discovered(conn, "IrishExaminer", u, hint):
                        inserted += 1
        conn.commit()
    return inserted

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
    published_dt = None
    for m in [soup.find("meta", attrs={"property": "article:published_time"}), soup.find("meta", attrs={"name": "parsely-pub-date"})]:
        if m and m.get("content"):
            published_dt = parse_datetime_maybe(str(m.get("content")))
            if published_dt:
                break
    published_date = published_dt.date() if published_dt else date_from_url_yyyy_mm_dd(url)
    container = soup.find("div", attrs={"data-testid": "article-body"}) or soup.find(attrs={"itemprop": "articleBody"}) or soup.find("article")
    texts = []
    if container:
        for node in container.find_all(["p", "li"], recursive=True):
            if not node.find_parent(["aside", "nav", "footer"]):
                t = node.get_text(" ", strip=True)
                if t and len(t) >= 20:
                    texts.append(t)
    body = "\n".join(texts).strip() or (container.get_text("\n", strip=True) if container else "")
    page_text = soup.get_text(" ", strip=True).lower()
    paywalled = len(body) < 300 and any(k in page_text for k in ("subscribe", "subscription", "sign in", "already a subscriber"))
    return title, body, published_date, paywalled

def process_single_article(url, source, session, cache_dir, start, end):
    result = {"url": url, "source": source, "status": "fetch_failed", "published_date": None, "title": "", "text_len": 0, "paywalled": False, "econ_hit": False, "unc_hit": False, "pol_hit": False, "match": False, "error": None}
    html = cached_get(session, url, cache_dir / "articles" / f"{safe_sha1(url)}.html", False)
    if not html:
        result["error"] = "fetch_failed"
        return result
    try:
        title, body, published_date, paywalled = extract_article(html, url, source)
        result.update({"published_date": published_date, "title": title, "text_len": len(body), "paywalled": paywalled})
        if published_date and not (start <= published_date <= end):
            result["status"] = "out_of_range"
            return result
        econ, unc, pol, match = evaluate_boolean_query(title, body)
        result.update({"econ_hit": econ, "unc_hit": unc, "pol_hit": pol, "match": match})
        result["status"] = "paywalled_or_empty" if paywalled and len(body) < 300 else ("empty_text" if not body else "ok")
    except Exception as e:
        result["error"] = str(e)
    return result

def update_db_batch(conn, results):
    now = iso_now()
    for r in results:
        pd_str = r["published_date"].isoformat() if r["published_date"] else None
        m_str = month_start(r["published_date"]).isoformat() if r["published_date"] else None
        if r["status"] == "fetch_failed":
            conn.execute("UPDATE articles SET status='fetch_failed', attempts_fetch_fail=attempts_fetch_fail+1, last_error=?, updated_at=? WHERE url=?", (r.get("error"), now, r["url"]))
        else:
            conn.execute("""UPDATE articles SET published_date=COALESCE(?,published_date), month=COALESCE(?,month), title=?, text_len=?, paywalled=?, econ_hit=?, uncertainty_hit=?, policy_hit=?, match=?, status=?, updated_at=? WHERE url=?""",
                (pd_str, m_str, r["title"], r["text_len"], int(r["paywalled"]), int(r["econ_hit"]), int(r["unc_hit"]), int(r["pol_hit"]), int(r["match"]), r["status"], now, r["url"]))
    conn.commit()

def fetch_and_process(conn, session, cache_dir, start, end, sources):
    pending = db_get_pending(conn, sources, start, end, PROCESS_LIMIT_PER_RUN)
    if not pending:
        return 0
    LOGGER.info("Pending: %d", len(pending))
    results, processed = [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_single_article, str(r["url"]), str(r["source"]), session, cache_dir, start, end): r for r in pending}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            results.append(f.result())
            processed += 1
            if processed % PROCESS_COMMIT_EVERY == 0:
                update_db_batch(conn, results)
                results = []
    if results:
        update_db_batch(conn, results)
    return processed

def main():
    configure_logging(VERBOSE)
    start, end = parse_yyyy_mm_dd(START_DATE), parse_yyyy_mm_dd(END_DATE)
    outdir, cache_dir = Path(OUTDIR), Path(OUTDIR) / "cache"
    ensure_dir(outdir / "state")
    ensure_dir(cache_dir / "sitemaps")
    ensure_dir(cache_dir / "articles")
    conn = db_connect(outdir / "state" / STATE_DB_NAME)
    db_init(conn)
    session = build_session()
    LOGGER.info("Scraping January 2026")
    for src, fn in [("IrishTimes", discover_irish_times), ("IrishExaminer", discover_irish_examiner)]:
        if src in RUN_SOURCES:
            n = fn(session, conn, cache_dir, start, end)
            LOGGER.info("%s: %d new URLs", src, n)
    fetch_and_process(conn, session, cache_dir, start, end, RUN_SOURCES)
    cur = conn.execute("SELECT COUNT(*) as total, SUM(match) as matched FROM articles WHERE month='2026-01-01' AND status NOT IN ('discovered','fetch_failed','out_of_range')")
    row = cur.fetchone()
    print(f"\nJanuary 2026: {row[1]} matched / {row[0]} total ({row[1]/row[0]*100:.2f}%)")
    conn.close()

if __name__ == "__main__":
    main()
