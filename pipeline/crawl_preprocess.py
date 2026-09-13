"""
PHASE 1 & 2: CRAWL + PREPROCESS
- Crawl from the VBPL API (with checkpoint resume)
- Filter data (valid effStatus)
- Merge cleaned data + index → processed_data.json ready for use
"""

import requests
import json
import math
import sys
import time
import os
from datetime import datetime
from slugify import slugify
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import get_settings


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

_settings = get_settings()
_crawl_settings = _settings.crawl

DATA_DIR = _settings.data.data_dir
os.makedirs(DATA_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.json")
CRAWL_DATA_FILE = os.path.join(DATA_DIR, "data.jsonl")
CRAWL_INDEX_FILE = os.path.join(DATA_DIR, "index_data.jsonl")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_data.json")
ERROR_ITEMS_FILE = os.path.join(DATA_DIR, "error_items.json")


MAX_PAGE_RETRIES = 5
MAX_DOC_RETRIES = 3

PAGE_SIZE = _crawl_settings.page_size
MAX_DOCS = _crawl_settings.max_docs


def _issue_date_from(value: str | None) -> str | None:
    """CRAWL_DATE_FROM: 'YYYY-MM-DD' -> start of day. Already has 'T' -> keep as-is."""
    if not value:
        return None
    return value if "T" in value else f"{value}T00:00:00"


def _issue_date_to(value: str | None) -> str | None:
    """CRAWL_DATE_TO: 'YYYY-MM-DD' -> end of day (inclusive)."""
    if not value:
        return None
    return value if "T" in value else f"{value}T23:59:59"


ISSUE_DATE_FROM = _issue_date_from(_crawl_settings.date_from)
ISSUE_DATE_TO = _issue_date_to(_crawl_settings.date_to)


# ─────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    """Session with automatic retry.

    vbpl.vn sits behind a WAF (Wangsu) that blocks any request missing a
    browser User-Agent (403 "Ws-Action: bot"), even the very first request —
    unrelated to rate-limiting or a JS challenge. requests defaults to a
    "python-requests/x.x" UA, so set a browser-like UA here for the whole session.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    return session


# ─────────────────────────────────────────────────────────────
# Headers
# ─────────────────────────────────────────────────────────────

def get_index_headers(doc_id: str, title_slug: str) -> dict:
    """Headers for the index API"""
    url_id = f"{title_slug}--{doc_id}"
    return {
        "accept": "text/x-component",
        "accept-language": "en-US,en;q=0.9",
        "cache-control": "no-cache",
        "content-type": "text/plain;charset=UTF-8",
        "next-action": "94635012466e8fede44782d4237c10fe75501920",
        "next-router-state-tree": f'%5B%22%22%2C%7B%22children%22%3A%5B%22van-ban%22%2C%7B%22children%22%3A%5B%5B%22category%22%2C%22chi-tiet%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%5B%22id%22%2C%22{url_id}%22%2C%22d%22%5D%2C%7B%22children%22%3A%5B%22__PAGE__%22%2C%7B%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%5D%7D%2Cnull%2Cnull%2Ctrue%5D',
        "pragma": "no-cache",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    }


# ─────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    """Load checkpoint or create a new one"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
            checkpoint.setdefault("docs_done", 0)
            return checkpoint
    return {"last_page": 0, "failed_docs": [], "docs_done": 0}


def save_checkpoint(last_page: int, failed_docs: list, docs_done: int):
    """Save checkpoint"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(
            {"last_page": last_page, "failed_docs": failed_docs, "docs_done": docs_done},
            f,
            indent=2,
        )


# ─────────────────────────────────────────────────────────────
# Process Doc
# ─────────────────────────────────────────────────────────────

def process_doc(session: requests.Session, doc_id: str, data_f, index_f) -> bool:
    """Fetch detail + index data for 1 doc"""
    for attempt in range(1, MAX_DOC_RETRIES + 1):
        try:
            # 1. Fetch detail
            detail_response = session.get(
                f"https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/{doc_id}",
                timeout=30,
            )
            detail_response.raise_for_status()
            data_detail = detail_response.json().get("data", {})
            data_f.write(json.dumps(data_detail, ensure_ascii=False) + "\n")

            # 2. Fetch index
            title_slug = slugify(data_detail.get("title", ""))
            index_url = f"https://vbpl.vn/van-ban/chi-tiet/{title_slug}--{doc_id}"

            index_response = session.post(
                url=index_url,
                headers=get_index_headers(doc_id, title_slug),
                data=f'["{doc_id}"]',
                timeout=30,
            )
            index_response.encoding = "utf-8"
            index_response.raise_for_status()

            for line in index_response.text.splitlines():
                if line.startswith("1:"):
                    index_data = json.loads(line[2:])
                    index_f.write(
                        json.dumps(
                            {"doc_id": doc_id, "index_data": index_data},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    break

            return True

        except Exception as e:
            wait = 2 ** attempt
            print(f"  [doc {doc_id}] lỗi lần {attempt}/{MAX_DOC_RETRIES}: {e} — chờ {wait}s")
            time.sleep(wait)

    print(f"  [doc {doc_id}] bỏ qua sau {MAX_DOC_RETRIES} lần thất bại")
    return False


# ─────────────────────────────────────────────────────────────
# PHASE 1: CRAWL
# ─────────────────────────────────────────────────────────────

def build_list_payload(page_num: int) -> dict:
    """Payload for the doc/all API. Applies the date filter if CRAWL_DATE_FROM/TO is set."""
    payload = {
        "pageSize": PAGE_SIZE,
        "sortDirection": "desc",
        "sortBy": "viewCount",
        "sortByViewCount": True,
        "pageNumber": page_num,
    }
    if ISSUE_DATE_FROM:
        payload["issueDateFrom"] = ISSUE_DATE_FROM
    if ISSUE_DATE_TO:
        payload["issueDateTo"] = ISSUE_DATE_TO
    return payload


def phase1_crawl():
    """Crawl data from the API.

    Crawl scope is limited by CRAWL_DATE_FROM/CRAWL_DATE_TO (server-side filter
    via issueDateFrom/issueDateTo) and/or CRAWL_MAX_DOCS (stops once enough docs
    have been crawled) — unset means crawl everything as before.
    """
    print("=" * 80)
    print("PHASE 1: CRAWL DATA")
    if ISSUE_DATE_FROM or ISSUE_DATE_TO:
        print(f"Lọc issueDate: {ISSUE_DATE_FROM or '-inf'} .. {ISSUE_DATE_TO or '+inf'}")
    if MAX_DOCS:
        print(f"Giới hạn CRAWL_MAX_DOCS: {MAX_DOCS}")
    print("=" * 80)

    checkpoint = load_checkpoint()
    start_page = checkpoint["last_page"]
    failed_docs = checkpoint["failed_docs"]
    docs_done = checkpoint["docs_done"]

    session = make_session()
    total_pages = None  # known only after the first response (total depends on the date filter)

    with open(CRAWL_DATA_FILE, "a", encoding="utf-8") as data_f, \
         open(CRAWL_INDEX_FILE, "a", encoding="utf-8") as index_f:

        page_num = start_page
        while total_pages is None or page_num < total_pages:
            if MAX_DOCS and docs_done >= MAX_DOCS:
                print(f"\nĐã crawl đủ {docs_done}/{MAX_DOCS} docs (CRAWL_MAX_DOCS), dừng.")
                break

            page_num += 1

            # --- Retry the page ---
            items = None
            for attempt in range(1, MAX_PAGE_RETRIES + 1):
                try:
                    response = session.post(
                        "https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/all",
                        json=build_list_payload(page_num),
                        timeout=30,
                    )
                    response.raise_for_status()
                    data = response.json().get("data", {})
                    items = data.get("items", [])
                    if total_pages is None:
                        total = data.get("total", 0)
                        total_pages = max(1, math.ceil(total / PAGE_SIZE))
                        print(f"Tổng số văn bản khớp filter: {total} ({total_pages} trang)")
                    break

                except Exception as e:
                    wait = 2 ** attempt
                    print(f"[page {page_num}] lỗi lần {attempt}/{MAX_PAGE_RETRIES}: {e} — chờ {wait}s")
                    time.sleep(wait)

            if items is None:
                print(f"[page {page_num}] bỏ qua cả trang sau {MAX_PAGE_RETRIES} lần thất bại")
                save_checkpoint(page_num, failed_docs, docs_done)
                continue

            # --- Process each doc ---
            for item in items:
                if MAX_DOCS and docs_done >= MAX_DOCS:
                    break

                doc_id = item.get("id")
                if not doc_id:
                    continue

                success = process_doc(session, doc_id, data_f, index_f)
                if success:
                    docs_done += 1
                else:
                    failed_docs.append(doc_id)

            # --- Checkpoint after each page ---
            data_f.flush()
            index_f.flush()
            save_checkpoint(page_num, failed_docs, docs_done)
            print(f"Page {page_num}/{total_pages} | Docs OK: {docs_done} | Docs lỗi: {len(failed_docs)}")

        # --- Retry failed docs (only if CRAWL_MAX_DOCS hasn't been reached) ---
        if failed_docs and not (MAX_DOCS and docs_done >= MAX_DOCS):
            print(f"\nRetry {len(failed_docs)} docs lỗi...")
            still_failed = []
            for doc_id in failed_docs:
                success = process_doc(session, doc_id, data_f, index_f)
                if success:
                    docs_done += 1
                else:
                    still_failed.append(doc_id)
                data_f.flush()
                index_f.flush()

            save_checkpoint(page_num, still_failed, docs_done)
            if still_failed:
                print(f"Vẫn còn {len(still_failed)} docs lỗi, xem checkpoint.json")
            else:
                print("Tất cả docs đã xử lý xong!")
        else:
            print(f"\nHoàn tất! Docs OK: {docs_done} | Docs lỗi còn lại: {len(failed_docs)}")


# ─────────────────────────────────────────────────────────────
# PHASE 2: PREPROCESS & FILTER
# ─────────────────────────────────────────────────────────────

def is_valid_document(item: dict, index_data_dict: dict) -> tuple:
    """
    Check whether a doc is valid.

    Rules:
    1. Must have a valid effStatus
    2. Only accept: "Còn hiệu lực", "Hết hiệu lực một phần", "Chưa có hiệu lực"
    3. Must have index metadata

    Return: (is_valid, reason)
    """
    doc_id = str(item.get("id", ""))
    
    # Check effStatus
    if "effStatus" not in item:
        return False, "missing_effStatus"
    
    if item["effStatus"] is None:
        return False, "effStatus_is_None"
    
    if not isinstance(item["effStatus"], dict):
        return False, "effStatus_not_dict"
    
    if "name" not in item["effStatus"]:
        return False, "effStatus_missing_name"
    
    eff_status_name = item["effStatus"]["name"]
    
    # KEEP: "Còn hiệu lực" (in effect), "Hết hiệu lực một phần" (partially expired), "Chưa có hiệu lực" (not yet in effect)
    # REMOVE: "Hết hiệu lực toàn bộ" (fully expired), "Không còn phù hợp" (no longer applicable), "Ngưng hiệu lực" (suspended)
    if eff_status_name not in {"Còn hiệu lực", "Hết hiệu lực một phần", "Chưa có hiệu lực"}:
        return False, f"unsupported_effStatus: {eff_status_name}"
    
    # Check index metadata
    if doc_id not in index_data_dict or not index_data_dict[doc_id]:
        return False, "no_index_metadata"
    
    return True, "valid"


def phase2_preprocess():
    """Filter & merge data"""
    print("\n" + "=" * 80)
    print("PHASE 2: PREPROCESS & FILTER")
    print("=" * 80)
    
    # Load crawled data
    print(f"\n1. Đọc {CRAWL_DATA_FILE}...")
    try:
        with open(CRAWL_DATA_FILE, "r", encoding="utf-8") as f:
            all_items = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File {CRAWL_DATA_FILE} không tồn tại! Chạy phase 1 trước.")
        return
    
    print(f"   Tổng docs crawled: {len(all_items)}")
    
    # Load index data
    print(f"\n2. Đọc {CRAWL_INDEX_FILE}...")
    try:
        with open(CRAWL_INDEX_FILE, "r", encoding="utf-8") as f:
            index_data_list = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"File {CRAWL_INDEX_FILE} không tồn tại! Chạy phase 1 trước.")
        return
    
    # Build index data map
    index_data_dict = {}
    for entry in index_data_list:
        doc_id = entry.get("doc_id")
        index_metadata = entry.get("index_data", [])
        index_data_dict[doc_id] = index_metadata
    
    print(f"   Tổng index entries: {len(index_data_dict)}")
    
    # Filter & merge
    print(f"\n3. Lọc & merge dữ liệu...")
    
    cleaned_data = []
    error_items = []
    
    for item in all_items:
        doc_num = item.get("docNum", "N/A")
        doc_id = str(item.get("id", ""))
        
        is_valid, reason = is_valid_document(item, index_data_dict)
        
        if not is_valid:
            error_items.append({
                "docNum": doc_num,
                "doc_id": doc_id,
                "title": item.get("title"),
                "effStatus": item.get("effStatus"),
                "effFrom": item.get("effFrom"),
                "reason": reason
            })
            continue
        
        # Add index metadata
        item["metadata"] = index_data_dict.get(doc_id, [])
        cleaned_data.append(item)
    
    print(f"   Docs hợp lệ: {len(cleaned_data)}")
    print(f"   Docs lỗi: {len(error_items)}")
    
    # Save results
    print(f"\n4. Lưu kết quả...")
    
    with open(PROCESSED_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"   {PROCESSED_DATA_FILE} ({len(cleaned_data)} docs)")
    
    with open(ERROR_ITEMS_FILE, "w", encoding="utf-8") as f:
        json.dump(error_items, f, ensure_ascii=False, indent=2)
    print(f"   {ERROR_ITEMS_FILE} ({len(error_items)} errors)")
    
    # Summary by reason
    print(f"\n5. Breakdown lỗi:")
    error_reasons = {}
    for err in error_items:
        reason = err["reason"]
        error_reasons[reason] = error_reasons.get(reason, 0) + 1
    
    for reason, count in sorted(error_reasons.items(), key=lambda x: -x[1]):
        print(f"   - {reason}: {count}")
    
    print(f"\nHoàn tất! Ready cho phase 3 (embedding).")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "phase2":
        # Run phase 2 only
        phase2_preprocess()
    else:
        # Run both phases
        phase1_crawl()
        phase2_preprocess()
