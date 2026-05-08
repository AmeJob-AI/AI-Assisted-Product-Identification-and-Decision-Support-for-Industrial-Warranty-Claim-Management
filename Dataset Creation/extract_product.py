
import csv
import re
import time
from pathlib import Path
from typing import Tuple

import requests
from bs4 import BeautifulSoup
import html as ihtml

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd  # NEW


# --------- CONFIG ----------
ITEMS_FILE = "item_numbers.txt"

CSV_OUT = "product_details_out.csv"
XLSX_OUT = "product_details_out.xlsx"   # NEW
DEBUG_DIR = Path("debug_html")

PRODUCT_URL_TEMPLATE = "https://rexel.se/swe/p/{}"

SLEEP_SEC = 0.5
TIMEOUT = 30

CSV_DELIMITER = ";"
# ---------------------------


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "sv-SE,sv;q=0.9,en;q=0.8",
    })

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def clean(s: str) -> str:
    s = ihtml.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def first_digits(s: str, min_len: int = 8, max_len: int = 14) -> str:
    if not s:
        return ""
    m = re.search(rf"\b\d{{{min_len},{max_len}}}\b", s)
    return m.group(0) if m else ""


def find_value_after_label(soup: BeautifulSoup, label_patterns: str) -> str:
    label_node = soup.find(string=re.compile(label_patterns, re.IGNORECASE))
    if not label_node:
        return ""

    label_tag = label_node.parent
    if not label_tag:
        return ""

    if label_tag.name == "dt":
        dd = label_tag.find_next("dd")
        if dd:
            return clean(dd.get_text(" ", strip=True))

    nxt = label_tag.find_next_sibling()
    if nxt:
        return clean(nxt.get_text(" ", strip=True))

    nxt2 = label_tag.find_next()
    if nxt2 and nxt2 != label_tag:
        txt = clean(nxt2.get_text(" ", strip=True))
        if txt and not re.search(label_patterns, txt, re.IGNORECASE):
            return txt

    return ""


def extract_rexel_fields(html_text: str, input_item: str) -> Tuple[str, str, str, str, str, str]:
    soup = BeautifulSoup(html_text, "html.parser")

    title = ""
    h1 = soup.find("h1")
    if h1:
        title = clean(h1.get_text(" ", strip=True))
    elif soup.title:
        title = clean(soup.title.get_text(" ", strip=True))

    item_number = input_item

    description = ""
    dd_short = soup.select_one("dd.short-product-description")
    if dd_short:
        description = clean(dd_short.get_text(" ", strip=True))
    else:
        description = find_value_after_label(soup, r"\bBeskrivning\b")

    manufacturer_item_no = ""

    long_desc_div = soup.select_one("div.long-product-description[data-manufacturervariable]")
    if long_desc_div and long_desc_div.has_attr("data-manufacturervariable"):
        manufacturer_item_no = clean(long_desc_div["data-manufacturervariable"])

    if not manufacturer_item_no:
        text = clean(soup.get_text(" ", strip=True))
        m = re.search(r"Manufacturer'?s item no\.?\s*[:：]\s*([A-Za-z0-9\-_/.]+)", text, re.IGNORECASE)
        if m:
            manufacturer_item_no = clean(m.group(1))

    ean = ""
    ean_raw = find_value_after_label(soup, r"\bEAN(?:-kod)?\b")
    if ean_raw:
        ean = first_digits(ean_raw, 8, 14)

    if not ean:
        text = clean(soup.get_text(" ", strip=True))
        m = re.search(r"\bEAN(?:-kod| code)?\b\s*[:：]?\s*(\d{8,14})\b", text, re.IGNORECASE)
        if m:
            ean = m.group(1)

    missing = []
    if not title:
        missing.append("title")
    if not manufacturer_item_no:
        missing.append("manufacturer_item_no")
    if not ean:
        missing.append("ean")
    if not description:
        missing.append("description")

    status = "OK" if not missing else "MISSING_" + "_".join(missing)
    return title, item_number, manufacturer_item_no, ean, description, status


def main():
    items_path = Path(ITEMS_FILE)
    if not items_path.exists():
        raise FileNotFoundError(f"Missing {ITEMS_FILE}. Put item numbers there, one per line.")

    items = [x.strip() for x in items_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not items:
        raise ValueError(f"{ITEMS_FILE} is empty.")

    DEBUG_DIR.mkdir(exist_ok=True)

    session = build_session()

    # We'll collect rows for Excel output too
    excel_rows = []  # NEW

    with open(CSV_OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=CSV_DELIMITER)
        header = [
            "input_item",
            "product_url",
            "title",
            "item_number",
            "manufacturer_item_no",
            "ean",
            "description",
            "status"
        ]
        w.writerow(header)

        for idx, item in enumerate(items, start=1):
            url = PRODUCT_URL_TEMPLATE.format(item)

            try:
                r = session.get(url, timeout=TIMEOUT, allow_redirects=True)

                if r.status_code != 200:
                    print(f"[{idx}] {item} -> HTTP_{r.status_code}")
                    row = [item, url, "", "", "", "", "", f"HTTP_{r.status_code}"]
                    w.writerow(row)
                    excel_rows.append(dict(zip(header, row)))
                    time.sleep(SLEEP_SEC)
                    continue

                html_text = r.content.decode("utf-8", errors="replace")
                title, item_no, manuf_no, ean, desc, status = extract_rexel_fields(html_text, item)

                if status != "OK":
                    (DEBUG_DIR / f"{item}.html").write_text(html_text, encoding="utf-8")

                # Keep EAN as text for Excel (prevents scientific notation)
                ean_csv = f"'{ean}" if ean else ""

                row = [item, url, title, item_no, manuf_no, ean_csv, desc, status]
                w.writerow(row)
                excel_rows.append(dict(zip(header, row)))

                print(f"[{idx}] {item} -> {status} | {title[:60]}")
                time.sleep(SLEEP_SEC)

            except requests.exceptions.RequestException as e:
                print(f"[{idx}] {item} -> REQ_ERROR: {type(e).__name__}: {e}")
                row = [item, url, "", "", "", "", "", f"REQ_ERROR_{type(e).__name__}"]
                w.writerow(row)
                excel_rows.append(dict(zip(header, row)))
                time.sleep(SLEEP_SEC)

            except Exception as e:
                print(f"[{idx}] {item} -> ERROR: {type(e).__name__}: {e}")
                row = [item, url, "", "", "", "", "", f"ERROR_{type(e).__name__}"]
                w.writerow(row)
                excel_rows.append(dict(zip(header, row)))
                time.sleep(SLEEP_SEC)

    # ---- NEW: Save XLSX ----
    df = pd.DataFrame(excel_rows)
    # ensure all columns exist in order
    df = df[header]
    df.to_excel(XLSX_OUT, index=False)
    # ------------------------

    print("\nDone Saved:", Path(CSV_OUT).resolve())
    print("Also saved :", Path(XLSX_OUT).resolve())
    print("If any row shows MISSING_*, open ./debug_html/<item>.html")


if __name__ == "__main__":
    main()
