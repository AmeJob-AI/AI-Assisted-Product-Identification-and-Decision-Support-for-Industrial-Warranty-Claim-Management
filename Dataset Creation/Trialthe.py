import os
import csv
import time
import re
import requests
from pathlib import Path

APIM_TEMPLATE = "https://apim-rxlse-prod-int-01.azure-api.net/pimv2/article/media/{}/EXTRA_LARGE"
ITEMS_FILE = "item_numbers.txt"

OUT_DIR = Path("dataset_images")
CSV_OUT = "image_mapping.csv"

APIM_SUBSCRIPTION_KEY = None  # put key here if needed

print("Running from folder:", os.getcwd())
print("Looking for:", os.path.join(os.getcwd(), ITEMS_FILE))

OUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

if APIM_SUBSCRIPTION_KEY:
    session.headers.update({"Ocp-Apim-Subscription-Key": APIM_SUBSCRIPTION_KEY})

def guess_ext(content_type: str, url: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    return ".jpg"

# Read items
with open(ITEMS_FILE, "r", encoding="utf-8") as f:
    item_numbers = [line.strip() for line in f if line.strip()]

rows = []
for idx, item in enumerate(item_numbers, start=1):
    api_url = APIM_TEMPLATE.format(item)
    try:
        r = session.get(api_url, timeout=60, allow_redirects=True)

        if r.status_code != 200:
            print(f"[{idx}] {item} -> FAILED {r.status_code}")
            rows.append([item, api_url, "", f"HTTP_{r.status_code}"])
            continue

        final_url = r.url
        ext = guess_ext(r.headers.get("Content-Type", ""), final_url)
        out_path = OUT_DIR / f"{item}{ext}"

        if out_path.exists():
            print(f"[{idx}] {item} -> SKIP (already exists)")
            rows.append([item, api_url, final_url, out_path.as_posix()])
            continue

        with open(out_path, "wb") as imgf:
            imgf.write(r.content)

        print(f"[{idx}] {item} -> OK saved {out_path.name}")
        rows.append([item, api_url, final_url, out_path.as_posix()])

        time.sleep(0.2)

    except Exception as e:
        print(f"[{idx}] {item} -> ERROR {e}")
        rows.append([item, api_url, "", f"ERROR_{e}"])


with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["item_number", "apim_url", "final_image_url", "saved_path_or_error"])
    writer.writerows(rows)

print("\nDone ✅")
print("Images folder:", OUT_DIR.resolve())
print("Mapping CSV:", Path(CSV_OUT).resolve())
