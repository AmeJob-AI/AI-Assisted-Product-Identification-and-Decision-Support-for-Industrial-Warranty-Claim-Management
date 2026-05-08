import csv
import time
import requests
from pathlib import Path

ITEMS_FILE = "item_numbers.txt"
IMAGES_DIR = Path("dataset_images")
OUT_CSV = "image_mapping.csv"   # will APPEND new rows

SIZES = ["EXTRA_LARGE", "LARGE", "MEDIUM", "SMALL"]  # fallback order
BASE = "https://apim-rxlse-prod-int-01.azure-api.net/pimv2/article/media/{}/{}"

SLEEP = 0.2
TIMEOUT = 60

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def existing_items():
    existing = set()
    for p in IMAGES_DIR.glob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            existing.add(p.stem.strip())
    return existing

def guess_ext(content_type: str) -> str:
    ct = (content_type or "").lower()
    if "png" in ct: return ".png"
    if "webp" in ct: return ".webp"
    if "jpeg" in ct or "jpg" in ct: return ".jpg"
    return ".jpg"

all_items = [line.strip() for line in open(ITEMS_FILE, "r", encoding="utf-8") if line.strip()]
IMAGES_DIR.mkdir(exist_ok=True)

existing = existing_items()
missing = [it for it in all_items if it not in existing]

print("Total items:", len(all_items))
print("Already have images:", len(existing))
print("Missing to download:", len(missing))

with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)

    for idx, item in enumerate(missing, start=1):
        saved = False

        for size in SIZES:
            api_url = BASE.format(item, size)

            try:
                r = session.get(api_url, timeout=TIMEOUT, allow_redirects=True)
                if r.status_code != 200:
                    continue

                ext = guess_ext(r.headers.get("Content-Type", ""))
                out_path = IMAGES_DIR / f"{item}{ext}"   # one image per product

                with open(out_path, "wb") as imgf:
                    imgf.write(r.content)

                w.writerow([item, api_url, r.url, out_path.as_posix()])
                print(f"[{idx}/{len(missing)}] {item} -> OK ({size})")
                saved = True
                time.sleep(SLEEP)
                break

            except Exception:
                continue

        if not saved:
            w.writerow([item, "", "", "NO_IMAGE_ANY_SIZE"])
            print(f"[{idx}/{len(missing)}] {item} -> NO IMAGE (all sizes failed)")