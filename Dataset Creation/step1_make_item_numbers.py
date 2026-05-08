import re
from pathlib import Path
import pandas as pd

# ========= CONFIG =========
INPUT_FILE = "Cleaned Products.xlsx"   
SHEET_NAME = None                      

# Set exact column name here if you know it (recommended):
# Example: ITEM_COL = "item_no"
ITEM_COL = None

# If ITEM_COL is None, it will auto-detect from these (case-insensitive):
ITEM_COL_CANDIDATES = [
    "item_no", "itemno", "artikelnummer", "artikelnr",
    "article_number", "article_no", "article no", "article",
    "e-number", "e number", "E-number/Article number"
]

OUT_TXT = Path("item_numbers.txt")
# ==========================

def normalize_item(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)     
    s = re.sub(r"\D+", "", s)       
    return s

def read_input_as_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        if SHEET_NAME is None:
            xls = pd.ExcelFile(path)
            first_sheet = xls.sheet_names[0]
            return pd.read_excel(path, sheet_name=first_sheet)
        return pd.read_excel(path, sheet_name=SHEET_NAME)
    return pd.read_csv(path)

def main():
    path = Path(INPUT_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path.resolve()}")

    df = read_input_as_df(path)

    # Choose column
    if ITEM_COL:
        if ITEM_COL not in df.columns:
            raise ValueError(f"Column '{ITEM_COL}' not found. Columns: {list(df.columns)}")
        col = ITEM_COL
    else:
        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        col = None
        for cand in ITEM_COL_CANDIDATES:
            if cand.lower() in cols_lower:
                col = cols_lower[cand.lower()]
                break
        if not col:
            raise ValueError(
                "Could not find item_no column. Available columns:\n"
                + "\n".join(map(str, df.columns))
            )

    items = df[col].apply(normalize_item)
    items = sorted(set([x for x in items.tolist() if x]))

    OUT_TXT.write_text("\n".join(items) + "\n", encoding="utf-8")

    print("✅ STEP 1 DONE")
    print(f"Used column: {col}")
    print(f"Unique item numbers: {len(items)}")
    print(f"Saved: {OUT_TXT.resolve()}")

if __name__ == "__main__":
    main()
