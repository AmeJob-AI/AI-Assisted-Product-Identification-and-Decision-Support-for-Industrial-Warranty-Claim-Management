"""
=====================================
Run: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from generatepdf import generate_alem09_pdf
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from jose import jwt
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import faiss
import torch
from transformers import AutoImageProcessor, AutoModel
import clip
import io
import pandas as pd
import pickle
import logging
import time
from pathlib import Path
from typing import Optional
import zxingcpp
from database import get_db

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(title="API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────
SECRET_KEY = "Api-secret-key-change-this"
security   = HTTPBearer()

# ── PATHS ─────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
CSV_FILE   = BASE_DIR / "final_metadata.csv"
FAISS_FILE = BASE_DIR / "faiss_index" / "product.index"
MAP_FILE   = BASE_DIR / "faiss_index" / "item_map.pkl"
CLIP_CACHE = BASE_DIR / "faiss_index" / "clip_cache.pkl"
#IMG_DIR    = Path(r"C:\Users\jobsy\Desktop\Thesis\THESIS_LATEST\Final_Dataset")
IMG_DIR    = Path(r"C:\Users\jobsy\Desktop\Thesis\Rexel_Dataset_full\Rexel_Dataset\dataset_images")

DINO_TOP  = 30
FINAL_TOP = 5

# ── Load AI Models ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")

logger.info("Loading DINOv2 Large ...")
dino_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
dino_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
dino_model.eval()

logger.info("Loading CLIP ViT-L/14 ...")
clip_model, clip_prep = clip.load("ViT-L/14", device=device)
clip_model.eval()
logger.info(" Models loaded.")

# ── Data globals ──────────────────────────────────────────────────
products_df  : pd.DataFrame = pd.DataFrame()
faiss_index  = None
item_map     : list         = []
clip_cache_d : dict         = {}

def load_data():
    global products_df, faiss_index, item_map, clip_cache_d
    if CSV_FILE.exists():
        products_df = pd.read_csv(str(CSV_FILE), dtype=str)
        products_df.columns = [
            c.strip().lower().replace(" ","_").replace("/","_")
             .replace("'","").replace(".","")
            for c in products_df.columns
        ]
        products_df = products_df.apply(
            lambda col: col.str.strip() if col.dtype=="object" else col
        )
        for col in ["item_no","ean"]:
            if col in products_df.columns:
                products_df[col] = products_df[col].str.replace(r"\.0$","",regex=True)
        logger.info(f"CSV: {len(products_df)} products")
    else:
        logger.error(f"CSV not found: {CSV_FILE}")

    if FAISS_FILE.exists() and MAP_FILE.exists():
        faiss_index = faiss.read_index(str(FAISS_FILE))
        with open(str(MAP_FILE),"rb") as f:
            item_map = pickle.load(f)
        logger.info(f" FAISS: {faiss_index.ntotal} embeddings")
    else:
        logger.warning("  FAISS index not found — run Build_index.py")

    if CLIP_CACHE.exists():
        with open(str(CLIP_CACHE),"rb") as f:
            clip_cache_d = pickle.load(f)
        logger.info(f" CLIP cache: {len(clip_cache_d)} products")

@app.on_event("startup")
async def on_startup():
    load_data()
    logger.info(" Server ready!")

# ── Embedding helpers ─────────────────────────────────────────────
def embed_dino(img):
    with torch.no_grad():
        inp = dino_proc(images=img, return_tensors="pt").to(device)
        out = dino_model(**inp)
        e   = out.last_hidden_state[:,0,:].cpu().numpy()
    e = e / (np.linalg.norm(e)+1e-8)
    return e.astype(np.float32)

def embed_clip(img):
    with torch.no_grad():
        t = clip_prep(img).unsqueeze(0).to(device)
        e = clip_model.encode_image(t).cpu().numpy().astype(np.float32)
    e = e / (np.linalg.norm(e)+1e-8)
    return e

# ── CSV helpers ───────────────────────────────────────────────────
def _clean(v):
    s = str(v).strip()
    return "" if s.lower() in ("nan","none","na","") else s

def _row_to_dict(row):
    mfr   = next((c for c in row.index if "manufacturer" in c and "item" in c),"")
    brand = next((c for c in row.index if "brand" in c),"")
    url   = next((c for c in row.index if "url" in c),"")
    return {
        "item_no":              _clean(row.get("item_no","")),
        "title":                _clean(row.get("title","")),
        "manufacturer_item_no": _clean(row.get(mfr,"")) if mfr else "",
        "description":          _clean(row.get("description","")),
        "ean":                  _clean(row.get("ean","")),
        "brand":                _clean(row.get(brand,"")) if brand else "",
        "product_url":          _clean(row.get(url,"")) if url else "",
    }

def find_by_item_no(item_no):
    if products_df.empty: return None
    m = products_df[products_df["item_no"]==str(item_no).strip()]
    return _row_to_dict(m.iloc[0]) if not m.empty else None

def find_by_ean(ean):
    if products_df.empty: return None
    ean = str(ean).strip().replace(".0","")
    m   = products_df[products_df["ean"]==ean]
    return _row_to_dict(m.iloc[0]) if not m.empty else None

def img_url(item_no):
    return f"http://localhost:8000/catalog-image/{item_no}"

# ── Barcode ───────────────────────────────────────────────────────
def detect_barcode(img):
    try:
        results = zxingcpp.read_barcodes(np.array(img.convert("RGB")))
        for r in results:
            t = r.text.strip()
            if t: return t
        return None
    except Exception as e:
        logger.error(f"Barcode error: {e}")
        return None

# ── FAISS search ──────────────────────────────────────────────────
def faiss_search(emb, top_k):
    if faiss_index is None or faiss_index.ntotal==0: return []
    dists, idxs = faiss_index.search(emb, top_k)
    seen, results = set(), []
    for dist, idx in zip(dists[0], idxs[0]):
        if idx==-1: continue
        item_no = item_map[idx]
        if item_no in seen: continue
        seen.add(item_no)
        p = find_by_item_no(item_no)
        if p:
            p["dino_score"] = round(max(0.0,(1.0-float(dist)/2.0)*100),1)
            p["similarity"] = p["dino_score"]
            p["image_url"]  = img_url(item_no)
            results.append(p)
    return results


# ── FAISS filtered search — only searches order item indices ──────
def faiss_search_filtered(emb, allowed_items: set, top_k: int):
    """
    Search FAISS but only return results whose item_no is in allowed_items.
    Used when an order number is provided — restricts results to order products only.
    """
    if faiss_index is None or faiss_index.ntotal == 0:
        return []

    # Search broader pool first then filter — take min(300, total) candidates
    search_k = min(faiss_index.ntotal, max(top_k * 10, 300))
    dists, idxs = faiss_index.search(emb, search_k)

    seen, results = set(), []
    for dist, idx in zip(dists[0], idxs[0]):
        if idx == -1:
            continue
        item_no = item_map[idx]
        if item_no in seen:
            continue
        if item_no not in allowed_items:
            continue   # ← skip anything not in the order
        seen.add(item_no)
        p = find_by_item_no(item_no)
        if p:
            p["dino_score"] = round(max(0.0, (1.0 - float(dist) / 2.0) * 100), 1)
            p["similarity"] = p["dino_score"]
            p["image_url"]  = img_url(item_no)
            results.append(p)
        if len(results) >= top_k:
            break

    # If we found fewer than top_k via similarity, pad with remaining order items
    if len(results) < top_k:
        found_items = {r["item_no"] for r in results}
        for item_no in allowed_items:
            if item_no in found_items:
                continue
            p = find_by_item_no(item_no)
            if p:
                p["dino_score"] = 0.0
                p["similarity"] = 0.0
                p["image_url"]  = img_url(item_no)
                results.append(p)
            if len(results) >= top_k:
                break

    return results

# ── CLIP rerank ───────────────────────────────────────────────────
def clip_rerank(query_img, candidates, top_k):
    if not clip_cache_d: return candidates[:top_k]
    query_emb = embed_clip(query_img)
    scored = []
    for c in candidates:
        cat_emb = clip_cache_d.get(c["item_no"])
        if cat_emb is not None:
            clip_sim = float(np.dot(query_emb.flatten(), cat_emb.flatten()))
            combined = (0.5*c["dino_score"]/100 + 0.5*max(0.0,clip_sim))*100
        else:
            combined = c["dino_score"]
        cp = dict(c)
        cp["similarity"] = round(min(100.0,max(0.0,combined)),1)
        scored.append(cp)
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]

# ════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "products": len(products_df),
        "faiss_embeddings": faiss_index.ntotal if faiss_index else 0,
        "faiss_ready": faiss_index is not None,
        "device": device,
    }

# ── Catalog image ─────────────────────────────────────────────────
@app.get("/catalog-image/{item_no}")
async def catalog_image(item_no: str):
    for ext in ["jpg","jpeg","png","JPG","JPEG","PNG"]:
        p = IMG_DIR / f"{item_no}.{ext}"
        if p.exists():
            return FileResponse(str(p))
    raise HTTPException(status_code=404, detail=f"No image for {item_no}")

# ════════════════════════════════════════════════════════
# AUTH — LOGIN
# ════════════════════════════════════════════════════════
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/auth/login")
def login(req: LoginRequest, db=Depends(get_db)):
    cursor = db.cursor()

    # sp_LoginTechnician selects:
    # [0]Tech_id, [1]technician_name, [2]password,
    # [3]company_name, [4]phone_number, [5]mail_id
    cursor.execute("EXEC sp_LoginTechnician1 @username=?", req.username)
    row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # pyodbc.Row only supports index access — NOT attribute access
    tech_id         = row[0]
    technician_name = row[1] or ""
    db_password     = row[2] or ""
    company_name    = row[3] or ""
    phone_number    = row[4] or ""
    mail_id         = row[5] or ""

    # Plain-text password comparison (no bcrypt)
    if req.password != db_password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = jwt.encode(
        {
            "sub":  str(tech_id),
            "name": technician_name,
            "exp":  datetime.utcnow() + timedelta(hours=8),
        },
        SECRET_KEY, algorithm="HS256"
    )

    return {
        "access_token":    token,
        "technician_name": technician_name,
        "technician_id":   tech_id,
        "company_name":    company_name,
        "phone_number":    phone_number,
        "mail_id":         mail_id,
    }

# ════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════
@app.get("/claims/dashboard")
def get_dashboard(credentials=Depends(security), db=Depends(get_db)):
    cursor = db.cursor()
    cursor.execute("EXEC sp_GetDashboard1")

    # First result-set: stats row  (total, approved, rejected, pending)
    stats_row = cursor.fetchone()
    stats = {
        "total":    stats_row[0] or 0,
        "approved": stats_row[1] or 0,
        "rejected": stats_row[2] or 0,
        "pending":  stats_row[3] or 0,
    }

    # Second result-set: recent claims
    cursor.nextset()
    columns = [col[0] for col in cursor.description]
    claims  = []
    for row in cursor.fetchall():
        claim = dict(zip(columns, row))
        if claim.get("Created_at"):
            claim["Created_at"] = str(claim["Created_at"])[:10]
        claims.append(claim)

    return {"stats": stats, "recent_claims": claims}

# ════════════════════════════════════════════════════════
# GET ORDER DETAILS — lookup by ordobjekt (order number)
# ════════════════════════════════════════════════════════
@app.get("/order/{order_number}")
def get_order_details(
    order_number: str,
    credentials=Depends(security),
    db=Depends(get_db),
):
    """
    Queries Orderrad → Order → Kunder using ordobjekt.
    Returns:
      - order_info: customer name, address, phone, faktnr, dates
      - item_numbers: list of all artnr in this order (for FAISS filter)
    """
    cursor = db.cursor()
    cursor.execute("EXEC sp_GetOrderDetails2 @ordobjekt=?", order_number)

    # Result-set 1: order + customer info
    info_row = cursor.fetchone()
    if not info_row:
        raise HTTPException(
            status_code=404,
            detail=f"Order '{order_number}' not found in Orderrad table"
        )

    order_info = {
        "customer_name":    info_row[0] or "",
        "customer_address": info_row[1] or "",
        "customer_phone":   str(info_row[2] or ""),
        "faktnr":           str(info_row[3] or ""),
        "projekt":          str(info_row[4] or ""),
        "objekt":           str(info_row[5] or ""),
        "odatum":           str(info_row[6])[:10] if info_row[6] else "",
        "levdat":           str(info_row[7])[:10] if info_row[7] else "",
    }

    # Result-set 2: all item numbers + quantity per artnr
    cursor.nextset()
    item_rows    = cursor.fetchall()
    item_numbers = []
    faktnr_map   = {}   # artnr -> faktnr
    quantity_map = {}   # artnr -> quantity (COUNT of rows in order)

    for row in item_rows:
        artnr    = str(row[0]).strip() if row[0] else ""
        faktnr   = str(row[1]).strip() if row[1] else ""
        quantity = int(row[2]) if row[2] else 1
        if artnr:
            item_numbers.append(artnr)
            faktnr_map[artnr]   = faktnr
            quantity_map[artnr] = quantity

    logger.info(f"Order {order_number}: {len(item_numbers)} items found")
    return {
        "success":      True,
        "order_number": order_number,
        "order_info":   order_info,
        "item_numbers": item_numbers,
        "faktnr_map":   faktnr_map,
        "quantity_map": quantity_map,
    }


# ════════════════════════════════════════════════════════
# IDENTIFY
# ════════════════════════════════════════════════════════
@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    order_items: Optional[str] = Form(None),  # ← Form() so FastAPI reads it from multipart
):
    """
    Identify product from image.
    If order_items is provided, FAISS search AND CLIP rerank are restricted
    to only those item numbers — nothing outside the order can appear.
    order_items format: "1315210,1377065,1820292,2106000"
    """
    t0 = time.time()
    try:
        data  = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")

        # Parse order filter
        allowed_items = None
        if order_items and str(order_items).strip():
            allowed_items = set(
                x.strip() for x in str(order_items).split(",") if x.strip()
            )
            logger.info(f"✅ Order filter active: {len(allowed_items)} items → {list(allowed_items)[:5]}...")

        # ── Barcode first ─────────────────────────────────────────
        barcode = detect_barcode(image)
        if barcode:
            product = find_by_ean(barcode) or find_by_item_no(barcode)
            if product:
                in_order = (allowed_items is None) or (product["item_no"] in allowed_items)
                if in_order:
                    product.update({
                        "similarity":   100.0,
                        "match_method": "barcode",
                        "image_url":    img_url(product["item_no"]),
                    })
                    return JSONResponse({
                        "success": True, "match_method": "barcode",
                        "barcode_value": barcode, "matches": [product],
                        "auto_confirm":  True,
                        "message": f"Barcode match: {product['item_no']}",
                    })
                else:
                    logger.info(f"Barcode {product['item_no']} not in order — falling to visual search")

        # ── FAISS ────────────────────────────────────────────────
        if faiss_index is None or faiss_index.ntotal == 0:
            return JSONResponse({
                "success": False, "match_method": "none",
                "matches": [], "auto_confirm": False,
                "message": "FAISS index not ready.",
            })

        dino_emb = embed_dino(image)

        if allowed_items:
            # FILTERED — only items from the order
            candidates = faiss_search_filtered(
                dino_emb.reshape(1, -1),
                allowed_items=allowed_items,
                top_k=len(allowed_items),   # get ALL order items ranked
            )
        else:
            candidates = faiss_search(dino_emb.reshape(1, -1), top_k=DINO_TOP)

        # ── CLIP rerank — enforce filter again so clip never adds outside items ──
        if allowed_items:
            # Only rerank candidates already in allowed_items (they all are, but be safe)
            candidates = [c for c in candidates if c["item_no"] in allowed_items]
            results = clip_rerank(image, candidates, top_k=min(FINAL_TOP, len(candidates)))
            # Final safety check — strip anything outside order
            results = [r for r in results if r["item_no"] in allowed_items]
        else:
            results = clip_rerank(image, candidates, top_k=FINAL_TOP)

        for r in results:
            r["match_method"] = "visual_search"

        total = (time.time() - t0) * 1000
        mode  = f"order-filtered ({len(allowed_items)} items)" if allowed_items else "full dataset"
        logger.info(f"Results: {[r['item_no'] for r in results]}")
        return JSONResponse({
            "success": True, "match_method": "visual_search",
            "matches": results, "auto_confirm": False,
            "message": f"Found {len(results)} matches in {total:.0f}ms [{mode}]",
        })

    except Exception as e:
        logger.error(f"Identify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ════════════════════════════════════════════════════════
# GENERATE PDF — returns filled ALEM09 PDF for download
# ════════════════════════════════════════════════════════
class PDFRequest(BaseModel):
    # All claim fields needed for PDF — mirrors ClaimData + technician info
    claimid:                        Optional[int]  = None
    # Buyer
    BuyerName:                      Optional[str]  = ""
    BuyerContactPerson:             Optional[str]  = ""
    BuyerPhone:                     Optional[str]  = ""
    BuyerEmail:                     Optional[str]  = ""
    # Seller
    Seller_name:                    Optional[str]  = ""
    Seller_contact:                 Optional[str]  = ""
    Seller_phone:                   Optional[str]  = ""
    Seller_mail:                    Optional[str]  = ""
    # Dates
    Complaint_date:                 Optional[str]  = ""
    Odatum:                         Optional[str]  = ""
    Faktnr:                         Optional[str]  = ""
    Levdat:                         Optional[str]  = ""
    Comm_date:                      Optional[str]  = ""
    # Agreement
    AB04:                           Optional[int]  = 0
    ABT06:                          Optional[int]  = 0
    EL10:                           Optional[int]  = 0
    ABS09:                          Optional[int]  = 0
    Agreement_HandymanForm:         Optional[int]  = 0
    Agreement_OtherText:            Optional[str]  = ""
    ContractAttached:               Optional[int]  = 0
    BuyerWarrantyMonths:            Optional[int]  = None
    EndCustomerOrFacilityOwner:     Optional[str]  = ""
    Install_site:                   Optional[str]  = ""
    InvestigationReportRequested:   Optional[int]  = 0
    CurrentProductLocation:         Optional[str]  = ""
    ReturnDeliveryAddress:          Optional[str]  = ""
    # Product
    Quantity:                       Optional[str]  = ""
    Artnr:                          Optional[str]  = ""
    EAN:                            Optional[str]  = ""
    Serienum:                       Optional[str]  = ""
    Type_designation:               Optional[str]  = ""
    Descfault:                      Optional[str]  = ""
    Add_info:                       Optional[str]  = ""
    # Installation
    InstalledIndoors:               Optional[int]  = 0
    InstalledOutdoors:              Optional[int]  = 0
    HumidEnvironment:               Optional[int]  = 0
    DryEnvironment:                 Optional[int]  = 0
    DustyEnvironment:               Optional[int]  = 0
    PermanentlyHeated:              Optional[int]  = 0
    AmbientTempMinMax:              Optional[str]  = ""
    AggressiveEnvironment:          Optional[int]  = 0
    VibrationsPresent:              Optional[int]  = 0
    InstalledToStandard:            Optional[int]  = 0
    InstalledToInstructions:        Optional[int]  = 0
    MaintainedToInstructions:       Optional[int]  = 0
    CircuitDiagramStatus:           Optional[str]  = "No"
    OperationLogStatus:             Optional[str]  = "No"
    Claim_status:                   Optional[str]  = "pending"
    Created_at:                     Optional[str]  = ""
    # Technician (filled by section)
    technician_name:                Optional[str]  = ""
    company_name:                   Optional[str]  = ""
    phone_number:                   Optional[str]  = ""
    mail_id:                        Optional[str]  = ""


@app.get("/identify-by-artnr")
def identify_by_artnr(
    item_no: str,
    credentials=Depends(security),
    db=Depends(get_db),
):
    if not item_no or not item_no.strip():
        raise HTTPException(status_code=400, detail="item_no is required")

    q = item_no.strip()

    # ── Step 1: Try CSV first (full details + image) ──────────────
    product = find_by_item_no(q) or find_by_ean(q)
    if product:
        product["image_url"]    = img_url(product["item_no"])
        product["similarity"]   = 100.0
        product["match_method"] = "manual"
        product["dino_score"]   = 100.0
        product["source"]       = "csv"
        logger.info(f" Manual lookup (CSV): '{q}' → {product['item_no']}")
        return JSONResponse({
            "success":      True,
            "match_method": "manual",
            "source":       "csv",
            "product":      product,
            "message":      f"Found in catalogue: {product['item_no']} — {product['title']}",
        })

    
    try:
        cursor = db.cursor()
        cursor.execute("""
            SELECT TOP 1
                r.artnr,
                r.faktnr,
                r.ordobjekt
            FROM Orderrad r
            WHERE r.artnr = ?
              AND r.artnr IS NOT NULL
              AND r.artnr NOT IN ('T', 'SN', 'ARBDV', '')
        """, q)
        row = cursor.fetchone()

        if row:
            # Found in DB — return minimal product dict, no image
            db_product = {
                "item_no":              str(row[0] or "").strip(),
                "title":                "",          # not available in Orderrad
                "manufacturer_item_no": "",          # not available
                "description":          "",          # not available
                "ean":                  "",          # not available
                "brand":                "",          # not available
                "product_url":          "",
                "image_url":            "",          # no image — frontend shows 
                "similarity":           100.0,
                "match_method":         "manual",
                "dino_score":           100.0,
                "source":               "db",
            }
            logger.info(f" Manual lookup (DB fallback): '{q}' found in Orderrad")
            return JSONResponse({
                "success":      True,
                "match_method": "manual",
                "source":       "db",
                "product":      db_product,
                "message":      f"Item '{q}' found in database (not in product catalogue). "
                                f"Please fill in product details manually.",
            })
    except Exception as e:
        logger.error(f"DB fallback error for '{q}': {e}")

    # ── Step 3: Not found anywhere ────────────────────────────────
    logger.info(f" Manual lookup: '{q}' not found in CSV or DB")
    return JSONResponse({
        "success": False,
        "source":  "not_found",
        "message": f"No product found for article number '{q}'. "
                   f"Check the number and try again.",
    })



@app.get("/validate-order-item")
def validate_order_item(
    order_number: str,
    item_no: str,
    credentials=Depends(security),
    db=Depends(get_db),
):
    if not order_number.strip() or not item_no.strip():
        raise HTTPException(status_code=400, detail="order_number and item_no are required")

    try:
        cursor = db.cursor()
        cursor.execute("""
            SELECT COUNT(*)
            FROM Orderrad
            WHERE ordobjekt = ?
              AND artnr     = ?
              AND artnr IS NOT NULL
              AND artnr NOT IN ('T', 'SN', 'ARBDV', '')
        """, order_number.strip(), item_no.strip())
        row = cursor.fetchone()
        count = int(row[0]) if row else 0

        if count > 0:
            return JSONResponse({
                "valid":   True,
                "message": f"Item '{item_no}' is part of order '{order_number}'.",
            })
        else:
            return JSONResponse({
                "valid":   False,
                "message": f"Article '{item_no}' is not part of order '{order_number}'. "
                           f"Please check the article number or order number.",
            })
    except Exception as e:
        logger.error(f"validate-order-item error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/claims/generate-pdf")
def generate_claim_pdf(
    req: PDFRequest,
    credentials=Depends(security),
):
    """
    Generates and returns a filled ALEM09 English PDF for download.
    Called from frontend immediately after successful claim submit.
    """
    try:
        data = req.dict()
        pdf_bytes = generate_alem09_pdf(data)
        filename  = f"ALEM09_Claim_{req.claimid or 'draft'}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(pdf_bytes)),
            }
        )
    except Exception as e:
        logger.error(f"PDF generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ════════════════════════════════════════════════════════
# SUBMIT CLAIM  — matches Claim_Table exactly
# ════════════════════════════════════════════════════════
class ClaimData(BaseModel):
    # Technician
    Tech_id:                        int

    # Customer (from top section of NewClaim form)
    Customer_name:                  Optional[str] = ""
    Customer_phone:                 Optional[str] = ""
    Customer_Address:               Optional[str] = ""

    # Product
    Artnr:                          Optional[str] = ""
    EAN:                            Optional[str] = ""
    Serienum:                       Optional[str] = ""
    Type_designation:               Optional[str] = ""

    # Fault
    Descfault:                      Optional[str] = ""
    Add_info:                       Optional[str] = ""

    # Seller
    Seller_name:                    Optional[str] = ""
    Seller_contact:                 Optional[str] = ""
    Seller_phone:                   Optional[str] = ""
    Seller_mail:                    Optional[str] = ""

    # Dates
    Complaint_date:                 Optional[str] = None
    Odatum:                         Optional[str] = None
    Faktnr:                         Optional[str] = ""
    Levdat:                         Optional[str] = None
    Comm_date:                      Optional[str] = None

    # Agreement (BIT → int 0/1 from frontend)
    AB04:                           Optional[int] = 0
    ABT06:                          Optional[int] = 0
    EL10:                           Optional[int] = 0
    ABS09:                          Optional[int] = 0
    Agreement_HandymanForm:         Optional[int] = 0
    Agreement_OtherText:            Optional[str] = ""
    ContractAttached:               Optional[int] = 0
    BuyerWarrantyMonths:            Optional[int] = None

    # End customer
    EndCustomerOrFacilityOwner:     Optional[str] = ""
    Install_site:                   Optional[str] = ""

    # Investigation
    InvestigationReportRequested:   Optional[int] = 0
    CurrentProductLocation:         Optional[str] = ""
    ReturnDeliveryAddress:          Optional[str] = ""

    # Installation (BIT → int)
    InstalledIndoors:               Optional[int] = 0
    InstalledOutdoors:              Optional[int] = 0
    HumidEnvironment:               Optional[int] = 0
    DryEnvironment:                 Optional[int] = 0
    DustyEnvironment:               Optional[int] = 0
    PermanentlyHeated:              Optional[int] = 0
    AmbientTempMinMax:              Optional[str] = ""
    AggressiveEnvironment:          Optional[int] = 0
    VibrationsPresent:              Optional[int] = 0

    # Compliance (BIT → int)
    InstalledToStandard:            Optional[int] = 0
    InstalledToInstructions:        Optional[int] = 0
    MaintainedToInstructions:       Optional[int] = 0

    # Documentation
    CircuitDiagramStatus:           Optional[str] = ""
    OperationLogStatus:             Optional[str] = ""

    # Status
    Claim_status:                   Optional[str] = "pending"


@app.post("/claims/submit")
def submit_claim(
    claim: ClaimData,
    credentials=Depends(security),
    db=Depends(get_db),
):
    cursor = db.cursor()

    # Convert empty date strings to None so SQL doesn't reject them
    def d(v):
        return v if v and str(v).strip() not in ("", "null", "None") else None

    cursor.execute("""
        INSERT INTO Claim_Table (
            Tech_id,
            Customer_name, Customer_phone, Customer_Address,
            Artnr, EAN, Serienum, Type_designation,
            Descfault, Add_info,
            Seller_name, Seller_contact, Seller_phone, Seller_mail,
            Complaint_date, Odatum, Faktnr, Levdat, Comm_date,
            AB04, ABT06, EL10, ABS09,
            Agreement_HandymanForm, Agreement_OtherText,
            ContractAttached, BuyerWarrantyMonths,
            EndCustomerOrFacilityOwner, Install_site,
            InvestigationReportRequested,
            CurrentProductLocation, ReturnDeliveryAddress,
            InstalledIndoors, InstalledOutdoors,
            HumidEnvironment, DryEnvironment, DustyEnvironment,
            PermanentlyHeated, AmbientTempMinMax,
            AggressiveEnvironment, VibrationsPresent,
            InstalledToStandard, InstalledToInstructions,
            MaintainedToInstructions,
            CircuitDiagramStatus, OperationLogStatus,
            Claim_status
        )
        OUTPUT INSERTED.claimid
        VALUES (
            ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?,
            ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?
        )
    """,
        claim.Tech_id,
        claim.Customer_name,  claim.Customer_phone,   claim.Customer_Address,
        claim.Artnr,          claim.EAN,              claim.Serienum,         claim.Type_designation,
        claim.Descfault,      claim.Add_info,
        claim.Seller_name,    claim.Seller_contact,   claim.Seller_phone,     claim.Seller_mail,
        d(claim.Complaint_date), d(claim.Odatum),     claim.Faktnr,           d(claim.Levdat), d(claim.Comm_date),
        claim.AB04,           claim.ABT06,            claim.EL10,             claim.ABS09,
        claim.Agreement_HandymanForm, claim.Agreement_OtherText,
        claim.ContractAttached,       claim.BuyerWarrantyMonths,
        claim.EndCustomerOrFacilityOwner, claim.Install_site,
        claim.InvestigationReportRequested,
        claim.CurrentProductLocation,     claim.ReturnDeliveryAddress,
        claim.InstalledIndoors,   claim.InstalledOutdoors,
        claim.HumidEnvironment,   claim.DryEnvironment,   claim.DustyEnvironment,
        claim.PermanentlyHeated,  claim.AmbientTempMinMax,
        claim.AggressiveEnvironment, claim.VibrationsPresent,
        claim.InstalledToStandard, claim.InstalledToInstructions, claim.MaintainedToInstructions,
        claim.CircuitDiagramStatus, claim.OperationLogStatus,
        claim.Claim_status,
    )

    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=500, detail="Insert failed — no claimid returned")

    db.commit()
    logger.info(f" Claim #{row[0]} saved — status: {claim.Claim_status}")
    return {"claimid": row[0], "status": claim.Claim_status}