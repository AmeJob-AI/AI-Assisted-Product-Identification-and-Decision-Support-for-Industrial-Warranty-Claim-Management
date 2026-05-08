
import logging
import pickle
import numpy as np
import faiss
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# PATHS  —  same structure as main.py
# ═══════════════════════════════════════════════════════════════════
BASE_DIR    = Path(__file__).resolve().parent.parent.parent   # → THESIS_LATEST/
IMG_DIR     = BASE_DIR / "Final_Dataset"
FAISS_FILE  = BASE_DIR / "faiss_index_Aug" / "product.index"
MAP_FILE    = BASE_DIR / "faiss_index_Aug" / "item_map.pkl"

AUG_PER_IMAGE = 7    # augmented copies per image (done in RAM, not saved to disk)


# ═══════════════════════════════════════════════════════════════════
# DETECT ALBUMENTATIONS VERSION
# Handles API differences between older and newer albumentations.
# ═══════════════════════════════════════════════════════════════════
def _get_albumentations_version():
    try:
        import albumentations as A
        ver = tuple(int(x) for x in A.__version__.split(".")[:2])
        return ver
    except Exception:
        return (0, 0)


# ═══════════════════════════════════════════════════════════════════
# IN-MEMORY AUGMENTATION
# Images are augmented in RAM and discarded after embedding.
# No extra disk space is used.
# ═══════════════════════════════════════════════════════════════════
def get_augmentation_pipeline():
    """
    Version-safe albumentations pipeline (v1.x + v2.x compatible)
    Fixes:
      - ImageCompression API mismatch
      - Downscale interpolation warning
      - GaussNoise API differences
    """
    try:
        import albumentations as A
        import cv2

        ver = _get_albumentations_version()
        logger.info(f"albumentations version: {'.'.join(str(v) for v in ver)}")

        # ─────────────────────────────────────────────
        # GaussNoise (v1 vs v2)
        # ─────────────────────────────────────────────
        if ver >= (2, 0):
            gauss_noise = A.GaussNoise(std_range=(0.02, 0.08), p=0.5)
        else:
            gauss_noise = A.GaussNoise(var_limit=(0.0004, 0.0064), p=0.5)

        # ─────────────────────────────────────────────
        # Downscale (v1 vs v2 + interpolation fix)
        # ─────────────────────────────────────────────
        if ver >= (2, 0):
            downscale = A.Downscale(
                scale=(0.5, 0.85),
                interpolation=cv2.INTER_LINEAR,
                p=0.3
            )
        else:
            downscale = A.Downscale(
                scale_min=0.5,
                scale_max=0.85,
                interpolation=cv2.INTER_LINEAR,
                p=0.3
            )

        # ─────────────────────────────────────────────
        # ImageCompression (FIXED ERROR HERE)
        # ─────────────────────────────────────────────
        try:
            # New API (>=1.4)
            compression = A.ImageCompression(quality_range=(50, 90), p=0.3)
        except TypeError:
            # Old API fallback
            compression = A.ImageCompression(quality_lower=50, quality_upper=90, p=0.3)

        # ─────────────────────────────────────────────
        # FULL PIPELINE
        # ─────────────────────────────────────────────
        pipeline = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.8
            ),

            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=30,
                val_shift_limit=30,
                p=0.5
            ),

            gauss_noise,

            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
            ], p=0.4),

            A.Rotate(
                limit=25,
                border_mode=cv2.BORDER_REFLECT,
                p=0.6
            ),

            A.Perspective(
                scale=(0.04, 0.12),
                p=0.4
            ),

            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=3,
                p=0.3
            ),

            A.CLAHE(
                clip_limit=4.0,
                p=0.3
            ),

            downscale,
            compression,
        ])

        logger.info(" Augmentation pipeline ready")

        # Optional debug (very useful)
        logger.info(
            "Transforms: " +
            ", ".join([type(t).__name__ for t in pipeline.transforms])
        )

        return pipeline

    except ImportError:
        logger.warning(" albumentations not installed — augmentation DISABLED")
        logger.warning("Install: pip install albumentations opencv-python-headless")
        return None

    except Exception as ex:
        logger.error(f" Failed to build augmentation pipeline: {ex}")
        logger.warning("Augmentation DISABLED — continuing without it")
        return None
    


def augment_in_memory(pil_img: Image.Image, pipeline, n: int) -> list:
    """
    Generate n augmented versions of an image in RAM.

    Changes vs original:
      - Logs the FIRST augmentation failure per call (was silent before).
      - Tracks how many augmentations failed so the caller can report them.
      - Still falls back to original image on failure — safe and non-crashing.

    Returns:
        list of PIL Images (length == n, always)
    """
    import cv2

    img_bgr  = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    results  = []
    failures = 0

    for _ in range(n):
        try:
            aug     = pipeline(image=img_bgr)["image"]
            aug_pil = Image.fromarray(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
            results.append(aug_pil)
        except Exception as ex:
            failures += 1
            if failures == 1:
                # Log only the first failure to avoid log spam
                logger.warning(f"   ⚠️  Augmentation step failed (using original): {ex}")
            results.append(pil_img)   # fallback to original

    return results, failures


# ═══════════════════════════════════════════════════════════════════
# EMBEDDING
# ═══════════════════════════════════════════════════════════════════
def embed(img: Image.Image, processor, model, device) -> np.ndarray:
    """Embed a single PIL image using DINOv2 and L2-normalise the vector."""
    with torch.no_grad():
        inp = processor(images=img, return_tensors="pt").to(device)
        out = model(**inp)
        e   = out.last_hidden_state[:, 0, :].cpu().numpy()
    e = e / (np.linalg.norm(e) + 1e-8)
    return e.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():

    # ── Check image folder ────────────────────────────────────────
    logger.info(f"Image folder: {IMG_DIR}")

    if not IMG_DIR.exists():
        logger.error(f" Image folder not found: {IMG_DIR}")
        logger.error("   Check that Final_Dataset/ exists in THESIS_LATEST/")
        return

    image_files = (list(IMG_DIR.glob("*.jpg")) +
                   list(IMG_DIR.glob("*.jpeg")) +
                   list(IMG_DIR.glob("*.png")))

    if not image_files:
        logger.error(f" No images found in {IMG_DIR}")
        return

    logger.info(f"Found: {len(image_files)} images")

    # ── Setup augmentation ────────────────────────────────────────
    aug_pipeline       = get_augmentation_pipeline()
    aug_count          = AUG_PER_IMAGE if aug_pipeline else 0
    total_expected     = len(image_files) * (1 + aug_count)
    total_aug_failures = 0   # ← NEW: track augmentation failures globally

    logger.info(f"Augmentation: {aug_count} per image (in-memory, no disk space used)")
    logger.info(f"Total embeddings to build: {total_expected}")
    logger.info("")

    # ── Load DINOv2 ───────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info("Loading DINOv2 Large ...")

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    model     = AutoModel.from_pretrained("facebook/dinov2-large").to(device)
    model.eval()

    logger.info(" DINOv2 loaded!")
    logger.info("-" * 50)
    logger.info("Processing images ... (be patient)")
    logger.info("-" * 50)

    # ── Process all images ────────────────────────────────────────
    embeddings   = []
    item_numbers = []
    failed       = 0

    for i, img_path in enumerate(image_files):
        try:
            item_no  = img_path.stem
            original = Image.open(img_path).convert("RGB")

            # Embed original image
            e = embed(original, processor, model, device)
            embeddings.append(e)
            item_numbers.append(item_no)

            # Embed augmented versions (in memory only)
            if aug_pipeline:
                # ── FIX: augment_in_memory now returns (results, failures) ──
                aug_images, aug_failures = augment_in_memory(
                    original, aug_pipeline, aug_count
                )
                total_aug_failures += aug_failures

                for aug_img in aug_images:
                    e = embed(aug_img, processor, model, device)
                    embeddings.append(e)
                    item_numbers.append(item_no)   # same item_no for all versions

            del original   # free memory

        except Exception as ex:
            logger.warning(f"  Skipping {img_path.name}: {ex}")
            failed += 1

        # Progress every 50 images
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            pct = (i + 1) / len(image_files) * 100
            logger.info(
                f"  {i+1}/{len(image_files)} images ({pct:.0f}%) "
                f"→ {len(embeddings)} embeddings"
                + (f"  [aug failures so far: {total_aug_failures}]"
                   if total_aug_failures else "")
            )

    if not embeddings:
        logger.error(" No embeddings created. Check your image files.")
        return

    # ── Sanity check ──────────────────────────────────────────────
    # ── NEW: warn if we got far fewer embeddings than expected ────
    if len(embeddings) < total_expected * 0.9:
        logger.warning(
            f"  Expected ~{total_expected} embeddings but only got {len(embeddings)}. "
            "Augmentation may have been partially failing — check warnings above."
        )

    # ── Build FAISS index ─────────────────────────────────────────
    logger.info("-" * 50)
    logger.info(f"Building FAISS index for {len(embeddings)} embeddings ...")

    matrix = np.vstack(embeddings).astype(np.float32)
    dim    = matrix.shape[1]   # 1024 for DINOv2-Large
    n      = len(embeddings)

    # Auto-select index type based on size
    if n < 10_000:
        # Exact search — perfect for 14 000 embeddings
        index      = faiss.IndexFlatL2(dim)
        index_type = "FlatL2 (exact search)"

    elif n < 500_000:
        # Approximate cluster search — for 10k to 500k
        n_clusters = min(4096, n // 39)
        n_clusters = max(n_clusters, 10)
        quantizer  = faiss.IndexFlatL2(dim)
        index      = faiss.IndexIVFFlat(quantizer, dim, n_clusters)
        logger.info(f"Training IVFFlat with {n_clusters} clusters ...")
        index.train(matrix)
        index.nprobe = min(64, n_clusters)
        index_type   = f"IVFFlat (clusters={n_clusters})"

    else:
        # Compressed — for 500k to millions
        n_clusters   = 8192
        m_subvectors = 64
        quantizer    = faiss.IndexFlatL2(dim)
        index        = faiss.IndexIVFPQ(quantizer, dim, n_clusters, m_subvectors, 8)
        logger.info(f"Training IVFPQ ({n_clusters} clusters) ...")
        index.train(matrix)
        index.nprobe = 128
        index_type   = f"IVFPQ (compressed, clusters={n_clusters})"

    index.add(matrix)

    # ── Save to disk ──────────────────────────────────────────────
    FAISS_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_FILE))
    with open(str(MAP_FILE), "wb") as f:
        pickle.dump(item_numbers, f)

    size_mb = FAISS_FILE.stat().st_size / 1024 / 1024

    logger.info("=" * 50)
    logger.info(" FAISS INDEX BUILT SUCCESSFULLY!")
    logger.info(f"   Original images   : {len(image_files)}")
    logger.info(f"   Aug per image     : {aug_count} (in RAM, no disk space used)")
    logger.info(f"   Total embeddings  : {index.ntotal}")
    logger.info(f"   Aug step failures : {total_aug_failures}")   # ← NEW
    logger.info(f"   Failed images     : {failed}")
    logger.info(f"   Index type        : {index_type}")
    logger.info(f"   Index file size   : {size_mb:.1f} MB")
    logger.info(f"   Saved to          : {FAISS_FILE}")
    logger.info("=" * 50)
    logger.info("")
    logger.info("NEXT STEP:")
    logger.info("   uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    main()