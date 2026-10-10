from pathlib import Path
import json
import re
import difflib
import io
import os

import cv2
import numpy as np
import pytesseract

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent

DB_PATH = BASE / "data" / "ingredients.json"
FRONTEND_PATH = ROOT / "frontend" / "index.html"


# ============================================================
# LOAD INGREDIENT DATABASE
# ============================================================

try:
    with open(DB_PATH, "r", encoding="utf-8") as f:
        INGREDIENTS = json.load(f)

except Exception as e:
    print(f"WARNING: Could not load ingredient database: {e}")
    INGREDIENTS = []


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Ingredient Explainer API",
    version="0.3"
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    x.strip()
    for x in os.getenv(
        "ALLOWED_ORIGINS",
        ""
    ).split(",")
    if x.strip()
]

if allowed_origins:

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(s: str) -> str:

    if not s:
        return ""

    s = s.lower().strip()

    s = re.sub(
        r"[^a-z0-9]+",
        " ",
        s
    )

    return re.sub(
        r"\s+",
        " ",
        s
    ).strip()


# ============================================================
# SPLIT INGREDIENTS
# ============================================================

def split_ingredients(text: str):

    if not text:
        return []

    text = text.strip()

    # Remove heading
    text = re.sub(
        r"\b(ingredients?|contents?)\s*[:\-]?\s*",
        "",
        text,
        flags=re.I
    )

    # Remove common OCR punctuation artifacts
    text = text.replace(
        "•",
        ","
    )

    # Split
    parts = re.split(
        r"[,;\n]+",
        text
    )

    result = []

    for part in parts:

        part = part.strip(
            " .:-|"
        )

        if part:
            result.append(part)

    return result


# ============================================================
# INGREDIENT DATABASE
# ============================================================

def build_alias_map():

    result = {}

    for item in INGREDIENTS:

        names = (
            [item.get("name", "")]
            + item.get("aliases", [])
            + item.get("codes", [])
        )

        for name in names:

            if name:

                result[
                    normalize(name)
                ] = item

    return result


ALIAS_MAP = build_alias_map()


def match_ingredient(raw: str):

    n = normalize(raw)

    if not n:
        return None

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if n in ALIAS_MAP:
        return ALIAS_MAP[n]

    # --------------------------------------------------------
    # Code / E-number matching
    # --------------------------------------------------------

    code = re.sub(
        r"\s+",
        "",
        n
    ).upper()

    for item in INGREDIENTS:

        for c in item.get("codes", []):

            if re.sub(
                r"\s+",
                "",
                c
            ).upper() == code:

                return item

    # --------------------------------------------------------
    # Partial match
    # --------------------------------------------------------

    for alias, item in sorted(
        ALIAS_MAP.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        if len(alias) >= 4 and alias in n:
            return item

    return None


# ============================================================
# ANALYZE INGREDIENT TEXT
# ============================================================

def analyze_ingredient_text(text: str):

    parts = split_ingredients(
        text
    )

    results = []

    for raw in parts:

        item = match_ingredient(
            raw
        )

        results.append({
            "raw": raw,
            "matched": item is not None,
            "ingredient": item
        })

    return results


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():

    if FRONTEND_PATH.exists():

        return FileResponse(
            FRONTEND_PATH
        )

    return {
        "name": "Ingredient Explainer API",
        "version": "0.3"
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "0.3"
    }


@app.get("/ingredients")
def ingredients():

    return {
        "count": len(INGREDIENTS),
        "items": INGREDIENTS
    }


@app.post("/analyze-text")
def analyze_text(payload: dict):

    text = payload.get(
        "text",
        ""
    )

    if not isinstance(
        text,
        str
    ) or not text.strip():

        raise HTTPException(
            status_code=400,
            detail="text is required"
        )

    return {
        "input": text,
        "ingredients": analyze_ingredient_text(
            text
        )
    }


# ============================================================
# READ UPLOADED IMAGE
# ============================================================

async def read_image(
    file: UploadFile
):

    data = await file.read()

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    try:

        pil_image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image"
        )

    rgb = np.array(
        pil_image
    )

    image = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return image


# ============================================================
# RESIZE
# ============================================================

def resize_for_ocr(image, target_width=1600):
    """Keep OCR input bounded for speed while preserving small label text."""
    height, width = image.shape[:2]
    if width > target_width:
        scale = target_width / float(width)
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    elif width < 1000:
        scale = min(1.5, 1200 / float(width))
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)
    return image


# ============================================================
# PREPROCESSING
# ============================================================

def enhance_green_channel(image):

    """
    Extract and enhance the green channel.

    Yellow text on red/orange packaging generally has
    substantially better contrast in the green channel
    than in normal grayscale.
    """

    b, g, r = cv2.split(
        image
    )

    gray = g

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    # Mild sharpening
    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.0
    )

    gray = cv2.addWeighted(
        gray,
        1.5,
        blur,
        -0.5,
        0
    )

    return gray


def make_yellow_text_mask(image):

    """
    Create a mask for yellow/yellow-green text.

    This is particularly useful for labels with yellow
    lettering on red/orange backgrounds.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [12, 40, 70],
        dtype=np.uint8
    )

    upper = np.array(
        [50, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# FIND "INGREDIENTS"
# ============================================================

def is_ingredients_word(text):

    if not text:
        return False

    cleaned = re.sub(
        r"[^A-Za-z]",
        "",
        text
    ).upper()

    # Exact / near-exact forms
    possible = [
        "INGREDIENT",
        "INGREDIENTS",
        "INGREDlENT",
        "INGREDlENTS",
        "INGREDENTS",
        "INGREDIENTS"
    ]

    for word in possible:

        if word in cleaned:
            return True

    # Handle OCR errors with high similarity
    if (
        "INGRED" in cleaned
        and len(cleaned) >= 7
    ):
        return True

    return False


def find_ingredients_word(
    image
):

    """
    Locate the INGREDIENTS word using OCR bounding boxes.

    Returns:
        x, y, width, height

    or None if not found.
    """

    data = pytesseract.image_to_data(
        image,
        config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT
    )

    count = len(
        data["text"]
    )

    best = None

    for i in range(count):

        text = data["text"][i].strip()

        if not text:
            continue

        if not is_ingredients_word(
            text
        ):
            continue

        try:

            confidence = float(
                data["conf"][i]
            )

        except Exception:

            confidence = 0

        x = int(
            data["left"][i]
        )

        y = int(
            data["top"][i]
        )

        w = int(
            data["width"][i]
        )

        h = int(
            data["height"][i]
        )

        candidate = (
            confidence,
            x,
            y,
            w,
            h
        )

        if (
            best is None
            or confidence > best[0]
        ):

            best = candidate

    if best is None:
        return None

    return (
        best[1],
        best[2],
        best[3],
        best[4]
    )


# ============================================================
# FIND INGREDIENT REGION
# ============================================================

def locate_ingredient_region(image):
    """
    Find the heading anywhere in the image with one OCR layout pass.
    Returns x, y, width, height in the supplied image's coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Avoid expensive multi-method preprocessing for the heading pass.
    data = pytesseract.image_to_data(
        gray,
        config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT,
        timeout=12,
    )

    best = None
    for i, raw in enumerate(data.get("text", [])):
        if not is_ingredients_word(raw.strip()):
            continue
        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError, KeyError):
            confidence = 0
        candidate = (
            confidence,
            int(data["left"][i]), int(data["top"][i]),
            int(data["width"][i]), int(data["height"][i])
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return None
    return best[1], best[2], best[3], best[4]


# ============================================================
# CROP INGREDIENT REGION
# ============================================================

def crop_ingredient_region(image, location):
    h, w = image.shape[:2]
    x, y, word_w, word_h = location

    # Ingredient lists may be in any part of the package. Capture a broad
    # area below and to the right of the heading without hard-coded 950 px.
    x1 = max(0, x - int(w * 0.015))
    x2 = min(w, max(x + int(w * 0.65), x + word_w + int(w * 0.10)))
    y1 = max(0, y - int(word_h * 0.8))
    y2 = min(h, y + max(int(h * 0.42), int(word_h * 12)))
    return image[y1:y2, x1:x2]


# ============================================================
# OCR CLEANUP
# ============================================================

def clean_ocr_text(
    text
):

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.replace(
        "\t",
        " "
    )

    # Common OCR artifacts
    replacements = {
        "®": "",
        "™": "",
        "©": "",
        "¢": "c",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Normalize spaces
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if re.fullmatch(
            r"[^A-Za-z0-9]+",
            line
        ):
            continue

        lines.append(
            line
        )

    return "\n".join(
        lines
    )


# ============================================================
# EXTRACT INGREDIENT SECTION
# ============================================================

def extract_ingredients_section(
    text
):

    if not text:
        return ""

    text = clean_ocr_text(
        text
    )

    # --------------------------------------------------------
    # Locate INGREDIENTS heading
    # --------------------------------------------------------

    match = re.search(
        r"ingredients?\s*[:\-]?",
        text,
        flags=re.I
    )

    if match:

        text = text[
            match.end():
        ].strip()

    # --------------------------------------------------------
    # Remove everything after common following headings
    # --------------------------------------------------------

    stop_patterns = [
        r"\n\s*allergen\s+advice",
        r"\n\s*allergen",
        r"\n\s*contains\s+",
        r"\n\s*nutrition",
        r"\n\s*nutritional",
        r"\n\s*directions",
        r"\n\s*storage",
        r"\n\s*manufactured",
        r"\n\s*distributed",
        r"\n\s*net\s+weight",
        r"\n\s*serving\s+size"
    ]

    for pattern in stop_patterns:

        stop = re.search(
            pattern,
            text,
            flags=re.I
        )

        if stop:

            text = text[
                :stop.start()
            ]

    return clean_ocr_text(
        text
    ).strip()


# ============================================================
# DATABASE-ASSISTED OCR CORRECTION
# ============================================================

def _ingredient_terms():
    """Build a compact list of database names/aliases once at startup."""
    terms = []
    for item in INGREDIENTS:
        for value in ([item.get("name", "")] + item.get("aliases", []) + item.get("codes", [])):
            value = str(value or "").strip()
            normalized = normalize(value)
            if normalized:
                terms.append((normalized, value, item.get("name", value)))
    # Longest terms first; retain database terminology.
    return sorted(set(terms), key=lambda x: len(x[0]), reverse=True)


INGREDIENT_TERMS = _ingredient_terms()


def correct_ocr_with_database(text):
    """
    Correct clear OCR misspellings using ingredients.json only.
    Operates on comma/semicolon/newline-delimited ingredient phrases and
    preserves surrounding qualifiers. It deliberately avoids replacing a
    longer phrase (for example 'soy sauce') with a shorter database term.
    """
    if not text or not INGREDIENT_TERMS:
        return text

    output = []
    for line in text.splitlines():
        pieces = re.split(r"([,;])", line)
        for idx in range(0, len(pieces), 2):
            phrase = pieces[idx]
            stripped = phrase.strip()
            if not stripped:
                continue

            # Keep bracketed percentages, quantities, and other qualifiers.
            core = re.sub(r"\([^)]*\)|\b\d+(?:\.\d+)?\s*%?", " ", stripped)
            core_norm = normalize(core)
            if not core_norm or len(core_norm) < 3:
                continue

            # Exact aliases anywhere in a phrase are already good; don't
            # collapse the phrase to a shorter ingredient name.
            if any(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", core_norm)
                   for term, _, _ in INGREDIENT_TERMS):
                continue

            # Fuzzy correction is intentionally conservative and token-count
            # matched to avoid false positives on compound ingredients.
            tokens = core_norm.split()
            if len(tokens) > 3:
                continue
            candidates = [
                (difflib.SequenceMatcher(None, core_norm, term).ratio(), term, display, canonical)
                for term, display, canonical in INGREDIENT_TERMS
                if len(term.split()) == len(tokens) and abs(len(term) - len(core_norm)) <= 3
            ]
            if not candidates:
                continue
            ratio, term, display, canonical = max(candidates, key=lambda x: x[0])
            if ratio < 0.86:
                continue

            # Replace only the core phrase, retaining punctuation/qualifiers
            # by replacing the phrase only when it is essentially just that term.
            if len(normalize(stripped).split()) == len(tokens) and len(normalize(stripped)) <= len(term) + 5:
                phrase = re.sub(re.escape(stripped), canonical, phrase, count=1, flags=re.I)
                pieces[idx] = phrase

        output.append("".join(pieces))
    return "\n".join(output)


# ============================================================
# OCR SCORE
# ============================================================

def score_ocr_text(
    text
):

    if not text:
        return -999

    score = 0

    lower = text.lower()

    # Strong signal
    if "ingredient" in lower:
        score += 30

    # Food-related vocabulary
    keywords = [
        "sunflower",
        "oil",
        "chilli",
        "chillies",
        "garlic",
        "onion",
        "celery",
        "soy",
        "sauce",
        "salt",
        "sugar",
        "spice",
        "flour",
        "wheat",
        "milk",
        "starch",
        "preservative",
        "flavour",
        "flavor",
        "condiment"
    ]

    for word in keywords:

        if word in lower:
            score += 5

    # Useful text length
    if len(text) > 30:
        score += 5

    if len(text) > 100:
        score += 5

    # Alphanumeric ratio
    alphanumeric = sum(
        c.isalnum()
        for c in text
    )

    if len(text):

        ratio = (
            alphanumeric
            / len(text)
        )

        if ratio > 0.50:
            score += 5

        if ratio > 0.70:
            score += 5

    return score


# ============================================================
# OCR INGREDIENT REGION
# ============================================================

def ocr_ingredient_region(crop):
    """Run one primary OCR pass, with one thresholded retry only if needed."""
    if crop is None or crop.size == 0:
        return ""

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Upscale only small crops: helps tiny printed text without bloating all OCR.
    if gray.shape[1] < 1100:
        scale = min(1.7, 1100 / float(gray.shape[1]))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold is useful for uneven package lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    text = pytesseract.image_to_string(
        enhanced,
        config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
        timeout=15,
    )
    text = clean_ocr_text(text)

    # Only retry when the primary pass yielded very little useful text.
    if len(re.sub(r"[^A-Za-z0-9]", "", text)) < 18:
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 11
        )
        retry = pytesseract.image_to_string(
            binary,
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
            timeout=15,
        )
        retry = clean_ocr_text(retry)
        if score_ocr_text(retry) > score_ocr_text(text):
            text = retry

    return text


def perform_ocr(image):
    """
    Locate the Ingredients heading anywhere in the image, OCR a targeted crop,
    then use the ingredient database for conservative OCR autocorrection.
    """
    image = resize_for_ocr(image)
    location = locate_ingredient_region(image)

    if location is not None:
        crop = crop_ingredient_region(image, location)
        text = ocr_ingredient_region(crop)
        ingredients = extract_ingredients_section(text)
        if len(ingredients) >= 12 and len(ingredients.split()) >= 2:
            return correct_ocr_with_database(ingredients)

    # Fallback: one OCR pass on the full image (not just its lower half), so
    # packages with the ingredient panel near the top or middle still work.
    fallback = ocr_ingredient_region(image)
    extracted = extract_ingredients_section(fallback)
    result = extracted if extracted else fallback
    return correct_ocr_with_database(result)


# ============================================================
# OCR ENDPOINT
# ============================================================

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...)
):

    image = await read_image(
        file
    )

    try:

        text = perform_ocr(
            image
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {e}"
        )

    return {
        "text": text,
        "ingredients": split_ingredients(
            text
        )
    }


# ============================================================
# OCR + ANALYZE ENDPOINT
# ============================================================

@app.post("/ocr-and-analyze")
async def ocr_and_analyze(
    file: UploadFile = File(...)
):

    image = await read_image(
        file
    )

    try:

        text = perform_ocr(
            image
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {e}"
        )

    results = analyze_ingredient_text(
        text
    )

    return {
        "ocr_text": text,
        "ingredients": results
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        )
    )
