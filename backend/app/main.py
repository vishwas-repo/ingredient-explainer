from pathlib import Path
import io
import json
import os
import re

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
    INGREDIENTS = []
    print(f"WARNING: Could not load ingredient database: {e}")


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

else:

    # Useful during development.
    # For production, set ALLOWED_ORIGINS in Render.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(s: str) -> str:

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
# INGREDIENT DATABASE LOOKUP
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

    # Exact match
    if n in ALIAS_MAP:
        return ALIAS_MAP[n]

    # E-number / code matching
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

    # Partial match
    for alias, item in sorted(
        ALIAS_MAP.items(),
        key=lambda x: len(x[0]),
        reverse=True
    ):

        if len(alias) >= 4 and alias in n:
            return item

    return None


# ============================================================
# INGREDIENT SPLITTING
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

    # Convert bullets to separators
    text = text.replace(
        "•",
        ","
    )

    # Split ingredients
    parts = re.split(
        r"[,;\n]+",
        text
    )

    result = []

    for part in parts:

        part = part.strip(
            " .:-"
        )

        if part:
            result.append(part)

    return result


# ============================================================
# ANALYZE TEXT
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
# BASIC ROUTES
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


# ============================================================
# ANALYZE TEXT ENDPOINT
# ============================================================

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
# IMAGE READING
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

    # Convert PIL RGB → NumPy
    rgb = np.array(
        pil_image
    )

    # RGB → OpenCV BGR
    image = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return image


# ============================================================
# FAST IMAGE RESIZE
# ============================================================

def resize_for_ocr(image):

    height, width = image.shape[:2]

    # Don't make huge phone photos unnecessarily large.
    #
    # Target width around 1800 px is generally sufficient
    # for ingredient-label OCR.

    max_width = 1800

    if width > max_width:

        scale = (
            max_width
            / float(width)
        )

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    elif width < 1000:

        scale = (
            1400
            / float(width)
        )

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    return image


# ============================================================
# FAST OCR PREPROCESSING
# ============================================================

def preprocess_for_ocr(image):

    image = resize_for_ocr(
        image
    )

    # Grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE improves text against uneven package backgrounds.
    #
    # This is considerably cheaper than running several
    # denoising/thresholding pipelines.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # Mild sharpening
    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        1.2
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.35,
        blur,
        -0.35,
        0
    )

    return sharpened


# ============================================================
# OCR TEXT CLEANUP
# ============================================================

def clean_ocr_text(text):

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
        "¢": "c",
        "©": "C",
        "®": "",
        "™": "",
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

    cleaned = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Ignore lines containing only symbols
        if re.fullmatch(
            r"[^A-Za-z0-9]+",
            line
        ):
            continue

        cleaned.append(
            line
        )

    return "\n".join(
        cleaned
    )


# ============================================================
# OCR QUALITY CHECK
# ============================================================

def score_ocr_text(text):

    if not text:
        return -999

    score = 0

    length = len(text)

    # Reasonable amount of text
    if length >= 20:
        score += 5

    if length >= 100:
        score += 5

    if length >= 250:
        score += 5

    # Letters/numbers vs garbage
    alphanumeric = sum(
        c.isalnum()
        for c in text
    )

    if length:

        ratio = (
            alphanumeric
            / length
        )

        if ratio > 0.50:
            score += 5

        if ratio > 0.70:
            score += 5

    # Useful food-label vocabulary
    keywords = [
        "ingredient",
        "ingredients",
        "sugar",
        "salt",
        "milk",
        "wheat",
        "flour",
        "oil",
        "water",
        "protein",
        "starch",
        "preservative",
        "flavour",
        "flavor",
        "spice",
        "acid",
        "citric",
        "sodium",
        "calcium",
        "vitamin",
        "extract",
        "emulsifier",
        "colour",
        "color"
    ]

    lower = text.lower()

    for word in keywords:

        if word in lower:
            score += 10

    # Penalize excessive strange symbols
    garbage = len(
        re.findall(
            r"[^A-Za-z0-9\s,.;:()/%&+\-'\"[\]]",
            text
        )
    )

    if garbage > 10:
        score -= 5

    if garbage > 30:
        score -= 10

    return score


# ============================================================
# INGREDIENT SECTION EXTRACTION
# ============================================================

def extract_ingredients_section(text):

    if not text:
        return ""

    # Sometimes OCR recognizes "Ingredients" imperfectly.
    pattern = re.compile(
        r"\bingredients?\s*[:\-]?",
        re.IGNORECASE
    )

    match = pattern.search(
        text
    )

    if not match:

        return text

    section = text[
        match.end():
    ].strip()

    # Common headings that usually follow the ingredient list
    stop_patterns = [
        r"\n\s*allergen",
        r"\n\s*contains",
        r"\n\s*nutrition",
        r"\n\s*nutritional",
        r"\n\s*directions",
        r"\n\s*storage",
        r"\n\s*manufactured",
        r"\n\s*distributed",
        r"\n\s*net\s*weight",
        r"\n\s*serving\s*size"
    ]

    for pattern in stop_patterns:

        stop = re.search(
            pattern,
            section,
            re.IGNORECASE
        )

        if stop:

            section = section[
                :stop.start()
            ]

    section = section.strip()

    if len(section) >= 20:
        return section

    return text


# ============================================================
# FAST OCR
# ============================================================

def perform_ocr(image):

    # --------------------------------------------------------
    # FAST PREPROCESSING
    # --------------------------------------------------------

    processed = preprocess_for_ocr(
        image
    )

    # --------------------------------------------------------
    # PASS 1
    #
    # PSM 6 works well when the user photographs the
    # ingredient paragraph fairly closely.
    # --------------------------------------------------------

    config = (
        "--oem 3 --psm 6 "
        "-c preserve_interword_spaces=1"
    )

    text = pytesseract.image_to_string(
        processed,
        config=config
    )

    text = clean_ocr_text(
        text
    )

    score = score_ocr_text(
        text
    )

    # --------------------------------------------------------
    # FAST PATH
    #
    # If OCR looks good, don't run Tesseract again.
    # --------------------------------------------------------

    if score >= 15:

        return extract_ingredients_section(
            text
        )

    # --------------------------------------------------------
    # PASS 2
    #
    # Only run when the first result looks poor.
    #
    # PSM 11 is useful for labels where text is separated
    # into multiple areas.
    # --------------------------------------------------------

    config = (
        "--oem 3 --psm 11 "
        "-c preserve_interword_spaces=1"
    )

    text2 = pytesseract.image_to_string(
        processed,
        config=config
    )

    text2 = clean_ocr_text(
        text2
    )

    score2 = score_ocr_text(
        text2
    )

    if score2 > score:

        text = text2

    return extract_ingredients_section(
        text
    )


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
# OCR + ANALYSIS ENDPOINT
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
