from pathlib import Path
import json
import re
import io
import os

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import pytesseract


# ============================================================
# Paths
# ============================================================

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent

DB_PATH = BASE / "data" / "ingredients.json"
FRONTEND_PATH = ROOT / "frontend" / "index.html"


# ============================================================
# Load ingredient database
# ============================================================

try:
    with open(DB_PATH, "r", encoding="utf-8") as f:
        INGREDIENTS = json.load(f)
except Exception as e:
    INGREDIENTS = []
    print(f"WARNING: Could not load ingredient database: {e}")


# ============================================================
# FastAPI application
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
    for x in os.getenv("ALLOWED_ORIGINS", "").split(",")
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
# Ingredient matching
# ============================================================

def normalize(s: str) -> str:
    """
    Normalize text for ingredient matching.
    """

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


def split_ingredients(text: str):
    """
    Split OCR/input text into individual ingredients.
    """

    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Remove the heading if present
    text = re.sub(
        r"\b(ingredients?|contents?)\s*[:\-]?\s*",
        "",
        text,
        flags=re.I
    )

    # Split on common ingredient separators
    parts = re.split(
        r"[,;\n•]+",
        text
    )

    cleaned = []

    for p in parts:

        p = p.strip(
            " .:-"
        )

        if p:
            cleaned.append(p)

    return cleaned


def build_alias_map():
    """
    Build lookup map from ingredient names,
    aliases and INS/E-number codes.
    """

    m = {}

    for item in INGREDIENTS:

        names = (
            [item["name"]]
            + item.get("aliases", [])
            + item.get("codes", [])
        )

        for name in names:

            m[normalize(name)] = item

    return m


ALIAS_MAP = build_alias_map()


def match_ingredient(raw: str):
    """
    Match OCR text against the ingredient database.
    """

    n = normalize(raw)

    if not n:
        return None

    # Exact match
    if n in ALIAS_MAP:
        return ALIAS_MAP[n]

    # Match ingredient codes such as E330
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
# Routes
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
            400,
            "text is required"
        )

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

    return {
        "input": text,
        "ingredients": results
    }


# ============================================================
# OCR preprocessing
# ============================================================

def resize_for_ocr(image):
    """
    Resize image so that small ingredient text becomes
    easier for Tesseract to recognize.
    """

    height, width = image.shape[:2]

    # We want the smallest dimension to be reasonably large.
    target_min_dimension = 1600

    current_min = min(
        height,
        width
    )

    if current_min < target_min_dimension:

        scale = (
            target_min_dimension
            / current_min
        )

        # Don't make images unnecessarily huge
        scale = min(
            scale,
            3.0
        )

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    else:

        # A modest upscale can still help OCR
        image = cv2.resize(
            image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

    return image


def preprocess_for_ocr(image):
    """
    Generate multiple versions of the image.

    Different label photographs benefit from different
    preprocessing techniques.
    """

    # --------------------------------------------------------
    # 1. Resize
    # --------------------------------------------------------

    image = resize_for_ocr(
        image
    )

    # --------------------------------------------------------
    # 2. Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 3. CLAHE local contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # 4. Mild denoising
    # --------------------------------------------------------

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # --------------------------------------------------------
    # 5. Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    # --------------------------------------------------------
    # 6. Otsu threshold
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # --------------------------------------------------------
    # 7. Sharpened grayscale
    # --------------------------------------------------------

    blur = cv2.GaussianBlur(
        denoised,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.5,
        blur,
        -0.5,
        0
    )

    # --------------------------------------------------------
    # 8. Inverted adaptive threshold
    #
    # Useful when text is light on a dark package.
    # --------------------------------------------------------

    adaptive_inverse = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11
    )

    return [
        (
            "enhanced",
            enhanced
        ),
        (
            "adaptive",
            adaptive
        ),
        (
            "otsu",
            otsu
        ),
        (
            "sharpened",
            sharpened
        ),
        (
            "adaptive_inverse",
            adaptive_inverse
        )
    ]


# ============================================================
# OCR cleanup
# ============================================================

def clean_ocr_text(text):
    """
    Clean common OCR artifacts while preserving useful text.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Replace tabs
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

    # Normalize excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Ignore lines consisting only of symbols
        if re.fullmatch(
            r"[^A-Za-z0-9]+",
            line
        ):
            continue

        cleaned_lines.append(
            line
        )

    return "\n".join(
        cleaned_lines
    )


# ============================================================
# OCR quality scoring
# ============================================================

def score_ocr_text(text):
    """
    Estimate which OCR result is most useful.

    This is not an AI confidence score. It is a heuristic
    used to select between several Tesseract results.
    """

    if not text:
        return -999

    score = 0

    character_count = len(
        text
    )

    # Prefer useful amounts of text
    if character_count > 20:
        score += 5

    if character_count > 100:
        score += 5

    if character_count > 300:
        score += 5

    # Ratio of letters/numbers to total characters
    alphanumeric = sum(
        c.isalnum()
        for c in text
    )

    if character_count > 0:

        ratio = (
            alphanumeric
            / character_count
        )

        if ratio > 0.50:
            score += 5

        if ratio > 0.70:
            score += 5

    # Food-label vocabulary
    ingredient_words = [
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

    for word in ingredient_words:

        if word in lower:
            score += 10

    # Penalize excessive unusual characters
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
# Ingredient section extraction
# ============================================================

def extract_ingredients_section(text):
    """
    Attempt to identify the part of the OCR result
    containing the ingredient list.

    If no 'Ingredients' heading is found, return the
    complete OCR result.
    """

    if not text:
        return text

    pattern = re.compile(
        r"\bingredients?\s*[:\-]?",
        re.IGNORECASE
    )

    match = pattern.search(
        text
    )

    if not match:
        return text

    start = match.end()

    section = text[
        start:
    ].strip()

    # Common headings that indicate the ingredient
    # section has ended.
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

    for stop_pattern in stop_patterns:

        stop = re.search(
            stop_pattern,
            section,
            re.IGNORECASE
        )

        if stop:

            section = section[
                :stop.start()
            ]

    section = section.strip()

    # Don't return a tiny accidental match
    if len(section) < 20:
        return text

    return section


# ============================================================
# Main OCR function
# ============================================================

def perform_ocr(image):
    """
    Run multiple preprocessing methods and Tesseract
    configurations, then select the most useful result.
    """

    processed_images = preprocess_for_ocr(
        image
    )

    results = []

    # --------------------------------------------------------
    # Tesseract page segmentation modes
    #
    # 6  = Uniform block of text
    # 11 = Sparse text
    # 12 = Sparse text + OSD
    # --------------------------------------------------------

    psm_modes = [
        6,
        11,
        12
    ]

    for image_name, processed in processed_images:

        for psm in psm_modes:

            config = (
                f"--oem 3 --psm {psm} "
                "-c preserve_interword_spaces=1"
            )

            try:

                text = pytesseract.image_to_string(
                    processed,
                    config=config
                )

            except Exception as e:

                print(
                    f"Tesseract error "
                    f"({image_name}, PSM {psm}): {e}"
                )

                continue

            text = clean_ocr_text(
                text
            )

            score = score_ocr_text(
                text
            )

            results.append({
                "text": text,
                "score": score,
                "image": image_name,
                "psm": psm
            })

    if not results:
        return ""

    # Highest-scoring OCR result
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = results[0]["text"]

    # Try to isolate ingredient section
    ingredients = extract_ingredients_section(
        best
    )

    if ingredients:

        return ingredients

    return best


# ============================================================
# Image upload
# ============================================================

async def read_image(file: UploadFile):
    """
    Read uploaded image and convert it to OpenCV BGR format.
    """

    data = await file.read()

    if not data:
        raise HTTPException(
            400,
            "Uploaded file is empty"
        )

    try:

        pil_image = Image.open(
            io.BytesIO(data)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            400,
            "Uploaded file is not a valid image"
        )

    # PIL RGB -> NumPy
    rgb = np.array(
        pil_image
    )

    # RGB -> OpenCV BGR
    image = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    return image


# ============================================================
# OCR endpoint
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
            500,
            f"OCR failed: {e}"
        )

    return {
        "text": text,
        "ingredients": split_ingredients(
            text
        )
    }


# ============================================================
# OCR + ingredient analysis endpoint
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
            500,
            f"OCR failed: {e}"
        )

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

    return {
        "ocr_text": text,
        "ingredients": results
    }


# ============================================================
# Application entry point
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
