from pathlib import Path
import json
import re
import io
import os
import difflib

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

else:

    # Development/default configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ============================================================
# NORMALIZATION
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
# BUILD INGREDIENT DICTIONARY
# ============================================================

def build_alias_map():

    result = {}

    for item in INGREDIENTS:

        if not isinstance(item, dict):
            continue

        names = []

        name = item.get(
            "name",
            ""
        )

        if name:
            names.append(name)

        aliases = item.get(
            "aliases",
            []
        )

        if isinstance(
            aliases,
            list
        ):
            names.extend(
                aliases
            )

        codes = item.get(
            "codes",
            []
        )

        if isinstance(
            codes,
            list
        ):
            names.extend(
                codes
            )

        for value in names:

            if not isinstance(
                value,
                str
            ):
                continue

            n = normalize(
                value
            )

            if n:

                result[n] = item

    return result


ALIAS_MAP = build_alias_map()


# ============================================================
# OCR CORRECTION DICTIONARY
# ============================================================

def build_ocr_dictionary():

    """
    Build a dictionary of words/phrases that are likely to
    appear in ingredient lists.

    We use:
        - ingredient names
        - aliases
        - ingredient codes

    Multi-word ingredient names are retained as phrases.
    Individual words are also extracted for word-level OCR
    correction.
    """

    phrases = set()
    words = set()

    for item in INGREDIENTS:

        if not isinstance(item, dict):
            continue

        values = []

        name = item.get(
            "name",
            ""
        )

        if name:
            values.append(name)

        aliases = item.get(
            "aliases",
            []
        )

        if isinstance(
            aliases,
            list
        ):
            values.extend(
                aliases
            )

        codes = item.get(
            "codes",
            []
        )

        if isinstance(
            codes,
            list
        ):
            values.extend(
                codes
            )

        for value in values:

            if not isinstance(
                value,
                str
            ):
                continue

            normalized = normalize(
                value
            )

            if not normalized:
                continue

            phrases.add(
                normalized
            )

            for word in normalized.split():

                # Ignore very short words because correcting
                # "oil" or "salt" from a single bad character
                # can easily create false positives.
                if len(word) >= 4:

                    words.add(
                        word
                    )

    return (
        sorted(
            phrases,
            key=len,
            reverse=True
        ),
        sorted(
            words,
            key=len,
            reverse=True
        )
    )


OCR_DICTIONARY_PHRASES, OCR_DICTIONARY_WORDS = (
    build_ocr_dictionary()
)


# ============================================================
# COMMON FOOD-LABEL WORDS
# ============================================================
#
# These are useful even if they are not present in the
# current ingredients.json.
#
# They are deliberately limited to common label vocabulary.
# ============================================================

COMMON_FOOD_WORDS = {
    "ingredients",
    "ingredient",
    "refined",
    "sunflower",
    "oil",
    "red",
    "chilli",
    "chillies",
    "garlic",
    "onion",
    "celery",
    "soy",
    "sauce",
    "iodized",
    "iodised",
    "salt",
    "sugar",
    "mixed",
    "spices",
    "spice",
    "condiments",
    "condiment",
    "wheat",
    "milk",
    "flour",
    "starch",
    "preservative",
    "flavour",
    "flavor",
    "emulsifier",
    "acid",
    "citric",
    "sodium",
    "calcium",
    "vitamin",
    "extract",
    "colour",
    "color",
    "contains",
    "allergen",
}


# ============================================================
# INGREDIENT MATCHING
# ============================================================

def match_ingredient(raw: str):

    n = normalize(
        raw
    )

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

        if not isinstance(
            item,
            dict
        ):
            continue

        for c in item.get(
            "codes",
            []
        ):

            if not isinstance(
                c,
                str
            ):
                continue

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

        if len(alias) >= 4:

            if alias in n:
                return item

    return None


# ============================================================
# FUZZY OCR CORRECTION
# ============================================================

def fuzzy_correct_word(
    word: str,
    threshold: float = 0.84
):
    """
    Correct a single OCR word against the ingredient
    dictionary.

    Example:

        sunflowar -> sunflower
        chillles  -> chillies
        iodizcd   -> iodized

    The correction is conservative.

    Returns:
        corrected_word, confidence

    or:
        original_word, 0.0
    """

    if not word:
        return word, 0.0

    # Preserve punctuation around the word
    prefix_match = re.match(
        r"^[^A-Za-z0-9]*",
        word
    )

    suffix_match = re.search(
        r"[^A-Za-z0-9]*$",
        word
    )

    prefix = (
        prefix_match.group(0)
        if prefix_match
        else ""
    )

    suffix = (
        suffix_match.group(0)
        if suffix_match
        else ""
    )

    core = word[
        len(prefix):
        len(word) - len(suffix)
        if suffix
        else len(word)
    ]

    if not core:
        return word, 0.0

    normalized = normalize(
        core
    )

    if not normalized:
        return word, 0.0

    # Never modify very short words.
    if len(normalized) < 5:
        return word, 0.0

    # Already a known word.
    if normalized in COMMON_FOOD_WORDS:
        return word, 1.0

    if normalized in OCR_DICTIONARY_WORDS:
        return word, 1.0

    # --------------------------------------------------------
    # Candidate list
    # --------------------------------------------------------

    candidates = set(
        OCR_DICTIONARY_WORDS
    )

    candidates.update(
        COMMON_FOOD_WORDS
    )

    # Don't compare a word against completely unrelated
    # word lengths.
    candidates = [
        candidate
        for candidate in candidates
        if abs(
            len(candidate)
            - len(normalized)
        ) <= 3
    ]

    if not candidates:
        return word, 0.0

    # --------------------------------------------------------
    # Find closest candidate
    # --------------------------------------------------------

    best = None
    best_score = 0.0

    for candidate in candidates:

        score = difflib.SequenceMatcher(
            None,
            normalized,
            candidate
        ).ratio()

        if score > best_score:

            best_score = score
            best = candidate

    if (
        best is None
        or best_score < threshold
    ):

        return word, best_score

    # Preserve original capitalization style
    if core.isupper():

        corrected = best.upper()

    elif core[:1].isupper():

        corrected = (
            best[:1].upper()
            + best[1:]
        )

    else:

        corrected = best

    return (
        prefix
        + corrected
        + suffix,
        best_score
    )


# ============================================================
# CORRECT OCR TEXT
# ============================================================

def correct_ocr_text(
    text: str
):
    """
    Correct OCR spelling errors using the ingredient
    dictionary.

    This operates on individual words and is intentionally
    conservative.

    Multi-word ingredient names are handled later by the
    ingredient matcher.
    """

    if not text:
        return text

    lines = []

    for line in text.splitlines():

        words = re.findall(
            r"[A-Za-z0-9]+|[^A-Za-z0-9]+",
            line
        )

        corrected_parts = []

        for part in words:

            # Only attempt correction on alphabetic words.
            if re.fullmatch(
                r"[A-Za-z]+",
                part
            ):

                corrected, score = (
                    fuzzy_correct_word(
                        part
                    )
                )

                corrected_parts.append(
                    corrected
                )

            else:

                corrected_parts.append(
                    part
                )

        lines.append(
            "".join(
                corrected_parts
            )
        )

    return "\n".join(
        lines
    )


# ============================================================
# PHRASE-LEVEL OCR CORRECTION
# ============================================================

def correct_common_ocr_phrases(
    text: str
):
    """
    Correct common multi-word OCR errors using known
    ingredient phrases.

    Example:

        "sun flowar oil"
             ->
        "sunflower oil"

    This is conservative and only applies to phrases that
    are already in the ingredient dictionary.
    """

    if not text:
        return text

    result = text

    # Process longer phrases first.
    for phrase in OCR_DICTIONARY_PHRASES:

        if len(
            phrase.split()
        ) < 2:

            continue

        # Only consider phrases with reasonably meaningful
        # lengths.
        if len(phrase) < 8:
            continue

        words = phrase.split()

        # Build a flexible OCR pattern.
        pattern_parts = []

        for word in words:

            pattern_parts.append(
                r"\b"
                + re.escape(
                    word
                )
                + r"\b"
            )

        pattern = r"\s+".join(
            pattern_parts
        )

        try:

            result = re.sub(
                pattern,
                phrase,
                result,
                flags=re.I
            )

        except re.error:
            continue

    return result


# ============================================================
# COMPLETE OCR CORRECTION
# ============================================================

def apply_ocr_correction(
    text: str
):
    """
    Run phrase and word level correction.
    """

    if not text:
        return ""

    # First correct individual words.
    corrected = correct_ocr_text(
        text
    )

    # Then normalize known phrases.
    corrected = correct_common_ocr_phrases(
        corrected
    )

    return corrected


# ============================================================
# SPLIT INGREDIENTS
# ============================================================

def split_ingredients(
    text: str
):

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

    # Remove common OCR bullet
    text = text.replace(
        "•",
        ","
    )

    # Split on commas, semicolons and line breaks.
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
            result.append(
                part
            )

    return result


# ============================================================
# ANALYZE INGREDIENT TEXT
# ============================================================

def analyze_ingredient_text(
    text: str
):

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
# READ IMAGE
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

def resize_for_ocr(
    image,
    target_width=1800
):

    height, width = image.shape[:2]

    if width > target_width:

        scale = (
            target_width
            / float(width)
        )

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    elif width < 900:

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
# GREEN CHANNEL ENHANCEMENT
# ============================================================

def enhance_green_channel(
    image
):

    """
    Yellow text on red/orange labels tends to have better
    contrast in the green channel.
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


# ============================================================
# YELLOW TEXT MASK
# ============================================================

def make_yellow_text_mask(
    image
):

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
# CLEAN OCR TEXT
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

    text = re.sub(
        r"[ ]{2,}",
        " ",
        text
    )

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
# DETECT INGREDIENTS WORD
# ============================================================

def is_ingredients_word(
    text
):

    if not text:
        return False

    cleaned = re.sub(
        r"[^A-Za-z]",
        "",
        text
    ).upper()

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

    # Tolerate some OCR errors.
    if (
        "INGRED" in cleaned
        and len(cleaned) >= 7
    ):

        return True

    return False


# ============================================================
# LOCATE INGREDIENTS WORD
# ============================================================

def find_ingredients_word(
    image
):

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
# LOCATE INGREDIENT REGION
# ============================================================

def locate_ingredient_region(
    image
):

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # First search lower 55%.
    # This avoids processing the whole image in the usual
    # case.
    # --------------------------------------------------------

    lower_y = int(
        h * 0.45
    )

    lower = image[
        lower_y:h,
        0:w
    ]

    enhanced = enhance_green_channel(
        lower
    )

    found = find_ingredients_word(
        enhanced
    )

    if found is not None:

        x, y, fw, fh = found

        absolute_y = (
            lower_y + y
        )

        return (
            x,
            absolute_y,
            fw,
            fh
        )

    # --------------------------------------------------------
    # Search entire image if necessary.
    # --------------------------------------------------------

    enhanced = enhance_green_channel(
        image
    )

    found = find_ingredients_word(
        enhanced
    )

    if found is not None:
        return found

    return None


# ============================================================
# CROP INGREDIENT REGION
# ============================================================

def crop_ingredient_region(
    image,
    location
):

    h, w = image.shape[:2]

    x, y, word_w, word_h = location

    # Give some margin before heading.
    x1 = max(
        0,
        x - 50
    )

    # Ingredient lists can span most of the package.
    x2 = min(
        w,
        x + 1250
    )

    y1 = max(
        0,
        y - 30
    )

    # Allow several lines after INGREDIENTS.
    y2 = min(
        h,
        y + 480
    )

    return image[
        y1:y2,
        x1:x2
    ]


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

    if "ingredient" in lower:
        score += 30

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
        "iodized",
        "iodised",
        "salt",
        "sugar",
        "spice",
        "spices",
        "condiment",
        "condiments",
        "wheat",
        "milk",
        "flour",
        "starch",
        "preservative",
        "flavour",
        "flavor",
        "emulsifier",
        "acid",
        "citric",
        "sodium",
        "calcium",
        "vitamin",
        "extract",
        "colour",
        "color"
    ]

    for word in keywords:

        if word in lower:
            score += 5

    if len(text) > 30:
        score += 5

    if len(text) > 100:
        score += 5

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
# OCR TARGET REGION
# ============================================================

def ocr_ingredient_region(
    crop
):

    # --------------------------------------------------------
    # PASS 1: Green channel
    # --------------------------------------------------------

    green = enhance_green_channel(
        crop
    )

    text1 = pytesseract.image_to_string(
        green,
        config=(
            "--oem 3 "
            "--psm 6 "
            "-c preserve_interword_spaces=1"
        )
    )

    text1 = clean_ocr_text(
        text1
    )

    score1 = score_ocr_text(
        text1
    )

    # --------------------------------------------------------
    # If clearly good, don't perform another expensive OCR.
    # --------------------------------------------------------

    if (
        "ingredient" in text1.lower()
        and score1 >= 35
    ):

        return text1

    # --------------------------------------------------------
    # PASS 2: Yellow mask
    # --------------------------------------------------------

    mask = make_yellow_text_mask(
        crop
    )

    text2 = pytesseract.image_to_string(
        mask,
        config=(
            "--oem 3 "
            "--psm 11 "
            "-c preserve_interword_spaces=1"
        )
    )

    text2 = clean_ocr_text(
        text2
    )

    score2 = score_ocr_text(
        text2
    )

    if score2 > score1:
        return text2

    return text1


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
    # Find heading
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
    # Stop at allergen / nutrition / other headings
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
# MAIN OCR PIPELINE
# ============================================================

def perform_ocr(
    image
):

    # --------------------------------------------------------
    # Resize once
    # --------------------------------------------------------

    image = resize_for_ocr(
        image
    )

    # --------------------------------------------------------
    # Locate INGREDIENTS
    # --------------------------------------------------------

    location = locate_ingredient_region(
        image
    )

    if location is not None:

        crop = crop_ingredient_region(
            image,
            location
        )

        raw_text = ocr_ingredient_region(
            crop
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Correct OCR before extracting individual
        # ingredients.
        # ----------------------------------------------------

        corrected_text = apply_ocr_correction(
            raw_text
        )

        ingredients = extract_ingredients_section(
            corrected_text
        )

        if (
            len(ingredients) >= 20
            and len(
                ingredients.split()
            ) >= 3
        ):

            return ingredients

    # ========================================================
    # FALLBACK
    # ========================================================

    h, w = image.shape[:2]

    lower = image[
        int(h * 0.45):h,
        0:w
    ]

    lower_green = enhance_green_channel(
        lower
    )

    fallback = pytesseract.image_to_string(
        lower_green,
        config=(
            "--oem 3 "
            "--psm 11 "
            "-c preserve_interword_spaces=1"
        )
    )

    fallback = clean_ocr_text(
        fallback
    )

    fallback = apply_ocr_correction(
        fallback
    )

    ingredients = extract_ingredients_section(
        fallback
    )

    if ingredients:
        return ingredients

    return fallback


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