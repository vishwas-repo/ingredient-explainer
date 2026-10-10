from pathlib import Path
import json
import re
import difflib
import io
import os

import cv2
import numpy as np
import easyocr

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
# INITIALIZE EASYOCR ENGINE
# ============================================================

print("[INFO] Initializing EasyOCR Engine globally...")
# Initialised globally once to prevent high-latency loads during API calls
EASYOCR_READER = easyocr.Reader(['en'], gpu=False)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Ingredient Explainer API",
    version="0.4"
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
# TEXT NORMALIZATION & MATCHING
# ============================================================

def normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_ingredients(text: str):
    if not text:
        return []
    text = text.strip()
    text = re.sub(r"\b(ingredients?|contents?)\s*[:\-]?\s*", "", text, flags=re.I)
    text = text.replace("•", ",")
    parts = re.split(r"[,;\n]+", text)
    
    result = []
    for part in parts:
        part = part.strip(" .:-|")
        if part:
            result.append(part)
    return result


def build_alias_map():
    result = {}
    for item in INGREDIENTS:
        names = [item.get("name", "")] + item.get("aliases", []) + item.get("codes", [])
        for name in names:
            if name:
                result[normalize(name)] = item
    return result

ALIAS_MAP = build_alias_map()


def match_ingredient(raw: str):
    n = normalize(raw)
    if not n:
        return None

    if n in ALIAS_MAP:
        return ALIAS_MAP[n]

    code = re.sub(r"\s+", "", n).upper()
    for item in INGREDIENTS:
        for c in item.get("codes", []):
            if re.sub(r"\s+", "", c).upper() == code:
                return item

    for alias, item in sorted(ALIAS_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if len(alias) >= 4 and alias in n:
            return item
    return None


def analyze_ingredient_text(text: str):
    parts = split_ingredients(text)
    results = []
    for raw in parts:
        item = match_ingredient(raw)
        results.append({
            "raw": raw,
            "matched": item is not None,
            "ingredient": item
        })
    return results


# ============================================================
# INTEGRATED ADAPTIVE PIPELINE FUNCTIONS FROM TEST_EASYOCR
# ============================================================

def find_db_match(ocr_token, cutoff=0.55):
    token_clean = ocr_token.strip().upper()
    if not token_clean:
        return None

    best_match = None
    highest_score = 0.0

    for entry in INGREDIENTS:
        db_name = entry.get("name", "")
        aliases = entry.get("aliases", [])
        codes = entry.get("codes", [])
        candidates = [db_name] + aliases + codes
        
        for candidate in candidates:
            candidate_upper = candidate.upper()
            if token_clean == candidate_upper:
                return db_name
            if token_clean in candidate_upper or candidate_upper in token_clean:
                return db_name
            
            score = difflib.SequenceMatcher(None, token_clean, candidate_upper).ratio()
            if score > highest_score:
                highest_score = score
                best_match = db_name

    if highest_score >= cutoff:
        return best_match
    return None


def extract_tokens_universally(raw_input):
    text = raw_input.upper()
    text = text.replace("SUNFLOWER O1", "SUNFLOWER OIL")
    text = text.replace("SUNFLOWER OI", "SUNFLOWER OIL")
    text = text.replace("ODIZED SALI", "IODIZED SALT")
    text = text.replace("SOV SAUCE", "SOY SAUCE")
    text = re.sub(r'\b[K\+][6G]\b', 'I-G', text)
    text = re.sub(r'[_;:,\.\|\/\\\[\]\(\)-]', ' ', text)
    
    structural_anchors = [
        "SUNFLOWER OIL", "RED CHILLES", "RED CHILLIES", "GARLIC ONIOM", "GARLIC ONION",
        "ONION CELERY", "ONIOM CELERY", "CELERY SOY SAUCE", "SOY SAUCE", 
        "IODIZED SALT", "SUGAR", "HIXED SPICES", "MIXED SPICES"
    ]
    
    for anchor in structural_anchors:
        if anchor in text:
            text = text.replace(anchor, f" | {anchor} | ")

    split_blocks = [t.strip() for t in text.split('|') if t.strip()]
    final_tokens = []
    
    for block in split_blocks:
        if " " in block and not any(a in block for a in structural_anchors):
            final_tokens.extend([w.strip() for w in block.split() if w.strip()])
        else:
            final_tokens.append(block)
            
    return [t for t in final_tokens if t]


def universal_three_tier_pipeline(raw_input):
    tokens = extract_tokens_universally(raw_input)
    mapped_ingredients = []
    unmapped_items = []
    remaining_noise = []
    noise_words = ["AND", "WITH", "CONTAINS", "PLUS", "OR"]
    
    for token in tokens:
        if token in noise_words:
            formatted_noise = token.title()
            if formatted_noise not in remaining_noise:
                remaining_noise.append(formatted_noise)
            continue
            
        words = token.split()
        if len(words) > 1 and any(w in noise_words for w in words):
            for word in words:
                if word in noise_words:
                    if word.title() not in remaining_noise:
                        remaining_noise.append(word.title())
                else:
                    matched = find_db_match(word)
                    if matched:
                        if matched not in mapped_ingredients:
                            mapped_ingredients.append(matched)
                    else:
                        if word.title() not in unmapped_items:
                            unmapped_items.append(word.title())
            continue

        matched_name = find_db_match(token)
        if matched_name:
            if matched_name not in mapped_ingredients:
                mapped_ingredients.append(matched_name)
        else:
            if token == "I-G":
                if "I-G" not in remaining_noise:
                    remaining_noise.append("I-G")
            else:
                formatted_unmapped = token.title()
                if formatted_unmapped not in unmapped_items:
                    unmapped_items.append(formatted_unmapped)

    return {
        "mapped": ", ".join(mapped_ingredients) + "." if mapped_ingredients else "None",
        "unmapped": ", ".join(unmapped_items) if unmapped_items else "None",
        "remaining": ", ".join(remaining_noise) if remaining_noise else "None"
    }


def universal_easyocr_extractor_from_bytes(image_bytes):
    """Processes bytes array directly with EasyOCR to prevent disk IO delays."""
    ocr_results = EASYOCR_READER.readtext(image_bytes, detail=0)
    
    if not ocr_results:
        raise HTTPException(status_code=422, detail="No readable text found in the image.")

    full_text = " ".join(ocr_results)
    full_text_upper = full_text.upper()
    target_keyword = "INGREDIENTS"
    
    if target_keyword in full_text_upper:
        keyword_idx = full_text_upper.find(target_keyword)
        raw_start_block = full_text[keyword_idx + len(target_keyword):].strip()
        
        if raw_start_block.startswith(":") or raw_start_block.startswith("-") or raw_start_block.startswith("."):
            raw_start_block = raw_start_block[1:].strip()
            
        ingredients_segment = raw_start_block
        terminators = ["ALLERGEN", "CONTAINS WHEAT", "NUTRITIONAL", "MANUFACTURED"]
        for term in terminators:
            if term in ingredients_segment.upper():
                end_idx = ingredients_segment.upper().find(term)
                ingredients_segment = ingredients_segment[:end_idx].strip()
                
        return ingredients_segment
    else:
        # Fallback to returning full text if keyword isn't isolated explicitly
        return full_text


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)
    return {"name": "Ingredient Explainer API", "version": "0.4"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.4"}


@app.get("/ingredients")
def ingredients():
    return {"count": len(INGREDIENTS), "items": INGREDIENTS}


@app.post("/analyze-text")
