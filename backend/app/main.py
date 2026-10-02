from pathlib import Path
import json
import re
import io
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, ImageEnhance
import pytesseract

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
DB_PATH = BASE / "data" / "ingredients.json"
FRONTEND_PATH = ROOT / "frontend" / "index.html"

with open(DB_PATH, "r", encoding="utf-8") as f:
    INGREDIENTS = json.load(f)

app = FastAPI(title="Ingredient Explainer API", version="0.3")

# Same-origin deployment needs no CORS. If the frontend is hosted separately
# (for example on Netlify), set ALLOWED_ORIGINS to its URL(s), comma-separated.
allowed_origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_ingredients(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(ingredients?|contents?)\s*[:\-]?\s*", "", text, flags=re.I)
    parts = re.split(r"[,;\n•]+", text)
    return [p.strip(" .:-") for p in parts if p.strip(" .:-")]


def build_alias_map():
    m = {}
    for item in INGREDIENTS:
        names = [item["name"]] + item.get("aliases", []) + item.get("codes", [])
        for name in names:
            m[normalize(name)] = item
    return m


ALIAS_MAP = build_alias_map()


def match_ingredient(raw: str):
    n = normalize(raw)
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


@app.get("/")
def root():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)
    return {"name": "Ingredient Explainer API", "version": "0.3"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.3"}


@app.get("/ingredients")
def ingredients():
    return {"count": len(INGREDIENTS), "items": INGREDIENTS}


@app.post("/analyze-text")
def analyze_text(payload: dict):
    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "text is required")
    parts = split_ingredients(text)
    results = []
    for raw in parts:
        item = match_ingredient(raw)
        results.append({"raw": raw, "matched": item is not None, "ingredient": item})
    return {"input": text, "ingredients": results}


async def read_and_prepare_image(file: UploadFile):
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Uploaded file is not a valid image")
    gray = ImageOps.grayscale(image)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(1.4)
    return gray


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    gray = await read_and_prepare_image(file)
    try:
        text = pytesseract.image_to_string(gray, config="--psm 6")
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")
    return {"text": text, "ingredients": split_ingredients(text)}


@app.post("/ocr-and-analyze")
async def ocr_and_analyze(file: UploadFile = File(...)):
    gray = await read_and_prepare_image(file)
    try:
        text = pytesseract.image_to_string(gray, config="--psm 6")
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {e}")

    parts = split_ingredients(text)
    results = []
    for raw in parts:
        item = match_ingredient(raw)
        results.append({"raw": raw, "matched": item is not None, "ingredient": item})
    return {"ocr_text": text, "ingredients": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
