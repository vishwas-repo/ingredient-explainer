# Ingredient Explainer MVP v0.3 — Deployment Ready

This version supports:
- Take Photo on phones
- Upload Image
- OCR with Tesseract
- Editable OCR text
- Ingredient analysis using `backend/data/ingredients.json`
- A `/health` endpoint
- Same-origin deployment: the FastAPI backend serves `frontend/index.html`
- Docker deployment with Tesseract installed inside the container

## Recommended free deployment: Render

The included `render.yaml` and `backend/Dockerfile` are ready for a Render Web Service.

1. Put this project in a GitHub repository.
2. In Render, choose **New → Web Service** and connect the repository.
3. Render should detect `render.yaml`, or choose **Docker** manually.
4. Select the **Free** plan.
5. Deploy.
6. Open the generated `https://...onrender.com` URL. The frontend and API are served by the same service.

The service listens on `0.0.0.0` and uses Render's `PORT` environment variable.

## Local Windows test

Backend:

```text
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

For local OCR, Tesseract must be installed separately on Windows and `tesseract.exe` must be on PATH.

Then open:

```text
http://localhost:8000/
```

You no longer need `python -m http.server 5500` for the combined deployment.

## Separate Netlify frontend (optional)

The frontend uses:

```js
window.INGREDIENT_API_URL || ""
```

For a separately hosted frontend, add a small script before `index.html`'s main script:

```html
<script>window.INGREDIENT_API_URL="https://YOUR-BACKEND.onrender.com";</script>
```

Then set the backend environment variable `ALLOWED_ORIGINS` to your Netlify URL, for example:

```text
https://your-site.netlify.app
```

## Important free-tier behavior

A free Render web service can sleep after inactivity, so the first request after sleeping may take around a minute. This is normal for the free tier.

The application data is bundled in `backend/data/ingredients.json`; it is not a persistent database.
