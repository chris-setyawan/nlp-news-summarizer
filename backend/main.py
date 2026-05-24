"""
backend/main.py
===============
FastAPI backend untuk Indonesian News Summarizer + NER.
Jalankan dengan: uvicorn backend.main:app --reload --port 8000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from scraper.news_scraper import scrape_article
from models.summarizer import Summarizer
from models.ner import NERExtractor, LABEL_CONFIG


# ── Global model instances ─────────────────────────────────────────────────────
summarizer: Summarizer = None
ner_extractor: NERExtractor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global summarizer, ner_extractor
    print("[API] Loading models...")
    summarizer = Summarizer()
    summarizer.load()
    ner_extractor = NERExtractor()
    ner_extractor.load()
    print("[API] Models ready!")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Indonesian News Summarizer API",
    description="Summarization + NER untuk berita berbahasa Indonesia",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class SummarizeRequest(BaseModel):
    url: str
    length: str = "sedang"  # pendek | sedang | panjang


class EntityItem(BaseModel):
    text: str
    label: str
    label_display: str
    emoji: str
    color: str
    score: float


class SummarizeResponse(BaseModel):
    success: bool
    error: str = None
    # Article info
    title: str = ""
    source: str = ""
    word_count: int = 0
    # Summary
    summary: str = ""
    summary_word_count: int = 0
    compression_pct: int = 0
    # NER
    entities: list[EntityItem] = []
    entity_count: int = 0


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": summarizer is not None}


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL tidak valid.")

    # 1. Scrape
    article = scrape_article(req.url)
    if not article.success:
        return SummarizeResponse(success=False, error=article.error_message)

    # 2. Summarize
    summary_result = summarizer.summarize(article.content, length=req.length)

    # DEBUG
    print(f"\n[DEBUG] Summary success: {summary_result.success}")
    print(f"[DEBUG] Summary error: {summary_result.error_message}")
    print(f"[DEBUG] Summary text: {repr(summary_result.summary)}")
    print(f"[DEBUG] Summary length: {len(summary_result.summary) if summary_result.summary else 0}")

    # 3. NER
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ner_result = ner_extractor.extract(article.content)
    ner_result = ner_extractor.extract(article.content)

    # Build entity list
    entities = []
    if ner_result.success:
        for e in ner_result.entities:
            entities.append(EntityItem(
                text=e.text,
                label=e.label,
                label_display=e.label_display,
                emoji=e.emoji,
                color=e.color,
                score=e.score,
            ))

    word_count = len(article.content.split())
    summary_words = len(summary_result.summary.split()) if summary_result.success else 0
    compression = round((1 - summary_words / max(word_count, 1)) * 100) if summary_result.success else 0

    return SummarizeResponse(
        success=True,
        title=article.title,
        source=article.source,
        word_count=word_count,
        summary=summary_result.summary if summary_result.success else "",
        summary_word_count=summary_words,
        compression_pct=compression,
        entities=entities,
        entity_count=len(entities),
    )


# ── Serve frontend static files ────────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def landing():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/app")
    def main_app():
        return FileResponse(os.path.join(frontend_dir, "app.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
