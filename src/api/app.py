"""
Phase 9 & 10: Model Deployment — FastAPI REST API
Serves the fine-tuned BART-base radiology summarization model.

Endpoints:
    POST /summarize  — Takes {"findings": "<text>"}, returns {"impression": "<summary>"}
    GET  /health     — Returns {"status": "ok", "model_loaded": true/false}

Environment variables:
    MODEL_DIR       — Local path to saved model directory (default: "model")
    HF_MODEL_NAME   — HuggingFace Hub model name (overrides MODEL_DIR if set)
    PORT            — Port to bind (set automatically by Render; default 8000)

Usage (local):
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_device = None
_model_loaded = False

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
NUM_BEAMS = 4


def load_model():
    """Load model and tokenizer from local directory or HuggingFace Hub."""
    global _model, _tokenizer, _device, _model_loaded

    hf_model_name = os.getenv("HF_MODEL_NAME", "")
    model_dir = os.getenv("MODEL_DIR", "model")

    source = hf_model_name if hf_model_name else model_dir
    logger.info(f"Loading model from: {source}")

    try:
        _tokenizer = AutoTokenizer.from_pretrained(source)
        _model = AutoModelForSeq2SeqLM.from_pretrained(source)

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()

        _model_loaded = True
        logger.info(f"Model loaded successfully on {_device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _model_loaded = False
        raise


# ---------------------------------------------------------------------------
# FastAPI lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # Shutdown cleanup (optional)
    logger.info("Shutting down API")


app = FastAPI(
    title="Radiology Report Summarizer",
    description=(
        "Fine-tuned BART-base model that generates a concise radiological IMPRESSION "
        "from free-text FINDINGS of a chest X-ray report."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class SummarizeRequest(BaseModel):
    findings: str

    @field_validator("findings")
    @classmethod
    def findings_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("findings must not be empty")
        return v.strip()


class SummarizeResponse(BaseModel):
    impression: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize radiology findings",
    responses={
        200: {"description": "Successful summarization"},
        400: {"description": "Missing or empty findings field"},
        500: {"description": "Internal server error during inference"},
    },
)
def summarize(request: SummarizeRequest):
    """
    Takes free-text radiology FINDINGS and returns a concise IMPRESSION summary.
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please try again later.")

    try:
        inputs = _tokenizer(
            request.findings,
            return_tensors="pt",
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                num_beams=NUM_BEAMS,
                max_length=MAX_TARGET_LENGTH,
                early_stopping=True,
            )

        impression = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return SummarizeResponse(impression=impression)

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during inference.")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health():
    """Returns service health status and whether the model is loaded."""
    return HealthResponse(status="ok", model_loaded=_model_loaded)


@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Radiology Report Summarizer API",
        "docs": "/docs",
        "health": "/health",
    }
