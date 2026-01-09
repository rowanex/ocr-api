from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .utils import ocr_image, detect_language, summarize_text, translate_text, SUPPORTED_LANGS
from . import models
import logging
from typing import Literal, Union
import time
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

START_TIME = time.time()

app = FastAPI(title="OCR & Summarization API")


@app.on_event("startup")
def load_models():
    load_flag = os.getenv("LOAD_MODELS", "1")
    if load_flag == "0":
        logger.info("LOAD_MODELS=0 -> skip loading ML models")
        return

    try:
        logger.info("Loading ML models...")

        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel, pipeline as hf_pipeline

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models.ocr_processor = NougatProcessor.from_pretrained(
            "facebook/nougat-base"
        )
        models.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            "facebook/nougat-base"
        ).to(DEVICE)
        models.ocr_model.eval()

        models.lang_detect = hf_pipeline(
            "text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device=0 if torch.cuda.is_available() else -1
        )

        models.summarizer = hf_pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1
        )

        models.models_loaded = True
        logger.info("Models loaded successfully")

    except Exception as e:
        models.load_error = str(e)
        logger.exception("Failed to load models")


# ====================
# Pydantic модели для ответов
# ====================
class ExtractTextResponse(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "Exapmle text from image"})
    language: str = Field(..., json_schema_extra={"example": "en"})


class SummarizedExtractTextResponse(BaseModel):
    original_language: str = Field(..., json_schema_extra={"example": "en"})
    summary: str = Field(..., json_schema_extra={"example": "Краткое содержание текста на выбранном языке"})


class HealthReadyResponse(BaseModel):
    status: Literal["ready"]
    models: dict[str, str]
    uptime_seconds: int


class HealthNotReadyResponse(BaseModel):
    status: Literal["not_ready"]
    missing_models: list[str]
    error: str | None = None


# ====================
# Роуты
# ====================
@app.get(
    "/health/live",
    tags=["Health"],
    summary="Проверка доступности API",
    description="Проверяет, что API запущено и отвечает на HTTP-запросы"
)
def liveness():
    return {"status": "alive"}


@app.get(
    "/health/ready",
    tags=["Health"],
    response_model=Union[HealthReadyResponse, HealthNotReadyResponse],
    summary="Проверка готовности API принимать OCR запросы",
    description="Возвращает информацию по статусу API и загрузки моделей"
)
def readiness():
    missing = []

    if models.ocr_processor is None:
        missing.append("ocr_processor")
    if models.ocr_model is None:
        missing.append("ocr_model")
    if models.lang_detect is None:
        missing.append("language_detector")
    if models.summarizer is None:
        missing.append("summarizer")

    if missing:
        return JSONResponse(
            status_code=503,
            content=HealthNotReadyResponse(
                status="not_ready",
                missing_models=missing,
                error=models.load_error,
            ).model_dump(),
        )

    return HealthReadyResponse(
        status="ready",
        models={
            "ocr": "facebook/nougat-base",
            "language_detection": "papluca/xlm-roberta-base-language-detection",
            "summarization": "facebook/bart-large-cnn",
            "translation": "lazy-load",
        },
        uptime_seconds=int(time.time() - START_TIME),
    )


@app.post(
    "/extract-text",
    response_model=ExtractTextResponse,
    summary="Извлечение текста с изображения",
    description="Принимает изображение и возвращает распознанный текст и язык оригинала"
)
async def extract_text(image: UploadFile = File(..., description="Изображение для распознавания текста")):
    try:
        image_bytes = await image.read()
        text = ocr_image(image_bytes)
        language = detect_language(text)
        return {"text": text, "language": language}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post(
    "/summarized-extract-text",
    response_model=SummarizedExtractTextResponse,
    summary="Извлечение текста и генерация резюме",
    description="Принимает изображение, извлекает текст, определяет язык и возвращает краткое резюме на выбранном языке"
)
async def summarized_extract_text(
    image: UploadFile = File(..., description="Изображение для распознавания текста"),
    summary_language: Literal[
        "en", "ru", "de", "fr", "es", "it", "pt", "nl"
    ] = Query(
        "en",
        description="Язык summary. Поддерживаемые значения: en, ru, de, fr, es, it, pt, nl"
    )
):
    if summary_language not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported summary_language. Supported languages: {sorted(SUPPORTED_LANGS)}"
        )

    try:
        image_bytes = await image.read()
        text = ocr_image(image_bytes)
        original_language = detect_language(text)
        summary = summarize_text(text)

        translated_summary = translate_text(summary, src_lang=original_language, tgt_lang=summary_language)

        return {"original_language": original_language, "summary": translated_summary}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
