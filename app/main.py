from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .utils import ocr_image, detect_language, summarize_text, translate_text

app = FastAPI(title="OCR & Summarization API")


# ====================
# Pydantic модели для ответов
# ====================
class ExtractTextResponse(BaseModel):
    text: str = Field(..., example="Пример текста с изображения")
    language: str = Field(..., example="ru")


class SummarizedExtractTextResponse(BaseModel):
    original_language: str = Field(..., example="ru")
    summary: str = Field(..., example="Краткое содержание текста на выбранном языке")


# ====================
# Роуты
# ====================
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
    summary_language: str = Query("en", description="Язык, на котором вернуть summary (например 'en', 'ru')")
):
    try:
        image_bytes = await image.read()
        text = ocr_image(image_bytes)
        original_language = detect_language(text)
        summary = summarize_text(text)

        translated_summary = translate_text(summary, src_lang=original_language, tgt_lang=summary_language)

        return {"original_language": original_language, "summary": translated_summary}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
