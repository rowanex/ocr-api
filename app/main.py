from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from .utils import ocr_image, detect_language


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
