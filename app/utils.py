from PIL import Image
import io
from .models import ocr_processor, ocr_model, lang_detect, summarizer
from transformers import pipeline

# Кэш моделей перевода, чтобы не загружать их каждый раз
translation_cache = {}

def ocr_image(image_bytes: bytes) -> str:
    """Распознаем текст с изображения"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values

    # Генерация текста
    generated_ids = ocr_model.generate(pixel_values)
    text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text

def detect_language(text: str) -> str:
    """Определяем язык текста"""
    result = lang_detect(text)
    return result[0]['label']

def summarize_text(text: str, max_length: int = 150) -> str:
    """Создаем краткое резюме текста"""
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Перевод текста с src_lang на tgt_lang"""
    if src_lang == tgt_lang:
        return text  # перевод не нужен

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    if model_name not in translation_cache:
        try:
            translation_cache[model_name] = pipeline("translation", model=model_name)
        except Exception:
            # Если модель для пары языков не найдена, возвращаем исходный текст
            return text

    translator = translation_cache[model_name]
    translated = translator(text, max_length=512)
    return translated[0]['translation_text']