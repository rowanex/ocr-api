from PIL import Image
import io
import torch
from transformers import pipeline
from . import models

# Кэш моделей перевода
translation_cache = {}
SUPPORTED_LANGS = {"en", "ru", "de", "fr", "es", "it", "pt", "nl"}


def ocr_image(image_bytes: bytes) -> str:
    """Распознаем текст с изображения"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    generated_ids = ocr_model.generate(pixel_values)
    text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def detect_language(text: str) -> str:
    """Определяем язык текста"""
    if not text.strip():
        return "unknown"

    result = models.lang_detect(text[:512])
    return result[0]["label"]


def summarize_text(text: str, max_length: int = 150) -> str:
    """Создаем краткое резюме текста"""

    if not text.strip():
        return ""
    
    text = text[:3000]

    max_input_length = 1024
    chunks = [
        text[i:i + max_input_length]
        for i in range(0, len(text), max_input_length)
    ]

    summaries = []

    for chunk in chunks:
        with torch.no_grad():
            summary = models.summarizer(
                chunk,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
        summaries.append(summary[0]["summary_text"])

    return " ".join(summaries)


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Перевод текста с src_lang на tgt_lang"""

    if not text.strip():
        return ""

    if src_lang not in SUPPORTED_LANGS or tgt_lang not in SUPPORTED_LANGS:
        return text

    if src_lang == tgt_lang:
        return text

    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    if model_name not in translation_cache:
        try:
            translation_cache[model_name] = pipeline(
                "translation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception:
            return text

    translator = translation_cache[model_name]

    with torch.no_grad():
        translated = translator(text, max_length=512)

    return translated[0]["translation_text"]