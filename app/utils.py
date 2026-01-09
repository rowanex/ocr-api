from PIL import Image
import io
import torch
from transformers import pipeline
from . import models

MAX_SUMMARIZE_TEXT_LENGTH = 3000
SUMMARIZE_CHUNK_SIZE = 1024
SUMMARY_MIN_LENGTH = 30
SUMMARY_MAX_LENGTH = 150

OCR_MAX_LENGTH = 3584
OCR_MIN_LENGTH = 1
OCR_NUM_BEAMS = 4
OCR_EARLY_STOPPING = True

TRANSLATION_MAX_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translation_cache = {}

SUPPORTED_LANGS = {"en", "ru", "de", "fr", "es", "it", "pt", "nl"}


def ocr_image(image_bytes: bytes) -> str:
    """
    Распознаем текст с изображения OCR.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pixel_values = models.ocr_processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():

        generate_kwargs = {
            "max_length": OCR_MAX_LENGTH,
            "min_length": OCR_MIN_LENGTH,
            "num_beams": OCR_NUM_BEAMS,
            "early_stopping": OCR_EARLY_STOPPING
        }

        if hasattr(models.ocr_processor, 'tokenizer') and hasattr(models.ocr_processor.tokenizer, 'unk_token_id'):
            unk_id = models.ocr_processor.tokenizer.unk_token_id
            if unk_id is not None:
                generate_kwargs["bad_words_ids"] = [[unk_id]]

        generated_ids = models.ocr_model.generate(pixel_values, **generate_kwargs)

    text = models.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text.strip()


def detect_language(text: str) -> str:
    """Определяем язык текста"""
    if not text.strip():
        return "unknown"

    result = models.lang_detect(text[:512])
    return result[0]["label"]


def summarize_text(text: str, max_length: int = SUMMARY_MAX_LENGTH) -> str:
    """Создаем краткое резюме текста"""

    if not text.strip():
        return ""

    text = text[:MAX_SUMMARIZE_TEXT_LENGTH]

    chunks = [
        text[i:i + SUMMARIZE_CHUNK_SIZE]
        for i in range(0, len(text), SUMMARIZE_CHUNK_SIZE)
    ]

    summaries = []

    for chunk in chunks:
        with torch.no_grad():
            summary = models.summarizer(
                chunk,
                max_length=max_length,
                min_length=SUMMARY_MIN_LENGTH,
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
        translated = translator(text, max_length=TRANSLATION_MAX_LENGTH)

    return translated[0]["translation_text"]
