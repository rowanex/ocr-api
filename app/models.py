from transformers import VisionEncoderDecoderModel, TrOCRProcessor, pipeline


# ====================
# OCR модель
# ====================
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")


# ====================
# Определение языка текста
# ====================
lang_detect = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")


# ====================
# Суммаризация текста
# ====================
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
