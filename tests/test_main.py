import io
from fastapi.testclient import TestClient
from app.main import app
from app import utils


client = TestClient(app)


def create_test_image():
    from PIL import Image
    img = Image.new("RGB", (1, 1), color="white")
    byte_arr = io.BytesIO()
    img.save(byte_arr, format="PNG")
    byte_arr.seek(0)
    return byte_arr


# ====================
# /extract-text
# ====================

def test_extract_text_success_and_empty_ocr(monkeypatch):
    monkeypatch.setattr("app.main.ocr_image", lambda _: "")
    monkeypatch.setattr("app.main.detect_language", lambda _: "en")

    files = {"image": ("test.png", create_test_image(), "image/png")}
    response = client.post("/extract-text", files=files)

    assert response.status_code == 200
    assert response.json() == {
        "text": "",
        "language": "en",
    }


def test_extract_text_no_file():
    response = client.post("/extract-text")
    assert response.status_code == 422


def test_extract_text_invalid_file_type():
    files = {"image": ("test.txt", io.BytesIO(b"not image"), "text/plain")}
    response = client.post("/extract-text", files=files)

    assert response.status_code == 500
    assert "error" in response.json()


# ====================
# /summarized-extract-text
# ====================

def test_summarized_extract_text_success(monkeypatch):
    monkeypatch.setattr("app.main.ocr_image", lambda _: "hello world")
    monkeypatch.setattr("app.main.detect_language", lambda _: "en")
    monkeypatch.setattr("app.main.summarize_text", lambda _: "short summary")
    monkeypatch.setattr(
        "app.main.translate_text",
        lambda text, src_lang, tgt_lang: text,
    )

    files = {"image": ("test.png", create_test_image(), "image/png")}
    response = client.post(
        "/summarized-extract-text?summary_language=en",
        files=files,
    )

    assert response.status_code == 200
    assert response.json() == {
        "original_language": "en",
        "summary": "short summary",
    }


def test_summarized_extract_text_default_language(monkeypatch):
    monkeypatch.setattr("app.main.ocr_image", lambda _: "hello world")
    monkeypatch.setattr("app.main.detect_language", lambda _: "en")
    monkeypatch.setattr("app.main.summarize_text", lambda _: "short summary")
    monkeypatch.setattr(
        "app.main.translate_text",
        lambda text, src_lang, tgt_lang: text,
    )

    files = {"image": ("test.png", create_test_image(), "image/png")}
    response = client.post("/summarized-extract-text", files=files)

    assert response.status_code == 200
    assert response.json()["summary"] == "short summary"


def test_summarized_extract_text_invalid_file():
    files = {"image": ("test.txt", io.BytesIO(b"not image"), "text/plain")}
    response = client.post(
        "/summarized-extract-text?summary_language=en",
        files=files,
    )

    assert response.status_code == 500
    assert "error" in response.json()


def test_summarized_extract_text_no_file():
    response = client.post("/summarized-extract-text")
    assert response.status_code == 422


# ====================
# app.utils
# ====================

def test_translate_text_same_language():
    result = utils.translate_text("привет", src_lang="ru", tgt_lang="ru")
    assert result == "привет"


def test_translate_text_different_language_ru(monkeypatch):
    def fake_translator(text, max_length=512):
        return [{"translation_text": "hello"}]

    monkeypatch.setattr(
        utils,
        "pipeline",
        lambda *args, **kwargs: fake_translator,
    )

    result = utils.translate_text("привет", src_lang="ru", tgt_lang="en")
    assert result == "hello"


def test_translate_text_model_not_found(monkeypatch):
    def broken_pipeline(*args, **kwargs):
        raise Exception("model not found")

    monkeypatch.setattr(utils, "pipeline", broken_pipeline)

    result = utils.translate_text("привет", src_lang="ru", tgt_lang="xx")
    assert result == "привет"


def test_detect_language(monkeypatch):
    monkeypatch.setattr(
        utils,
        "lang_detect",
        lambda text: [{"label": "ru"}],
    )

    assert utils.detect_language("привет") == "ru"
