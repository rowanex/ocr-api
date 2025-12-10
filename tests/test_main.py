import io
import pytest
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def create_test_image():
    from PIL import Image
    img = Image.new("RGB", (1, 1), color="white")
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr


# ====================
# /extract-text
# ====================

def test_extract_text_success_and_empty_ocr():
    """
    OCR должен вернуть пустую строку.
    """
    image_bytes = create_test_image()
    files = {"image": ("test.png", image_bytes, "image/png")}

    response = client.post("/extract-text", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "text" in data
    assert "language" in data
    assert isinstance(data["text"], str)
    assert data["text"].strip() == ""
    assert isinstance(data["language"], str)


def test_extract_text_no_file():
    """
    Отсутствие required параметра File(...)
    FastAPI возвращает 422 Unprocessable Entity
    """
    response = client.post("/extract-text")
    assert response.status_code == 422


def test_extract_text_invalid_file_type():
    """
    Передаем файл, который не является изображением.
    Обработчик должен вернуть 500, потому что ваш try/except перехватывает всё.
    """
    files = {"image": ("test.txt", io.BytesIO(b"not image"), "text/plain")}
    response = client.post("/extract-text", files=files)
    assert response.status_code == 500
    assert "error" in response.json()


# ====================
# /summarized-extract-text
# ====================


def test_summarized_extract_text_success():
    image_bytes = create_test_image()
    files = {"image": ("test.png", image_bytes, "image/png")}

    response = client.post("/summarized-extract-text?summary_language=en", files=files)

    assert response.status_code == 200
    data = response.json()

    assert "original_language" in data
    assert "summary" in data
    assert isinstance(data["summary"], str)
    assert len(data["summary"].strip()) > 0


def test_summarized_extract_text_default_language():
    """
    summary_language по умолчанию = en
    Проверяем что можно вызвать без параметра
    """
    image_bytes = create_test_image()
    files = {"image": ("test.png", image_bytes, "image/png")}

    response = client.post("/summarized-extract-text", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data


def test_summarized_extract_text_invalid_file():
    files = {"image": ("test.txt", io.BytesIO(b"not image"), "text/plain")}
    response = client.post("/summarized-extract-text?summary_language=en", files=files)

    assert response.status_code == 500
    assert "error" in response.json()


def test_summarized_extract_text_no_file():
    response = client.post("/summarized-extract-text")
    assert response.status_code == 422