# OCR & Summarization API

API для распознавания текста с изображений и генерации краткого резюме на выбранном языке.

## Роуты

1. **/extract-text**  
   - POST, UploadFile
   - Ответ: 
     ```json
     {
       "text": "Пример текста с изображения",
       "language": "ru"
     }
     ```

2. **/summarized-extract-text**  
   - POST, UploadFile, query параметр `summary_language`
   - Извлекает текст, определяет язык, создаёт summary и переводит его на выбранный язык.
   - Ответ:
     ```json
     {
       "original_language": "ru",
       "summary": "Краткое содержание текста на выбранном языке"
     }
     ```

## Запуск локально
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
#requirments
pip install -r requirements.txt 
# OR (homebrew pyhton)
pip install --break-system-packages -r requirements.txt
uvicorn app.main:app --reload

## Запуск docker
```bash
docker build -t ocr-api:latest .
docker run -p 8000:8000 ocr-api:latest