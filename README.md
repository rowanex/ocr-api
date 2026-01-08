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

3. **/health/live**
   - GET
   - Liveness-probe. Проверяет, что приложение запущено и отвечает.
   - Ответ:
    ```json
    {
      "status": "alive"
    }
    ```

4. **/health/ready**  
   - GET
   - Readiness-probe. Проверяет, что все ML-модели загружены и сервис готов обрабатывать запросы.
   - Ответ 200:
    ```json
    {
      "status": "ready",
      "models": {
        "ocr": "naver-clova-ix/donut-base",
        "language_detection": "papluca/xlm-roberta-base-language-detection",
        "summarization": "facebook/bart-large-cnn",
        "translation": "lazy-load"
      },
      "uptime_seconds": 123,
    }
    ```


## Запуск локально
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

#requirments
pip install -r requirements.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

#start
uvicorn app.main:app --reload

## Запуск docker
```bash
docker build -t ocr-api:latest .
docker run -p 8000:8000 ocr-api:latest