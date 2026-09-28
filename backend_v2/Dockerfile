FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY backend_v2/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend_v2/app.py backend_v2/stitch_worker.py ./

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
