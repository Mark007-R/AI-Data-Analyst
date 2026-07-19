# DataAI — Hugging Face Spaces (sdk: docker, app_port: 7860)
FROM python:3.11-slim

WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces runs as a non-root user; keep dataset files in a writable dir
RUN mkdir -p /home/user/app/data && chmod -R 777 /home/user/app/data
ENV DATAAI_DATA_DIR=/home/user/app/data

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
