FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/api-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generated/ generated/
COPY services/api-gateway/ services/api-gateway/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-u", "services/api-gateway/main.py"]
