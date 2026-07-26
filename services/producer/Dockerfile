FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        librdkafka-dev gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/producer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generated/ generated/
COPY services/producer/ services/producer/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 9090

CMD ["python", "-u", "services/producer/main.py"]
