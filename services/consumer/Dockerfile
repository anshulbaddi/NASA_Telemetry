FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/consumer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generated/ generated/
COPY services/consumer/ services/consumer/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 9091

CMD ["python", "-u", "services/consumer/main.py"]
