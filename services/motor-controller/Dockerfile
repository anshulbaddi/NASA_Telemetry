FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/motor-controller/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generated/ generated/
COPY services/motor-controller/ services/motor-controller/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 50052 9094

CMD ["python", "-u", "services/motor-controller/main.py"]
