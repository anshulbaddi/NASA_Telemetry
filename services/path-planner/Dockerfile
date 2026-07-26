FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/path-planner/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generated/ generated/
COPY services/path-planner/ services/path-planner/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 50054 9092

CMD ["python", "-u", "services/path-planner/main.py"]
