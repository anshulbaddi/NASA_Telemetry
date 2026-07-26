FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/obstacle-detector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generated/ generated/
COPY services/obstacle-detector/ services/obstacle-detector/

# Build the Cython LIDAR extension in-place
RUN cd services/obstacle-detector && \
    python setup.py build_ext --inplace --quiet && \
    echo "lidar_cluster extension built"

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 50051 9093

CMD ["python", "-u", "services/obstacle-detector/main.py"]
