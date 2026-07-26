FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        librdkafka-dev gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY services/fdir/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generated/ generated/
COPY services/fdir/ services/fdir/

# Build the Cython/C++ rolling-buffer extension in-place
RUN cd services/fdir && \
    python setup.py build_ext --inplace --quiet && \
    echo "fdir_native extension built"

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 50053 2112

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:2112/metrics')" || exit 1

CMD ["python", "-u", "services/fdir/main.py"]
