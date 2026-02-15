FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ALL files first
COPY . .

# Install ALL dependencies from pyproject.toml
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    python-dotenv \
    opencv-python \
    pillow \
    pytesseract \
    pdf2image \
    rank-bm25 \
    anthropic \
    datasets \
    evaluate \
    pydantic \
    opentelemetry-api \
    opentelemetry-sdk \
    tqdm \
    requests \
    beautifulsoup4 \
    xgboost \
    lightgbm \
    rouge-score \
    prometheus-client \
    boto3 \
    sqlalchemy \
    pytest \
    pytest-cov \
    jupyter \
    notebook \
    llama-index-core \
    faiss-cpu \
    llama-index \
    evidently \
    ragas \
    sagemaker \
    transformers \
    sentence-transformers \
    FlagEmbedding \
    langchain \
    langchain-anthropic \
    langgraph \
    "numpy<2" \
    "ray[default]"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING="1"

# Create directories
RUN mkdir -p data/images data/risk_memos models/cache

# Entry point
ENTRYPOINT ["python", "run_e2e.py"]
CMD ["--dry-run"]