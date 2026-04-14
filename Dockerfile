FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Pre-download model during build if using HuggingFace Hub
# This avoids a long startup delay on first request.
# Set HF_MODEL_NAME build arg to your hub model, e.g.:
#   docker build --build-arg HF_MODEL_NAME=your-username/radiology-summarizer .
ARG HF_MODEL_NAME=""
ENV HF_MODEL_NAME=${HF_MODEL_NAME}
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

RUN if [ -n "$HF_MODEL_NAME" ]; then \
    python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; \
               AutoTokenizer.from_pretrained('${HF_MODEL_NAME}'); \
               AutoModelForSeq2SeqLM.from_pretrained('${HF_MODEL_NAME}')"; \
    fi

# If using a local model directory instead of HuggingFace Hub, uncomment:
# COPY model/ ./model/
# ENV MODEL_DIR=/app/model

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
