# ───────── 1. Base image ─────────
FROM python:3.10-slim

# Optional: fix a deterministic home for HF cache
ENV HF_HOME=/root/.cache/huggingface

# ───────── 2. Copy project files ─────────
WORKDIR /app
COPY ./src /app/src
COPY ./requirements.txt /app/requirements.txt

# ───────── 3. Install system deps ─────────
# (add poppler-utils, libmagic, etc. only if your code needs them)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

# ───────── 4. Python deps  ─────────
#   1️⃣  CPU-only PyTorch wheels
#   2️⃣  sentence-transformers wrapper
#   3️⃣  All remaining requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir sentence-transformers && \
    pip install --no-cache-dir -r requirements.txt && \
    # 🧹 remove pip wheel cache to save space
    pip cache purge

# ───────── 5. Expose port & run ─────────
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
