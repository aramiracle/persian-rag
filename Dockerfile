# ---------------- BUILDER IMAGE ----------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
# Added: git (requested), cmake (for quantization compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual env
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .

# Install requirements
# Note: Ensure 'huggingface_hub' is in your requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# ---------------- RUN IMAGE ----------------
FROM python:3.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the venv from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy Application Code and Models
COPY backend ./backend
COPY models/ ./models

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]