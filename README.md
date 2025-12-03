# 🇮🇷 Persian News RAG System

![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square)
![Milvus](https://img.shields.io/badge/Vector_DB-Milvus-00a1ea?style=flat-square)
![Elasticsearch](https://img.shields.io/badge/Search-Elasticsearch-005571?style=flat-square)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00?style=flat-square)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ed?style=flat-square)

Production-ready RAG system for Persian news with hybrid search using RRF fusion: dense vectors (Milvus) + sparse BM25 (Elasticsearch).

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────┐
│  Gradio UI      │────▶│  FastAPI Backend                     │
│  (Port 7860)    │     │  - RAG Service (Hybrid Search + RRF) │
└─────────────────┘     │  - LLM Service (OpenAI)              │
                        │  - Embedding Service (GGUF)          │
                        └──────────┬───────────────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        ▼                     ▼
                ┌───────────────┐    ┌────────────────┐
                │    Milvus     │    │ Elasticsearch  │
                │  (Vectors)    │    │  (BM25+ Index) │
                └───────────────┘    └────────────────┘
```

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Frontend | Gradio | UI for search, Q&A, upload |
| Backend | FastAPI | API orchestration |
| Vector DB | Milvus 2.6.5 | Dense embeddings (IVF_PQ) |
| Search Engine | Elasticsearch 9.2.0 | BM25+ + Persian analyzer |
| Embeddings | llama.cpp + GGUF | Local quantized inference |
| LLM | OpenAI API | Answer generation |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- 20GB+ free disk space

### 1. Model Setup

Download and quantize the Qwen3 embedding model:

```bash
chmod +x quantize.sh
./quantize.sh
```

This script will:
1. Download `Qwen/Qwen3-Embedding-0.6B` from Hugging Face
2. Build llama.cpp from source
3. Convert model to GGUF format
4. Quantize to Q4_K_M (reduces size by ~75%)
5. Place final model at `models/qwen3-embed-0.6b-q4_k_m.gguf`

### 2. Environment Configuration

Create `.env` file:

```env
# --- APPLICATION ---
APP_NAME="Persian News RAG System"
APP_VERSION=2.0.0

# --- MILVUS ---
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=persian_news
MILVUS_DIMENSION=1024
MILVUS_INDEX_TYPE=IVF_PQ
MILVUS_NLIST=65536
MILVUS_NPROBE=32
MILVUS_M=64
MILVUS_NBITS=8

# --- ELASTICSEARCH ---
ES_HOST=http://elasticsearch:9200
ES_INDEX_NAME=persian_news_v1
ES_B=0.75
ES_K1=1.2

# --- EMBEDDING ---
EMBEDDING_MODEL_PATH=models/qwen3-embed-0.6b-q4_k_m.gguf
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=64
EMBEDDING_DEVICE=cpu

# --- LLM ---
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-5.1
LLM_TEMPERATURE=0.1
LLM_MAX_COMPLETION_TOKENS=2048

# --- RAG ---
RAG_TOP_K=10
RAG_RRF_K=60
RAG_ALPHA=0.3
RAG_RETRIEVAL_MULTIPLIER=5

# --- UPLOAD ---
UPLOAD_PANDAS_CHUNK_SIZE=256
UPLOAD_TEMP_DIR=storage/temp_uploads
```

### 3. Launch Services

Start the complete stack:

```bash
docker compose up -d --build
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:7860 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Elasticsearch | http://localhost:9200 |
| MinIO Console | http://localhost:9001 |

---

## 📥 Data Ingestion

### CSV Format

| Column | Required | Description |
|--------|----------|-------------|
| `content` | ✅ | Article body |
| `title` | ✅ | Headline |
| `url` | ✅ | Source link |
| `news_agency_name` | ✅ | Publisher |
| `published_at` | ✅ | ISO datetime |
| `topic` | ❌ | Category |
| `categories` | ❌ | Tags |

### Upload

Via UI: Upload Data tab → Select CSV → Start Upload

Via API:
```bash
curl -X POST "http://localhost:8000/api/v1/upload/csv" -F "file=@data.csv"
```

Returns `job_id` for status tracking.

---

## 🔍 Hybrid Search Engine

### Search Algorithm

1. **Parallel Retrieval**: Fetch top-50 from Dense (Milvus) and Sparse (Elasticsearch)
2. **Cross-Scoring**: Fill missing scores for complete coverage
   - Missing dense: Fetch vectors → compute cosine similarity
   - Missing sparse: Query ES with ID filter → get BM25 scores
3. **RRF Fusion**:
   - Convert BM25 scores → ranks → RRF scores: `1/(k + rank)`
   - Normalize both dense and sparse to [0,1]
   - Final: `α × dense + (1-α) × sparse`
4. **Ranking**: Sort by final score, return top-K

### Key Parameters

```env
RAG_ALPHA=0.5           # 0.0=pure BM25, 1.0=pure vector
RAG_RRF_K=60            # RRF smoothing constant
RAG_RETRIEVAL_MULTIPLIER=5  # Fetch 5×top_k candidates
```

### Why Cross-Scoring?

Without cross-scoring, documents only in one retriever get unfairly penalized with score=0. Cross-scoring ensures every document has **both** dense and sparse scores for fair ranking.

---

## 🤖 LLM Integration

System enforces strict RAG guidelines:
- Use ONLY provided documents
- MUST cite with [1], [2] notation
- If no answer: respond "اطلاعات مربوطه در اسناد موجود نیست"
- Dynamic context windowing to fit token limits

---

## 📡 API Reference

### Search

```bash
POST /api/v1/rag/search
```

```json
{
  "query": "تحولات اقتصادی",
  "top_k": 10,
  "min_score": 0.0
}
```

### Q&A

```bash
POST /api/v1/rag/ask
```

```json
{
  "question": "نرخ تورم چقدر است؟",
  "top_k": 5
}
```

### Upload

```bash
POST /api/v1/upload/csv
Content-Type: multipart/form-data
```

Returns `job_id`. Check status:

```bash
GET /api/v1/upload/status/{job_id}
```

### System

```bash
GET /api/v1/system/health
GET /api/v1/system/stats
GET /api/v1/documents/preview?limit=10
```

---

## 📂 Project Structure

```
persian-news-rag/
├── backend/
│   ├── api/
│   │   ├── dependencies.py           # Dependency injection
│   │   └── routers/
│   │       ├── documents.py          # Document preview API
│   │       ├── rag.py                # Search & QA endpoints
│   │       ├── system.py             # Health & stats
│   │       └── upload.py             # CSV upload handler
│   ├── core/
│   │   ├── config.py                 # Pydantic settings
│   │   └── ranking.py                # RRF fusion logic
│   ├── schemas/
│   │   └── schemas.py                # Request/Response models
│   ├── services/
│   │   ├── elasticsearch_service.py  # ES client wrapper
│   │   ├── embedding_service.py      # GGUF inference
│   │   ├── llm_service.py            # OpenAI client
│   │   ├── milvus_client.py          # Milvus operations
│   │   ├── rag_service.py            # Hybrid search orchestration
│   │   └── upload_service.py         # Background CSV processing
│   └── main.py                       # FastAPI application
│
├── frontend/
│   └── gradio_app.py                 # Gradio web UI
│
├── models/                           # GGUF model storage
│   └── qwen3-embed-0.6b-q4_k_m.gguf
│
├── storage/
│   └── temp_uploads/                 # Temporary CSV storage
│
├── volumes/                          # Docker persistent data
│   ├── milvus/
│   ├── elasticsearch/
│   ├── etcd/
│   └── minio/
│
├── docker-compose.yml                # Service orchestration
├── Dockerfile                        # Backend container
├── Dockerfile.frontend               # Frontend container
├── quantize.sh                       # Model setup script
├── requirements.txt                  # Backend dependencies
├── requirements-frontend.txt         # Frontend dependencies
├── .env                              # Configuration
└── README.md
```

---

## ⚙️ Configuration Tuning

### Milvus Index

```env
# Small dataset (<1M)
MILVUS_INDEX_TYPE=IVF_SQ8
MILVUS_NLIST=4096

# Large dataset (>1M, default)
MILVUS_INDEX_TYPE=IVF_PQ
MILVUS_NLIST=65536
MILVUS_M=64
```

### Elasticsearch Memory

```yaml
# docker-compose.yml
ES_JAVA_OPTS=-Xms4g -Xmx4g  # Increase for large datasets
```

### Embedding Performance

```env
# CPU
EMBEDDING_BATCH_SIZE=32
EMBEDDING_DEVICE=cpu

# GPU
EMBEDDING_BATCH_SIZE=128
EMBEDDING_DEVICE=cuda
```

---

## 🐛 Troubleshooting

**ES permission denied**
```bash
docker compose run --rm setup
```

**Milvus "index not found"**  
Wait for more data or lower threshold:
```env
MILVUS_INDEX_BUILD_THRESHOLD=100
```

**OOM during upload**  
Reduce chunk size:
```env
UPLOAD_PANDAS_CHUNK_SIZE=64
```

---

## 📄 License

MIT License