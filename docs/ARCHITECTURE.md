# MedArchive RAG Architecture

**Enterprise-Grade Clinical Decision Support System**

---

## System Overview

MedArchive RAG is a **Retrieval-Augmented Generation (RAG)** system designed specifically for clinical decision support. Unlike traditional chatbots that may hallucinate medical information, this system grounds every answer in verified institutional guidelines with full source attribution.

### Core Principles

1. **Zero-Hallucination Design**: All responses must cite source documents
2. **Sub-Second Performance**: 300ms target latency for clinical workflows
3. **Table-Aware Intelligence**: Preserve complex medical tables (dosage charts, lab ranges)
4. **Microservices Architecture**: Independent scaling of API and ingestion
5. **Dev/Prod Parity**: Same Docker containers from local dev to production AKS

---

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Production System                             │
│                     (Azure Kubernetes Service)                         │
└───────────────────────────────────────────────────────────────────────┘

                              Internet
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Azure Load Balancer   │
                    │    (TLS Termination)    │
                    └────────┬───────────────┘
                             │
                ┌────────────┴───────────────┐
                │                            │
                ▼                            ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │   API Service Pods   │   │ Ingestion Worker Pod │
    │   (HPA: 2-10 pods)   │   │   (StatefulSet: 1)   │
    │                      │   │                      │
    │  ┌───────────────┐  │   │  ┌───────────────┐  │
    │  │ FastAPI       │  │   │  │ Background    │  │
    │  │ Embedding     │  │   │  │ PDF Processor │  │
    │  │ Reranker      │  │   │  │ LlamaParse    │  │
    │  │ Groq Client   │  │   │  │ Chunking      │  │
    │  └───────┬───────┘  │   │  └───────┬───────┘  │
    └──────────┼──────────┘   └──────────┼──────────┘
               │                          │
               │                          │
               └──────────┬───────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Qdrant Cluster       │
              │   (3-node Sharding)    │
              │   Binary Quantization  │
              └────────────────────────┘
                          │
                          │ (Persistent Volumes)
                          ▼
              ┌────────────────────────┐
              │   Azure Blob Storage   │
              │   (Document Store)     │
              └────────────────────────┘

     ┌──────────────────────────────────────────┐
     │         External Services                 │
     ├──────────────────────────────────────────┤
     │  • Groq API (LLM Inference)              │
     │  • LlamaParse (PDF Processing)           │
     │  • Azure Key Vault (Secrets)             │
     │  • Azure Monitor (Observability)         │
     │  • Arize Phoenix (RAG Tracing) [Phase 5] │
     └──────────────────────────────────────────┘
```

---

## Data Flow

### Query Path (API Service)

```
1. User Query Received
   └─▶ FastAPI Endpoint (/api/v1/query)

2. Query Embedding
   └─▶ sentence-transformers (BAAI/bge-large-en-v1.5)
       └─▶ 1024-dimensional dense vector

3. Stage 1: Wide Net Retrieval
   └─▶ Hybrid Search in Qdrant
       ├─▶ Dense Vector Search (semantic)
       └─▶ Sparse Vector Search (BM25 keywords)
       └─▶ Returns top 50 chunks

4. Stage 2: Reranking
   └─▶ BGE-Reranker-v2-m3
       └─▶ Re-scores based on query-chunk semantic alignment
       └─▶ Returns top 5 chunks

5. Context Injection
   └─▶ Format chunks into LLM prompt
       └─▶ System: "You are a clinical assistant. Cite sources."
       └─▶ Context: [Chunk 1] [Source: Doc X, p. Y]
       └─▶ Query: User's question

6. LLM Generation
   └─▶ Groq Llama-3.3-70B (280 tokens/sec)
       └─▶ Streaming response
       └─▶ Citation extraction

7. Response Assembly
   └─▶ QueryResponse model
       ├─▶ answer: Generated text
       ├─▶ citations: [Document, Page, Snippet, Score]
       ├─▶ latency_ms: Performance tracking
       └─▶ timestamp: For audit logs
```

### Ingestion Path (Background Worker)

```
1. File Discovery
   └─▶ Watch /data/document_store
       └─▶ Compute MD5 hash of each PDF
       └─▶ Compare with stored hashes (incremental sync)

2. PDF Parsing (LlamaParse)
   └─▶ Upload PDF to LlamaParse API
       └─▶ Result: Markdown with preserved tables
       Example:
       | Drug | Pediatric Dose | Adult Dose |
       |------|---------------|------------|
       | Amox | 15mg/kg BID   | 500mg TID  |

3. Metadata Extraction
   └─▶ pypdf: Extract title, author, page count
   └─▶ Custom logic: Parse department, effective date
   └─▶ Create DocumentMetadata record

4. Semantic Chunking
   └─▶ RecursiveCharacterTextSplitter
       ├─▶ Target: 1024 tokens
       ├─▶ Overlap: 200 tokens
       ├─▶ Split on: Headers, paragraphs, sentences
       └─▶ Keep sections intact (e.g., "Adverse Effects")

5. Metadata Enrichment
   └─▶ For each chunk:
       ├─▶ section_path: "Cardiology > Heart Failure > Acute"
       ├─▶ heading: "Acute Decompensation Protocol"
       ├─▶ page_numbers: [23, 24]
       └─▶ document_id: UUID reference

6. Embedding Generation
   └─▶ sentence-transformers (BAAI/bge-large-en-v1.5)
       └─▶ Batch processing (32 chunks/batch)
       └─▶ Output: float32[1024] vectors

7. Vector Indexing (Qdrant)
   └─▶ Upsert to collection "medarchive_documents"
       ├─▶ Vector: Dense embedding
       ├─▶ Payload: ChunkMetadata JSON
       └─▶ Binary Quantization applied

8. Atomic Update
   └─▶ If file changed: Delete old chunks, insert new
   └─▶ Update DocumentMetadata.updated_at
   └─▶ Set ProcessingStatus.COMPLETED
```

---

## Technology Stack Rationale

| Technology | Why This Choice? | Alternatives Considered |
|------------|------------------|------------------------|
| **FastAPI** | Async-native, auto-validation, OpenAPI docs | Flask (sync), Django (heavy) |
| **Qdrant** | Binary quantization, hybrid search, easy deployment | Pinecone (vendor lock-in), Milvus (complex) |
| **Groq** | 280 tok/sec (10x faster than OpenAI), LLaMA 3.3 | OpenAI (slow, expensive), Local LLama (infra heavy) |
| **LlamaParse** | Best-in-class table extraction, medical doc optimized | Unstructured.io (less accurate), pypdf (no tables) |
| **BGE-Large** | Top-ranked on MTEB, medical domain strong | OpenAI embeddings (cost), Instructor-XL (slower) |
| **Poetry** | Modern dependency management, lockfile determinism | pip-tools (manual), conda (slower) |
| **Pydantic v2** | Runtime validation, 17x faster than v1, JSON Schema | Marshmallow (manual), dataclasses (no validation) |

---

## Microservices Design

### API Service

**Responsibilities:**
- Serve query requests from users/frontends
- Orchestrate retrieval pipeline
- Handle LLM inference
- Return structured responses

**Scaling Strategy:**
- Horizontal Pod Autoscaler (HPA) in AKS
- Scale on CPU > 70% or Request Rate > 100/sec
- Stateless: Can have 2-10 pods

**Resource Requirements:**
- CPU: 1-2 cores per pod
- Memory: 4GB (embedding model in memory)
- Disk: Minimal (no persistence)

### Ingestion Service

**Responsibilities:**
- Monitor document store for changes
- Parse PDFs through LlamaParse
- Chunk and embed documents
- Index into Qdrant

**Scaling Strategy:**
- StatefulSet with 1 replica (file sync coordination)
- Can add workers for parallel processing in future

**Resource Requirements:**
- CPU: 2-4 cores (embedding computation)
- Memory: 8GB (large PDFs + model)
- Disk: 50GB (document cache)

---

## Phase Roadmap

### **Phase 1: Foundation** **COMPLETE**
- Git repository and project structure
- Poetry dependency management
- Docker infrastructure (multi-stage builds)
- Shared models and utilities
- API/Ingestion scaffolding
- Testing infrastructure

### **Phase 2: Ingestion Pipeline** (Next)
**Goal:** Convert PDFs into searchable semantic chunks

**Tasks:**
1. LlamaParse integration for table-aware parsing
2. RecursiveCharacterTextSplitter for semantic chunking
3. Metadata enrichment (section paths, page numbers)
4. File hashing for incremental sync
5. Embedding model integration (sentence-transformers)
6. Basic Qdrant indexing (without BQ initially)

**Deliverables:**
- Functional PDF → Qdrant pipeline
- Ingestion worker processes documents in `/data/document_store`
- Unit tests for parsing and chunking
- Integration test: End-to-end ingestion of sample PDF

### **Phase 3: Vector Storage & Retrieval**
**Goal:** Sub-millisecond semantic search

**Tasks:**
1. Qdrant collection setup with schema
2. Binary Quantization configuration
3. Hybrid search (dense + sparse vectors)
4. BM25 integration for keyword matching
5. Retrieval API endpoint implementation
6. Performance benchmarking

**Deliverables:**
- Query endpoint returns top-K chunks
- Hybrid search functional
- Latency < 100ms for retrieval

### **Phase 4: Two-Stage Retrieval & LLM**
**Goal:** Fast, accurate answers with citations

**Tasks:**
1. Reranker integration (BGE-Reranker-v2-m3)
2. Groq client setup
3. Prompt engineering for citations
4. Streaming response implementation
5. Citation extraction logic
6. Query endpoint complete

**Deliverables:**
- End-to-end RAG working
- Streaming responses
- Citation accuracy > 95%

### **Phase 5: Evaluation & Safety**
**Goal:** Prove system is safe for clinical use

**Tasks:**
1. RAGAS evaluation framework
2. Faithfulness tests (hallucination detection)
3. Context precision metrics
4. Arize Phoenix tracing integration
5. Feedback loop (thumbs up/down)
6. Benchmark suite with clinical Q&A

**Deliverables:**
- RAGAS score > 0.85
- Observability dashboard
- Automated regression tests

### **Phase 6: Production Deployment**
**Goal:** Deploy to Azure Kubernetes Service

**Tasks:**
1. Helm charts for AKS
2. Azure DevOps CI/CD pipeline
3. Horizontal Pod Autoscaler
4. Azure Key Vault integration
5. Azure Monitor / Application Insights
6. Rolling update strategy
7. Disaster recovery plan

**Deliverables:**
- Production-ready AKS deployment
- CI/CD pipeline: git push → production
- SLA monitoring

---

## Security Considerations

### Phase 1-4 (Development)
- **Secrets Management**: `.env` files (git-ignored)
- **API Keys**: Groq, LlamaParse in environment variables
- **Network**: Docker bridge network isolation

### Phase 6 (Production)
- **Secrets Management**: Azure Key Vault
- **Authentication**: OAuth2 / Azure AD
- **Network**: Private AKS cluster, no public IPs
- **Encryption**: TLS 1.3 in transit, encrypted PVs at rest
- **RBAC**: Kubernetes role-based access control
- **Scanning**: Trivy for container vulnerabilities
- **Audit Logs**: Azure Monitor for compliance

---

## Monitoring & Observability

### Metrics to Track
- **Latency**: P50, P95, P99 query times
- **Throughput**: Queries per second
- **Error Rate**: 5xx responses / total requests
- **Retrieval Quality**: Citation accuracy, relevance scores
- **Ingestion**: Documents processed/hour, parse failures

### Phase 5+ Tooling
- **Application Insights**: API metrics, distributed tracing
- **Qdrant Dashboard**: Vector DB health, index size
- **Arize Phoenix**: RAG-specific tracing (query → retrieval → generation)
- **Prometheus + Grafana**: Custom dashboards (Phase 6)

---

## Cost Estimation (Monthly)

### Development (Local)
- **Compute**: Free (local Docker)
- **Groq API**: ~$5 (generous free tier)
- **LlamaParse**: ~$10 (1000 pages/mo)
- **Total**: ~$15/month

### Production (AKS - Assuming 1000 queries/day)
- **AKS Cluster**: $150 (2x Standard_D4s_v3 nodes)
- **Qdrant**: $50 (self-hosted on AKS)
- **Groq API**: $50 (assuming 30K queries × $0.27/1M tokens)
- **LlamaParse**: $100 (10K pages/mo for updates)
- **Azure Storage**: $20 (Blob + PV)
- **Total**: ~$370/month

**Cost per Query**: $0.012 (vs. $0.10+ for GPT-4 RAG)

---

## Future Enhancements (Post-Phase 6)

1. **Multi-Modal Support**: Process medical images (X-rays, charts)
2. **Federated Search**: Query multiple hospitals' knowledge bases
3. **Real-Time Updates**: WebSocket for guideline change notifications
4. **Mobile App**: iOS/Android with offline mode
5. **Voice Interface**: Speech-to-text for hands-free queries
6. **FHIR Integration**: Link to patient records for personalized guidance

---

## References

- **RAG Best Practices**: [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/rag/)
- **Medical NLP**: [ClinicalBERT Paper](https://arxiv.org/abs/1904.03323)
- **Vector DB Comparison**: [Qdrant Benchmarks](https://qdrant.tech/benchmarks/)
- **Groq Performance**: [Groq vs OpenAI Latency](https://groq.com)

---

**Next Steps**: Proceed to [DEVELOPMENT.md](DEVELOPMENT.md) for local setup instructions.
