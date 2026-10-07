# ResearchMind AI - Backend (Phase 7: Global Optimization & Verification)

ResearchMind AI is an intelligent research paper retrieval, ranking, and knowledge synthesis system.

## Features Across Phases

- **Backend Foundation (Phase 1):**
  - Modular FastAPI application architecture
  - Centralized environment configuration via Pydantic Settings
  - Global API versioning under `/api/v1`
  - CORS middleware supporting Next.js frontend
  - Automated interactive OpenAPI documentation (`/docs`, `/redoc`)
  - Health check (`/api/v1/health`) and root endpoint (`/`)
- **Scholarly Source Adapters (Phase 2):**
  - **Semantic Scholar Adapter:** Asynchronous query fetching metadata, authors, DOI, citations, and venue
  - **OpenAlex Adapter:** Asynchronous query fetching works with abstract reconstruction from `abstract_inverted_index`
  - **Resilient Transport:** Configurable timeouts, exponential backoff retries, and HTTP 429 rate-limit handling with `Retry-After` parsing
  - **Provider Failure Isolation:** Asynchronous parallel execution via `asyncio.gather` so failure of one provider does not crash the search
  - **Security & Privacy:** Backend-only API keys, automatic header sanitization in logs
- **Canonical Research Data Layer (Phase 3):**
  - **Canonical Model (`ResearchDocument`):** Standard representation across all scholarly providers with full metadata
  - **Normalization Pipeline (`normalize.py`):** Title cleaning, DOI prefix stripping, year coercion, author deduplication, code repository detection, and open-access extraction
  - **Deterministic IDs:** Deterministic UUID generation based on DOI, title, or provider tokens
  - **Deduplication Engine (`deduplicate.py`):** 5-stage priority matching:
    1. DOI
    2. Provider Identifier
    3. Normalized exact title
    4. Title + Author + Year similarity
    5. Semantic similarity safeguard
  - **Provenance Preservation:** Retains complete multi-source audit trails (`ProvenanceRecord`) when merging records
  - **Query Cache Abstraction (`cache.py`):** In-memory TTL cache keyed by normalized provider, query, filters, and pagination
- **Hybrid Retrieval Pipeline (Phase 4):**
  - **BM25 Lexical Retrieval (`bm25.py`):** Okapi BM25 implementation supporting exact technical terms, hyphenated multi-token compounds, uppercase acronyms, and identifiers (DOIs, ArXiv IDs).
  - **SBERT Semantic Retrieval (`sbert.py`):** Dense semantic embeddings with cosine similarity. Pluggable design supports `sentence-transformers` model weights with an instant, deterministic, zero-download fallback.
  - **Reciprocal Rank Fusion (`rrf.py`):** Scale-invariant rank fusion:
    $$RRF\_score(d) = \sum_{m \in M} \frac{w_m}{k + rank_m(d)}$$
    Prevents raw BM25 score magnitudes from overpowering dense semantic similarities ($k=60$ default).
  - **Bounded Candidate Pool (`pipeline.py`):** Configurable candidate bounds (`RETRIEVAL_CANDIDATE_LIMIT`, default 20) feeding downstream rerankers.
  - **Retrieval Metadata (`RetrievalMetadata`):** Every ranked candidate includes comprehensive audit metrics (`bm25_rank`, `bm25_score`, `sbert_rank`, `sbert_score`, `rrf_rank`, `rrf_score`).
- **Cross-Encoder Reranking & Features (Phase 5):**
  - **Cross-Encoder Reranker (`reranker.py`):** Evaluates joint query-document cross-attention pairs to rerank the bounded Phase 4 candidate pool.
  - **Batch Inference & Device Support:** Configurable batching (`batch_size=16`) supporting CPU and CUDA GPU execution.
  - **Score Normalization:** Sigmoid normalization $\sigma(x) = \frac{1}{1 + e^{-x}}$ mapping raw logits to $[0.0, 1.0]$.
  - **Execution Telemetry:** Returns `pre_rerank_rank`, `reranker_score`, `post_rerank_rank`, and `model_metadata` with latency measurements in milliseconds.
  - **Graceful Fault Tolerance:** Catches hardware OOM or model errors, preserving pre-rerank RRF ordering with `is_fallback=True`.
  - **Intermediate Relevance Features (`features.py`):** Formulates Cross-Encoder score as an intermediate relevance feature alongside BM25, SBERT, RRF, citation log-velocity, and exponential recency decay.
- **Feedback, Preferences, Constraints & Utility (Phase 6):**
  - **User Feedback Layer (`app/feedback/`):**
    - Supported feedback: `relevant`, `not_relevant`, `prefer_similar`, `exclude_similar`, and `pairwise_preference`.
    - Strict contradiction detector (`validator.py`): Rejects direct pointwise conflicts, cycle preferences, and inversion contradictions without silent resolution.
    - Matching engine (`matcher.py`): Scores candidates based on direct and similarity feedback signals.
  - **8 Preference Dimensions (`app/preferences/`):** Relevance, implementation relevance, recency, compute requirements (low_compute, balanced, high_compute), dataset availability, source-code availability, pretrained model availability, and source quality.
  - **Constraints & Penalties (`app/optimization/constraints.py`):** Hard required filters (`require_code`, `require_open_access`, `min_year`, `max_year`, `min_citations`, `required_venues`), hard exclusions, and soft constraints with calibrated penalties.
  - **Formal Multi-Criteria Utility Model (`app/optimization/utility.py`):**
    $$U(R) = w_1 \cdot \text{relevance} + w_2 \cdot \text{preference\_match} + w_3 \cdot \text{feedback\_match} + w_4 \cdot \text{buildability} + w_5 \cdot \text{source\_quality} + w_6 \cdot \text{recency} - \text{penalties}$$
  - **Audit Run Persistence (`UtilityRunStore`):** Persists exact inputs, weights, and component evaluations for each ranking run under immutable `run_id`.
- **Global Optimization & Independent Verification (Phase 7):**
  - **Formal Problem:**
    $$R^* = \arg\max_{R \in \mathcal{S}_n} U(R \mid q, D, F, P, C)$$
  - **Epistemic Guarantee Scope:**
    *Strictly globally optimal over the retrieved candidate set under the explicitly defined utility and constraints. Does not claim universal optimality over all documents or undefined human preferences.*
  - **Stage 1 (Linear Assignment Solver):**
    Solves linear positional ranking using the Hungarian algorithm via `scipy.optimize.linear_sum_assignment`.
  - **Stage 2 (Pairwise Diversity & Richer Constraints):**
    Formulates redundancy penalties $-\lambda \sum_{j < l} S_{i, m} \gamma_j \gamma_l$ and required document inclusion as a Mixed-Integer Linear Program (MILP) solved via `scipy.optimize.milp` with McCormick envelope linearization.
  - **Ground Truth Baseline (`ExhaustivePermutationSolver`):**
    Evaluates all permutations for $N \le 10$ to prove exact optimality.
  - **Approximation Gap Recording:**
    Quantifies greedy/approximate solution distance from exact optimum:
    $$\Delta U = U(R_{\text{exact}}) - U(R_{\text{approx}}) \ge 0.0$$
  - **Independent Verification (`OptimizationVerifier`):**
    Re-evaluates objective values independently, enforces hard constraint satisfaction, and records comprehensive telemetry.
  - **Endpoints:** `/api/v1/optimize`, `/api/v1/search` (with `optimize=True`).
- **Research Intelligence (Phase 8):**
  - **Explainability (`explainer.py`):**
    - Decomposes utility scores into 7 verified components: relevance, preference match, feedback match, buildability, source quality, recency, and penalties.
    - Strictly generated ONLY from actual stored ranking inputs and run audit records.
  - **Buildability (`buildability.py`):**
    - 9-dimensional engineering assessment: dataset availability, source-code availability, pretrained model availability, documentation, hardware requirements, dependency complexity, license/accessibility, reproducibility, implementation completeness.
    - Returns scalar scores plus concrete, factual evidence and reproducibility tiers (High, Moderate, Low, Blocked).
  - **Knowledge Graph (`graph.py`):**
    - Multi-entity graph representing: Paper -> Dataset, Paper -> Model, Paper -> Method, Paper -> Repository, Paper -> Related Paper (citations & conceptual similarity).
    - Preserves data provenance (`GraphProvenance`) for every node and edge.
  - **Engineering Roadmap (`roadmap.py`):**
    - 7 ordered execution milestones: dataset preparation, baseline, model training, evaluation, backend, frontend integration, deployment.
    - Explicitly links milestones to supporting research papers, datasets, checkpoints, and code repositories.
  - **Endpoints:** `/api/v1/intelligence/explain`, `/api/v1/intelligence/buildability`, `/api/v1/intelligence/graph`, `/api/v1/intelligence/roadmap`, `/api/v1/intelligence/report`.

---

## Project Structure

```
backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entrypoint & CORS configuration
│   ├── config.py            # Environment settings with Pydantic
│   │
│   ├── api/                 # API route controllers
│   │   ├── __init__.py      # Router aggregation
│   │   ├── health.py        # Health check endpoint
│   │   ├── search.py        # POST /api/v1/search endpoint
│   │   ├── utility.py       # Feedback validation & utility evaluation endpoints
│   │   ├── optimization.py  # POST /api/v1/optimize endpoint
│   │   └── intelligence.py  # Phase 8 research intelligence endpoints
│   │
│   ├── ingestion/           # Data ingestion & canonical normalization
│   │   ├── __init__.py
│   │   ├── normalize.py     # Clean & map to canonical schema
│   │   ├── deduplicate.py   # Multi-stage deduplication & merging
│   │   └── cache.py         # Provider query caching abstraction
│   │
│   ├── retrieval/           # Hybrid retrieval & reranking (Phases 4 & 5)
│   │   ├── __init__.py
│   │   ├── bm25.py          # BM25 lexical retriever with technical tokenization
│   │   ├── sbert.py         # Dense semantic retriever & cosine similarity
│   │   ├── rrf.py           # Reciprocal Rank Fusion combiner
│   │   ├── reranker.py      # Cross-Encoder reranker with batching & fallback
│   │   ├── features.py      # Multi-modal intermediate relevance feature extraction
│   │   └── pipeline.py      # End-to-end hybrid retrieval & reranking pipeline
│   │
│   ├── feedback/            # User feedback models, validation & matching (Phase 6)
│   │   ├── __init__.py
│   │   ├── models.py        # FeedbackType, FeedbackItem, FeedbackCollection
│   │   ├── validator.py     # Contradiction detection & validation
│   │   └── matcher.py       # Feedback matching score computation
│   │
│   ├── preferences/         # User multi-dimensional preferences (Phase 6)
│   │   ├── __init__.py
│   │   ├── models.py        # UserPreferences across 8 dimensions
│   │   └── evaluator.py     # Preference match evaluator
│   │
│   ├── optimization/        # Optimization layer & verification (Phase 7)
│   │   ├── __init__.py
│   │   ├── constraints.py   # Hard filters, exclusions, soft penalties & contradictions
│   │   ├── utility.py       # Formal utility function U(R), evaluation & run persistence
│   │   ├── models.py        # OptimizationProblem, OptimizationResult, VerificationReport
│   │   ├── solvers.py       # LinearAssignment, PairwiseDiversityILP, Exhaustive & Greedy
│   │   ├── verification.py  # Independent verification engine & approximation gap audit
│   │   └── optimizer.py     # GlobalRankingOptimizer dispatcher
│   │
│   ├── schemas/             # Pydantic models
│   │   ├── __init__.py
│   │   ├── common.py        # HealthResponse & RootResponse
│   │   ├── paper.py         # Canonical ResearchDocument & ProvenanceRecord
│   │   ├── retrieval.py     # RetrievalMetadata, RerankMetadata, RetrievedCandidate
│   │   └── source.py        # SearchRequest, SearchResponse, SourcePaper
│   │
│   ├── services/            # Business logic layer
│   │   ├── __init__.py
│   │   └── health_service.py
│   │
│   └── sources/             # Scholarly source adapters
│       ├── __init__.py
│       ├── base.py          # BaseSourceAdapter, retry, backoff, 429 & errors
│       ├── semantic_scholar.py # Semantic Scholar Academic Graph adapter
│       └── openalex.py      # OpenAlex Works adapter
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Session test fixtures
│   ├── test_health.py       # Health & root route tests
│   ├── test_sources.py      # Adapter unit tests (mocked HTTP, retries, 429)
│   ├── test_search.py       # Search API & provider failure isolation tests
│   ├── test_ingestion.py    # Normalization, deduplication, cache & provenance tests
│   ├── test_retrieval.py    # BM25, SBERT, RRF, candidate limits & determinism tests
│   ├── test_reranker.py     # Cross-Encoder reranking, batching, limits & fallback tests
│   ├── test_feedback_utility.py # Feedback validation & utility run audit tests
│   └── test_optimization.py # Stage 1, Stage 2 MILP, Exhaustive & Verification tests
│
├── .env                     # Local environment file (git-ignored)
├── .env.example             # Template environment variables
├── .gitignore               # Git ignore rules
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── Dockerfile               # Production container image
```

---

## API Endpoints

### 1. Global Ranking Optimization (`POST /api/v1/optimize`)

- **Method:** `POST`
- **Path:** `/api/v1/optimize`
- **Request Body:**
  ```json
  {
    "candidates": [...],
    "k": 5,
    "objective_type": "pairwise_diversity",
    "diversity_lambda": 0.3,
    "preferences": {
      "importance_relevance": 0.8,
      "importance_recency": 0.4
    },
    "constraints": {
      "min_year": 2020,
      "require_code": true
    }
  }
  ```
- **Response Body:**
  ```json
  {
    "ranked_candidates": [...],
    "objective_value": 1.4047,
    "solver_used": "ilp_diversity",
    "solver_status": "optimal",
    "verification": {
      "is_verified": true,
      "verification_status": "verified_optimal",
      "solver_status": "optimal",
      "candidate_count": 4,
      "k": 2,
      "solver_used": "ilp_diversity",
      "objective_type": "pairwise_diversity",
      "reported_objective": 1.4047,
      "recomputed_objective": 1.4047,
      "is_objective_match": true,
      "constraints_satisfied": true,
      "constraint_violations": [],
      "exhaustive_comparison_performed": true,
      "exact_reference_objective": 1.4047,
      "approximation_gap": 0.0,
      "details": {
        "algorithm": "MILP (HiGHS)",
        "num_variables": 20,
        "num_constraints": 18
      }
    },
    "guarantee_scope": "globally optimal over the retrieved candidate set under the explicitly defined utility and constraints"
  }
  ```

- **Research Intelligence (Phase 8):**
  - **Explainability (`app/intelligence/explainer.py`):** Decomposes utility scores into individual component contributions with grounded narrative explanations generated strictly from stored ranking inputs.
  - **Buildability (`app/intelligence/buildability.py`):** Comprehensive 9-dimensional engineering feasibility scoring (datasets, code repositories, pretrained checkpoints, dependencies, hardware envelopes, license compatibility) returning calibrated scores $[0.0, 1.0]$ and concrete evidence.
  - **Knowledge Graph (`app/intelligence/graph.py`):** Multi-entity knowledge graph connecting Papers $\to$ Datasets, Models, Methods, Repositories, and Citations with explicit provenance and confidence weights.
  - **Implementation Roadmap (`app/intelligence/roadmap.py`):** 7-stage engineering implementation plan spanning Dataset Preparation, Baselines, Model Training, Evaluation, Backend, Frontend Integration, and Deployment.
- **Production Integration & Evaluation (Phase 9):**
  - **PostgreSQL Persistence (`app/db/`):** Full async database models and repository via SQLAlchemy & `asyncpg` (with SQLite async fallback) persisting `ResearchDocument`, multi-source provenance, saved papers library, user feedback, user preferences, utility runs, optimization results, and research roadmaps.
  - **Vector Layer (`app/vector/`):** High-performance normalized cosine similarity index (`VectorIndex`) supporting document ID mapping, embedding model versioning, index versioning, disk serialization, and dynamic rebuild capability.
  - **Connected Frontend Functionality:**
    - Full paper details: `GET /api/v1/papers/{id}`
    - Saved papers library: `GET /api/v1/papers/saved`, `POST /api/v1/papers/saved`, `DELETE /api/v1/papers/saved/{id}`
    - User feedback: `GET /api/v1/feedback`, `POST /api/v1/feedback`
    - Multi-dimensional preferences: `GET /api/v1/preferences`, `PUT /api/v1/preferences`
    - Intelligence reports: `POST /api/v1/intelligence/full-report`
  - **Security & Hardening:**
    - Upstream scholarly provider keys isolated server-side
    - RateLimiterMiddleware (`RATE_LIMIT_REQUESTS_PER_MINUTE=120`)
    - Strict input validation via Pydantic v2 schemas
    - Safe logging with masked credentials and headers
    - Production CORS configurations
  - **Observability & Telemetry (`app/observability/`):**
    - Real-time latency tracking across source retrieval, hybrid retrieval, Cross-Encoder reranking, and global optimization
    - Cache hit rate monitoring
    - Provider error logging and breakdown
    - Candidate pool statistics and solver status tracking via `GET /api/v1/observability/metrics`
  - **IR Evaluation & Benchmarks (`app/evaluation/`):**
    - Precision@K, Recall@K, and NDCG@K
    - Search success rate across query sessions
    - User satisfaction scoring from feedback signals
    - Optimization verification rate
    - Buildability accuracy against ground truth benchmarks via `POST /api/v1/evaluation/benchmark`
  - **Docker & Deployment:**
    - Multi-stage hardened production `Dockerfile` with non-root security and container healthchecks
    - `docker-compose.yml` orchestrating FastAPI backend and PostgreSQL 16 Alpine service
    - Deployment documentation in `docs/deployment.md`

---

## Production Deployment with Docker Compose

To launch the backend API and PostgreSQL services in production:

```bash
docker compose up -d --build
```

### Health & Telemetry Verification:
```bash
# Check service and database status
curl -X GET "http://localhost:8000/api/v1/health"

# Inspect database connectivity
curl -X GET "http://localhost:8000/api/v1/health/db"

# View real-time observability telemetry
curl -X GET "http://localhost:8000/api/v1/observability/metrics"
```

---

## Running Automated Tests

Run the full automated test suite covering all 9 phases:

```bash
pytest -v
```
All 99 unit, integration, and end-to-end tests pass cleanly.
