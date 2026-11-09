.PHONY: help setup install ingest index eval api ui test lint fmt typecheck seed docker-build docker-up docker-down clean

PYTHON := python
UV := uv
SRC := src/hybrid_rag
STRATEGY ?= structural
DATE := $(shell python -c "import datetime; print(datetime.date.today().isoformat())")

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────────────────────
setup: ## Create venv and install all dependencies
	$(UV) sync --all-extras
	@echo "✓ Environment ready. Activate with: .venv\\Scripts\\activate"

install: setup ## Alias for setup

# ── Data & Index ──────────────────────────────────────────────────────────────
download: ## Download pinned HF docs corpus
	$(UV) run python scripts/download_corpus.py

demo-bank: ## Generate synthetic refusal-test corpus
	$(UV) run python scripts/make_demo_bank.py

ingest: ## Normalize raw docs → data/processed/
	$(UV) run python -m hybrid_rag.ingest.normalize

index: ## Build both indexes (STRATEGY=structural|fixed|semantic)
	$(UV) run python -m hybrid_rag.index.build --strategy $(STRATEGY)

index-all: ## Build indexes for all three chunking strategies
	$(UV) run python -m hybrid_rag.index.build --strategy fixed
	$(UV) run python -m hybrid_rag.index.build --strategy structural
	$(UV) run python -m hybrid_rag.index.build --strategy semantic

seed: download ingest index ## Full pipeline: download → ingest → index (structural)
	$(UV) run python -m hybrid_rag.cli smoke-test

# ── Evaluation ────────────────────────────────────────────────────────────────
eval-retrieval: ## Retrieval-only eval on golden set
	$(UV) run python -m hybrid_rag.eval.run_eval --retrieval-only --out reports/retrieval_$(DATE).md

eval-full: ## Full eval: retrieval + generation + citations + confidence
	$(UV) run python -m hybrid_rag.eval.run_eval --out reports/full_$(DATE).md

eval: eval-retrieval eval-full ## Run all evaluations

# ── Serving ───────────────────────────────────────────────────────────────────
api: ## Start FastAPI server (dev mode)
	$(UV) run uvicorn hybrid_rag.api.main:app --reload --host 0.0.0.0 --port 8080

ui: ## Start Streamlit dashboard
	$(UV) run streamlit run app/dashboard.py --server.port 8501

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run all tests
	$(UV) run pytest tests/ -v

test-unit: ## Run unit tests only (no LLM calls)
	$(UV) run pytest tests/ -v -m "not integration"

test-cov: ## Run tests with coverage report
	$(UV) run pytest tests/ --cov=$(SRC) --cov-report=html --cov-report=term-missing

# ── Code Quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	$(UV) run ruff check $(SRC) tests scripts app

fmt: ## Auto-format with ruff
	$(UV) run ruff format $(SRC) tests scripts app
	$(UV) run ruff check --fix $(SRC) tests scripts app

typecheck: ## Run mypy type checker
	$(UV) run mypy $(SRC)

check: lint typecheck test ## Run lint + typecheck + tests

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -t hybrid-rag-hf-docs:latest .

docker-up: ## Start all services with docker-compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Follow logs
	docker compose logs -f

docker-seed: ## Seed data inside running containers
	docker compose exec api python -m hybrid_rag.cli seed

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove generated artifacts (keeps data/)
	rm -rf .chroma/ chroma_data/ htmlcov/ .coverage .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete

clean-all: clean ## Remove everything including data/processed
	rm -rf data/processed/ data/raw/
	@echo "⚠  Raw and processed data removed. Run 'make seed' to rebuild."
