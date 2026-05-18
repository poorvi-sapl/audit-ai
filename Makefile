.PHONY: up down logs ps smoke install install-dev spacy-model env-check

COMPOSE = docker compose -f docker/docker-compose.yml

# ── Docker services ───────────────────────────────────────────────────────────

up:
	$(COMPOSE) up -d
	@echo "Waiting for services to be healthy..."
	@$(COMPOSE) ps

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

# ── Python environment ────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

install-dev:
	pip install -e ".[dev,ocr]"

spacy-model:
	python -m spacy download en_core_web_lg

env-check:
	@test -f .env || (echo "ERROR: .env not found. Copy config/.env.template to .env and fill in values." && exit 1)
	@echo ".env found."

# ── Tests ─────────────────────────────────────────────────────────────────────

smoke: env-check
	pytest tests/test_smoke.py -v

test:
	pytest tests/ -v --cov=auditai_data_normalization --cov=engineering_benchmark --cov=raw_to_training_pair --cov-report=term-missing

# ── Ollama model pull (run once on dev machine) ───────────────────────────────

pull-llm:
	ollama pull gemma3:12b

# ── DVC & JSONL versioning ────────────────────────────────────────────────────

dvc-init:
	dvc init
	dvc remote add -d local_remote /tmp/auditai_dvc_remote
