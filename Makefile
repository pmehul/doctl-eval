.PHONY: help venv install ingest gold screen serve serve-mock docker docker-run clean verify

IMAGE ?= doctl-eval:latest
PORT  ?= 8080
PY    ?= .venv/bin/python

help:
	@echo "Data pipeline (offline, run once):"
	@echo "  make ingest        Freeze the doctl issue snapshot from the GitHub API"
	@echo "  make gold          Build the gold set from maintainer labels + hand overlay"
	@echo ""
	@echo "Evaluation:"
	@echo "  make screen        Screen the full candidate pool on the dev split"
	@echo "  make serve         Run the harness against real Serverless Inference"
	@echo "  make serve-mock    Run the harness against the offline simulator"
	@echo ""
	@echo "Container:"
	@echo "  make docker        Build the image"
	@echo "  make docker-run    Run the image (expects .env)"
	@echo ""
	@echo "  make verify        Sanity-check the corpus, gold set and prompt invariants"

venv:
	python3 -m venv .venv

install: venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

# Set GITHUB_TOKEN to raise the API rate limit from 60/hr to 5000/hr.
ingest:
	$(PY) scripts/ingest_issues.py

gold:
	$(PY) scripts/build_ground_truth.py

# Selection runs on dev. Reporting runs on test, once, in the application.
screen:
	$(PY) scripts/screen_models.py --split dev

serve:
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload

serve-mock:
	PROVIDER=mock $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload

docker:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8080 --env-file .env -v "$$PWD/data/runs:/app/data/runs" $(IMAGE)

verify:
	$(PY) scripts/verify.py

clean:
	rm -rf __pycache__ app/__pycache__ scripts/__pycache__ .pytest_cache
	find . -name '*.pyc' -delete
