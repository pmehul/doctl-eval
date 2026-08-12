.PHONY: help venv install ingest gold screen serve serve-mock docker docker-run clean verify env

IMAGE ?= doctl-eval:latest
PORT  ?= 8080
PY    ?= .venv/bin/python

# Screening knobs. Empty means "leave it to the script's own default", so an unset
# variable here does not turn into `--concurrency ` on the command line.
SPLIT       ?= dev
CONCURRENCY ?=
MODELS      ?=
REPEAT      ?=

# Load .env into the environment for any target that talks to the API.
#
# Without this, `make screen` fails with "DO_INFERENCE_API_KEY is not set" even
# though .env sits right there with the key in it, because make does not read .env
# on its own. `docker-run` already passed --env-file .env, so the project had a
# .env.example that only half the targets honoured. This closes that gap.
#
# It is sourced in a subshell per recipe rather than with make's `include`, because
# include would have make itself parse the file, and a value containing a `#` or a
# space would then break in ways that are irritating to debug.
LOADENV = set -a; if [ -f .env ]; then . ./.env; fi; set +a;

# Re-apply anything set explicitly, after .env has been sourced.
#
# Sourcing .env overwrites variables that make already placed in the recipe
# environment, so .env quietly beat the command line. `MAX_ISSUES=40 make screen`
# ran all 536 issues because .env pins MAX_ISSUES=0, and `make screen
# CONCURRENCY=2` measured at the 8 pinned in .env while reporting 8. Neither
# failed; they just did something other than what was asked, which is the worst
# way for a knob to be wrong when the output is a latency and cost comparison.
#
# Precedence is now the usual one for dotenv files: an explicit setting wins, and
# .env supplies defaults for everything else. Make imports the ambient environment
# as variables too, so `export MAX_ISSUES=40` in the shell also wins.
ENV_KNOBS = PROVIDER CONCURRENCY MAX_ISSUES PERSIST_RUNS SCORED_SPLIT \
            REQUEST_TIMEOUT_S MAX_RETRIES MAX_TOKENS REASONING_MAX_TOKENS \
            TEMPERATURE RUNS_DIR
OVERRIDE = env $(foreach v,$(ENV_KNOBS),$(if $($(v)),$(v)='$($(v))'))

help:
	@echo "First time:"
	@echo "  make install       Create .venv and install dependencies"
	@echo "  make env           Create .env from the template"
	@echo ""
	@echo "Data pipeline (offline, run once). Both artefacts are already committed:"
	@echo "  make ingest        Download and freeze the doctl issues"
	@echo "  make gold          Build the answer key"
	@echo ""
	@echo "Evaluation (reads .env automatically):"
	@echo "  make screen        Test all 11 candidate models on the dev split"
	@echo "                       MODELS=a,b      only these model ids"
	@echo "                       CONCURRENCY=n   requests in flight"
	@echo "                       SPLIT=dev|test  which half of the answer key"
	@echo "                       REPEAT=n        run each model n times, report mean +/- spread"
	@echo "  make serve         Run the app against real Serverless Inference"
	@echo "  make serve-mock    Run the app against the offline simulator, no key needed"
	@echo ""
	@echo "Container:"
	@echo "  make docker        Build the image"
	@echo "  make docker-run    Run the image"
	@echo ""
	@echo "  make verify        Run the invariant checks"

venv:
	python3 -m venv .venv

install: venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

# Creates .env if it is missing. Never overwrites an existing one, because that
# would silently destroy a working key.
env:
	@if [ -f .env ]; then \
		echo ".env already exists, leaving it alone."; \
	else \
		cp .env.example .env; \
		echo "Created .env from the template."; \
		echo "Now open it and set DO_INFERENCE_API_KEY."; \
	fi

# Set GITHUB_TOKEN to raise the API rate limit from 60/hr to 5000/hr.
ingest:
	@$(LOADENV) $(OVERRIDE) $(PY) scripts/ingest_issues.py

gold:
	$(PY) scripts/build_ground_truth.py

# Model selection runs on dev. The test split is scored once, in the application.
# Knobs are forwarded as explicit flags rather than left to the environment.
# LOADENV sources .env *after* make has put its command-line variables into the
# recipe environment, so `make screen CONCURRENCY=2` was overwritten by whatever
# CONCURRENCY .env pins: the run reported .env's number and measured at it. For a
# latency comparison that is not a cosmetic bug, it silently attaches your p50/p95
# to a concurrency you did not choose. An explicit --concurrency beats both.
screen:
	@$(LOADENV) $(OVERRIDE) $(PY) scripts/screen_models.py --split $(SPLIT) \
	  $(if $(CONCURRENCY),--concurrency $(CONCURRENCY),) \
	  $(if $(MODELS),--models $(MODELS),) \
	  $(if $(REPEAT),--repeat $(REPEAT),)

serve:
	@$(LOADENV) $(OVERRIDE) $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload

serve-mock:
	@PROVIDER=mock $(PY) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload

# --load matters on any machine where buildx defaults to the docker-container
# driver, which is the norm on Docker Desktop. Without it the build succeeds, warns
# "build result will only remain in the build cache", and leaves nothing for
# `docker run` to find, so `make docker && make docker-run` fails on a missing
# image after an apparently clean build.
docker:
	docker build --load -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):8080 --env-file .env -v "$$PWD/data/runs:/app/data/runs" $(IMAGE)

verify:
	@$(LOADENV) $(OVERRIDE) $(PY) scripts/verify.py

clean:
	rm -rf __pycache__ app/__pycache__ scripts/__pycache__ .pytest_cache
	find . -name '*.pyc' -delete
