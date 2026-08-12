# One stage, on the slim base. A multi-stage build would save maybe 40 MB and cost
# the reader clarity. Nothing here needs compiling and there's no build-time secret
# to keep out of a layer.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so editing the code doesn't force a reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The code, plus the saved data. The issues and the answer key are baked in on
# purpose: the container has to sort the same 536 issues every time, and you
# shouldn't need GitHub access or a token to reproduce a run. Downloading is a
# separate offline step.
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/corpus/ ./data/corpus/
COPY data/ground_truth/ ./data/ground_truth/

# Runs get written here. Declared as a volume so results outlive the container, and
# chowned so the non-root user can write to it.
RUN mkdir -p /app/data/runs /app/data/screening

# Non-root. The process needs no privileges, and this container holds an API key in
# its environment, so keeping it out of root's process space costs nothing.
RUN useradd --create-home --uid 10001 harness \
    && chown -R harness:harness /app
USER harness

# ---- Settings --------------------------------------------------------------
# Everything is an environment variable, so concurrency, timeouts and which half of
# the answer key to score can change without rebuilding. The exercise asks for that,
# and it's why CONCURRENCY isn't a build arg.
#
#   BASIC_AUTH_PASSWORD    login password. Empty means no login, which is only
#                          safe on localhost. Always set it on a public URL.
#   BASIC_AUTH_USERNAME    login username (default "reviewer")
#   DO_INFERENCE_API_KEY   needed when PROVIDER=digitalocean. Never baked in.
#   PROVIDER               digitalocean | mock
#   CONCURRENCY            requests in the air at once (default 16, measured)
#   REQUEST_TIMEOUT_S      how long to wait for one reply (default 120)
#   MAX_RETRIES            attempts per call, retryable failures only (default 3)
#   TEMPERATURE            0 so runs reproduce
#   MAX_TOKENS             output cap for normal models (default 96)
#   REASONING_MAX_TOKENS   output cap for reasoning models (default 1400)
#   SCORED_SPLIT           test | dev | all (default test)
#   MAX_ISSUES             0 = all 536; above 0 takes an even spread
#   PORT                   listen port (default 8080)
ENV PROVIDER=digitalocean \
    BASIC_AUTH_USERNAME=reviewer \
    CONCURRENCY=16 \
    REQUEST_TIMEOUT_S=120 \
    MAX_RETRIES=3 \
    TEMPERATURE=0 \
    SCORED_SPLIT=test \
    RUNS_DIR=/app/data/runs \
    PORT=8080

EXPOSE 8080
VOLUME ["/app/data/runs"]

# /api/health reports settings and data problems without calling a model, so it's
# safe to poll and costs nothing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=8s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
u=f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8080\")}/api/health'; \
sys.exit(0 if urllib.request.urlopen(u,timeout=4).status==200 else 1)"

# Shell form so $PORT expands. One worker on purpose: the app allows one run at a
# time using an in-process lock, and extra workers would each hold their own lock and
# their own concurrency budget. Two runs at once could double the real number of
# in-flight requests while both reported the configured figure. Scaling out wouldn't
# help anyway, since the work is waiting on the network and already async.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
