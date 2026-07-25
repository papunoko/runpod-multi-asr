# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:ac55d124da4882b497f732d8dfd9a702d5447a5f29d08d56da6f64f0a1eb34bc

ARG UBUNTU_SNAPSHOT=20260723T000000Z
ARG SOURCE_REVISION=unknown
ARG IMAGE_VERSION=dev

LABEL org.opencontainers.image.source="https://github.com/papunoko/runpod-multi-asr" \
      org.opencontainers.image.description="Offline three-model Japanese meeting ASR worker for Runpod Pods" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      org.opencontainers.image.revision="$SOURCE_REVISION" \
      org.opencontainers.image.version="$IMAGE_VERSION"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy \
    ASR_RUNTIME_VERSION=1.0.0 \
    ASR_JOB_ROOT=/workspace/multi-asr/jobs \
    ASR_MODEL_MANIFEST=/opt/multi-asr/MODELS.json \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000

COPY --from=uv /uv /uvx /usr/local/bin/

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    if [ -f /run/secrets/http_proxy ]; then export HTTP_PROXY="$(cat /run/secrets/http_proxy)" http_proxy="$(cat /run/secrets/http_proxy)"; fi \
    && if [ -f /run/secrets/https_proxy ]; then export HTTPS_PROXY="$(cat /run/secrets/https_proxy)" https_proxy="$(cat /run/secrets/https_proxy)"; fi \
    && if [ -f /run/secrets/no_proxy ]; then export NO_PROXY="$(cat /run/secrets/no_proxy)" no_proxy="$(cat /run/secrets/no_proxy)"; fi \
    && find /etc/apt -maxdepth 2 -type f \( -name '*.list' -o -name '*.sources' \) \
        -exec sed -Ei \
        "s#https?://(archive|security)\\.ubuntu\\.com/ubuntu/?#https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#g" {} + \
    && rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/cuda*.sources \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ /tmp/requirements/

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    if [ -f /run/secrets/http_proxy ]; then export HTTP_PROXY="$(cat /run/secrets/http_proxy)" http_proxy="$(cat /run/secrets/http_proxy)"; fi \
    && if [ -f /run/secrets/https_proxy ]; then export HTTPS_PROXY="$(cat /run/secrets/https_proxy)" https_proxy="$(cat /run/secrets/https_proxy)"; fi \
    && if [ -f /run/secrets/no_proxy ]; then export NO_PROXY="$(cat /run/secrets/no_proxy)" no_proxy="$(cat /run/secrets/no_proxy)"; fi \
    && uv python install 3.11.15 3.12.13 \
    && uv venv --python 3.11.15 /opt/venvs/whisper \
    && uv pip install --python /opt/venvs/whisper/bin/python \
        --require-hashes --requirement /tmp/requirements/whisper.txt \
    && /opt/venvs/whisper/bin/python -c "import aiohttp, faster_whisper"

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    if [ -f /run/secrets/http_proxy ]; then export HTTP_PROXY="$(cat /run/secrets/http_proxy)" http_proxy="$(cat /run/secrets/http_proxy)"; fi \
    && if [ -f /run/secrets/https_proxy ]; then export HTTPS_PROXY="$(cat /run/secrets/https_proxy)" https_proxy="$(cat /run/secrets/https_proxy)"; fi \
    && if [ -f /run/secrets/no_proxy ]; then export NO_PROXY="$(cat /run/secrets/no_proxy)" no_proxy="$(cat /run/secrets/no_proxy)"; fi \
    && uv venv --python 3.12.13 /opt/venvs/qwen \
    && uv pip install --python /opt/venvs/qwen/bin/python \
        --require-hashes \
        --index https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
        --requirement /tmp/requirements/qwen.txt \
    && /opt/venvs/qwen/bin/python -c "import qwen_asr, torch; assert torch.__version__ == '2.8.0+cu128'"

RUN --mount=type=secret,id=http_proxy,required=false \
    --mount=type=secret,id=https_proxy,required=false \
    --mount=type=secret,id=no_proxy,required=false \
    if [ -f /run/secrets/http_proxy ]; then export HTTP_PROXY="$(cat /run/secrets/http_proxy)" http_proxy="$(cat /run/secrets/http_proxy)"; fi \
    && if [ -f /run/secrets/https_proxy ]; then export HTTPS_PROXY="$(cat /run/secrets/https_proxy)" https_proxy="$(cat /run/secrets/https_proxy)"; fi \
    && if [ -f /run/secrets/no_proxy ]; then export NO_PROXY="$(cat /run/secrets/no_proxy)" no_proxy="$(cat /run/secrets/no_proxy)"; fi \
    && uv venv --python 3.11.15 /opt/venvs/reazon \
    && uv pip install --python /opt/venvs/reazon/bin/python \
        --require-hashes \
        --index https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match \
        --requirement /tmp/requirements/reazon.txt \
    && /opt/venvs/reazon/bin/python -c "import nemo.collections.asr, torch; assert torch.__version__ == '2.8.0+cu128'" \
    && rm -rf /root/.cache /tmp/requirements

WORKDIR /opt/multi-asr

COPY MODELS.json /opt/multi-asr/MODELS.json

# Keep each model in its own sub-10-GB OCI layer and before frequently changed
# worker code. This preserves model-layer cache across runtime-only releases.
COPY models/whisper/ /opt/models/whisper/
COPY model-manifests/whisper.sha256 /opt/models/whisper/MODEL_MANIFEST.sha256
RUN cd /opt/models/whisper \
    && sha256sum --check --strict MODEL_MANIFEST.sha256 \
    && find . -type f -exec chmod 0444 {} +

COPY models/reazon/ /opt/models/reazon/
COPY model-manifests/reazon.sha256 /opt/models/reazon/MODEL_MANIFEST.sha256
RUN cd /opt/models/reazon \
    && sha256sum --check --strict MODEL_MANIFEST.sha256 \
    && find . -type f -exec chmod 0444 {} +

COPY models/qwen/ /opt/models/qwen/
COPY model-manifests/qwen.sha256 /opt/models/qwen/MODEL_MANIFEST.sha256
RUN cd /opt/models/qwen \
    && sha256sum --check --strict MODEL_MANIFEST.sha256 \
    && find . -type f -exec chmod 0444 {} +

COPY pipeline/ /opt/multi-asr/pipeline/
COPY worker_core.py server.py /opt/multi-asr/worker/
COPY THIRD_PARTY_NOTICES.md /opt/multi-asr/THIRD_PARTY_NOTICES.md
COPY LICENSE /opt/multi-asr/LICENSE

RUN chmod 0555 /opt/multi-asr/pipeline/deploy/run_backend.sh \
    && /opt/venvs/whisper/bin/python -m compileall -q /opt/multi-asr \
    && mkdir -p /workspace/multi-asr/jobs

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/venvs/whisper/bin/python", "/opt/multi-asr/worker/server.py"]
