# Runpod Multi-ASR Worker

An immutable, fixed-action HTTP worker that runs three Japanese meeting-ASR
backends sequentially on one Runpod GPU Pod. Python environments, runtime code,
and exact model snapshots are all baked into the image:

```text
/opt/venvs/whisper   Python 3.11
/opt/venvs/qwen      Python 3.12
/opt/venvs/reazon    Python 3.11
/opt/models/{whisper,qwen,reazon}
/opt/multi-asr/worker
```

Pod startup performs no package installation and no model download. Readiness
requires a visible NVIDIA GPU, successful CUDA imports in all three isolated
environments, and SHA-256 verification of every model file.

The authenticated service exposes only these fixed actions:

- `GET /health`
- `PUT /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/result`
- `DELETE /v1/jobs/{job_id}`

The client chooses the 32-hexadecimal job ID, making upload recovery
idempotent. The server checks declared size, SHA-256, MP3 format, and duration
before accepting a job. It then prepares bounded chunks and runs, in order,
`faster_whisper`, `reazon_nemo`, and `qwen3`. A result is released only when all
three fixed model identities, all expected chunks, transcript text, Whisper
timestamps, and per-backend timings pass the worker contract. One job runs at a
time to bound GPU memory use.

## Required environment

- `FAST_ASR_TOKEN`: random bearer token of at least 32 characters
- NVIDIA GPU with a driver compatible with the CUDA 12.8 runtime

Optional settings include `ASR_JOB_ROOT`, `ASR_MAX_AUDIO_BYTES`, and `PORT`.
Never put audio, credentials, meeting names, or transcripts in this repository
or image.

## Reproduce the build

Model weights are intentionally excluded from Git. Fetch only the reviewed
immutable revisions and verify their file manifests before building:

```bash
./scripts/prepare_models.sh
docker buildx build \
  --platform linux/amd64 \
  --tag runpod-multi-asr:1.0.0 \
  --load .
```

`scripts/publish_image.sh` passes configured `HTTP_PROXY`, `HTTPS_PROXY`, and
`NO_PROXY` values as ephemeral BuildKit secrets when they are present; proxy
values are not Docker build arguments, image environment, or provenance data.

`scripts/prepare_models.sh` rejects symlinks, unexpected files, missing files,
and any SHA-256 mismatch. The Dockerfile also repeats manifest verification
inside each separate model layer. The CUDA and uv base images are digest-pinned,
Ubuntu packages use a fixed snapshot, Python patch releases are exact, and all
transitive Python dependencies are hash-locked per environment.

## Publication contract

Release publication binds an evidence commit to the exact source commit built
into the image. From a clean source commit, run:

```bash
./scripts/publish_image.sh v1.0.0
```

The script performs one BuildKit build into a private GHCR staging tag, with an
SBOM and maximum provenance. It pushes OCI zstd layers directly to the registry
without importing the large image into the builder's local image store. Runtime
compatibility is accepted only after a representative Runpod digest-pull and
startup smoke test. The script reads
the resulting digest, proves the staging manifest is not anonymously readable,
and scans that exact remote digest for
fixable HIGH/CRITICAL OS and Python-library CVEs. Model snapshot directories are
excluded from Trivy because their files are not installed packages and are
independently verified against committed SHA-256 manifests. Only after a clean
scan does the script copy the same index digest to the semantic-version and
source-revision tags. It then generates
`release-evidence/v1.0.0/{release.json,trivy.json}`.

Commit only those two evidence files in the immediately following commit and tag
that evidence commit `v1.0.0`. Dispatch `container.yml` on the tag with the exact
image digest.
The workflow requires the tag commit to contain only the two evidence files,
requires its parent to be the source commit recorded in the image labels and
provenance, and requires the complete public history to use the repository
owner's GitHub noreply identity. It verifies the remote SBOM and independently
rescans the exact remote digest before signing it with GitHub OIDC. The GHCR
package stays private until this gate and signature verification have succeeded.

Runpod clients must use the digest-pinned reference recorded in their reviewed
allowlist; a mutable tag alone is never accepted.

Because the three model snapshots and CUDA Python environments make this image
large, publication is performed from a controlled builder with sufficient disk
instead of a standard hosted GitHub runner. Source tests still run in GitHub
Actions. Changing a model revision, dependency lock, base digest, runtime
contract, or model manifest requires a new version.

For repeated Pods in one datacenter, models can instead be supplied as a
separately verified, digest-addressed Network Volume bundle. The security,
lifecycle, and A/B measurement contract for that alternative is documented in
[`docs/model-storage-options.md`](docs/model-storage-options.md). Per-Pod model
downloads are not part of the fast-start design.
