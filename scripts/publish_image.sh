#!/usr/bin/env bash
# Publish one private staging artifact, scan that exact digest, then add release tags.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd -- "$REPO_ROOT"

for command_name in curl docker git jq sha256sum trivy; do
  command -v "$command_name" >/dev/null || {
    printf 'Required command is unavailable: %s\n' "$command_name" >&2
    exit 2
  }
done
if [[ $# -ne 1 || ! "$1" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf 'Usage: %s vMAJOR.MINOR.PATCH\n' "$0" >&2
  exit 2
fi
VERSION_TAG="$1"
if [[ -n "$(git status --porcelain=v1)" ]]; then
  printf 'Release source tree must be clean.\n' >&2
  exit 2
fi
if git rev-parse --verify --quiet "refs/tags/$VERSION_TAG" >/dev/null; then
  printf 'Release tag already exists: %s\n' "$VERSION_TAG" >&2
  exit 2
fi

SOURCE_REVISION="$(git rev-parse --verify HEAD)"
IMAGE_VERSION="${VERSION_TAG#v}"
IMAGE_REPOSITORY="ghcr.io/papunoko/runpod-multi-asr"
STAGING_TAG="staging-$SOURCE_REVISION"
BUILDER_NAME="${MULTI_ASR_BUILDER:-multi-asr-release}"
ARTIFACT_DIR="$REPO_ROOT/artifacts/releases/$VERSION_TAG"
EVIDENCE_DIR="$REPO_ROOT/release-evidence/$VERSION_TAG"
TRIVY_REPORT="$ARTIFACT_DIR/trivy.json"
PROVENANCE_REPORT="$ARTIFACT_DIR/provenance.json"
SBOM_REPORT="$ARTIFACT_DIR/sbom.json"
IMAGE_REPORT="$ARTIFACT_DIR/image.json"
METADATA_REPORT="$ARTIFACT_DIR/build-metadata.json"

if [[ -e "$ARTIFACT_DIR" || -e "$EVIDENCE_DIR" ]]; then
  printf 'Release output already exists for %s; inspect it before retrying.\n' \
    "$VERSION_TAG" >&2
  exit 2
fi
mkdir -p -- "$ARTIFACT_DIR" "$EVIDENCE_DIR"

PROXY_SECRET_ARGS=()
BUILDER_DRIVER_OPTS=()
if [[ -n "${HTTP_PROXY:-}" ]]; then
  PROXY_SECRET_ARGS+=(--secret "id=http_proxy,env=HTTP_PROXY")
  BUILDER_DRIVER_OPTS+=(--driver-opt "env.HTTP_PROXY=$HTTP_PROXY")
fi
if [[ -n "${HTTPS_PROXY:-}" ]]; then
  PROXY_SECRET_ARGS+=(--secret "id=https_proxy,env=HTTPS_PROXY")
  BUILDER_DRIVER_OPTS+=(--driver-opt "env.HTTPS_PROXY=$HTTPS_PROXY")
fi
if [[ -n "${NO_PROXY:-}" ]]; then
  PROXY_SECRET_ARGS+=(--secret "id=no_proxy,env=NO_PROXY")
fi

"$SCRIPT_DIR/verify_models.sh"
docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1 || \
  docker buildx create --name "$BUILDER_NAME" --driver docker-container \
    "${BUILDER_DRIVER_OPTS[@]}"
docker buildx inspect "$BUILDER_NAME" --bootstrap >/dev/null

# This is the only release build. Push directly from BuildKit without importing
# the very large image into the daemon's containerd store. Fast zstd compression
# keeps the model layers practical to export for zstd-capable OCI runtimes.
# The pushed index includes the runnable image, BuildKit provenance, and SBOM
# attestations under one immutable digest.
docker buildx build \
  --builder "$BUILDER_NAME" \
  --platform linux/amd64 \
  "${PROXY_SECRET_ARGS[@]}" \
  --build-arg "SOURCE_REVISION=$SOURCE_REVISION" \
  --build-arg "IMAGE_VERSION=$IMAGE_VERSION" \
  --provenance=mode=max \
  --sbom=true \
  --output "type=image,name=$IMAGE_REPOSITORY:$STAGING_TAG,push=true,store=false,compression=zstd,compression-level=1,oci-mediatypes=true" \
  --metadata-file "$METADATA_REPORT" .

IMAGE_DIGEST="$(jq -r '."containerimage.digest"' "$METADATA_REPORT")"
if [[ ! "$IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  printf 'Published image digest is missing from BuildKit metadata.\n' >&2
  exit 2
fi
EXACT_IMAGE="$IMAGE_REPOSITORY@$IMAGE_DIGEST"

# A new GHCR package is expected to remain private through scanning/signing.
# Prove that an anonymous registry token cannot read the staged manifest.
ANONYMOUS_TOKEN="$(
  curl -sS --get \
    --data-urlencode "scope=repository:papunoko/runpod-multi-asr:pull" \
    --data-urlencode "service=ghcr.io" \
    https://ghcr.io/token | jq -r '.token // empty'
)"
ANONYMOUS_STATUS="000"
if [[ -n "$ANONYMOUS_TOKEN" ]]; then
  ANONYMOUS_STATUS="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      -H "Authorization: Bearer $ANONYMOUS_TOKEN" \
      -H 'Accept: application/vnd.oci.image.index.v1+json' \
      "https://ghcr.io/v2/papunoko/runpod-multi-asr/manifests/$IMAGE_DIGEST"
  )"
fi
if [[ "$ANONYMOUS_STATUS" == "200" ]]; then
  printf 'Refusing to scan: the staging artifact is anonymously readable.\n' >&2
  exit 2
fi
if [[ ! "$ANONYMOUS_STATUS" =~ ^(000|401|403|404)$ ]]; then
  printf 'Could not prove private staging visibility (HTTP %s).\n' \
    "$ANONYMOUS_STATUS" >&2
  exit 2
fi

# Scan the exact remote subject that will later be allowlisted and signed.
trivy image \
  --image-src remote \
  --parallel 1 \
  --skip-dirs /opt/models \
  --exit-code 1 \
  --ignore-unfixed \
  --scanners vuln \
  --vuln-type os,library \
  --severity HIGH,CRITICAL \
  --format json \
  --output "$TRIVY_REPORT" \
  "$EXACT_IMAGE"
VULNERABILITY_COUNT="$(
  jq '[.Results[]?.Vulnerabilities[]?] | length' "$TRIVY_REPORT"
)"
if [[ "$VULNERABILITY_COUNT" != "0" ]]; then
  printf 'Exact digest has fixable HIGH/CRITICAL vulnerabilities.\n' >&2
  exit 2
fi

docker buildx imagetools inspect \
  --format '{{json .Provenance}}' "$EXACT_IMAGE" >"$PROVENANCE_REPORT"
docker buildx imagetools inspect \
  --format '{{json .SBOM}}' "$EXACT_IMAGE" >"$SBOM_REPORT"
docker buildx imagetools inspect \
  --format '{{json .Image}}' "$EXACT_IMAGE" >"$IMAGE_REPORT"
jq -e 'type == "object" and length > 0' "$PROVENANCE_REPORT" >/dev/null
jq -e 'type == "object" and length > 0' "$SBOM_REPORT" >/dev/null
grep -Fq -- "$SOURCE_REVISION" "$PROVENANCE_REPORT"
grep -Fq -- "$SOURCE_REVISION" "$IMAGE_REPORT"

# A single index source is copied byte-for-byte; verify both public-facing tags
# still resolve to the scanned digest before producing committed evidence.
docker buildx imagetools create \
  --tag "$IMAGE_REPOSITORY:$VERSION_TAG" \
  --tag "$IMAGE_REPOSITORY:sha-$SOURCE_REVISION" \
  "$EXACT_IMAGE"
for published_tag in "$VERSION_TAG" "sha-$SOURCE_REVISION"; do
  TAG_DIGEST="$(
    docker buildx imagetools inspect \
      --format '{{json .Manifest}}' "$IMAGE_REPOSITORY:$published_tag" |
      jq -r '.digest'
  )"
  if [[ "$TAG_DIGEST" != "$IMAGE_DIGEST" ]]; then
    printf 'Release tag %s does not resolve to the scanned digest.\n' \
      "$published_tag" >&2
    exit 2
  fi
done

cp -- "$TRIVY_REPORT" "$EVIDENCE_DIR/trivy.json"
TRIVY_REPORT_SHA256="$(sha256sum "$EVIDENCE_DIR/trivy.json" | cut -d' ' -f1)"
PROVENANCE_SHA256="$(sha256sum "$PROVENANCE_REPORT" | cut -d' ' -f1)"
SBOM_SHA256="$(sha256sum "$SBOM_REPORT" | cut -d' ' -f1)"
MODELS_SHA256="$(sha256sum MODELS.json | cut -d' ' -f1)"
TRIVY_VERSION="$(trivy --version | sed -n '1s/^Version: //p')"
VERIFIED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

jq -n \
  --arg tag "$VERSION_TAG" \
  --arg sourceCommit "$SOURCE_REVISION" \
  --arg imageRepository "$IMAGE_REPOSITORY" \
  --arg imageDigest "$IMAGE_DIGEST" \
  --arg image "$IMAGE_REPOSITORY:$VERSION_TAG@$IMAGE_DIGEST" \
  --arg scanSubject "$EXACT_IMAGE" \
  --arg trivyVersion "$TRIVY_VERSION" \
  --arg trivyReport "release-evidence/$VERSION_TAG/trivy.json" \
  --arg trivyReportSha256 "$TRIVY_REPORT_SHA256" \
  --arg provenanceSha256 "$PROVENANCE_SHA256" \
  --arg sbomSha256 "$SBOM_SHA256" \
  --arg modelsSha256 "$MODELS_SHA256" \
  --arg verifiedAt "$VERIFIED_AT" \
  '{
    schemaVersion: 1,
    tag: $tag,
    sourceCommit: $sourceCommit,
    imageRepository: $imageRepository,
    imageDigest: $imageDigest,
    image: $image,
    modelsSha256: $modelsSha256,
    stagingVisibilityDuringScan: "private",
    scan: {
      subject: $scanSubject,
      trivyVersion: $trivyVersion,
      report: $trivyReport,
      reportSha256: $trivyReportSha256,
      fixableHighOrCriticalCves: 0
    },
    attestations: {
      provenancePresent: true,
      provenanceSha256: $provenanceSha256,
      sbomPresent: true,
      sbomSha256: $sbomSha256
    },
    verifiedAt: $verifiedAt
  }' >"$EVIDENCE_DIR/release.json"

printf '%s\n' "$IMAGE_REPOSITORY:$VERSION_TAG@$IMAGE_DIGEST"
printf 'Commit only release-evidence/%s, tag that evidence commit as %s, then dispatch container.yml with image_digest=%s.\n' \
  "$VERSION_TAG" "$VERSION_TAG" "$IMAGE_DIGEST"
