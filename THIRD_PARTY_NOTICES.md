# Third-party notices

The published container includes third-party software and model artifacts. Its
SBOM attached to the exact OCI digest is the authoritative component inventory;
each component remains subject to its own license.

- `Systran/faster-whisper-large-v3` is fixed to revision
  `edaa852ec7e145841d8ffdb056a99866b5f0a478`. Review both the conversion model
  card and the upstream Whisper model/license notices before redistribution.
- `Qwen/Qwen3-ASR-1.7B` is fixed to revision
  `7278e1e70fe206f11671096ffdd38061171dd6e5`; its model card declares
  Apache-2.0.
- `reazon-research/reazonspeech-nemo-v2` is fixed to revision
  `33693408be76b7cba9fd4a7546a0a8772430211b`; its model card declares
  Apache-2.0.
- faster-whisper, CTranslate2, Qwen-ASR, PyTorch, NeMo, and their dependencies
  remain under their upstream licenses. Exact package versions and hashes are
  recorded under `requirements/` and in the image SBOM.
- The NVIDIA CUDA base image is subject to NVIDIA's container and CUDA license
  terms.

Review the license metadata and notices attached to the exact image digest
before redistributing the image.
