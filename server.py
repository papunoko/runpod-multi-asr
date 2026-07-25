"""Authenticated, fixed-action HTTP service for isolated three-backend ASR."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import aiofiles
from aiohttp import web

from worker_core import (
    BACKEND_ORDER,
    MODEL_PATHS,
    ContractError,
    atomic_write_json,
    build_result,
    load_model_identities,
    new_job_record,
    read_job_record,
    sha256_file,
    transition_job,
    utc_now,
    validate_backend_output,
    validate_job_id,
    validate_meeting_key,
    validate_prepared_run,
    validate_sha256,
    verify_model_manifest,
)


RUNTIME_VERSION = os.environ.get("ASR_RUNTIME_VERSION", "1.0.0")
JOB_ROOT = Path(os.environ.get("ASR_JOB_ROOT", "/workspace/multi-asr/jobs"))
MAX_AUDIO_BYTES = int(os.environ.get("ASR_MAX_AUDIO_BYTES", str(256 * 1024**2)))
MAX_JOB_BYTES = int(os.environ.get("ASR_MAX_JOB_BYTES", str(8 * 1024**3)))
MAX_JOBS = int(os.environ.get("ASR_MAX_JOBS", "8"))
MAX_AUDIO_DURATION_SECONDS = float(
    os.environ.get("ASR_MAX_AUDIO_DURATION_SECONDS", str(8 * 60 * 60))
)
UPLOAD_TIMEOUT_SECONDS = int(os.environ.get("ASR_UPLOAD_TIMEOUT_SECONDS", "900"))
PREPARE_TIMEOUT_SECONDS = int(os.environ.get("ASR_PREPARE_TIMEOUT_SECONDS", "1800"))
MAX_BACKEND_SECONDS = int(os.environ.get("ASR_MAX_BACKEND_SECONDS", "7200"))
FFPROBE_PATH = os.environ.get("ASR_FFPROBE_PATH", "/usr/bin/ffprobe")
PORT = int(os.environ.get("PORT", "8000"))
TOKEN = os.environ.get("FAST_ASR_TOKEN", "")

LAUNCHER_PATH = Path("/opt/multi-asr/pipeline/deploy/run_backend.sh")
MODEL_IDENTITIES_PATH = Path("/opt/multi-asr/MODELS.json")
NVIDIA_SMI_COMMAND = (
    "/usr/bin/nvidia-smi",
    "--query-gpu=name",
    "--format=csv,noheader",
)
VENV_PYTHONS = (
    Path("/opt/venvs/whisper/bin/python"),
    Path("/opt/venvs/reazon/bin/python"),
    Path("/opt/venvs/qwen/bin/python"),
)
IMPORT_COMMANDS = (
    (
        "/opt/venvs/whisper/bin/python",
        "-I",
        "-c",
        "import aiohttp, faster_whisper",
    ),
    (
        "/opt/venvs/reazon/bin/python",
        "-I",
        "-c",
        "import nemo.collections.asr, torch; "
        "assert torch.__version__ == '2.8.0+cu128'; "
        "assert torch.cuda.is_available()",
    ),
    (
        "/opt/venvs/qwen/bin/python",
        "-I",
        "-c",
        "import qwen_asr, torch; "
        "assert torch.__version__ == '2.8.0+cu128'; "
        "assert torch.cuda.is_available()",
    ),
)
REAZON_MODEL_FILE = Path("/opt/models/reazon/reazonspeech-nemo-v2.nemo")
ATOMIC_TEMP_RE = re.compile(r"\.(?:job|result)\.json\.[A-Za-z0-9_-]{6,64}\.tmp")
PIPELINE_TEMP_RE = re.compile(
    r"\.(?:manifest\.json|chunks\.jsonl|silences\.json|backend\.json|"
    r"chunk_[0-9]{3,6}\.json)\.[A-Za-z0-9_-]{6,64}"
)
CHUNK_FILE_RE = re.compile(r"chunk_[0-9]{3,6}")
SUBPROCESS_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PATH",
        "TMPDIR",
        "TZ",
    }
)


def _safe_log(event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True), flush=True)


def _offline_environment() -> dict[str, str]:
    # Backend runners never need the HTTP bearer token, Runpod/API credentials,
    # proxy credentials, or request metadata inherited by the worker process.
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in SUBPROCESS_ENV_ALLOWLIST
    }
    environment.update(
        {
            "ASR_ENV_ROOT": "/opt/venvs",
            "ASR_CACHE_ROOT": "/workspace/multi-asr/cache",
            "HF_HOME": "/workspace/multi-asr/cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/workspace/multi-asr/cache/huggingface/hub",
            "TORCH_HOME": "/workspace/multi-asr/cache/torch",
            "XDG_CACHE_HOME": "/workspace/multi-asr/cache/xdg",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "ASR_DEVICE": "cuda",
            "WHISPER_COMPUTE_TYPE": "float16",
        }
    )
    return environment


def _probe_uploaded_audio(path: Path, declared_duration: float) -> float:
    try:
        completed = subprocess.run(
            [
                FFPROBE_PATH,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "format=duration,format_name",
                "-of",
                "json",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError("audio probe failed") from exc
    if completed.returncode != 0 or len(completed.stdout) > 64 * 1024:
        raise ContractError("audio probe rejected the upload")
    try:
        payload = json.loads(
            completed.stdout.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value: {value}")
            ),
        )
        metadata = payload["format"]
        duration = float(metadata["duration"])
        format_name = metadata["format_name"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ContractError("audio probe returned invalid metadata") from exc
    if (
        not math.isfinite(duration)
        or duration <= 0
        or duration > MAX_AUDIO_DURATION_SECONDS
        or format_name != "mp3"
        or abs(duration - declared_duration) > max(5.0, declared_duration * 0.01)
    ):
        raise ContractError("audio probe contract mismatch")
    return duration


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


class FastAsrRuntime:
    """Serialized controller; model weights live only in child processes."""

    def __init__(self) -> None:
        if len(TOKEN) < 32:
            raise RuntimeError("FAST_ASR_TOKEN must contain at least 32 characters")
        if MAX_AUDIO_BYTES <= 0 or MAX_AUDIO_BYTES > 256 * 1024**2:
            raise RuntimeError("ASR_MAX_AUDIO_BYTES is outside the supported range")
        if MAX_JOB_BYTES < MAX_AUDIO_BYTES or MAX_JOB_BYTES > 16 * 1024**3:
            raise RuntimeError("ASR_MAX_JOB_BYTES is outside the supported range")
        if MAX_JOBS <= 0 or MAX_JOBS > 32:
            raise RuntimeError("ASR_MAX_JOBS is outside the supported range")
        if MAX_AUDIO_DURATION_SECONDS <= 0 or MAX_AUDIO_DURATION_SECONDS > 8 * 60 * 60:
            raise RuntimeError(
                "ASR_MAX_AUDIO_DURATION_SECONDS is outside the supported range"
            )
        if UPLOAD_TIMEOUT_SECONDS < 30 or UPLOAD_TIMEOUT_SECONDS > 3600:
            raise RuntimeError("ASR_UPLOAD_TIMEOUT_SECONDS is outside the supported range")
        if PREPARE_TIMEOUT_SECONDS < 60 or PREPARE_TIMEOUT_SECONDS > 3600:
            raise RuntimeError("ASR_PREPARE_TIMEOUT_SECONDS is outside the supported range")
        if MAX_BACKEND_SECONDS < 300 or MAX_BACKEND_SECONDS > 7200:
            raise RuntimeError("ASR_MAX_BACKEND_SECONDS is outside the supported range")
        if not os.path.isabs(FFPROBE_PATH):
            raise RuntimeError("ASR_FFPROBE_PATH must be absolute")
        if not JOB_ROOT.is_absolute():
            raise RuntimeError("ASR_JOB_ROOT must be absolute")
        self._startup_error: str | None = None
        self._startup_checked = threading.Event()
        self._model_identities: tuple[dict[str, Any], ...] = ()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="multi-asr")
        self._upload_lock = threading.Lock()
        self._uploading: dict[str, int] = {}
        JOB_ROOT.mkdir(parents=True, exist_ok=True)
        os.chmod(JOB_ROOT, 0o700)
        self._recover_after_restart()

    @property
    def ready(self) -> bool:
        return (
            self._startup_checked.is_set()
            and self._startup_error is None
            and len(self._model_identities) == len(BACKEND_ORDER)
        )

    @property
    def model_error(self) -> str | None:
        return self._startup_error

    @property
    def model_identities(self) -> tuple[dict[str, Any], ...]:
        return self._model_identities

    def check_runtime(self) -> None:
        started = time.perf_counter()
        try:
            executable_paths = (
                Path(FFPROBE_PATH),
                Path(NVIDIA_SMI_COMMAND[0]),
                LAUNCHER_PATH,
                *VENV_PYTHONS,
            )
            if any(
                not path.is_file()
                or _is_link_like(path)
                or not os.access(path, os.X_OK)
                for path in executable_paths
            ):
                raise RuntimeError("required executable is missing or unsafe")
            gpu = subprocess.run(
                list(NVIDIA_SMI_COMMAND),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                env=_offline_environment(),
            )
            if gpu.returncode != 0 or not gpu.stdout or len(gpu.stdout) > 64 * 1024:
                raise RuntimeError("GPU readiness check failed")
            for command in IMPORT_COMMANDS:
                imported = subprocess.run(
                    list(command),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=120,
                    check=False,
                    env=_offline_environment(),
                )
                if imported.returncode != 0 or len(imported.stdout) > 64 * 1024:
                    raise RuntimeError("backend import readiness check failed")
            identities = load_model_identities(MODEL_IDENTITIES_PATH)
            for identity in identities:
                verify_model_manifest(
                    Path(identity["path"]),
                    Path(identity["manifestPath"]),
                    identity["manifestSha256"],
                )
            if (
                not REAZON_MODEL_FILE.is_file()
                or _is_link_like(REAZON_MODEL_FILE)
                or REAZON_MODEL_FILE.parent != Path(MODEL_PATHS["reazon_nemo"])
            ):
                raise RuntimeError("fixed Reazon model file is missing or unsafe")
            self._model_identities = identities
            _safe_log(
                "multi_asr_ready",
                runtimeVersion=RUNTIME_VERSION,
                backends=list(BACKEND_ORDER),
                checkSeconds=round(time.perf_counter() - started, 3),
            )
        except Exception as exc:
            self._startup_error = type(exc).__name__
            self._model_identities = ()
            _safe_log("runtime_check_failed", errorCode="runtime_check_failed")
        finally:
            self._startup_checked.set()

    # Preserve the old startup hook name without loading any model weights.
    load_model = check_runtime

    def submit(self, job_id: str) -> None:
        self._executor.submit(self._run_job, job_id)

    def _record_path(self, job_id: str) -> Path:
        return JOB_ROOT / job_id / "job.json"

    def _recover_after_restart(self) -> None:
        for job_dir in JOB_ROOT.iterdir():
            if not job_dir.is_dir() or _is_link_like(job_dir):
                continue
            try:
                job_id = validate_job_id(job_dir.name)
            except ContractError:
                continue
            self._remove_atomic_temps(job_dir)
            record_path = job_dir / "job.json"
            if not record_path.exists():
                try:
                    self._remove_known_job_files(job_dir, allow_record=False)
                except (ContractError, OSError):
                    _safe_log("unsafe_orphan_preserved", jobId=job_id)
                continue
            try:
                record = read_job_record(record_path)
                if record["status"] in {"accepted", "running"}:
                    for backend_id, state in record.get("backends", {}).items():
                        if state.get("status") == "running":
                            state = dict(state)
                            state.update(
                                {
                                    "status": "failed",
                                    "completedAt": utc_now(),
                                    "failureCount": 1,
                                }
                            )
                            record["backends"][backend_id] = state
                    record = transition_job(
                        record, "failed", error_code="worker_restarted"
                    )
                    atomic_write_json(record_path, record)
                    _safe_log("job_failed", jobId=job_id, errorType="WorkerRestart")
            except (ContractError, OSError):
                _safe_log("job_record_invalid", jobId=job_id)

    @staticmethod
    def _remove_atomic_temps(job_dir: Path) -> None:
        for entry in job_dir.rglob("*"):
            if _is_link_like(entry):
                continue
            if entry.is_file() and (
                ATOMIC_TEMP_RE.fullmatch(entry.name)
                or PIPELINE_TEMP_RE.fullmatch(entry.name)
            ):
                entry.unlink(missing_ok=True)

    @staticmethod
    def _validate_run_tree(run_dir: Path) -> None:
        if not run_dir.is_dir() or _is_link_like(run_dir):
            raise ContractError("run directory is unsafe")
        root = run_dir.resolve(strict=True)
        allowed_directories = {
            "audio",
            "audio/raw_stereo",
            "audio/enhanced_mono",
            "hypotheses",
            *(f"hypotheses/{backend_id}" for backend_id in BACKEND_ORDER),
        }
        allowed_fixed_files = {"manifest.json", "chunks.jsonl", "silences.json"}
        for entry in run_dir.rglob("*"):
            if _is_link_like(entry):
                raise ContractError("run directory contains a link")
            try:
                entry.resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise ContractError("run directory entry escaped its root") from exc
            relative = entry.relative_to(run_dir).as_posix()
            if entry.is_dir():
                if relative not in allowed_directories:
                    raise ContractError("run directory contains an unexpected directory")
                continue
            if not entry.is_file():
                raise ContractError("run directory contains a non-regular entry")
            parent = entry.parent.relative_to(run_dir).as_posix()
            allowed = relative in allowed_fixed_files
            if parent == "audio/raw_stereo":
                allowed = bool(re.fullmatch(r"chunk_[0-9]{3,6}\.mp3", entry.name))
            elif parent == "audio/enhanced_mono":
                allowed = bool(re.fullmatch(r"chunk_[0-9]{3,6}\.wav", entry.name))
            elif parent.startswith("hypotheses/") and parent.count("/") == 1:
                backend_id = parent.split("/", 1)[1]
                allowed = backend_id in BACKEND_ORDER and (
                    entry.name == "backend.json"
                    or bool(re.fullmatch(r"chunk_[0-9]{3,6}\.json", entry.name))
                )
            if PIPELINE_TEMP_RE.fullmatch(entry.name):
                allowed = parent in {".", *allowed_directories}
            if not allowed:
                raise ContractError("run directory contains an unexpected file")

    @classmethod
    def _remove_known_job_files(cls, job_dir: Path, *, allow_record: bool) -> None:
        if not job_dir.is_dir() or _is_link_like(job_dir):
            raise ContractError("job directory is unsafe")
        allowed = {
            "audio.mp3.part",
            "audio.mp3",
            "result.json",
            "result.json.sha256",
        }
        if allow_record:
            allowed.add("job.json")
        entries = list(job_dir.iterdir())
        for entry in entries:
            if _is_link_like(entry):
                raise ContractError("job directory contains a link")
            if entry.is_dir():
                if entry.name != "run":
                    raise ContractError("job directory contains an unexpected directory")
                cls._validate_run_tree(entry)
            elif (
                not entry.is_file()
                or (
                    entry.name not in allowed
                    and not ATOMIC_TEMP_RE.fullmatch(entry.name)
                )
            ):
                raise ContractError("job directory contains an unexpected entry")
        for entry in entries:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink(missing_ok=True)
        job_dir.rmdir()

    def claim_upload(
        self,
        *,
        job_id: str,
        meeting_key: str,
        audio_sha256: str,
        audio_bytes: int,
        audio_duration_seconds: float,
    ) -> tuple[str, dict[str, Any] | None]:
        with self._upload_lock:
            if job_id in self._uploading:
                return "uploading", None
            if self._uploading:
                return "upload_limit", None
            job_dir = JOB_ROOT / job_id
            record_path = job_dir / "job.json"
            if record_path.exists():
                record = read_job_record(record_path)
                if (
                    record["meetingKey"] != meeting_key
                    or record["audioSha256"] != audio_sha256
                    or record["audioBytes"] != audio_bytes
                    or abs(float(record["audioDurationSeconds"]) - audio_duration_seconds)
                    > 0.001
                ):
                    raise ContractError("idempotency key belongs to another input")
                return "existing", record
            if job_dir.exists():
                return "uploading", None
            records: list[dict[str, Any]] = []
            job_directory_count = 0
            for candidate in JOB_ROOT.iterdir():
                candidate_record = candidate / "job.json"
                if candidate.is_dir():
                    job_directory_count += 1
                if candidate.is_dir() and candidate_record.is_file():
                    try:
                        records.append(read_job_record(candidate_record))
                    except ContractError:
                        continue
            if job_directory_count >= MAX_JOBS:
                return "job_limit", None
            used_bytes = sum(int(record["audioBytes"]) for record in records)
            reserved_bytes = sum(self._uploading.values())
            if used_bytes + reserved_bytes + audio_bytes > MAX_JOB_BYTES:
                return "byte_limit", None
            try:
                free_bytes = shutil.disk_usage(JOB_ROOT).free
            except OSError:
                free_bytes = 0
            if free_bytes and free_bytes < audio_bytes + 512 * 1024 * 1024:
                return "disk_limit", None
            job_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
            self._uploading[job_id] = audio_bytes
            return "claimed", None

    def release_upload(self, job_id: str) -> None:
        with self._upload_lock:
            self._uploading.pop(job_id, None)

    @staticmethod
    def _commands(job_dir: Path) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        audio_path = str(job_dir / "audio.mp3")
        run_dir = str(job_dir / "run")
        launcher = "/opt/multi-asr/pipeline/deploy/run_backend.sh"
        prepare = (
            launcher,
            "core",
            "prepare",
            audio_path,
            "--run-dir",
            run_dir,
            "--target-seconds",
            "75",
            "--overlap-seconds",
            "3",
            "--search-window-seconds",
            "15",
        )
        backend_commands = {
            "faster_whisper": (
                launcher,
                "whisper",
                "--run-dir",
                run_dir,
                "--model",
                "/opt/models/whisper",
            ),
            "reazon_nemo": (
                launcher,
                "reazon",
                "--run-dir",
                run_dir,
                "--model",
                "/opt/models/reazon/reazonspeech-nemo-v2.nemo",
            ),
            "qwen3": (
                launcher,
                "qwen",
                "--run-dir",
                run_dir,
                "--model",
                "/opt/models/qwen",
                "--max-new-tokens",
                "2048",
            ),
        }
        return prepare, backend_commands

    @staticmethod
    def _run_subprocess(command: Sequence[str], timeout: int) -> None:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            env=_offline_environment(),
        )
        if completed.returncode != 0:
            raise RuntimeError("fixed ASR subprocess failed")

    def _run_job(self, job_id: str) -> None:
        record_path = self._record_path(job_id)
        current_backend: str | None = None
        backend_started_monotonic: float | None = None
        try:
            if not self.ready:
                raise RuntimeError("runtime is unavailable")
            record = read_job_record(record_path)
            record = transition_job(record, "running")
            atomic_write_json(record_path, record)
            _safe_log("job_running", jobId=job_id)
            job_started = time.monotonic()
            job_dir = record_path.parent
            run_dir = job_dir / "run"
            prepare_command, backend_commands = self._commands(job_dir)
            self._run_subprocess(prepare_command, PREPARE_TIMEOUT_SECONDS)
            validate_prepared_run(run_dir, record)

            outputs: dict[str, dict[str, Any]] = {}
            timings: dict[str, dict[str, Any]] = {}
            for backend_id in BACKEND_ORDER:
                current_backend = backend_id
                started_at = utc_now()
                backend_started_monotonic = time.monotonic()
                state = {"status": "running", "startedAt": started_at}
                record["backends"][backend_id] = state
                atomic_write_json(record_path, record)
                self._run_subprocess(backend_commands[backend_id], MAX_BACKEND_SECONDS)
                output = validate_backend_output(run_dir, backend_id)
                completed_at = utc_now()
                elapsed = time.monotonic() - backend_started_monotonic
                timing = {
                    "startedAt": started_at,
                    "completedAt": completed_at,
                    "processingSeconds": elapsed,
                }
                state = {
                    "status": "complete",
                    **timing,
                    "outputCount": output["summary"]["chunksCompleted"],
                    "failureCount": 0,
                }
                record["backends"][backend_id] = state
                atomic_write_json(record_path, record)
                outputs[backend_id] = output
                timings[backend_id] = timing
                backend_started_monotonic = None

            total_elapsed = time.monotonic() - job_started
            result = build_result(
                record=record,
                model_identities=self._model_identities,
                backend_outputs=outputs,
                backend_timings=timings,
                total_processing_seconds=total_elapsed,
            )
            result_path = job_dir / "result.json"
            atomic_write_json(result_path, result)
            digest = sha256_file(result_path)
            (job_dir / "result.json.sha256").write_text(
                f"{digest}  result.json\n", encoding="ascii"
            )
            record = transition_job(record, "complete")
            record["resultSha256"] = digest
            record["totalProcessingSeconds"] = total_elapsed
            atomic_write_json(record_path, record)
            _safe_log(
                "job_complete",
                jobId=job_id,
                backends=list(BACKEND_ORDER),
                totalProcessingSeconds=round(total_elapsed, 3),
            )
        except Exception as exc:
            try:
                record = read_job_record(record_path)
                if (
                    current_backend is not None
                    and record.get("backends", {}).get(current_backend, {}).get("status")
                    == "running"
                ):
                    state = dict(record["backends"][current_backend])
                    state.update(
                        {
                            "status": "failed",
                            "completedAt": utc_now(),
                            "processingSeconds": max(
                                0.0,
                                time.monotonic()
                                - (backend_started_monotonic or time.monotonic()),
                            ),
                            "outputCount": 0,
                            "failureCount": 1,
                            "errorCode": "backend_failed",
                        }
                    )
                    record["backends"][current_backend] = state
                if record["status"] in {"accepted", "running"}:
                    record = transition_job(record, "failed", error_code="asr_failed")
                    atomic_write_json(record_path, record)
            except Exception:
                pass
            _safe_log(
                "job_failed",
                jobId=job_id,
                backend=current_backend,
                errorType=type(exc).__name__,
            )


RUNTIME = FastAsrRuntime()


def _single_header(request: web.Request, name: str) -> str:
    values = request.headers.getall(name, [])
    if len(values) != 1:
        raise ContractError(f"{name} must appear exactly once")
    return values[0]


@web.middleware
async def authentication(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        supplied = _single_header(request, "Authorization")
    except ContractError:
        supplied = ""
    expected = f"Bearer {TOKEN}"
    if not hmac.compare_digest(supplied, expected):
        raise web.HTTPUnauthorized(
            text=json.dumps({"error": "unauthorized"}),
            content_type="application/json",
        )
    return await handler(request)


async def health(_: web.Request) -> web.Response:
    if RUNTIME.model_error:
        return web.json_response(
            {"state": "failed", "errorCode": "runtime_check_failed"}, status=500
        )
    if not RUNTIME.ready:
        return web.json_response({"state": "checking"}, status=503)
    return web.json_response(
        {
            "state": "multi_asr_ready",
            "runtimeVersion": RUNTIME_VERSION,
            "backendOrder": list(BACKEND_ORDER),
            "models": [
                {
                    key: identity[key]
                    for key in (
                        "id",
                        "label",
                        "repository",
                        "revision",
                        "manifestSha256",
                    )
                }
                for identity in RUNTIME.model_identities
            ],
        }
    )


async def create_job(request: web.Request) -> web.Response:
    if not RUNTIME.ready:
        return web.json_response({"error": "not_ready"}, status=503)
    try:
        job_id = validate_job_id(request.match_info["job_id"])
        meeting_key = validate_meeting_key(_single_header(request, "X-Meeting-Key"))
        expected_sha = validate_sha256(_single_header(request, "X-Audio-SHA256"))
        expected_bytes = int(_single_header(request, "X-Audio-Bytes"))
        expected_duration = float(_single_header(request, "X-Audio-Duration-Seconds"))
    except (ContractError, TypeError, ValueError):
        return web.json_response({"error": "invalid_headers"}, status=400)
    try:
        content_type = _single_header(request, "Content-Type")
    except ContractError:
        return web.json_response({"error": "invalid_content_type"}, status=415)
    if content_type.lower() != "application/octet-stream":
        return web.json_response({"error": "invalid_content_type"}, status=415)
    if expected_bytes <= 0 or expected_bytes > MAX_AUDIO_BYTES:
        return web.json_response({"error": "audio_size_rejected"}, status=413)
    if (
        not math.isfinite(expected_duration)
        or expected_duration <= 0
        or expected_duration > MAX_AUDIO_DURATION_SECONDS
    ):
        return web.json_response({"error": "audio_duration_rejected"}, status=413)
    if request.content_length is not None and request.content_length != expected_bytes:
        return web.json_response({"error": "content_length_mismatch"}, status=400)

    job_dir = JOB_ROOT / job_id
    try:
        claim, existing = RUNTIME.claim_upload(
            job_id=job_id,
            meeting_key=meeting_key,
            audio_sha256=expected_sha,
            audio_bytes=expected_bytes,
            audio_duration_seconds=expected_duration,
        )
    except ContractError:
        return web.json_response({"error": "idempotency_conflict"}, status=409)
    if claim == "existing":
        return web.json_response(
            {"jobId": job_id, "status": existing["status"]}, status=200
        )
    if claim == "uploading":
        return web.json_response({"error": "upload_in_progress"}, status=409)
    if claim == "upload_limit":
        return web.json_response({"error": "upload_limit"}, status=429)
    if claim in {"job_limit", "byte_limit", "disk_limit"}:
        return web.json_response({"error": claim}, status=507)

    temporary = job_dir / "audio.mp3.part"
    final = job_dir / "audio.mp3"
    digest = hashlib.sha256()
    received = 0
    try:
        async with asyncio.timeout(UPLOAD_TIMEOUT_SECONDS):
            async with aiofiles.open(temporary, "xb") as stream:
                async for block in request.content.iter_chunked(1024 * 1024):
                    received += len(block)
                    if received > expected_bytes or received > MAX_AUDIO_BYTES:
                        raise ContractError("received size exceeded the contract")
                    digest.update(block)
                    await stream.write(block)
                await stream.flush()
                await asyncio.to_thread(os.fsync, stream.fileno())
        if received != expected_bytes:
            raise ContractError("received size mismatch")
        if not hmac.compare_digest(digest.hexdigest(), expected_sha):
            raise ContractError("received hash mismatch")
        os.replace(temporary, final)
        probed_duration = await asyncio.to_thread(
            _probe_uploaded_audio, final, expected_duration
        )
        record = new_job_record(
            job_id=job_id,
            meeting_key=meeting_key,
            audio_sha256=expected_sha,
            audio_bytes=received,
            audio_duration_seconds=expected_duration,
        )
        record["probedAudioDurationSeconds"] = probed_duration
        atomic_write_json(job_dir / "job.json", record)
    except asyncio.CancelledError:
        temporary.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        raise
    except TimeoutError:
        temporary.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        return web.json_response({"error": "upload_timeout"}, status=408)
    except Exception:
        temporary.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        try:
            job_dir.rmdir()
        except OSError:
            pass
        return web.json_response({"error": "audio_rejected"}, status=400)
    finally:
        RUNTIME.release_upload(job_id)

    try:
        RUNTIME.submit(job_id)
    except Exception:
        record = read_job_record(job_dir / "job.json")
        record = transition_job(record, "failed", error_code="executor_unavailable")
        atomic_write_json(job_dir / "job.json", record)
        return web.json_response(
            {"jobId": job_id, "status": "failed", "errorCode": "executor_unavailable"},
            status=503,
        )
    _safe_log("audio_accepted", jobId=job_id, audioBytes=received)
    return web.json_response({"jobId": job_id, "status": "accepted"}, status=202)


async def job_status(request: web.Request) -> web.Response:
    try:
        job_id = validate_job_id(request.match_info["job_id"])
        record = read_job_record(JOB_ROOT / job_id / "job.json")
    except (ContractError, OSError):
        return web.json_response({"error": "job_not_found"}, status=404)
    response = {
        "jobId": record["jobId"],
        "status": record["status"],
        "errorCode": record.get("errorCode"),
        "backendOrder": list(BACKEND_ORDER),
        "backends": record.get("backends", {}),
    }
    for name in ("totalProcessingSeconds", "resultSha256"):
        if name in record:
            response[name] = record[name]
    return web.json_response(response)


async def job_result(request: web.Request) -> web.StreamResponse:
    try:
        job_id = validate_job_id(request.match_info["job_id"])
        record = read_job_record(JOB_ROOT / job_id / "job.json")
    except (ContractError, OSError):
        return web.json_response({"error": "job_not_found"}, status=404)
    if record["status"] != "complete":
        return web.json_response({"error": "result_not_ready"}, status=409)
    result_path = JOB_ROOT / job_id / "result.json"
    try:
        digest = await asyncio.to_thread(sha256_file, result_path)
    except OSError:
        return web.json_response({"error": "result_hash_mismatch"}, status=500)
    if not hmac.compare_digest(digest, record.get("resultSha256", "")):
        return web.json_response({"error": "result_hash_mismatch"}, status=500)
    return web.FileResponse(
        result_path,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-Result-SHA256": digest,
        },
    )


async def delete_job(request: web.Request) -> web.Response:
    try:
        job_id = validate_job_id(request.match_info["job_id"])
        job_dir = JOB_ROOT / job_id
        record = read_job_record(job_dir / "job.json")
        if record["status"] not in {"complete", "failed"}:
            return web.json_response({"error": "job_not_terminal"}, status=409)
        if record["status"] == "complete":
            digest = await asyncio.to_thread(sha256_file, job_dir / "result.json")
            if not hmac.compare_digest(digest, record.get("resultSha256", "")):
                return web.json_response({"error": "result_hash_mismatch"}, status=500)
        RUNTIME._remove_known_job_files(job_dir, allow_record=True)
    except (ContractError, OSError):
        return web.json_response({"error": "job_not_found_or_unsafe"}, status=404)
    _safe_log("job_deleted", jobId=job_id)
    return web.Response(status=204)


async def on_startup(_: web.Application) -> None:
    threading.Thread(
        target=RUNTIME.check_runtime, name="runtime-check", daemon=True
    ).start()


def build_app() -> web.Application:
    app = web.Application(
        middlewares=[authentication], client_max_size=MAX_AUDIO_BYTES
    )
    app.router.add_get("/health", health)
    app.router.add_put("/v1/jobs/{job_id}", create_job)
    app.router.add_get("/v1/jobs/{job_id}", job_status)
    app.router.add_get("/v1/jobs/{job_id}/result", job_result)
    app.router.add_delete("/v1/jobs/{job_id}", delete_job)
    app.on_startup.append(on_startup)
    return app


if __name__ == "__main__":
    web.run_app(build_app(), host="0.0.0.0", port=PORT, access_log=None)
