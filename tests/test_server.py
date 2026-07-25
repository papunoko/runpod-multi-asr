from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aiohttp.test_utils import TestClient, TestServer
from multidict import CIMultiDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_IMPORT_ROOT = tempfile.TemporaryDirectory(prefix="multi-asr-server-import-")
os.environ["FAST_ASR_TOKEN"] = "t" * 43
os.environ["ASR_JOB_ROOT"] = str(Path(_IMPORT_ROOT.name) / "jobs")
server = importlib.import_module("server")
server.RUNTIME._executor.shutdown(wait=False, cancel_futures=True)


def identities() -> tuple[dict[str, str], ...]:
    values = []
    for backend_id in server.BACKEND_ORDER:
        path = server.MODEL_PATHS[backend_id]
        values.append(
            {
                "id": backend_id,
                "label": backend_id,
                "repository": f"example/{backend_id}",
                "revision": "1" * 40,
                "path": path,
                "manifestPath": f"{path}/MODEL_MANIFEST.sha256",
                "manifestSha256": "a" * 64,
            }
        )
    return tuple(values)


def write_pipeline_outputs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "chunk_id": "chunk_000",
        "index": 0,
        "core_start_s": 0.0,
        "core_end_s": 1.5,
        "audio_start_s": 0.0,
        "audio_end_s": 1.5,
        "boundary_reason": "end",
        "raw_audio": "audio/raw_stereo/chunk_000.mp3",
        "enhanced_audio": "audio/enhanced_mono/chunk_000.wav",
    }
    (run_dir / "chunks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    server.atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "source": {
                "path": str(run_dir.parent / "audio.mp3"),
                "sha256": "1" * 64,
                "probe": {"format": {"duration": "1.5"}},
            },
            "preprocess": {
                "target_chunk_s": 75.0,
                "overlap_s": 3.0,
                "search_window_s": 15.0,
            },
            "chunk_count": 1,
        },
    )
    (run_dir / "silences.json").write_text("[]\n", encoding="utf-8")
    raw = run_dir / "audio" / "raw_stereo"
    enhanced = run_dir / "audio" / "enhanced_mono"
    raw.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (raw / "chunk_000.mp3").write_bytes(b"raw")
    (enhanced / "chunk_000.wav").write_bytes(b"wav")
    for backend_id in server.BACKEND_ORDER:
        directory = run_dir / "hypotheses" / backend_id
        directory.mkdir(parents=True)
        model = (
            "/opt/models/reazon/reazonspeech-nemo-v2.nemo"
            if backend_id == "reazon_nemo"
            else server.MODEL_PATHS[backend_id]
        )
        server.atomic_write_json(
            directory / "backend.json",
            {
                "schema_version": 1,
                "backend_id": backend_id,
                "model": model,
                "chunks_total": 1,
                "chunks_completed": 1,
                "failures": [],
                "context_snapshot_id": "none",
                "audio_variant": "enhanced",
                "seed_backend": None,
            },
        )
        server.atomic_write_json(
            directory / "chunk_000.json",
            {
                "schema_version": 1,
                "backend_id": backend_id,
                "model": model,
                "chunk_id": "chunk_000",
                "audio_start_s": 0.0,
                "language": "ja",
                "capabilities": ["text"],
                "text": backend_id,
                "segments": (
                    [{"start": 0.0, "end": 1.0, "text": "whisper"}]
                    if backend_id == "faster_whisper"
                    else []
                ),
                "selected_term_ids": [],
                "context_snapshot_id": "none",
                "metadata": {},
            },
        )


def make_record(job_dir: Path, job_id: str) -> None:
    job_dir.mkdir(parents=True)
    (job_dir / "audio.mp3").write_bytes(b"audio")
    record = server.new_job_record(
        job_id=job_id,
        meeting_key="meeting-20260725",
        audio_sha256="1" * 64,
        audio_bytes=5,
        audio_duration_seconds=1.5,
    )
    record["probedAudioDurationSeconds"] = 1.5
    server.atomic_write_json(job_dir / "job.json", record)


class _CancelledContent:
    async def iter_chunked(self, _: int):
        yield b"partial"
        raise asyncio.CancelledError


class _CancelledRequest:
    def __init__(self, job_id: str, headers: CIMultiDict[str]) -> None:
        self.match_info = {"job_id": job_id}
        self.headers = headers
        self.content_length = None
        self.content = _CancelledContent()


class ServerTests(unittest.IsolatedAsyncioTestCase):
    TOKEN = "z" * 43

    def test_backend_environment_excludes_credentials_and_request_metadata(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "FAST_ASR_TOKEN": "worker-secret",
                "RUNPOD_API_KEY": "provider-secret",
                "HTTPS_PROXY": "http://proxy-user:proxy-password@example.invalid",
                "REQUEST_ID": "request-identity",
                "CUDA_VISIBLE_DEVICES": "0",
                "PATH": os.environ.get("PATH", ""),
            },
            clear=True,
        ):
            environment = server._offline_environment()
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertIn("PATH", environment)
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("FAST_ASR_TOKEN", environment)
        self.assertNotIn("RUNPOD_API_KEY", environment)
        self.assertNotIn("HTTPS_PROXY", environment)
        self.assertNotIn("REQUEST_ID", environment)

    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="multi-asr-server-test-")
        server.JOB_ROOT = Path(self.temporary.name) / "jobs"
        server.TOKEN = self.TOKEN
        server.MAX_AUDIO_BYTES = 1024 * 1024
        server.MAX_JOB_BYTES = 8 * 1024 * 1024
        server.MAX_JOBS = 8
        server.MAX_AUDIO_DURATION_SECONDS = 3600.0
        server.UPLOAD_TIMEOUT_SECONDS = 30
        server.PREPARE_TIMEOUT_SECONDS = 60
        server.MAX_BACKEND_SECONDS = 7200
        self.runtime = server.FastAsrRuntime()
        self.runtime._model_identities = identities()
        self.runtime._startup_checked.set()
        self.runtime.submit = lambda _: None
        server.RUNTIME = self.runtime
        self.original_probe = server._probe_uploaded_audio
        server._probe_uploaded_audio = lambda path, declared: declared
        app = server.build_app()
        app.on_startup.clear()
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        server._probe_uploaded_audio = self.original_probe
        self.runtime._executor.shutdown(wait=False, cancel_futures=True)
        self.temporary.cleanup()

    def headers(
        self,
        body: bytes,
        *,
        meeting_key: str = "meeting-20260725",
        digest: str | None = None,
    ) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.TOKEN}",
            "Content-Type": "application/octet-stream",
            "X-Meeting-Key": meeting_key,
            "X-Audio-SHA256": digest or hashlib.sha256(body).hexdigest(),
            "X-Audio-Bytes": str(len(body)),
            "X-Audio-Duration-Seconds": "1.5",
        }

    async def test_authentication_and_multi_backend_health_label(self) -> None:
        response = await self.client.get("/health")
        self.assertEqual(response.status, 401)
        await response.read()
        duplicate = CIMultiDict()
        duplicate.add("Authorization", f"Bearer {self.TOKEN}")
        duplicate.add("Authorization", f"Bearer {self.TOKEN}")
        response = await self.client.get("/health", headers=duplicate)
        self.assertEqual(response.status, 401)
        await response.read()
        response = await self.client.get(
            "/health", headers={"Authorization": f"Bearer {self.TOKEN}"}
        )
        self.assertEqual(response.status, 200)
        payload = await response.json()
        self.assertEqual(payload["state"], "multi_asr_ready")
        self.assertEqual(payload["backendOrder"], list(server.BACKEND_ORDER))
        self.assertEqual(
            [item["id"] for item in payload["models"]], list(server.BACKEND_ORDER)
        )

    async def test_chunked_upload_is_accepted_idempotent_and_status_has_backends(self) -> None:
        body = b"test-audio-payload"
        job_id = "a" * 32

        async def chunks():
            yield body[:4]
            yield body[4:]

        response = await self.client.put(
            f"/v1/jobs/{job_id}", data=chunks(), headers=self.headers(body)
        )
        self.assertEqual(response.status, 202)
        response = await self.client.put(
            f"/v1/jobs/{job_id}", data=body, headers=self.headers(body)
        )
        self.assertEqual(response.status, 200)
        response = await self.client.get(
            f"/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {self.TOKEN}"},
        )
        payload = await response.json()
        self.assertEqual(payload["backendOrder"], list(server.BACKEND_ORDER))
        self.assertEqual(payload["backends"]["qwen3"]["status"], "pending")

    async def test_hash_mismatch_and_cancelled_upload_remove_partial_job(self) -> None:
        body = b"corrupt-for-declared-hash"
        job_id = "b" * 32
        response = await self.client.put(
            f"/v1/jobs/{job_id}",
            data=body,
            headers=self.headers(body, digest="0" * 64),
        )
        self.assertEqual(response.status, 400)
        self.assertFalse((server.JOB_ROOT / job_id).exists())
        cancelled_id = "c" * 32
        request = _CancelledRequest(cancelled_id, CIMultiDict(self.headers(b"partial")))
        with self.assertRaises(asyncio.CancelledError):
            await server.create_job(request)
        self.assertFalse((server.JOB_ROOT / cancelled_id).exists())

    async def test_submit_failure_is_terminal(self) -> None:
        self.runtime.submit = mock.Mock(side_effect=RuntimeError("executor stopped"))
        body = b"executor-failure"
        job_id = "d" * 32
        response = await self.client.put(
            f"/v1/jobs/{job_id}", data=body, headers=self.headers(body)
        )
        self.assertEqual(response.status, 503)
        record = server.read_job_record(server.JOB_ROOT / job_id / "job.json")
        self.assertEqual(record["errorCode"], "executor_unavailable")

    async def test_readiness_checks_fail_closed_then_succeed_without_loading_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable_paths = [root / f"exe-{index}" for index in range(6)]
            for path in executable_paths:
                path.write_text("x", encoding="ascii")
                path.chmod(0o755)
            reazon = root / "reazonspeech-nemo-v2.nemo"
            reazon.write_bytes(b"model")
            failed = server.FastAsrRuntime()
            with (
                mock.patch.object(server, "FFPROBE_PATH", str(executable_paths[0])),
                mock.patch.object(
                    server,
                    "NVIDIA_SMI_COMMAND",
                    (str(executable_paths[5]), "--query-gpu=name", "--format=csv,noheader"),
                ),
                mock.patch.object(server, "LAUNCHER_PATH", executable_paths[1]),
                mock.patch.object(server, "VENV_PYTHONS", tuple(executable_paths[2:5])),
                mock.patch.object(
                    server,
                    "IMPORT_COMMANDS",
                    tuple((str(path), "-I", "-c", "pass") for path in executable_paths[2:5]),
                ),
                mock.patch.object(server, "REAZON_MODEL_FILE", reazon),
                mock.patch.object(
                    server.subprocess,
                    "run",
                    return_value=mock.Mock(returncode=1, stdout=b""),
                ),
            ):
                failed.check_runtime()
            self.assertFalse(failed.ready)
            self.assertIsNotNone(failed.model_error)
            failed._executor.shutdown(wait=False, cancel_futures=True)

            ready = server.FastAsrRuntime()
            with (
                mock.patch.dict(
                    server.MODEL_PATHS,
                    {"reazon_nemo": str(reazon.parent)},
                ),
                mock.patch.object(server, "FFPROBE_PATH", str(executable_paths[0])),
                mock.patch.object(
                    server,
                    "NVIDIA_SMI_COMMAND",
                    (str(executable_paths[5]), "--query-gpu=name", "--format=csv,noheader"),
                ),
                mock.patch.object(server, "LAUNCHER_PATH", executable_paths[1]),
                mock.patch.object(server, "VENV_PYTHONS", tuple(executable_paths[2:5])),
                mock.patch.object(
                    server,
                    "IMPORT_COMMANDS",
                    tuple((str(path), "-I", "-c", "pass") for path in executable_paths[2:5]),
                ),
                mock.patch.object(server, "REAZON_MODEL_FILE", reazon),
                mock.patch.object(server, "load_model_identities", return_value=identities()),
                mock.patch.object(server, "verify_model_manifest") as verify,
                mock.patch.object(
                    server.subprocess,
                    "run",
                    return_value=mock.Mock(returncode=0, stdout=b"GPU\n"),
                ),
            ):
                ready.check_runtime()
            self.assertTrue(ready.ready)
            self.assertEqual(verify.call_count, 3)
            ready._executor.shutdown(wait=False, cancel_futures=True)

    async def test_job_runs_exact_commands_sequentially_and_builds_result_v2(self) -> None:
        job_id = "e" * 32
        job_dir = server.JOB_ROOT / job_id
        make_record(job_dir, job_id)
        write_pipeline_outputs(job_dir / "run")
        calls: list[tuple[tuple[str, ...], int]] = []

        def run(command, timeout):
            calls.append((tuple(command), timeout))

        self.runtime._run_subprocess = run
        self.runtime._run_job(job_id)
        expected_prepare, expected_backends = self.runtime._commands(job_dir)
        self.assertEqual(
            calls,
            [
                (expected_prepare, server.PREPARE_TIMEOUT_SECONDS),
                *(
                    (expected_backends[item], server.MAX_BACKEND_SECONDS)
                    for item in server.BACKEND_ORDER
                ),
            ],
        )
        record = server.read_job_record(job_dir / "job.json")
        self.assertEqual(record["status"], "complete")
        self.assertTrue(all(item["status"] == "complete" for item in record["backends"].values()))
        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["schemaVersion"], 2)
        self.assertEqual(result["backendOrder"], list(server.BACKEND_ORDER))
        self.assertEqual([item["id"] for item in result["backends"]], list(server.BACKEND_ORDER))
        self.assertTrue((job_dir / "result.json.sha256").is_file())
        response = await self.client.get(
            f"/v1/jobs/{job_id}/result",
            headers={"Authorization": f"Bearer {self.TOKEN}"},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["X-Result-SHA256"], record["resultSha256"])
        await response.read()
        (job_dir / "result.json").write_text("{}\n", encoding="utf-8")
        response = await self.client.get(
            f"/v1/jobs/{job_id}/result",
            headers={"Authorization": f"Bearer {self.TOKEN}"},
        )
        self.assertEqual(response.status, 500)

    async def test_backend_failure_stops_sequence_and_records_backend_failure(self) -> None:
        job_id = "f" * 32
        job_dir = server.JOB_ROOT / job_id
        make_record(job_dir, job_id)
        write_pipeline_outputs(job_dir / "run")
        calls: list[str] = []

        def run(command, timeout):
            calls.append(command[1])
            if command[1] == "reazon":
                raise RuntimeError("failure")

        self.runtime._run_subprocess = run
        self.runtime._run_job(job_id)
        record = server.read_job_record(job_dir / "job.json")
        self.assertEqual(calls, ["core", "whisper", "reazon"])
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["backends"]["faster_whisper"]["status"], "complete")
        self.assertEqual(record["backends"]["reazon_nemo"]["status"], "failed")
        self.assertEqual(record["backends"]["qwen3"]["status"], "pending")
        self.assertFalse((job_dir / "result.json").exists())

    async def test_restart_and_delete_accept_known_run_tree_but_refuse_unknown(self) -> None:
        job_id = "7" * 32
        job_dir = server.JOB_ROOT / job_id
        make_record(job_dir, job_id)
        write_pipeline_outputs(job_dir / "run")
        (job_dir / "run" / ".manifest.json.abcdefgh").write_bytes(b"temporary")
        recovered = server.FastAsrRuntime()
        recovered._executor.shutdown(wait=False, cancel_futures=True)
        record = server.read_job_record(job_dir / "job.json")
        self.assertEqual(record["errorCode"], "worker_restarted")
        self.assertFalse((job_dir / "run" / ".manifest.json.abcdefgh").exists())
        recovered._remove_known_job_files(job_dir, allow_record=True)
        self.assertFalse(job_dir.exists())

        unsafe = server.JOB_ROOT / ("8" * 32)
        unsafe.mkdir()
        run = unsafe / "run"
        run.mkdir()
        unknown = run / "unknown.txt"
        unknown.write_text("preserve", encoding="utf-8")
        with self.assertRaises(server.ContractError):
            recovered._remove_known_job_files(unsafe, allow_record=False)
        self.assertTrue(unknown.exists())

    async def test_audio_probe_keeps_fixed_argv_timeout_and_format_checks(self) -> None:
        valid = mock.Mock(
            returncode=0,
            stdout=b'{"format":{"duration":"12.25","format_name":"mp3"}}',
        )
        with mock.patch.object(server.subprocess, "run", return_value=valid) as called:
            duration = self.original_probe(Path("audio.mp3"), 12.0)
        self.assertEqual(duration, 12.25)
        self.assertEqual(called.call_args.kwargs["timeout"], 30)
        self.assertNotIn("shell", called.call_args.kwargs)
        wrong = mock.Mock(
            returncode=0,
            stdout=b'{"format":{"duration":"12.0","format_name":"wav"}}',
        )
        with mock.patch.object(server.subprocess, "run", return_value=wrong):
            with self.assertRaises(server.ContractError):
                self.original_probe(Path("audio.mp3"), 12.0)


if __name__ == "__main__":
    unittest.main()
