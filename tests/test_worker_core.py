from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "worker_core.py"
SPEC = importlib.util.spec_from_file_location("worker_core", MODULE_PATH)
worker_core = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(worker_core)


def identity(backend_id: str) -> dict[str, str]:
    path = worker_core.MODEL_PATHS[backend_id]
    return {
        "id": backend_id,
        "label": backend_id.replace("_", "-"),
        "repository": f"example/{backend_id}",
        "revision": "1" * 40,
        "path": path,
        "manifestPath": f"{path}/MODEL_MANIFEST.sha256",
        "manifestSha256": "a" * 64,
    }


def record() -> dict[str, object]:
    value = worker_core.new_job_record(
        job_id="a" * 32,
        meeting_key="meeting-20260725",
        audio_sha256="b" * 64,
        audio_bytes=1234,
        audio_duration_seconds=10.0,
    )
    value["probedAudioDurationSeconds"] = 10.0
    return worker_core.transition_job(value, "running")


def write_run(run_dir: Path, *, malformed_backend: str | None = None) -> None:
    run_dir.mkdir(parents=True)
    rows = [
        {
            "chunk_id": "chunk_000",
            "index": 0,
            "core_start_s": 0.0,
            "core_end_s": 10.0,
            "audio_start_s": 0.0,
            "audio_end_s": 10.0,
            "boundary_reason": "end",
            "raw_audio": "audio/raw_stereo/chunk_000.mp3",
            "enhanced_audio": "audio/enhanced_mono/chunk_000.wav",
        }
    ]
    (run_dir / "chunks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    raw = run_dir / "audio" / "raw_stereo"
    enhanced = run_dir / "audio" / "enhanced_mono"
    raw.mkdir(parents=True)
    enhanced.mkdir(parents=True)
    (raw / "chunk_000.mp3").write_bytes(b"raw")
    (enhanced / "chunk_000.wav").write_bytes(b"wav")
    worker_core.atomic_write_json(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "source": {
                "path": str(run_dir.parent / "audio.mp3"),
                "sha256": "b" * 64,
                "probe": {"format": {"duration": "10.0"}},
            },
            "preprocess": {
                "target_chunk_s": 75.0,
                "overlap_s": 3.0,
                "search_window_s": 15.0,
            },
            "chunk_count": 1,
        },
    )
    for backend_id in worker_core.BACKEND_ORDER:
        directory = run_dir / "hypotheses" / backend_id
        directory.mkdir(parents=True)
        model = worker_core.BACKEND_MODEL_ARGUMENTS[backend_id]
        worker_core.atomic_write_json(
            directory / "backend.json",
            {
                "schema_version": 1,
                "backend_id": backend_id,
                "model": model,
                "chunks_total": 1,
                "chunks_completed": 0 if malformed_backend == backend_id else 1,
                "failures": [],
                "context_snapshot_id": "none",
                "audio_variant": "enhanced",
                "seed_backend": None,
            },
        )
        segments = (
            [{"index": 0, "start": 0.0, "end": 1.0, "text": "テスト"}]
            if backend_id == "faster_whisper"
            else []
        )
        worker_core.atomic_write_json(
            directory / "chunk_000.json",
            {
                "schema_version": 1,
                "backend_id": backend_id,
                "model": model,
                "chunk_id": "chunk_000",
                "audio_start_s": 0.0,
                "language": "ja",
                "capabilities": ["text"],
                "text": f"{backend_id}の結果",
                "segments": segments,
                "selected_term_ids": [],
                "context_snapshot_id": "none",
                "metadata": {},
            },
        )


class WorkerCoreTests(unittest.TestCase):
    def test_job_lifecycle_initializes_fixed_backend_order(self) -> None:
        accepted = worker_core.new_job_record(
            job_id="a" * 32,
            meeting_key="meeting",
            audio_sha256="b" * 64,
            audio_bytes=1,
            audio_duration_seconds=1.0,
        )
        self.assertEqual(accepted["backendOrder"], list(worker_core.BACKEND_ORDER))
        self.assertEqual(
            [item["status"] for item in accepted["backends"].values()],
            ["pending", "pending", "pending"],
        )
        running = worker_core.transition_job(accepted, "running")
        complete = worker_core.transition_job(running, "complete")
        self.assertEqual(complete["status"], "complete")

    def test_invalid_transition_is_rejected(self) -> None:
        accepted = worker_core.new_job_record(
            job_id="a" * 32,
            meeting_key="meeting",
            audio_sha256="b" * 64,
            audio_bytes=1,
            audio_duration_seconds=1.0,
        )
        with self.assertRaises(worker_core.ContractError):
            worker_core.transition_job(accepted, "complete")

    def test_identity_manifest_is_exact_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "MODELS.json"
            worker_core.atomic_write_json(
                path,
                {
                    "schemaVersion": 1,
                    "backends": [identity(item) for item in worker_core.BACKEND_ORDER],
                },
            )
            values = worker_core.load_model_identities(path)
            self.assertEqual(tuple(item["id"] for item in values), worker_core.BACKEND_ORDER)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["backends"].reverse()
            worker_core.atomic_write_json(path, payload)
            with self.assertRaises(worker_core.ContractError):
                worker_core.load_model_identities(path)

    def test_manifest_verification_rejects_escape_and_accepts_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            root.mkdir()
            model = root / "model.bin"
            model.write_bytes(b"model")
            line = f"{hashlib.sha256(b'model').hexdigest()}  model.bin\n"
            manifest = root / "MODEL_MANIFEST.sha256"
            manifest.write_text(line, encoding="utf-8")
            worker_core.verify_model_manifest(
                root, manifest, worker_core.sha256_file(manifest)
            )
            manifest.write_text(f"{'0' * 64}  ../outside\n", encoding="utf-8")
            with self.assertRaises(worker_core.ContractError):
                worker_core.verify_model_manifest(
                    root, manifest, worker_core.sha256_file(manifest)
                )

    def test_backend_outputs_and_schema_v2_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            run_dir = job_dir / "run"
            write_run(run_dir)
            running = record()
            worker_core.validate_prepared_run(run_dir, running)
            outputs = {
                backend_id: worker_core.validate_backend_output(run_dir, backend_id)
                for backend_id in worker_core.BACKEND_ORDER
            }
            timings = {
                backend_id: {
                    "startedAt": "2026-07-25T00:00:00Z",
                    "completedAt": "2026-07-25T00:00:01Z",
                    "processingSeconds": 1.0,
                }
                for backend_id in worker_core.BACKEND_ORDER
            }
            result = worker_core.build_result(
                record=running,
                model_identities=[identity(item) for item in worker_core.BACKEND_ORDER],
                backend_outputs=outputs,
                backend_timings=timings,
                total_processing_seconds=4.0,
            )
            self.assertEqual(result["schemaVersion"], 2)
            self.assertEqual(result["audioSha256"], "b" * 64)
            self.assertNotIn("audio", result)
            self.assertEqual(result["backendOrder"], list(worker_core.BACKEND_ORDER))
            self.assertEqual([item["status"] for item in result["backends"]], ["complete"] * 3)
            self.assertEqual(result["backends"][0]["summary"]["segmentCount"], 1)
            self.assertEqual(result["backends"][1]["chunks"][0]["segments"], [])

    def test_incomplete_and_malformed_backend_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            write_run(run_dir, malformed_backend="reazon_nemo")
            with self.assertRaises(worker_core.ContractError):
                worker_core.validate_backend_output(run_dir, "reazon_nemo")

            whisper = run_dir / "hypotheses" / "faster_whisper" / "chunk_000.json"
            payload = json.loads(whisper.read_text(encoding="utf-8"))
            payload["segments"] = []
            worker_core.atomic_write_json(whisper, payload)
            with self.assertRaises(worker_core.ContractError):
                worker_core.validate_backend_output(run_dir, "faster_whisper")

            qwen = run_dir / "hypotheses" / "qwen3" / "chunk_000.json"
            qwen.unlink()
            with self.assertRaises(worker_core.ContractError):
                worker_core.validate_backend_output(run_dir, "qwen3")


if __name__ == "__main__":
    unittest.main()
