#!/usr/bin/env python3
"""Smoke test FIR/Gita OCR through the Gemini web app client.

This script intentionally keeps the Gemini App credentials out of source. It
can read them from environment variables or, for this local migration smoke
test, from the untracked notebook that demonstrated the method.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gemini_app_client as app_client  # noqa: E402
import indic_ocr_v1_pipeline as pipeline  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_pages(page_manifest: Path, fir_docs: int, gita_images: int) -> list[dict[str, Any]]:
    rows = load_jsonl(page_manifest)
    if not rows:
        raise RuntimeError(f"No page rows found in {page_manifest}")

    selected: list[dict[str, Any]] = []
    seen_fir_docs: set[str] = set()
    if fir_docs > 0:
        fir_candidates = sorted(
            (row for row in rows if row.get("corpus") == "fir" and int(row.get("page_index") or 0) == 1),
            key=lambda row: (str(row.get("source_path")), str(row.get("page_id"))),
        )
        for row in fir_candidates:
            document_id = str(row.get("document_id"))
            if document_id in seen_fir_docs:
                continue
            image_path = ROOT / str(row["render_path"])
            if not image_path.exists():
                continue
            selected.append({**row, "sample_kind": "fir_first_page"})
            seen_fir_docs.add(document_id)
            if len(seen_fir_docs) >= fir_docs:
                break

    gita_count = 0
    if gita_images > 0:
        gita_candidates = sorted(
            (row for row in rows if row.get("corpus") == "gita"),
            key=lambda row: (str(row.get("source_path")), str(row.get("page_id"))),
        )
        for row in gita_candidates:
            image_path = ROOT / str(row["render_path"])
            if not image_path.exists():
                continue
            selected.append({**row, "sample_kind": "gita_image"})
            gita_count += 1
            if gita_count >= gita_images:
                break

    if len(seen_fir_docs) < fir_docs or gita_count < gita_images:
        raise RuntimeError(
            f"Insufficient rendered inputs: found {len(seen_fir_docs)} FIR first pages "
            f"and {gita_count} Gita images."
        )
    return selected


async def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    output = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    report_dir = output / "reports"
    raw_path = report_dir / "gemini_app_raw_responses.jsonl"
    run_path = report_dir / "gemini_app_runs.jsonl"
    failure_path = report_dir / "gemini_app_failures.jsonl"
    selected_path = output / "manifests" / "selected_pages.jsonl"
    summary_path = report_dir / "gemini_app_smoke_summary.json"

    pages = select_pages(ROOT / args.page_manifest, args.fir_docs, args.gita_images)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline.jsonl_write(selected_path, pages)

    psid, psidts, credential_source = app_client.load_webapi_credentials(ROOT / args.notebook)
    previous_cookie_path = os.environ.get("GEMINI_COOKIE_PATH")
    temporary_cookie_cache: tempfile.TemporaryDirectory[str] | None = None
    if args.cookie_cache_dir:
        cookie_cache_path = Path(args.cookie_cache_dir).expanduser()
        cookie_cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["GEMINI_COOKIE_PATH"] = str(cookie_cache_path)
        cookie_cache_mode = "persistent_user_supplied"
    else:
        temporary_cookie_cache = tempfile.TemporaryDirectory(prefix="gemini_webapi_cookie_cache_")
        os.environ["GEMINI_COOKIE_PATH"] = temporary_cookie_cache.name
        cookie_cache_mode = "temporary_deleted_after_run"

    client = None
    try:
        client = await app_client.init_app_client(
            psid,
            psidts,
            proxy=args.proxy,
            init_timeout=args.init_timeout,
            verbose=args.verbose,
        )
        selected_model, model_info = app_client.resolve_requested_model(client, args.model)
        available_models = app_client.available_model_rows(client)
    except Exception:
        if temporary_cookie_cache:
            temporary_cookie_cache.cleanup()
        if previous_cookie_path is None:
            os.environ.pop("GEMINI_COOKIE_PATH", None)
        else:
            os.environ["GEMINI_COOKIE_PATH"] = previous_cookie_path
        raise

    min_confidence = float(args.min_confidence)
    successes = 0
    failures = 0
    validation_errors: dict[str, int] = {}
    sleep_values: list[float] = []
    started_all = time.time()

    try:
        for index, page in enumerate(pages, 1):
            delay = round(random.uniform(args.min_sleep, args.max_sleep), 3)
            sleep_values.append(delay)
            await asyncio.sleep(delay)

            started = time.time()
            raw_text = ""
            status: dict[str, Any]
            normalized: dict[str, Any] | None = None
            try:
                parsed, call_status, raw_text = await app_client.call_gemini_app_page(
                    page=page,
                    client=client,
                    selected_model=selected_model,
                    model_key=f"gemini_webapi:{args.model}",
                    model_info=model_info,
                    request_timeout=args.request_timeout,
                )
                if parsed is None:
                    status = {
                        **call_status,
                        "ok": False,
                        "status": "call_failed",
                        "validation_errors": [],
                    }
                else:
                    ok, errors, normalized = pipeline.validate_teacher_payload(parsed, min_confidence)
                    status = {
                        "ok": ok,
                        "status": "accepted" if ok else "invalid_json_or_quality",
                        "validation_errors": errors,
                        "confidence": normalized.get("confidence") if normalized else None,
                        "error_type": call_status.get("error_type"),
                        "error": call_status.get("error"),
                    }
            except Exception as exc:  # noqa: BLE001 - smoke runner must log exact API failures.
                status = {
                    "ok": False,
                    "status": "call_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "validation_errors": [],
                }

            elapsed = round(time.time() - started, 3)
            raw_row = {
                "created_at": utc_now(),
                "page_id": page["page_id"],
                "source_path": page["source_path"],
                "corpus": page["corpus"],
                "sample_kind": page["sample_kind"],
                "raw_response": raw_text,
                "response_hash": app_client.response_hash(raw_text) if raw_text else None,
            }
            status_for_row = dict(status)
            status_for_row.pop("model", None)
            status_for_row.pop("model_info", None)
            status_for_row.pop("transport", None)
            run_row = {
                "created_at": utc_now(),
                "index": index,
                "page_id": page["page_id"],
                "document_id": page["document_id"],
                "source_path": page["source_path"],
                "render_path": page["render_path"],
                "corpus": page["corpus"],
                "sample_kind": page["sample_kind"],
                "sleep_before_request_seconds": delay,
                "elapsed_seconds": elapsed,
                "transport": "gemini_webapi",
                "credential_source": credential_source,
                "model": model_info,
                **status_for_row,
            }
            append_jsonl(raw_path, raw_row)
            append_jsonl(run_path, run_row)
            if status["ok"]:
                successes += 1
            else:
                failures += 1
                append_jsonl(failure_path, run_row)
                for error in status.get("validation_errors") or []:
                    validation_errors[str(error)] = validation_errors.get(str(error), 0) + 1

            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(pages),
                        "page_id": page["page_id"],
                        "corpus": page["corpus"],
                        "ok": status["ok"],
                        "status": status["status"],
                        "elapsed_seconds": elapsed,
                        "sleep_seconds": delay,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        if client is not None:
            await client.close()
        if temporary_cookie_cache:
            temporary_cookie_cache.cleanup()
        if previous_cookie_path is None:
            os.environ.pop("GEMINI_COOKIE_PATH", None)
        else:
            os.environ["GEMINI_COOKIE_PATH"] = previous_cookie_path

    summary = {
        "created_at": utc_now(),
        "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "transport": "gemini_webapi",
        "credential_source": credential_source,
        "cookie_cache_mode": cookie_cache_mode,
        "model": model_info,
        "available_models": available_models,
        "page_manifest": args.page_manifest,
        "requested": {"fir_docs": args.fir_docs, "gita_images": args.gita_images},
        "attempted_pages": len(pages),
        "attempted_fir_first_pages": sum(1 for page in pages if page["sample_kind"] == "fir_first_page"),
        "attempted_gita_images": sum(1 for page in pages if page["sample_kind"] == "gita_image"),
        "successes": successes,
        "failures": failures,
        "success_rate": round(successes / max(1, len(pages)), 6),
        "min_sleep_seconds": args.min_sleep,
        "max_sleep_seconds": args.max_sleep,
        "actual_sleep_seconds": {
            "min": min(sleep_values) if sleep_values else None,
            "max": max(sleep_values) if sleep_values else None,
            "values": sleep_values,
        },
        "validation_error_counts": validation_errors,
        "elapsed_seconds": round(time.time() - started_all, 3),
        "artifacts": {
            "selected_pages": str(selected_path.relative_to(ROOT)),
            "runs": str(run_path.relative_to(ROOT)),
            "raw_responses": str(raw_path.relative_to(ROOT)),
            "failures": str(failure_path.relative_to(ROOT)),
            "summary": str(summary_path.relative_to(ROOT)),
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-manifest", default="artifacts/page_only_v1/manifests/page_manifest.jsonl")
    parser.add_argument("--output", default="artifacts/gemini_app_smoke_local")
    parser.add_argument("--notebook", default="Untitled3.ipynb")
    parser.add_argument("--fir-docs", type=int, default=10)
    parser.add_argument("--gita-images", type=int, default=10)
    parser.add_argument("--min-sleep", type=float, default=0.8)
    parser.add_argument("--max-sleep", type=float, default=2.2)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument(
        "--model",
        default="auto-flash",
        help="Use auto-flash to prefer gemini-3.5-flash if the account exposes it; use default for Gemini App default.",
    )
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--init-timeout", type=float, default=60)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument(
        "--cookie-cache-dir",
        default=None,
        help="Optional explicit gemini_webapi cookie cache directory. Defaults to a deleted temp dir.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("--max-sleep must be greater than or equal to --min-sleep")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    summary = asyncio.run(run_batch(args))
    return 0 if summary["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
