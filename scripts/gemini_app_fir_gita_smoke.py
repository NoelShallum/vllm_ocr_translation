#!/usr/bin/env python3
"""Smoke test FIR/Gita OCR through the Gemini web app client.

This script intentionally keeps the Gemini App credentials out of source. It
can read them from environment variables or, for this local migration smoke
test, from the untracked notebook that demonstrated the method.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
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

import indic_ocr_v1_pipeline as pipeline  # noqa: E402
from gemini_webapi import GeminiClient  # noqa: E402
from gemini_webapi.constants import Model  # noqa: E402


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


def dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def notebook_assignment_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    notebook = json.loads(path.read_text(encoding="utf-8"))
    values: dict[str, str] = {}
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or [])
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                continue
            if not isinstance(node.value.value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"Secure_1PSID", "Secure_1PSIDTS"}:
                    values[target.id] = node.value.value
    return values


def load_webapi_credentials(notebook_path: Path) -> tuple[str, str, str]:
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    psid = env.get("GEMINI_WEBAPI_SECURE_1PSID") or env.get("Secure_1PSID")
    psidts = env.get("GEMINI_WEBAPI_SECURE_1PSIDTS") or env.get("Secure_1PSIDTS")
    source = "environment"
    if not psid or not psidts:
        notebook_values = notebook_assignment_values(notebook_path)
        psid = psid or notebook_values.get("Secure_1PSID")
        psidts = psidts or notebook_values.get("Secure_1PSIDTS")
        source = f"notebook:{notebook_path.name}"
    if not psid or not psidts:
        raise RuntimeError(
            "Missing Gemini App credentials. Set GEMINI_WEBAPI_SECURE_1PSID and "
            "GEMINI_WEBAPI_SECURE_1PSIDTS, or pass --notebook with the local demo notebook."
        )
    return psid, psidts, source


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


def ocr_prompt() -> str:
    return (
        "You are an OCR expert for Indian legal, police, and Indic book pages. "
        "Transcribe only visible text from this full page image. Preserve script, spelling, "
        "line breaks, page numbers, tables, stamps, signatures, handwritten regions, form labels, "
        "and illegible markers. Return valid JSON only with exactly these top-level keys: "
        "plain_text, markdown, blocks, key_values, tables, quality_flags, confidence. "
        "Blocks must be an ordered list with text, type, bbox, language, and confidence when visible. "
        "Do not translate. Do not explain your reasoning. Do not include markdown fences or any text "
        "outside JSON. The first character of your response must be { and the last character must be }. "
        "For unclear handwriting, write your best single reading once and use [illegible] for unreadable spans."
    )


def response_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def available_model_rows(client: GeminiClient) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in client.list_models() or []:
        rows.append({
            "model_name": str(getattr(model, "model_name", "")),
            "display_name": str(getattr(model, "display_name", "")),
            "description": str(getattr(model, "description", "")),
            "is_available": bool(getattr(model, "is_available", True)),
            "advanced_only": bool(getattr(model, "advanced_only", False)),
        })
    return rows


def resolve_requested_model(client: GeminiClient, requested: str) -> tuple[Any, dict[str, Any]]:
    requested = requested.strip()
    if not requested or requested == "default":
        return Model.UNSPECIFIED, {"requested": requested or "default", "resolved": "unspecified", "source": "default"}
    if requested != "auto-flash":
        return requested, {"requested": requested, "resolved": requested, "source": "explicit"}

    models = [model for model in client.list_models() or [] if bool(getattr(model, "is_available", True))]
    flash_models = [
        model for model in models
        if "flash" in str(getattr(model, "model_name", "")).lower()
        or "flash" in str(getattr(model, "display_name", "")).lower()
    ]
    preferred_names = [
        "gemini-3.5-flash",
        "gemini-3-5-flash",
        "gemini-3.5-flash-plus",
        "gemini-3.5-flash-advanced",
        "gemini-3-flash",
        "gemini-3-flash-plus",
        "gemini-3-flash-advanced",
    ]
    by_name = {str(getattr(model, "model_name", "")): model for model in flash_models}
    for name in preferred_names:
        if name in by_name:
            model = by_name[name]
            return model, {
                "requested": "auto-flash",
                "resolved": str(getattr(model, "model_name", name)),
                "display_name": str(getattr(model, "display_name", "")),
                "source": "dynamic_registry",
            }
    if flash_models:
        non_thinking = [
            model for model in flash_models
            if "thinking" not in str(getattr(model, "model_name", "")).lower()
        ]
        candidates = non_thinking or flash_models
        model = sorted(candidates, key=lambda item: str(getattr(item, "model_name", "")))[-1]
        return model, {
            "requested": "auto-flash",
            "resolved": str(getattr(model, "model_name", "")),
            "display_name": str(getattr(model, "display_name", "")),
            "source": "dynamic_registry_fallback",
        }
    return Model.BASIC_FLASH, {
        "requested": "auto-flash",
        "resolved": Model.BASIC_FLASH.model_name,
        "source": "enum_fallback",
    }


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

    psid, psidts, credential_source = load_webapi_credentials(ROOT / args.notebook)
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

    client = GeminiClient(psid, psidts, proxy=args.proxy)
    try:
        await client.init(
            timeout=args.init_timeout,
            auto_close=False,
            close_delay=300,
            auto_refresh=True,
            verbose=args.verbose,
        )
        selected_model, model_info = resolve_requested_model(client, args.model)
        available_models = available_model_rows(client)
    except Exception:
        if temporary_cookie_cache:
            temporary_cookie_cache.cleanup()
        if previous_cookie_path is None:
            os.environ.pop("GEMINI_COOKIE_PATH", None)
        else:
            os.environ["GEMINI_COOKIE_PATH"] = previous_cookie_path
        raise

    prompt = ocr_prompt()
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

            image_path = ROOT / str(page["render_path"])
            started = time.time()
            raw_text = ""
            status: dict[str, Any]
            normalized: dict[str, Any] | None = None
            try:
                request_kwargs: dict[str, Any] = {"temporary": True}
                if args.request_timeout:
                    request_kwargs["timeout"] = args.request_timeout
                if selected_model is Model.UNSPECIFIED:
                    response = await client.generate_content(prompt, files=[image_path], **request_kwargs)
                else:
                    response = await client.generate_content(prompt, files=[image_path], model=selected_model, **request_kwargs)
                raw_text = response.text or ""
                parsed = pipeline.parse_json_response(raw_text)
                ok, errors, normalized = pipeline.validate_teacher_payload(parsed, min_confidence)
                status = {
                    "ok": ok,
                    "status": "accepted" if ok else "invalid_json_or_quality",
                    "validation_errors": errors,
                    "confidence": normalized.get("confidence") if normalized else None,
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
                "response_hash": response_hash(raw_text) if raw_text else None,
            }
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
                **status,
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
    parser.add_argument("--output", default="artifacts/gemini_app_smoke_20260615")
    parser.add_argument("--notebook", default="Untitled3.ipynb")
    parser.add_argument("--fir-docs", type=int, default=10)
    parser.add_argument("--gita-images", type=int, default=5)
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
