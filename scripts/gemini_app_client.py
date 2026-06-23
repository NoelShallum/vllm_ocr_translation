#!/usr/bin/env python3
"""Shared Gemini App API helpers for FIR/Gita OCR."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model


ROOT = Path(__file__).resolve().parents[1]
GEMINI_REQUIRED_FIELDS = {
    "plain_text",
    "markdown",
    "blocks",
    "key_values",
    "tables",
    "quality_flags",
    "confidence",
}
INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')


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


def load_webapi_credentials(notebook_path: Path | None = None) -> tuple[str, str, str]:
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    psid = env.get("GEMINI_WEBAPI_SECURE_1PSID") or env.get("Secure_1PSID")
    psidts = env.get("GEMINI_WEBAPI_SECURE_1PSIDTS") or env.get("Secure_1PSIDTS")
    source = "environment"
    if (not psid or not psidts) and notebook_path is not None:
        notebook_values = notebook_assignment_values(notebook_path)
        psid = psid or notebook_values.get("Secure_1PSID")
        psidts = psidts or notebook_values.get("Secure_1PSIDTS")
        source = f"notebook:{notebook_path.name}"
    if not psid or not psidts:
        raise RuntimeError(
            "Missing Gemini App credentials. Set GEMINI_WEBAPI_SECURE_1PSID and "
            "GEMINI_WEBAPI_SECURE_1PSIDTS."
        )
    return psid, psidts, source


def escape_json_string_controls(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if not in_string:
            out.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            out.append(char)
            escaped = False
        elif char == "\\":
            out.append(char)
            escaped = True
        elif char == '"':
            out.append(char)
            in_string = False
        elif char == "\n":
            out.append("\\n")
        elif char == "\r":
            out.append("\\r")
        elif char == "\t":
            out.append("\\t")
        else:
            out.append(char)
    return "".join(out)


def repair_invalid_json_escapes(text: str) -> str:
    return INVALID_JSON_ESCAPE_RE.sub(r"\\\\", text)


def parse_json_response(text: str) -> Any:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    variants = [
        stripped,
        escape_json_string_controls(stripped),
        repair_invalid_json_escapes(escape_json_string_controls(stripped)),
    ]
    candidates: list[Any] = []
    seen: set[str] = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        try:
            value = json.loads(variant)
            if isinstance(value, dict) and GEMINI_REQUIRED_FIELDS.issubset(value):
                return value
            candidates.append(value)
        except json.JSONDecodeError:
            pass

        for idx, char in enumerate(variant):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(variant[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and GEMINI_REQUIRED_FIELDS.issubset(value):
                return value
            if isinstance(value, list) and any(
                isinstance(item, dict) and GEMINI_REQUIRED_FIELDS.issubset(item)
                for item in value
            ):
                return value
            candidates.append(value)

    scored = [
        (len(GEMINI_REQUIRED_FIELDS.intersection(value)), value)
        for value in candidates
        if isinstance(value, dict)
    ]
    if scored:
        score, value = max(scored, key=lambda item: item[0])
        if score > 0:
            return value
    raise ValueError("Gemini App response did not contain parseable OCR JSON")


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


async def init_app_client(
    psid: str,
    psidts: str,
    *,
    proxy: str | None = None,
    init_timeout: float = 60,
    verbose: bool = False,
) -> GeminiClient:
    client = GeminiClient(psid, psidts, proxy=proxy)
    await client.init(
        timeout=init_timeout,
        auto_close=False,
        close_delay=300,
        auto_refresh=True,
        verbose=verbose,
    )
    return client


async def call_gemini_app_page(
    *,
    page: dict[str, Any],
    client: GeminiClient,
    selected_model: Any,
    model_key: str,
    model_info: dict[str, Any],
    request_timeout: float | None,
) -> tuple[Any | None, dict[str, Any], str]:
    image_path = ROOT / str(page["render_path"])
    started = time.time()
    raw_text = ""
    try:
        request_kwargs: dict[str, Any] = {"temporary": True}
        if request_timeout:
            request_kwargs["timeout"] = request_timeout
        if selected_model is Model.UNSPECIFIED:
            response = await client.generate_content(ocr_prompt(), files=[image_path], **request_kwargs)
        else:
            response = await client.generate_content(ocr_prompt(), files=[image_path], model=selected_model, **request_kwargs)
        raw_text = response.text or ""
        parsed = parse_json_response(raw_text)
        return parsed, {
            "ok": True,
            "model": model_key,
            "model_info": model_info,
            "transport": "gemini_webapi",
            "elapsed_seconds": round(time.time() - started, 3),
            "finish_reasons": [],
            "thought_part_count": 0,
            "response_hash": response_hash(raw_text) if raw_text else None,
        }, raw_text
    except Exception as exc:  # noqa: BLE001 - caller logs exact App API failures.
        return None, {
            "ok": False,
            "model": model_key,
            "model_info": model_info,
            "transport": "gemini_webapi",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
            "finish_reasons": [],
            "elapsed_seconds": round(time.time() - started, 3),
        }, raw_text
