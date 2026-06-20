#!/usr/bin/env python3
"""Safe diagnostics for the Gemini App browser-cookie transport.

The report intentionally records only booleans, counts, hashes, model names,
and failure classes. Cookie values, access tokens, upload identifiers, and raw
model responses are kept in memory only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gemini_app_browser_smoke as browser_smoke  # noqa: E402
import gemini_app_fir_gita_smoke as fir_gita_smoke  # noqa: E402
import indic_ocr_v1_pipeline as pipeline  # noqa: E402
from gemini_webapi import GeminiClient  # noqa: E402
from gemini_webapi.constants import Model  # noqa: E402
from gemini_webapi.exceptions import (  # noqa: E402
    APIError,
    AuthError,
    GeminiError,
    TemporarilyBlocked,
    TimeoutError as GeminiTimeoutError,
    UsageLimitExceeded,
)
from gemini_webapi.utils import upload_file  # noqa: E402


FORBIDDEN_REPORT_STRINGS = (
    "__Secure-1PSID",
    "__Secure-1PSIDTS",
    "Secure_1PSID",
    "Secure_1PSIDTS",
    "SECURE_1PSID",
    "SECURE_1PSIDTS",
    "secure_1psid",
    "secure_1psidts",
    "1PSID",
    "1PSIDTS",
    "SNlM0e",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative_or_absolute(path: Path) -> str:
    return str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def safe_error(exc: BaseException, secret_values: list[str] | None = None) -> str:
    text = f"{type(exc).__name__}: {exc}"[:2000]
    for value in secret_values or []:
        if value and len(value) >= 8:
            text = text.replace(value, "[redacted]")
    for forbidden in FORBIDDEN_REPORT_STRINGS:
        text = text.replace(forbidden, "[redacted]")
    text = re.sub(r"([A-Za-z0-9_./+=:-]{48,})", "[redacted-long-value]", text)
    return text[:1000]


def classify_exception(exc: BaseException) -> str:
    message = str(exc).lower()
    if isinstance(exc, GeminiTimeoutError) or "timeout" in message or "timed out" in message:
        return "timeout"
    if isinstance(exc, (UsageLimitExceeded, TemporarilyBlocked)) or "429" in message:
        return "rate_or_usage_block"
    if "quota" in message or "usage limit" in message or "temporarily blocked" in message:
        return "rate_or_usage_block"
    if isinstance(exc, AuthError) or "401" in message or "unauthorized" in message:
        return "stale_cookies"
    if "1100" in message:
        return "api_error_1100"
    if "snlm0e" in message or "access token" in message:
        return "missing_access_token"
    if isinstance(exc, APIError):
        return "api_error"
    if isinstance(exc, GeminiError):
        return "generation_error"
    return type(exc).__name__


class Recorder:
    def __init__(self, output: Path):
        self.output = output
        self.steps_path = output / "diagnostics_steps.jsonl"
        self.steps: list[dict[str, Any]] = []

    def add(self, name: str, status: str, **metadata: Any) -> None:
        row = {
            "created_at": utc_now(),
            "step": name,
            "status": status,
            **metadata,
        }
        self.steps.append(row)
        append_jsonl(self.steps_path, row)


def browser_cookie_status(cookie_env: dict[str, str]) -> dict[str, bool]:
    psid_present = bool(cookie_env.get("GEMINI_WEBAPI_SECURE_1PSID"))
    psidts_present = bool(cookie_env.get("GEMINI_WEBAPI_SECURE_1PSIDTS"))
    return {
        "primary_auth_cookie_present": psid_present,
        "timestamp_auth_cookie_present": psidts_present,
        "required_auth_cookies_present": psid_present and psidts_present,
    }


def poll_browser_cookies(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, bool], str]:
    deadline = time.time() + args.cookie_timeout
    last_status = "not_checked"
    cookie_env: dict[str, str] = {}

    while True:
        proc = browser_smoke.run_agent_browser(args, "--json", "cookies", "get", check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                cookie_env, _metadata = browser_smoke.extract_cookie_env(proc.stdout)
            except json.JSONDecodeError as exc:
                last_status = f"cookie_json_parse_failed:{type(exc).__name__}"
            else:
                status = browser_cookie_status(cookie_env)
                if status["required_auth_cookies_present"]:
                    return cookie_env, status, "ok"
                last_status = "required_browser_cookies_missing"
        else:
            last_status = f"cookie_read_failed:{proc.returncode}"

        if time.time() >= deadline:
            return cookie_env, browser_cookie_status(cookie_env), last_status
        time.sleep(args.cookie_poll_seconds)


def select_probe_page(page_manifest: Path) -> tuple[dict[str, Any], str]:
    failures: list[str] = []
    for fir_docs, gita_images, label in ((1, 0, "fir_first_page"), (0, 1, "gita_image")):
        try:
            pages = fir_gita_smoke.select_pages(page_manifest, fir_docs=fir_docs, gita_images=gita_images)
            if pages:
                return pages[0], label
        except Exception as exc:  # noqa: BLE001 - diagnostics should try the fallback corpus.
            failures.append(f"{label}:{type(exc).__name__}")
    raise RuntimeError(
        f"No rendered FIR/Gita probe image found in {page_manifest}. Attempts: {', '.join(failures)}"
    )


def model_names(rows: list[dict[str, Any]]) -> list[str]:
    names = []
    for row in rows:
        name = str(row.get("model_name") or "").strip()
        display = str(row.get("display_name") or "").strip()
        names.append(name or display)
    return sorted(name for name in set(names) if name)


async def generate_with_model(
    client: GeminiClient,
    prompt: str,
    selected_model: Any,
    *,
    files: list[Path] | None = None,
    timeout: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {"temporary": True}
    if timeout:
        kwargs["timeout"] = timeout
    if files:
        kwargs["files"] = files
    if selected_model is Model.UNSPECIFIED:
        return await client.generate_content(prompt, **kwargs)
    return await client.generate_content(prompt, model=selected_model, **kwargs)


async def run_client_checks(
    args: argparse.Namespace,
    recorder: Recorder,
    cookie_env: dict[str, str],
) -> tuple[dict[str, Any], str | None]:
    client: GeminiClient | None = None
    selected_model: Any = Model.UNSPECIFIED
    secret_values = list(cookie_env.values())
    previous_cookie_path = os.environ.get("GEMINI_COOKIE_PATH")
    report: dict[str, Any] = {
        "initialized": False,
        "access_token_present": False,
        "session_id_present": False,
        "push_id_present": False,
        "account_status": "unknown",
        "model_listing": {"ok": False, "model_count": 0, "model_names": []},
        "model_resolution": {},
        "text_generation": {"ok": False},
        "upload_probe": {"ok": False},
        "ocr_generation": {"ok": False},
        "cookie_cache_mode": "temporary_deleted_after_run",
    }
    failure_class: str | None = None

    try:
        with tempfile.TemporaryDirectory(prefix="gemini_app_diag_cookie_cache_") as cookie_cache:
            os.environ["GEMINI_COOKIE_PATH"] = cookie_cache
            try:
                try:
                    client = GeminiClient(
                        cookie_env["GEMINI_WEBAPI_SECURE_1PSID"],
                        cookie_env["GEMINI_WEBAPI_SECURE_1PSIDTS"],
                        proxy=args.proxy,
                    )
                    await client.init(
                        timeout=args.init_timeout,
                        auto_close=False,
                        auto_refresh=False,
                        verbose=args.verbose,
                    )
                    report.update(
                        {
                            "initialized": True,
                            "access_token_present": bool(client.access_token),
                            "session_id_present": bool(client.session_id),
                            "push_id_present": bool(client.push_id),
                            "account_status": str(client.account_status),
                        }
                    )
                    if not client.access_token:
                        failure_class = "missing_access_token"
                        recorder.add("client_init", "failed", failure_class=failure_class, client=report)
                        return report, failure_class
                    recorder.add(
                        "client_init",
                        "passed",
                        access_token_present=report["access_token_present"],
                        session_id_present=report["session_id_present"],
                        push_id_present=report["push_id_present"],
                        account_status=report["account_status"],
                    )
                except Exception as exc:  # noqa: BLE001 - failure classification is the diagnostic output.
                    failure_class = classify_exception(exc)
                    report["init_error"] = safe_error(exc, secret_values)
                    recorder.add("client_init", "failed", failure_class=failure_class, error=report["init_error"])
                    return report, failure_class

                try:
                    available_models = fir_gita_smoke.available_model_rows(client)
                    selected_model, model_info = fir_gita_smoke.resolve_requested_model(client, args.model)
                    names = model_names(available_models)
                    report["model_listing"] = {
                        "ok": bool(names),
                        "model_count": len(names),
                        "model_names": names,
                    }
                    report["model_resolution"] = {
                        key: value
                        for key, value in model_info.items()
                        if key in {"requested", "resolved", "display_name", "source"}
                    }
                    if not names:
                        failure_class = "model_drift"
                        recorder.add("model_listing", "failed", failure_class=failure_class, model_count=0)
                        return report, failure_class
                    recorder.add(
                        "model_listing",
                        "passed",
                        model_count=len(names),
                        model_names=names,
                        model_resolution=report["model_resolution"],
                    )
                except Exception as exc:  # noqa: BLE001
                    failure_class = classify_exception(exc)
                    report["model_listing"]["error"] = safe_error(exc, secret_values)
                    recorder.add(
                        "model_listing",
                        "failed",
                        failure_class=failure_class,
                        error=report["model_listing"]["error"],
                    )
                    return report, failure_class

                if args.health_only:
                    report["text_generation"] = {"ok": None, "status": "skipped_health_only"}
                    report["upload_probe"] = {"ok": None, "status": "skipped_health_only"}
                    report["ocr_generation"] = {"ok": None, "status": "skipped_health_only"}
                    recorder.add("generation_probes", "skipped", reason="health_only")
                    return report, failure_class

                try:
                    response = await generate_with_model(
                        client,
                        'Return exactly this JSON object and nothing else: {"diagnostic_ok": true}',
                        selected_model,
                        timeout=args.request_timeout,
                    )
                    text = response.text or ""
                    report["text_generation"] = {
                        "ok": bool(text.strip()),
                        "response_chars": len(text),
                        "response_hash": sha256_text(text) if text else None,
                        "contains_expected_key": "diagnostic_ok" in text,
                    }
                    if not report["text_generation"]["ok"]:
                        failure_class = "text_generation_empty"
                        recorder.add("text_generation", "failed", failure_class=failure_class)
                        return report, failure_class
                    recorder.add("text_generation", "passed", **report["text_generation"])
                except Exception as exc:  # noqa: BLE001
                    failure_class = classify_exception(exc)
                    report["text_generation"]["error"] = safe_error(exc, secret_values)
                    recorder.add(
                        "text_generation",
                        "failed",
                        failure_class=failure_class,
                        error=report["text_generation"]["error"],
                    )
                    return report, failure_class

                try:
                    probe_page, probe_kind = select_probe_page(ROOT / args.page_manifest)
                    image_path = ROOT / str(probe_page["render_path"])
                    report["probe_page"] = {
                        "page_id": str(probe_page.get("page_id") or ""),
                        "corpus": str(probe_page.get("corpus") or ""),
                        "sample_kind": probe_kind,
                        "render_path": relative_or_absolute(image_path),
                    }
                    recorder.add("probe_page_selection", "passed", probe_page=report["probe_page"])
                except Exception as exc:  # noqa: BLE001
                    failure_class = "missing_probe_image"
                    report["probe_page_error"] = safe_error(exc, secret_values)
                    recorder.add(
                        "probe_page_selection",
                        "failed",
                        failure_class=failure_class,
                        error=report["probe_page_error"],
                    )
                    return report, failure_class

                try:
                    upload_identifier = await upload_file(
                        image_path,
                        client=client.client,
                        push_id=client.push_id,
                        verbose=False,
                    )
                    report["upload_probe"] = {
                        "ok": bool(upload_identifier),
                        "identifier_present": bool(upload_identifier),
                    }
                    if not upload_identifier:
                        failure_class = "upload_failure"
                        recorder.add("upload_probe", "failed", failure_class=failure_class)
                        return report, failure_class
                    recorder.add("upload_probe", "passed", **report["upload_probe"])
                except Exception as exc:  # noqa: BLE001
                    classified = classify_exception(exc)
                    failure_class = "upload_failure" if classified not in {"timeout", "rate_or_usage_block"} else classified
                    report["upload_probe"]["error"] = safe_error(exc, secret_values)
                    recorder.add(
                        "upload_probe",
                        "failed",
                        failure_class=failure_class,
                        error=report["upload_probe"]["error"],
                    )
                    return report, failure_class

                try:
                    response = await generate_with_model(
                        client,
                        fir_gita_smoke.ocr_prompt(),
                        selected_model,
                        files=[image_path],
                        timeout=args.request_timeout,
                    )
                    raw_text = response.text or ""
                    parsed = fir_gita_smoke.parse_ocr_json_response(raw_text)
                    ok, validation_errors, normalized = pipeline.validate_teacher_payload(
                        parsed,
                        float(args.min_confidence),
                    )
                    report["ocr_generation"] = {
                        "ok": bool(ok),
                        "response_chars": len(raw_text),
                        "response_hash": sha256_text(raw_text) if raw_text else None,
                        "validation_error_count": len(validation_errors),
                        "validation_errors": [str(error)[:160] for error in validation_errors[:5]],
                        "confidence": normalized.get("confidence") if normalized else None,
                    }
                    if not ok:
                        failure_class = "ocr_validation_failed"
                        recorder.add("ocr_generation", "failed", failure_class=failure_class, **report["ocr_generation"])
                        return report, failure_class
                    recorder.add("ocr_generation", "passed", **report["ocr_generation"])
                except Exception as exc:  # noqa: BLE001
                    failure_class = classify_exception(exc)
                    if failure_class == "api_error":
                        failure_class = "ocr_generation_failed"
                    report["ocr_generation"]["error"] = safe_error(exc, secret_values)
                    recorder.add(
                        "ocr_generation",
                        "failed",
                        failure_class=failure_class,
                        error=report["ocr_generation"]["error"],
                    )
                    return report, failure_class
            finally:
                if client is not None:
                    await client.close()
    finally:
        if previous_cookie_path is None:
            os.environ.pop("GEMINI_COOKIE_PATH", None)
        else:
            os.environ["GEMINI_COOKIE_PATH"] = previous_cookie_path
    return report, failure_class


def scan_report_safety(output: Path, secret_values: list[str]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    exact_values = [value for value in secret_values if value and len(value) >= 8]
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for forbidden in FORBIDDEN_REPORT_STRINGS:
            if forbidden in text:
                findings.append({"file": relative_or_absolute(path), "kind": "forbidden_cookie_or_token_name"})
                break
        for value in exact_values:
            if value in text:
                findings.append({"file": relative_or_absolute(path), "kind": "exact_secret_value"})
                break
    return {
        "ok": not findings,
        "checked_files": sum(1 for path in output.rglob("*") if path.is_file()),
        "finding_count": len(findings),
        "findings": findings[:20],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser-executable", default=os.getenv("BRAVE_BROWSER"))
    parser.add_argument("--profile-dir", default="~/.cache/gemini-techpeek-cdp-brave")
    parser.add_argument("--cdp-port", type=int, default=9222)
    parser.add_argument("--authuser", default="0")
    parser.add_argument("--account-substring", default="techpeek.ai@gmail.com")
    parser.add_argument("--cdp-timeout", type=float, default=20)
    parser.add_argument("--cookie-timeout", type=float, default=15)
    parser.add_argument("--cookie-poll-seconds", type=float, default=2)
    parser.add_argument("--page-manifest", default="artifacts/page_only_v1/manifests/page_manifest.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="auto-flash")
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--init-timeout", type=float, default=60)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Check CDP/account/cookies/client init/model listing, but skip text/upload/OCR generation probes.",
    )
    args = parser.parse_args(argv)
    if args.cookie_timeout < 0:
        parser.error("--cookie-timeout must be non-negative")
    if args.cookie_poll_seconds <= 0:
        parser.error("--cookie-poll-seconds must be positive")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    output = Path(args.output) if args.output else ROOT / "artifacts" / f"gemini_app_diagnostics_{timestamp_slug()}"
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    recorder = Recorder(output)
    summary_path = output / "diagnostics_summary.json"
    failure_class: str | None = None
    cookie_env: dict[str, str] = {}

    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "ok": False,
        "failure_class": None,
        "output": relative_or_absolute(output),
        "profile_dir": str(Path(args.profile_dir).expanduser()),
        "cdp_port": args.cdp_port,
        "authuser": str(args.authuser),
        "mode": "health_only" if args.health_only else "full",
        "account_check": {
            "status": "not_run",
            "expected_substring": args.account_substring,
            "expected_substring_present": False,
        },
        "cookies": browser_cookie_status(cookie_env),
        "client": {},
        "artifacts": {
            "summary": relative_or_absolute(summary_path),
            "steps": relative_or_absolute(recorder.steps_path),
        },
    }

    try:
        gemini_url = f"https://gemini.google.com/app?authuser={args.authuser}"
        browser_smoke.launch_browser(args, gemini_url)
        browser_smoke.wait_for_cdp(args.cdp_port, args.cdp_timeout)
        recorder.add("cdp", "passed", port=args.cdp_port)
    except Exception as exc:  # noqa: BLE001
        failure_class = "cdp_unavailable"
        recorder.add("cdp", "failed", failure_class=failure_class, error=safe_error(exc))
        summary["failure_class"] = failure_class
        write_json(summary_path, summary)
        summary["secret_scan"] = scan_report_safety(output, [])
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    browser_smoke.open_gemini(args)
    account_status = browser_smoke.verify_account(args)
    expected_present = account_status.get("status") == "confirmed"
    summary["account_check"] = {
        "status": account_status.get("status", "unknown"),
        "authuser": str(args.authuser),
        "expected_substring": args.account_substring,
        "expected_substring_present": expected_present,
    }
    recorder.add("account_check", "passed" if expected_present else "failed", account_check=summary["account_check"])
    browser_smoke.open_gemini(args)

    cookie_env, cookie_status, cookie_poll_status = poll_browser_cookies(args)
    summary["cookies"] = {**cookie_status, "poll_status": cookie_poll_status}
    recorder.add(
        "browser_cookies",
        "passed" if cookie_status["required_auth_cookies_present"] else "failed",
        cookies=summary["cookies"],
    )

    if not cookie_status["primary_auth_cookie_present"]:
        failure_class = "signed_out_profile"
    elif not cookie_status["timestamp_auth_cookie_present"]:
        failure_class = "missing_session_rotation_cookie"
    elif not expected_present:
        failure_class = "authuser_mismatch"

    if failure_class:
        summary["failure_class"] = failure_class
        write_json(summary_path, summary)
        summary["secret_scan"] = scan_report_safety(output, list(cookie_env.values()))
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    client_report, client_failure = asyncio.run(run_client_checks(args, recorder, cookie_env))
    summary["client"] = client_report
    failure_class = client_failure
    if failure_class:
        summary["failure_class"] = failure_class

    summary["ok"] = failure_class is None
    write_json(summary_path, summary)
    summary["secret_scan"] = scan_report_safety(output, list(cookie_env.values()))
    if not summary["secret_scan"]["ok"] and summary["ok"]:
        summary["ok"] = False
        summary["failure_class"] = "diagnostic_report_secret_leak"

    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
