#!/usr/bin/env python3
"""Launch Brave through CDP, extract Gemini App cookies, and run the smoke test.

The extracted cookie values are passed only through the child process
environment. They are never printed and are not written to repo files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

COOKIE_NAMES = {
    "__Secure-1PSID": "GEMINI_WEBAPI_SECURE_1PSID",
    "__Secure-1PSIDTS": "GEMINI_WEBAPI_SECURE_1PSIDTS",
}


def find_browser(explicit: str | None) -> str:
    if explicit:
        return explicit
    for path in ("/opt/brave.com/brave/brave",):
        if Path(path).exists():
            return path
    for name in ("brave-browser-stable", "brave-browser", "chromium", "google-chrome"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("Could not find Brave/Chromium. Pass --browser-executable explicitly.")


def cdp_version_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/json/version"


def cdp_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(cdp_version_url(port), timeout=1.0) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_for_cdp(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cdp_ready(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"CDP did not become ready on port {port} within {timeout:.1f}s")


def launch_browser(args: argparse.Namespace, url: str) -> subprocess.Popen[str] | None:
    if cdp_ready(args.cdp_port):
        print(f"Using existing CDP browser on port {args.cdp_port}.", flush=True)
        return None

    browser = find_browser(args.browser_executable)
    profile_dir = Path(args.profile_dir).expanduser()
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        browser,
        f"--user-data-dir={profile_dir}",
        f"--remote-debugging-port={args.cdp_port}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--new-window",
        url,
    ]
    print(
        "Launching Brave CDP profile "
        f"{profile_dir} on port {args.cdp_port}. Complete login in that window if prompted.",
        flush=True,
    )
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )


def run_agent_browser(args: argparse.Namespace, *command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["agent-browser", "--cdp", str(args.cdp_port), *command]
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def safe_agent_output(proc: subprocess.CompletedProcess[str]) -> str:
    chunks = []
    if proc.stdout.strip():
        chunks.append(proc.stdout.strip())
    if proc.stderr.strip():
        chunks.append(proc.stderr.strip())
    return "\n".join(chunks)


def iter_cookie_dicts(value: Any):
    if isinstance(value, dict):
        if "name" in value and "value" in value:
            yield value
        for child in value.values():
            yield from iter_cookie_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_cookie_dicts(child)


def extract_cookie_env(cookie_json: str) -> tuple[dict[str, str], dict[str, Any]]:
    data = json.loads(cookie_json)
    found: dict[str, str] = {}
    metadata: dict[str, Any] = {"found_cookie_names": [], "domains": {}}

    for cookie in iter_cookie_dicts(data):
        name = str(cookie.get("name") or "")
        if name not in COOKIE_NAMES:
            continue
        domain = str(cookie.get("domain") or "")
        if "google.com" not in domain and "gemini.google.com" not in domain:
            continue
        value = str(cookie.get("value") or "")
        if not value:
            continue
        found[COOKIE_NAMES[name]] = value
        metadata["found_cookie_names"].append(name)
        metadata["domains"][name] = domain

    metadata["found_cookie_names"] = sorted(set(metadata["found_cookie_names"]))
    return found, metadata


def cookie_env_from_browser(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    deadline = time.time() + args.login_timeout
    last_error = ""
    while time.time() < deadline:
        proc = run_agent_browser(args, "--json", "cookies", "get", check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                cookie_env, metadata = extract_cookie_env(proc.stdout)
            except json.JSONDecodeError as exc:
                last_error = f"Could not parse cookie JSON: {exc}"
            else:
                if "GEMINI_WEBAPI_SECURE_1PSID" in cookie_env:
                    return cookie_env, metadata
                last_error = "Gemini __Secure-1PSID cookie not present yet."
        else:
            last_error = safe_agent_output(proc)
        print("Waiting for Gemini/Google login cookies from the CDP browser...", flush=True)
        time.sleep(args.cookie_poll_seconds)

    raise RuntimeError(
        "Timed out waiting for Gemini cookies. "
        "Open the launched Brave window, sign into the techpeek.ai account, "
        f"then re-run this command. Last status: {last_error[:500]}"
    )


def page_text(args: argparse.Namespace, max_chars: int = 6000) -> str:
    proc = run_agent_browser(args, "get", "text", "body", check=False)
    text = safe_agent_output(proc)
    return text[:max_chars]


def verify_account(args: argparse.Namespace) -> dict[str, Any]:
    if not args.account_substring:
        return {"status": "skipped", "reason": "--account-substring not set"}

    myaccount_url = f"https://myaccount.google.com/?authuser={args.authuser}"
    run_agent_browser(args, "open", myaccount_url, check=False)
    run_agent_browser(args, "wait", "--load", "networkidle", check=False)
    text = page_text(args)
    ok = args.account_substring.lower() in text.lower()
    return {
        "status": "confirmed" if ok else "not_confirmed",
        "authuser": args.authuser,
        "expected_substring": args.account_substring,
    }


def open_gemini(args: argparse.Namespace) -> None:
    gemini_url = f"https://gemini.google.com/app?authuser={args.authuser}"
    run_agent_browser(args, "open", gemini_url, check=False)
    run_agent_browser(args, "wait", "--load", "networkidle", check=False)


def run_smoke(args: argparse.Namespace, cookie_env: dict[str, str]) -> int:
    env = os.environ.copy()
    env.update(cookie_env)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "gemini_app_fir_gita_smoke.py"),
        "--page-manifest",
        args.page_manifest,
        "--output",
        args.output,
        "--fir-docs",
        str(args.fir_docs),
        "--gita-images",
        str(args.gita_images),
        "--min-sleep",
        str(args.min_sleep),
        "--max-sleep",
        str(args.max_sleep),
        "--model",
        args.model,
        "--init-timeout",
        str(args.init_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--notebook",
        args.no_notebook_path,
    ]
    if args.verbose:
        cmd.append("--verbose")
    if args.proxy:
        cmd.extend(["--proxy", args.proxy])

    print(
        "Running Gemini App smoke test with browser-derived cookies "
        f"for {args.fir_docs} FIR pages and {args.gita_images} Gita images.",
        flush=True,
    )
    return subprocess.run(cmd, env=env, cwd=ROOT).returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser-executable", default=os.getenv("BRAVE_BROWSER"))
    parser.add_argument("--profile-dir", default="~/.cache/gemini-techpeek-cdp-brave")
    parser.add_argument("--cdp-port", type=int, default=9222)
    parser.add_argument("--authuser", default="0")
    parser.add_argument("--account-substring", default="techpeek.ai@gmail.com")
    parser.add_argument("--login-timeout", type=float, default=300)
    parser.add_argument("--cookie-poll-seconds", type=float, default=5)
    parser.add_argument("--cdp-timeout", type=float, default=20)
    parser.add_argument("--page-manifest", default="artifacts/page_only_v1/manifests/page_manifest.jsonl")
    parser.add_argument("--output", default="artifacts/gemini_app_smoke_20_browser")
    parser.add_argument("--fir-docs", type=int, default=10)
    parser.add_argument("--gita-images", type=int, default=10)
    parser.add_argument("--min-sleep", type=float, default=0.8)
    parser.add_argument("--max-sleep", type=float, default=2.2)
    parser.add_argument("--model", default="auto-flash")
    parser.add_argument("--init-timeout", type=float, default=60)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-notebook-path",
        default="artifacts/nonexistent_no_notebook_fallback.ipynb",
        help="Nonexistent notebook path used to ensure browser cookies are the credential source.",
    )
    args = parser.parse_args(argv)
    if args.min_sleep < 0 or args.max_sleep < args.min_sleep:
        parser.error("--max-sleep must be greater than or equal to --min-sleep")
    if args.cookie_poll_seconds <= 0:
        parser.error("--cookie-poll-seconds must be positive")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    gemini_url = f"https://gemini.google.com/app?authuser={args.authuser}"
    launch_browser(args, gemini_url)
    wait_for_cdp(args.cdp_port, args.cdp_timeout)

    open_gemini(args)
    cookie_env, cookie_metadata = cookie_env_from_browser(args)
    print(
        "Extracted required Gemini cookie names from browser: "
        f"{', '.join(cookie_metadata['found_cookie_names'])}.",
        flush=True,
    )

    account_status = verify_account(args)
    if account_status["status"] == "confirmed":
        print(
            "Confirmed Google account selection via authuser="
            f"{args.authuser} and substring {args.account_substring!r}.",
            flush=True,
        )
    else:
        print(
            "Account check was not confirmed from visible My Account text; "
            "continuing with authuser URL and browser-derived cookies.",
            flush=True,
        )
    open_gemini(args)

    return run_smoke(args, cookie_env)


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - CLI wrapper should report cleanly.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
