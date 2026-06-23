#!/usr/bin/env python3
"""Page-only Indic OCR and OCR+translation data pipeline.

The v1 contract is intentionally page-first: every training example uses a
full-page image input and no crop-level rows are emitted.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "page_only_v1"
RAW_ROOT = ROOT / "data" / "raw" / "vllm_fine_tune_pdfs"
GOLD_SCHEMA_VERSION = "indic_ocr_gold_v1"
FT_SCHEMA_VERSION = "indic_ocr_ft_v1"
SPLIT_SEED = "2026-06-06:v1"
GEMINI_REQUIRED_FIELDS = {
    "plain_text",
    "markdown",
    "blocks",
    "key_values",
    "tables",
    "quality_flags",
    "confidence",
}
FIR_GITA_CORPORA = {"fir", "gita"}

def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def gemini_transport_from_args(args: argparse.Namespace | None = None) -> str:
    transport = str(getattr(args, "gemini_transport", None) or os.getenv("GEMINI_TRANSPORT", "app")).strip().lower()
    if transport not in {"app", "official"}:
        raise RuntimeError(f"Unsupported Gemini transport {transport!r}; expected app or official.")
    return transport


def gemini_app_model_from_args(args: argparse.Namespace | None = None) -> str:
    return str(getattr(args, "gemini_app_model", None) or os.getenv("GEMINI_APP_MODEL", "auto-flash")).strip() or "auto-flash"


def active_teacher_model(args: argparse.Namespace | None = None) -> str:
    if gemini_transport_from_args(args) == "app":
        if getattr(args, "gemini_app_model", None):
            return f"gemini_webapi:{gemini_app_model_from_args(args)}"
        return os.getenv("GEMINI_APP_TEACHER_MODEL_KEY") or f"gemini_webapi:{gemini_app_model_from_args(args)}"
    return os.getenv("GEMINI_TEACHER_MODEL", "gemma-4-31b-it")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def ensure_tools() -> None:
    missing = [tool for tool in ["file", "pdfinfo", "pdftoppm", "pdftotext"] if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing required system tools: {', '.join(missing)}")


def mkdirs(output: Path) -> dict[str, Path]:
    paths = {
        "output": output,
        "manifests": output / "manifests",
        "renders": output / "renders",
        "text": output / "text",
        "gold": output / "gold",
        "exports": output / "exports",
        "splits": output / "splits",
        "calibration": output / "calibration",
        "evaluation": output / "evaluation",
        "reports": output / "reports",
        "quarantine": output / "quarantine",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(value: str, n: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def rel_to_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def jsonl_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def file_mime(path: Path) -> str:
    cp = run_cmd(["file", "-L", "--mime-type", "-b", str(path)])
    return cp.stdout.strip() if cp.returncode == 0 else "application/octet-stream"


def pdf_info(path: Path) -> dict[str, Any]:
    cp = run_cmd(["pdfinfo", str(path)])
    info: dict[str, Any] = {"pdfinfo_ok": cp.returncode == 0, "pdfinfo_error": cp.stderr.strip()[:1000]}
    if cp.returncode != 0:
        return info
    for line in cp.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip().lower().replace(" ", "_")] = value.strip()
    pages = info.get("pages")
    info["page_count"] = int(pages) if isinstance(pages, str) and pages.isdigit() else None
    return info


def text_script_counts(text: str) -> dict[str, int]:
    return {
        "chars": len(text.strip()),
        "devanagari": sum(1 for ch in text if "\u0900" <= ch <= "\u097F"),
        "latin": sum(1 for ch in text if "a" <= ch.lower() <= "z"),
        "digits": sum(1 for ch in text if ch.isdigit()),
        "replacement": text.count("\ufffd"),
    }


def first_page_text_stats(path: Path) -> dict[str, Any]:
    cp = run_cmd(["pdftotext", "-layout", "-f", "1", "-l", "1", str(path), "-"], timeout=60)
    text = cp.stdout or ""
    counts = text_script_counts(text)
    return {
        "text_extract_ok": cp.returncode == 0,
        "first_page_text_chars": counts["chars"],
        "first_page_devanagari_chars": counts["devanagari"],
        "first_page_latin_chars": counts["latin"],
        "text_extract_error": cp.stderr.strip()[:500],
    }


def discover_external_extract_roots() -> list[Path]:
    roots: list[Path] = []
    for candidate in sorted(ROOT.parent.glob("Vllm_fine_tune_pdfs*")):
        if candidate.suffix.lower() == ".zip" or not candidate.is_dir():
            continue
        root = candidate / "Vllm_fine_tune_pdfs" if (candidate / "Vllm_fine_tune_pdfs").is_dir() else candidate
        if all((root / sub).is_dir() for sub in ["fir", "sc_english", "sc_hindi"]):
            roots.append(root)
    return roots


def organize_external_sources(paths: dict[str, Path]) -> dict[str, Any]:
    roots = discover_external_extract_roots()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    counts = {"fir": 0, "sc_english": 0, "sc_hindi": 0}
    conflicts: list[dict[str, Any]] = []
    for subdir in counts:
        (RAW_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    for source_root in roots:
        for subdir in counts:
            for src in sorted((source_root / subdir).glob("*.pdf")):
                dest = RAW_ROOT / subdir / src.name
                src_resolved = src.resolve()
                if dest.exists() or dest.is_symlink():
                    try:
                        if dest.resolve() == src_resolved:
                            status = "already_linked"
                        elif dest.exists() and sha256_file(dest) == sha256_file(src_resolved):
                            status = "duplicate_same_sha_kept_existing"
                        else:
                            status = "name_conflict"
                            conflicts.append({
                                "target": rel_to_root(dest),
                                "existing": str(dest.resolve()),
                                "incoming": str(src_resolved),
                            })
                    except FileNotFoundError:
                        dest.unlink()
                        dest.symlink_to(src_resolved)
                        status = "relinked_broken_symlink"
                else:
                    dest.symlink_to(src_resolved)
                    status = "linked"
                if status != "name_conflict":
                    counts[subdir] += 1
                manifest.append({
                    "source_root": str(source_root),
                    "source_path": str(src),
                    "organized_path": rel_to_root(dest),
                    "corpus_subdir": subdir,
                    "status": status,
                })

    result = {
        "created_at": utc_now(),
        "raw_root": rel_to_root(RAW_ROOT),
        "external_roots": [str(root) for root in roots],
        "counts": counts,
        "raw_pdf_counts": {
            subdir: len(list((RAW_ROOT / subdir).glob("*.pdf")))
            for subdir in counts
        },
        "conflicts": conflicts,
        "manifest_rows": len(manifest),
        "symlink_policy": "link external PDFs into data/raw/vllm_fine_tune_pdfs without copying bytes",
    }
    jsonl_write(paths["manifests"] / "organization_manifest.jsonl", manifest)
    write_json(paths["manifests"] / "organization_summary.json", result)
    return result


def parse_sc_filename(path: Path, language_variant: str | None = None) -> dict[str, Any]:
    stem = path.stem
    copy_match = re.search(r"\((\d+)\)$", stem)
    copy_index = int(copy_match.group(1)) if copy_match else 0
    canonical_stem = re.sub(r"\(\d+\)$", "", stem)
    m = re.match(r"^(S_)?(\d{4})_(\d+)_(\d+)_(\d+)_(EN|HIN)$", canonical_stem)
    if not m:
        return {"canonical_stem": canonical_stem, "copy_index": copy_index, "language_variant": language_variant}
    prefix, year, volume, start_page, end_page, language = m.groups()
    citation_key = f"{prefix or ''}{year}_{volume}_{start_page}_{end_page}"
    return {
        "canonical_stem": canonical_stem,
        "copy_index": copy_index,
        "sc_prefix": bool(prefix),
        "year": int(year),
        "volume": int(volume),
        "report_start_page": int(start_page),
        "report_end_page": int(end_page),
        "language_variant": language_variant or language,
        "citation_key": citation_key,
    }


def parse_fir_filename(path: Path) -> dict[str, Any]:
    stem = path.stem
    parts = stem.split("_")
    subtype = "general"
    low = stem.lower()
    if "cyber" in low:
        subtype = "cyber"
    elif "traffic" in low:
        subtype = "traffic"
    elif "sc_st" in low:
        subtype = "sc_st"
    elif "mahila" in low:
        subtype = "mahila"
    elif low.startswith("rail"):
        subtype = "rail"
    m = re.search(r"(.+?)[_-](\d{1,4})[-_](\d{2,4})$", stem)
    fir_no = None
    year = None
    if m:
        fir_no = m.group(2)
        raw_year = m.group(3)
        year = int(raw_year) if len(raw_year) == 4 else 2000 + int(raw_year)
    district = "_".join(parts[:2]) if parts and parts[0] in {"West", "East", "Rail"} and len(parts) > 1 else parts[0]
    return {
        "district_or_unit_raw": district,
        "fir_number_raw": fir_no,
        "year_normalized": year,
        "filename_subtype_hint": subtype,
        "filename_anomalies": [
            flag
            for flag, present in {
                "no_ps_token": "_PS_" not in stem and "_Ps_" not in stem,
                "hyphen_separator": "-" in stem,
                "double_underscore": "__" in stem,
            }.items()
            if present
        ],
    }


def source_specs() -> list[dict[str, Any]]:
    specs = [
        {"root": RAW_ROOT / "fir", "corpus": "fir", "source_set": "vllm_external_fir"},
        {"root": RAW_ROOT / "sc_english", "corpus": "sc", "source_set": "vllm_external_sc_english", "language_variant": "EN"},
        {"root": RAW_ROOT / "sc_hindi", "corpus": "sc", "source_set": "vllm_external_sc_hindi", "language_variant": "HIN"},
        {"root": ROOT / "SC100", "corpus": "sc", "source_set": "legacy_SC100"},
        {"root": ROOT / "fir_sample_100" / "fir_sample_100", "corpus": "fir", "source_set": "legacy_fir_sample_100"},
        {"root": ROOT / "gita20", "corpus": "gita", "source_set": "legacy_gita20"},
    ]
    return [spec for spec in specs if spec["root"].exists()]


def quick_pdf_page_count(path: Path) -> int | None:
    cp = run_cmd(["pdfinfo", str(path)], timeout=30)
    if cp.returncode != 0:
        return None
    m = re.search(r"^Pages:\s+(\d+)", cp.stdout, flags=re.M)
    return int(m.group(1)) if m else None


def smoke_allowed_paths(fir_docs: int, gita_pages: int, sc_pairs: int) -> set[Path]:
    allowed: set[Path] = set()
    allowed.update(sorted((RAW_ROOT / "fir").glob("*.pdf"))[:fir_docs])
    image_exts = {".jpg", ".jpeg", ".png"}
    allowed.update([p for p in sorted((ROOT / "gita20").iterdir()) if p.suffix.lower() in image_exts][:gita_pages] if (ROOT / "gita20").exists() else [])

    english = {p.stem.removesuffix("_EN"): p for p in sorted((RAW_ROOT / "sc_english").glob("*.pdf"))}
    hindi = {p.stem.removesuffix("_HIN"): p for p in sorted((RAW_ROOT / "sc_hindi").glob("*.pdf"))}
    same_page_keys: list[str] = []
    mismatch_keys: list[str] = []
    fallback_keys: list[str] = []
    for key in sorted(set(english) & set(hindi)):
        fallback_keys.append(key)
        english_pages = quick_pdf_page_count(english[key])
        hindi_pages = quick_pdf_page_count(hindi[key])
        if english_pages is None or hindi_pages is None:
            continue
        if english_pages == hindi_pages and len(same_page_keys) < max(1, sc_pairs - 1):
            same_page_keys.append(key)
        elif english_pages != hindi_pages and len(mismatch_keys) < 1:
            mismatch_keys.append(key)
        if len(same_page_keys) + len(mismatch_keys) >= sc_pairs:
            break
    selected_keys = (same_page_keys + mismatch_keys)[:sc_pairs]
    if len(selected_keys) < sc_pairs:
        selected_keys.extend([key for key in fallback_keys if key not in selected_keys][: sc_pairs - len(selected_keys)])
    for key in selected_keys:
        allowed.add(english[key])
        allowed.add(hindi[key])
    return {path.resolve() for path in allowed}


def fir_gita_smoke_allowed_paths(fir_docs: int, gita_pages: int) -> set[Path]:
    allowed: set[Path] = set()
    allowed.update(sorted((RAW_ROOT / "fir").glob("*.pdf"))[:fir_docs])
    image_exts = {".jpg", ".jpeg", ".png"}
    if (ROOT / "gita20").exists():
        allowed.update([p for p in sorted((ROOT / "gita20").iterdir()) if p.suffix.lower() in image_exts][:gita_pages])
    return {path.resolve() for path in allowed}


def discover_files(smoke_paths: set[Path] | None = None) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    by_sha: dict[str, list[str]] = {}
    for spec in source_specs():
        files = sorted(p for p in spec["root"].iterdir() if p.is_file() or p.is_symlink())
        if smoke_paths is not None:
            files = [p for p in files if p.resolve() in smoke_paths]
        for path in files:
            rel = rel_to_root(path)
            mime = file_mime(path)
            ext = path.suffix.lower()
            try:
                sha = sha256_file(path)
            except FileNotFoundError:
                sha = "missing"
            by_sha.setdefault(sha, []).append(rel)
            corpus = spec["corpus"]
            row: dict[str, Any] = {
                "source_path": rel,
                "source_real_path": str(path.resolve()) if path.exists() else None,
                "source_set": spec["source_set"],
                "corpus": corpus,
                "basename": path.name,
                "extension": ext,
                "mime_type": mime,
                "sha256": sha,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "valid_source": False,
                "quarantine_reason": None,
            }
            if corpus == "sc":
                language_variant = spec.get("language_variant")
                if language_variant is None:
                    if path.stem.endswith("_HIN"):
                        language_variant = "HIN"
                    elif path.stem.endswith("_EN"):
                        language_variant = "EN"
                row.update(parse_sc_filename(path, language_variant))
                row["source_category"] = "supreme_court_report"
                row["doc_type"] = "supreme_court_report"
                row["language_hint"] = "hi-Deva" if row.get("language_variant") == "HIN" else "en-Latn"
                row["sc_pair_key"] = row.get("citation_key")
            elif corpus == "fir":
                row.update(parse_fir_filename(path))
                row["source_category"] = "fir"
                row["doc_type"] = "fir"
                row["language_hint"] = "hi-Deva+en-Latn"
            else:
                row["source_category"] = "book_page"
                row["doc_type"] = "gita_page"
                row["language_hint"] = "hi-Deva+sa-Deva"

            if mime == "application/pdf":
                info = pdf_info(path)
                row.update(info)
                row.update(first_page_text_stats(path))
                row["valid_source"] = bool(info.get("pdfinfo_ok") and info.get("page_count"))
            elif mime in {"image/jpeg", "image/png"}:
                try:
                    with Image.open(path) as img:
                        row["image_width_px"], row["image_height_px"] = img.size
                    row["page_count"] = 1
                    row["valid_source"] = True
                except Exception as exc:
                    row["quarantine_reason"] = f"image_open_failed:{type(exc).__name__}"
            elif mime == "text/html":
                row["quarantine_reason"] = "html_masquerading_as_pdf"
            else:
                row["quarantine_reason"] = f"unsupported_mime:{mime}"

            if not row["valid_source"] and not row.get("quarantine_reason"):
                row["quarantine_reason"] = "invalid_or_unreadable_source"
            rows.append(row)

    duplicate_groups = {sha: sorted(paths) for sha, paths in by_sha.items() if sha != "missing" and len(paths) > 1}
    for row in rows:
        duplicate_id = "sha256:" + row["sha256"] if row["sha256"] in duplicate_groups else None
        row["duplicate_group_id"] = duplicate_id
        if row["corpus"] == "sc":
            row["split_group_id"] = f"sc_pair:{row.get('sc_pair_key') or row.get('canonical_stem') or row['basename']}"
        elif row["corpus"] == "fir":
            row["split_group_id"] = duplicate_id or f"fir:{row['sha256']}"
        else:
            row["split_group_id"] = duplicate_id or f"gita:{row['sha256']}"
    return rows, duplicate_groups


def filter_rows_by_corpus(
    rows: list[dict[str, Any]],
    duplicate_groups: dict[str, list[str]],
    corpora: set[str],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    selected = [row for row in rows if row.get("corpus") in corpora]
    selected_paths = {row["source_path"] for row in selected}
    filtered_duplicate_groups: dict[str, list[str]] = {}
    for sha, members in duplicate_groups.items():
        filtered_members = sorted(member for member in members if member in selected_paths)
        if len(filtered_members) > 1:
            filtered_duplicate_groups[sha] = filtered_members
    return selected, filtered_duplicate_groups


def apply_smoke_filter(rows: list[dict[str, Any]], *, fir_docs: int, gita_pages: int, sc_pairs: int) -> list[dict[str, Any]]:
    selected: set[str] = set()
    fir = [r for r in rows if r["corpus"] == "fir" and r["valid_source"]]
    gita = [r for r in rows if r["corpus"] == "gita" and r["valid_source"]]
    for row in sorted(fir, key=lambda r: r["source_path"])[:fir_docs]:
        selected.add(row["source_path"])
    for row in sorted(gita, key=lambda r: r["source_path"])[:gita_pages]:
        selected.add(row["source_path"])

    sc_by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["corpus"] != "sc" or not row["valid_source"] or not row.get("sc_pair_key"):
            continue
        sc_by_key.setdefault(row["sc_pair_key"], {})[row.get("language_variant") or ""] = row
    paired_keys = [key for key, values in sc_by_key.items() if values.get("HIN") and values.get("EN")]
    for key in sorted(paired_keys)[:sc_pairs]:
        selected.add(sc_by_key[key]["HIN"]["source_path"])
        selected.add(sc_by_key[key]["EN"]["source_path"])

    return [row for row in rows if row["source_path"] in selected or row.get("quarantine_reason")]


def write_ingestion_artifacts(paths: dict[str, Path], rows: list[dict[str, Any]], duplicate_groups: dict[str, list[str]]) -> None:
    quarantine = [row for row in rows if row.get("quarantine_reason")]
    jsonl_write(paths["manifests"] / "ingestion_manifest.jsonl", rows)
    jsonl_write(paths["manifests"] / "quarantine.jsonl", quarantine)
    sc_duplicate_groups = {
        sha: members
        for sha, members in duplicate_groups.items()
        if any("/sc_" in member or member.startswith("SC100/") for member in members)
    }
    write_json(paths["manifests"] / "sc_duplicate_groups.json", {
        "duplicate_groups": sc_duplicate_groups,
        "sc_duplicate_group_count": len(sc_duplicate_groups),
        "sc_duplicate_file_count": sum(len(members) for members in sc_duplicate_groups.values()),
    })


def rendered_page_number(name: str) -> int:
    m = re.search(r"-(\d+)\.jpg$", name)
    return int(m.group(1)) if m else 0


def rendered_pdf_paths(row_dir: Path, page_count: int) -> list[Path]:
    files = sorted(row_dir.glob("page-*.jpg"), key=lambda p: rendered_page_number(p.name))
    if len(files) == page_count:
        return files
    width = max(1, len(str(page_count)))
    padded = [row_dir / f"page-{i:0{width}d}.jpg" for i in range(1, page_count + 1)]
    if all(p.exists() for p in padded):
        return padded
    plain = [row_dir / f"page-{i}.jpg" for i in range(1, page_count + 1)]
    if all(p.exists() for p in plain):
        return plain
    return files


def classify_page(row: dict[str, Any], image_path: Path) -> tuple[list[str], str]:
    flags: list[str] = []
    page_class = "image"
    if row["mime_type"] == "application/pdf":
        chars = int(row.get("first_page_text_chars") or 0)
        page_class = "native_text_or_mixed" if chars > 500 else "scan_or_image_pdf"
        if chars > 500:
            flags.append("embedded_text_available")
        if row["corpus"] == "fir":
            flags.append("handwriting_likely")
        if row.get("language_variant") == "HIN":
            flags.append("devanagari")
        elif row.get("language_variant") == "EN":
            flags.append("latin")
    try:
        with Image.open(image_path) as img:
            if img.width * img.height > 80_000_000:
                flags.append("oversized_render")
                if img.width > img.height:
                    flags.append("landscape_or_rotated")
                return sorted(set(flags)), page_class
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            if stat.mean[0] > 248:
                flags.append("low_ink_or_blank_risk")
            if stat.stddev[0] < 25:
                flags.append("low_contrast")
            if img.width > img.height:
                flags.append("landscape_or_rotated")
    except Exception:
        flags.append("image_stats_failed")
    if row["corpus"] == "sc" and row.get("report_end_page") and row.get("report_start_page"):
        if int(row["report_end_page"]) - int(row["report_start_page"]) > 30:
            flags.append("long_judgment")
    return sorted(set(flags)), page_class


def document_id(row: dict[str, Any]) -> str:
    return f"{row['corpus']}:{short_hash(row['source_path'] + ':' + row['sha256'])}"


def render_sources(paths: dict[str, Path], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    page_rows: list[dict[str, Any]] = []
    valid_rows = [row for row in rows if row["valid_source"]]
    for idx, row in enumerate(valid_rows, 1):
        src = ROOT / row["source_path"]
        doc_id = document_id(row)
        row["document_id"] = doc_id
        row_dir = paths["renders"] / row["corpus"] / safe_id(doc_id)
        row_dir.mkdir(parents=True, exist_ok=True)
        if row["mime_type"] == "application/pdf":
            page_count = int(row["page_count"])
            dpi = 300 if row["corpus"] == "fir" else 200
            prefix = row_dir / "page"
            expected = rendered_pdf_paths(row_dir, page_count)
            if len(expected) != page_count or not all(p.exists() for p in expected):
                cp = run_cmd(
                    ["pdftoppm", "-jpeg", "-r", str(dpi), "-f", "1", "-l", str(page_count), str(src), str(prefix)],
                    timeout=max(180, page_count * 30),
                )
                if cp.returncode != 0:
                    raise RuntimeError(f"pdftoppm failed for {row['source_path']}: {cp.stderr[:1000]}")
                expected = rendered_pdf_paths(row_dir, page_count)
            if len(expected) != page_count:
                raise RuntimeError(f"Expected {page_count} rendered pages for {row['source_path']}, found {len(expected)}")
            for page_index, image_path in enumerate(expected, 1):
                with Image.open(image_path) as img:
                    width, height = img.size
                flags, page_class = classify_page(row, image_path)
                page_rows.append(page_row(row, doc_id, page_index, page_count, image_path, dpi, width, height, flags, page_class))
        elif row["mime_type"] in {"image/jpeg", "image/png"}:
            dest = row_dir / f"page-1{src.suffix.lower()}"
            if not dest.exists():
                shutil.copy2(src, dest)
            with Image.open(dest) as img:
                width, height = img.size
            flags, page_class = classify_page(row, dest)
            page_rows.append(page_row(row, doc_id, 1, 1, dest, None, width, height, flags, page_class))
        if idx % 25 == 0:
            print(f"rendered/linked {idx}/{len(valid_rows)} valid sources", flush=True)
    jsonl_write(paths["manifests"] / "page_manifest.jsonl", page_rows)
    return page_rows


def page_row(
    row: dict[str, Any],
    doc_id: str,
    page_index: int,
    page_count: int,
    image_path: Path,
    dpi: int | None,
    width: int,
    height: int,
    flags: list[str],
    page_class: str,
) -> dict[str, Any]:
    data = {
        "page_id": f"{doc_id}:p{page_index:04d}",
        "document_id": doc_id,
        "source_path": row["source_path"],
        "source_set": row["source_set"],
        "corpus": row["corpus"],
        "doc_type": row["doc_type"],
        "split_group_id": row["split_group_id"],
        "duplicate_group_id": row.get("duplicate_group_id"),
        "page_index": page_index,
        "page_count": page_count,
        "render_path": rel_to_root(image_path),
        "render_dpi": dpi,
        "image_width_px": width,
        "image_height_px": height,
        "image_sha256": sha256_file(image_path),
        "bbox_coord_space": "normalized_1000",
        "language_hint": row.get("language_hint"),
        "page_class": page_class,
        "quality_flags": flags,
        "visual_token_budget": int(os.getenv("GEMMA4_VISUAL_TOKENS", "1120")),
        "is_crop": False,
    }
    for key in ["language_variant", "citation_key", "sc_pair_key"]:
        if key in row:
            data[key] = row[key]
    return data


def assign_splits(paths: dict[str, Path], rows: list[dict[str, Any]], pages: list[dict[str, Any]]) -> dict[str, str]:
    groups = sorted({row["split_group_id"] for row in rows if row["valid_source"]})
    assignments: dict[str, str] = {}
    for group in groups:
        bucket = int(hashlib.sha256((SPLIT_SEED + ":" + group).encode()).hexdigest()[:8], 16) % 100
        assignments[group] = "train" if bucket < 80 else "val" if bucket < 90 else "test"
    jsonl_write(paths["splits"] / "split_assignments.jsonl", [
        {"split_group_id": group, "split": split} for group, split in sorted(assignments.items())
    ])
    write_json(paths["splits"] / "splits.json", {
        "seed": SPLIT_SEED,
        "counts_by_group": {name: sum(1 for s in assignments.values() if s == name) for name in ["train", "val", "test"]},
        "counts_by_page": {name: sum(1 for p in pages if assignments.get(p["split_group_id"]) == name) for name in ["train", "val", "test"]},
        "split_policy": "document_or_sc_pair_or_duplicate_group_level",
    })
    return assignments


def scripts_from_language_hint(language: str) -> list[str]:
    scripts = []
    if "Deva" in language:
        scripts.append("Deva")
    if "Latn" in language or "en" in language:
        scripts.append("Latn")
    if "Taml" in language:
        scripts.append("Taml")
    return scripts or ["Unknown"]


def extract_pdf_text_pages(path: Path, page_count: int) -> list[str]:
    cp = run_cmd(["pdftotext", "-layout", str(path), "-"], timeout=max(120, page_count * 10))
    if cp.returncode == 0:
        parts = cp.stdout.split("\f")
        if len(parts) >= page_count:
            return [parts[i].strip() for i in range(page_count)]
    texts: list[str] = []
    for page_index in range(1, page_count + 1):
        one = run_cmd(["pdftotext", "-layout", "-f", str(page_index), "-l", str(page_index), str(path), "-"], timeout=60)
        texts.append(one.stdout.strip() if one.returncode == 0 else "")
    return texts


def text_quality(text: str, expected_language: str) -> dict[str, Any]:
    counts = text_script_counts(text)
    lower = text.lower()
    legal_markers = [
        "supreme court",
        "judgment",
        "appeal",
        "petitioner",
        "respondent",
        "न्यायालय",
        "अपील",
        "याचिक",
        "प्रतिवादी",
    ]
    marker_count = sum(1 for marker in legal_markers if marker in lower)
    replacement_ratio = counts["replacement"] / max(1, counts["chars"])
    if expected_language == "EN":
        script_ok = counts["latin"] >= max(30, counts["devanagari"] * 2)
    elif expected_language == "HIN":
        script_ok = counts["devanagari"] >= max(20, counts["latin"] // 2)
    else:
        script_ok = counts["chars"] > 0
    ok = counts["chars"] >= 20 and replacement_ratio < 0.02 and script_ok
    return {
        **counts,
        "expected_language": expected_language,
        "legal_marker_count": marker_count,
        "replacement_ratio": replacement_ratio,
        "script_ok": script_ok,
        "non_empty": counts["chars"] > 0,
        "quality_ok": ok,
    }


def extract_sc_text(paths: dict[str, Path], rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    text_index: dict[tuple[str, int], dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    valid_sc = [r for r in rows if r["valid_source"] and r["corpus"] == "sc" and r["mime_type"] == "application/pdf"]
    for idx, row in enumerate(valid_sc, 1):
        doc_id = row.get("document_id") or document_id(row)
        row["document_id"] = doc_id
        page_count = int(row["page_count"])
        source = ROOT / row["source_path"]
        texts = extract_pdf_text_pages(source, page_count)
        text_dir = paths["text"] / "sc" / safe_id(doc_id)
        text_dir.mkdir(parents=True, exist_ok=True)
        expected_language = row.get("language_variant") or "UNK"
        for page_index, text in enumerate(texts[:page_count], 1):
            text_path = text_dir / f"page-{page_index:04d}.txt"
            text_path.write_text(text + "\n", encoding="utf-8")
            quality = text_quality(text, expected_language)
            item = {
                "document_id": doc_id,
                "source_path": row["source_path"],
                "page_index": page_index,
                "page_count": page_count,
                "language_variant": expected_language,
                "citation_key": row.get("citation_key"),
                "text_path": rel_to_root(text_path),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "extraction_tool": "pdftotext -layout",
                "quality": quality,
            }
            manifest.append(item)
            text_index[(doc_id, page_index)] = item
        if idx % 50 == 0:
            print(f"extracted SC text {idx}/{len(valid_sc)} documents", flush=True)
    jsonl_write(paths["manifests"] / "sc_text_extraction_manifest.jsonl", manifest)
    return text_index


def sc_text_index_from_manifest(paths: dict[str, Path]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (row["document_id"], int(row["page_index"])): row
        for row in jsonl_read(paths["manifests"] / "sc_text_extraction_manifest.jsonl")
    }


def sc_text_rows_for_document(text_index: dict[tuple[str, int], dict[str, Any]], document_id: str) -> list[dict[str, Any]]:
    return [
        row
        for (doc_id, _), row in sorted(text_index.items(), key=lambda item: item[0][1])
        if doc_id == document_id
    ]


def sc_text_rows_by_document(text_index: dict[tuple[str, int], dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_doc: dict[str, list[dict[str, Any]]] = {}
    for (doc_id, _), row in text_index.items():
        rows_by_doc.setdefault(doc_id, []).append(row)
    for rows in rows_by_doc.values():
        rows.sort(key=lambda row: int(row["page_index"]))
    return rows_by_doc


def sc_page_weight(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    quality = row.get("quality", {})
    chars = float(quality.get("chars") or 0)
    if chars < 20 or not quality.get("non_empty"):
        return 0.0
    return max(0.0, chars)


def sc_page_intervals(rows: list[dict[str, Any]]) -> dict[int, tuple[float, float, float]]:
    weights = [(int(row["page_index"]), sc_page_weight(row)) for row in rows]
    total = sum(weight for _, weight in weights)
    if total <= 0:
        count = max(1, len(weights))
        return {
            page_index: ((idx - 1) / count, idx / count, 1.0 / count)
            for idx, (page_index, _) in enumerate(weights, 1)
        }
    cursor = 0.0
    intervals: dict[int, tuple[float, float, float]] = {}
    for page_index, weight in weights:
        start = cursor / total
        cursor += weight
        end = cursor / total
        intervals[page_index] = (start, end, weight)
    return intervals


def interval_overlaps(
    source_interval: tuple[float, float, float],
    target_intervals: dict[int, tuple[float, float, float]],
) -> list[dict[str, Any]]:
    source_start, source_end, source_weight = source_interval
    source_span = max(0.000001, source_end - source_start)
    rows: list[dict[str, Any]] = []
    for page_index, target_interval in sorted(target_intervals.items()):
        target_start, target_end, target_weight = target_interval
        overlap = max(0.0, min(source_end, target_end) - max(source_start, target_start))
        if overlap <= 0:
            continue
        target_span = max(0.000001, target_end - target_start)
        rows.append({
            "page_index": page_index,
            "source_overlap_ratio": round(overlap / source_span, 6),
            "target_overlap_ratio": round(overlap / target_span, 6),
            "normalized_overlap": round(overlap, 6),
            "target_weight": round(target_weight, 3),
            "source_weight": round(source_weight, 3),
        })
    rows.sort(key=lambda row: (-row["source_overlap_ratio"], row["page_index"]))
    return rows


def significant_overlap_pages(overlaps: list[dict[str, Any]], min_ratio: float = 0.15) -> list[int]:
    pages = [row["page_index"] for row in overlaps if row["source_overlap_ratio"] >= min_ratio]
    if pages:
        return sorted(pages)
    return [overlaps[0]["page_index"]] if overlaps else []


def sc_verification_relation(
    source_page_index: int,
    source_page_count: int,
    target_page_count: int,
    target_pages: list[int],
    source_quality: dict[str, Any],
    target_qualities: list[dict[str, Any]],
) -> tuple[str, list[str], float, bool]:
    reasons: list[str] = []
    source_chars = int(source_quality.get("chars") or 0)
    target_quality_ok = bool(target_qualities) and all(q.get("quality_ok") for q in target_qualities)
    source_quality_ok = bool(source_quality.get("quality_ok"))
    same_page = source_page_count == target_page_count and target_pages == [source_page_index]

    if source_chars < 20:
        reasons.append("source_page_text_too_short")
    if not source_quality_ok:
        reasons.append("source_text_quality_failed")
    if not target_pages:
        reasons.append("no_target_page_overlap")
    if target_pages and not target_quality_ok:
        reasons.append("target_text_quality_failed")
    if source_page_count != target_page_count:
        reasons.append("page_count_mismatch")

    if not target_pages:
        return "unmatched_or_empty", reasons, 0.0, False
    if len(target_pages) > 1:
        return "source_page_spans_multiple_target_pages", reasons, 0.55, False
    if same_page and source_quality_ok and target_quality_ok:
        return "same_page_likely", reasons, 0.93, True
    if same_page:
        return "same_page_low_text_quality", reasons, 0.68, False
    if source_page_count == target_page_count:
        reasons.append("same_page_index_not_best_flow_match")
        return "same_count_shift_or_bleed", reasons, 0.45, False
    return "page_count_mismatch_single_target_page", reasons, 0.62, False


def build_directional_sc_content_verification(
    pair: dict[str, Any],
    rows_by_doc: dict[str, list[dict[str, Any]]],
    source_variant: str,
    target_variant: str,
) -> list[dict[str, Any]]:
    source_doc_key = "hindi_document_id" if source_variant == "HIN" else "english_document_id"
    target_doc_key = "english_document_id" if target_variant == "EN" else "hindi_document_id"
    source_count_key = "hindi_page_count" if source_variant == "HIN" else "english_page_count"
    target_count_key = "english_page_count" if target_variant == "EN" else "hindi_page_count"
    source_doc = pair.get(source_doc_key)
    target_doc = pair.get(target_doc_key)
    source_page_count = int(pair.get(source_count_key) or 0)
    target_page_count = int(pair.get(target_count_key) or 0)
    if not source_doc or not target_doc or source_page_count <= 0 or target_page_count <= 0:
        return []

    source_rows = rows_by_doc.get(source_doc, [])
    target_rows = rows_by_doc.get(target_doc, [])
    source_intervals = sc_page_intervals(source_rows)
    target_intervals = sc_page_intervals(target_rows)
    target_by_page = {int(row["page_index"]): row for row in target_rows}
    source_by_page = {int(row["page_index"]): row for row in source_rows}

    rows: list[dict[str, Any]] = []
    for page_index in range(1, source_page_count + 1):
        source_text = source_by_page.get(page_index)
        source_quality = source_text.get("quality", {}) if source_text else {}
        overlaps = interval_overlaps(source_intervals.get(page_index, (0.0, 0.0, 0.0)), target_intervals)
        target_pages = significant_overlap_pages(overlaps)
        target_qualities = [
            target_by_page.get(target_page, {}).get("quality", {})
            for target_page in target_pages
        ]
        relation, reasons, confidence, same_info = sc_verification_relation(
            page_index,
            source_page_count,
            target_page_count,
            target_pages,
            source_quality,
            target_qualities,
        )
        top_overlap = overlaps[0] if overlaps else {}
        rows.append({
            "verification_id": (
                f"{pair['pair_id']}:{source_variant.lower()}_p{page_index:04d}"
                f"_to_{target_variant.lower()}"
            ),
            "pair_id": pair["pair_id"],
            "citation_key": pair.get("citation_key"),
            "source_language_variant": source_variant,
            "target_language_variant": target_variant,
            "source_document_id": source_doc,
            "target_document_id": target_doc,
            "source_page_index": page_index,
            "source_page_count": source_page_count,
            "target_page_count": target_page_count,
            "source_text_path": source_text.get("text_path") if source_text else None,
            "source_text_sha256": source_text.get("text_sha256") if source_text else None,
            "source_text_quality": source_quality,
            "predicted_target_page_indices": target_pages,
            "top_target_page_index": top_overlap.get("page_index"),
            "top_source_overlap_ratio": top_overlap.get("source_overlap_ratio"),
            "top_target_overlap_ratio": top_overlap.get("target_overlap_ratio"),
            "overlap_candidates": overlaps[:5],
            "content_relation": relation,
            "same_information_likely": same_info,
            "bleed_likely": relation in {
                "source_page_spans_multiple_target_pages",
                "same_count_shift_or_bleed",
                "page_count_mismatch_single_target_page",
            },
            "needs_manual_review": not same_info,
            "confidence": confidence,
            "failure_reasons": reasons,
            "verification_method": "cumulative_text_flow_overlap_v1",
        })
    return rows


def build_sc_content_verification(
    paths: dict[str, Path],
    pair_rows: list[dict[str, Any]],
    text_index: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows_by_doc = sc_text_rows_by_document(text_index)
    for pair in pair_rows:
        if not pair.get("paired"):
            continue
        rows.extend(build_directional_sc_content_verification(pair, rows_by_doc, "HIN", "EN"))
        rows.extend(build_directional_sc_content_verification(pair, rows_by_doc, "EN", "HIN"))

    relation_counts = dict(Counter(row["content_relation"] for row in rows))
    review_rows = [row for row in rows if row.get("needs_manual_review")]
    same_info_rows = [row for row in rows if row.get("same_information_likely")]
    summary = {
        "created_at": utc_now(),
        "verification_method": "cumulative_text_flow_overlap_v1",
        "row_count": len(rows),
        "pair_count": len([pair for pair in pair_rows if pair.get("paired")]),
        "same_information_likely_rows": len(same_info_rows),
        "needs_manual_review_rows": len(review_rows),
        "relation_counts": relation_counts,
        "note": (
            "This is a deterministic page-flow verifier over extracted text. "
            "Rows marked same_information_likely are safe same-page candidates; "
            "other rows indicate probable bleed, page-count mismatch, low text quality, or uncertainty."
        ),
    }
    jsonl_write(paths["reports"] / "sc_page_content_verification.jsonl", rows)
    write_json(paths["reports"] / "sc_page_content_verification_summary.json", summary)
    return rows, summary


def build_sc_alignment_from_pairs(
    paths: dict[str, Path],
    pair_rows: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    text_index: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    verification_rows, _ = build_sc_content_verification(paths, pair_rows, text_index)
    h_to_e = [
        row for row in verification_rows
        if row["source_language_variant"] == "HIN" and row["target_language_variant"] == "EN"
    ]
    verification_by_key = {
        (row["pair_id"], int(row["source_page_index"])): row for row in h_to_e
    }
    page_lookup = {(p["document_id"], p["page_index"]): p for p in pages if p["corpus"] == "sc"}
    text_by_doc_page = {(doc_id, page_index): row for (doc_id, page_index), row in text_index.items()}
    pair_by_id = {row["pair_id"]: row for row in pair_rows}
    alignment_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for verify in h_to_e:
        pair = pair_by_id[verify["pair_id"]]
        h_page = page_lookup.get((pair["hindi_document_id"], int(verify["source_page_index"])))
        target_pages = verify.get("predicted_target_page_indices") or []
        english_page_index = target_pages[0] if len(target_pages) == 1 else None
        e_text = (
            text_by_doc_page.get((pair["english_document_id"], int(english_page_index)))
            if english_page_index is not None
            else None
        )
        verified = bool(
            verify.get("same_information_likely")
            and h_page
            and e_text
            and english_page_index == int(verify["source_page_index"])
        )
        reasons = list(verify.get("failure_reasons") or [])
        if not h_page:
            reasons.append("missing_hindi_rendered_page")
        if not e_text:
            reasons.append("missing_single_english_text_target")
        row = {
            "alignment_id": f"{verify['pair_id']}:p{int(verify['source_page_index']):04d}",
            "pair_id": verify["pair_id"],
            "citation_key": verify.get("citation_key"),
            "hindi_page_id": h_page["page_id"] if h_page else None,
            "hindi_render_path": h_page["render_path"] if h_page else None,
            "hindi_page_index": int(verify["source_page_index"]),
            "english_document_id": pair.get("english_document_id"),
            "english_source_path": pair.get("english_source_path"),
            "english_page_index": english_page_index,
            "english_text_path": e_text["text_path"] if e_text else None,
            "english_text_sha256": e_text["text_sha256"] if e_text else None,
            "alignment_method": "same_page_content_flow_verified_v1" if verified else verify["verification_method"],
            "alignment_verified": verified,
            "alignment_confidence": verify["confidence"] if verified else 0.0,
            "content_relation": verify["content_relation"],
            "predicted_english_page_indices": target_pages,
            "bleed_likely": verify["bleed_likely"],
            "failure_reasons": [] if verified else sorted(set(reasons)),
            "native_english_target": True,
            "gemini_used_for_translation": False,
            "content_verification_id": verify["verification_id"],
        }
        alignment_rows.append(row)
        if not verified:
            failures.append({**row, "failure_stage": "page_content_alignment"})

    pair_failures = [
        {**pair, "failure_stage": "pairing"}
        for pair in pair_rows
        if pair.get("failure_reasons")
    ]
    jsonl_write(paths["manifests"] / "sc_page_alignment.jsonl", alignment_rows)
    jsonl_write(paths["reports"] / "sc_alignment_failures.jsonl", pair_failures + failures)
    return alignment_rows


def build_sc_pairs_and_alignment(
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    text_index: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["corpus"] == "sc" and row["valid_source"] and row.get("sc_pair_key"):
            by_key.setdefault(row["sc_pair_key"], {})[row.get("language_variant") or ""] = row

    pair_rows: list[dict[str, Any]] = []
    for key in sorted(by_key):
        values = by_key[key]
        hin = values.get("HIN")
        eng = values.get("EN")
        pair_id = f"sc_pair:{key}"
        pair_ok = bool(hin and eng)
        failure_reasons: list[str] = []
        if not hin:
            failure_reasons.append("missing_hindi_pdf")
        if not eng:
            failure_reasons.append("missing_english_pdf")
        if pair_ok and int(hin["page_count"]) != int(eng["page_count"]):
            failure_reasons.append("page_count_mismatch_possible_offset_or_bleed")
        pair_row = {
            "pair_id": pair_id,
            "citation_key": key,
            "hindi_source_path": hin["source_path"] if hin else None,
            "english_source_path": eng["source_path"] if eng else None,
            "hindi_document_id": hin.get("document_id") if hin else None,
            "english_document_id": eng.get("document_id") if eng else None,
            "hindi_page_count": int(hin["page_count"]) if hin else None,
            "english_page_count": int(eng["page_count"]) if eng else None,
            "paired": pair_ok,
            "failure_reasons": failure_reasons,
            "split_group_id": f"sc_pair:{key}",
        }
        pair_rows.append(pair_row)
    jsonl_write(paths["manifests"] / "sc_pdf_pairs.jsonl", pair_rows)
    alignment_rows = build_sc_alignment_from_pairs(paths, pair_rows, pages, text_index)
    return pair_rows, alignment_rows


def image_inline_part(image_path: Path) -> dict[str, Any]:
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": "image/jpeg", "data": data}}


def gemini_ocr_response_schema() -> dict[str, Any]:
    return {
        "type": "OBJECT",
        "properties": {
            "plain_text": {"type": "STRING"},
            "markdown": {"type": "STRING"},
            "blocks": {"type": "ARRAY", "items": {"type": "OBJECT"}},
            "key_values": {"type": "ARRAY", "items": {"type": "OBJECT"}},
            "tables": {"type": "ARRAY", "items": {"type": "OBJECT"}},
            "quality_flags": {"type": "ARRAY", "items": {"type": "STRING"}},
            "confidence": {"type": "NUMBER"},
        },
        "required": sorted(GEMINI_REQUIRED_FIELDS),
        "propertyOrdering": ["plain_text", "markdown", "blocks", "key_values", "tables", "quality_flags", "confidence"],
    }


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def parse_json_response(text: str) -> Any:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        candidates: list[Any] = []
        for idx, char in enumerate(stripped):
            if char not in "{[":
                continue
            candidate_text = stripped[idx:]
            try:
                value, _ = decoder.raw_decode(candidate_text)
            except json.JSONDecodeError:
                try:
                    value, _ = decoder.raw_decode(escape_json_string_controls(candidate_text))
                except json.JSONDecodeError:
                    continue
            candidates.append(value)
            if isinstance(value, dict) and GEMINI_REQUIRED_FIELDS.issubset(value):
                return value
            if isinstance(value, list) and any(isinstance(item, dict) and GEMINI_REQUIRED_FIELDS.issubset(item) for item in value):
                return value
        scored = [
            (len(GEMINI_REQUIRED_FIELDS.intersection(value)), value)
            for value in candidates
            if isinstance(value, dict)
        ]
        if scored:
            score, value = max(scored, key=lambda item: item[0])
            if score > 0:
                return value
        raise


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


def call_gemini(page: dict[str, Any], model: str, api_key: str) -> tuple[Any | None, dict[str, Any], str]:
    image_path = ROOT / page["render_path"]
    prompt = (
        "You are an OCR expert for Indian legal, police, and Indic book pages. "
        "Transcribe only visible text from this full page. Preserve script, spelling, line breaks, "
        "page numbers, tables, stamps, signatures, handwritten regions, form labels, and illegible markers. "
        "Return valid JSON only with exactly these top-level keys: plain_text, markdown, blocks, "
        "key_values, tables, quality_flags, confidence. Blocks must be an ordered list with text, type, "
        "bbox, language, and confidence when visible. Do not translate. Do not explain your reasoning. "
        "Do not include markdown fences, analysis, bullet lists, or any text outside JSON. The first "
        "character of your response must be { and the last character must be }. For unclear handwriting, "
        "write your best single reading once and use [illegible] for unreadable spans; do not list alternatives, "
        "do not re-read lines, and do not repeat text."
    )
    generation_config = {
        "temperature": 0,
        "maxOutputTokens": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "16384")),
        "responseMimeType": "application/json",
    }
    if env_flag("GEMINI_USE_RESPONSE_SCHEMA", False):
        generation_config["responseSchema"] = gemini_ocr_response_schema()
    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}, image_inline_part(image_path)],
        }],
        "generationConfig": generation_config,
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    raw = ""
    finish_reasons: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "180"))) as response:
            raw = response.read().decode("utf-8")
        elapsed = round(time.time() - started, 3)
        payload = json.loads(raw)
        finish_reasons = [cand.get("finishReason") for cand in payload.get("candidates", []) if cand.get("finishReason")]
        text = ""
        thought_part_count = 0
        for cand in payload.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("thought"):
                    thought_part_count += 1
                    continue
                text += part.get("text", "")
        parsed = parse_json_response(text)
        return parsed, {
            "ok": True,
            "model": model,
            "elapsed_seconds": elapsed,
            "finish_reasons": finish_reasons,
            "thought_part_count": thought_part_count,
            "response_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }, raw
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        detail = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            try:
                raw = exc.read().decode("utf-8")
                detail = raw[:1000]
            except Exception:
                detail = str(exc)
        return None, {
            "ok": False,
            "model": model,
            "error_type": type(exc).__name__,
            "error": detail[:1000],
            "finish_reasons": finish_reasons,
            "elapsed_seconds": round(time.time() - started, 3),
        }, raw


def is_quota_status(status: dict[str, Any]) -> bool:
    detail = str(status.get("error") or "")
    detail_l = detail.lower()
    error_type = str(status.get("error_type") or "")
    error_type_l = error_type.lower()
    return (
        "RESOURCE_EXHAUSTED" in detail
        or "Quota exceeded" in detail
        or '"code": 429' in detail
        or "code\": 429" in detail
        or "status: 429" in detail_l
        or "usage limit exceeded" in detail_l
        or "rate limit" in detail_l
        or "too many requests" in detail_l
        or error_type == "UsageLimitExceeded"
        or error_type_l == "temporarilyblocked"
    )


def is_retry_neutral_transient_failure(status: dict[str, Any]) -> bool:
    detail = str(status.get("error") or "").lower()
    return (
        (
            status.get("error_type") == "URLError"
            and (
            "temporary failure in name resolution" in detail
            or "name or service not known" in detail
            )
        )
        or (
            status.get("error_type") == "HTTPError"
            and (
                '"code": 500' in detail
                or '"code": 503' in detail
                or '"status": "internal"' in detail
                or '"status": "unavailable"' in detail
                or "currently experiencing high demand" in detail
            )
        )
        or status.get("error_type") in {"TimeoutError", "ConnectionError"}
        or "temporarily unavailable" in detail
        or "try again later" in detail
    )


def retry_after_seconds(status: dict[str, Any]) -> float:
    detail = str(status.get("error") or "")
    m = re.search(r"retry in ([0-9.]+)s", detail, flags=re.I)
    if m:
        return float(m.group(1)) + 2.0
    return float(os.getenv("GEMINI_QUOTA_BACKOFF_SECONDS", "65"))


def quota_detail_fields(status: dict[str, Any]) -> dict[str, Any]:
    detail = str(status.get("error") or "")
    metric = None
    limit = None
    metric_match = re.search(r"Quota exceeded for metric:\s*([^,\n]+)", detail)
    if metric_match:
        metric = metric_match.group(1).strip()
    limit_match = re.search(r"\blimit:\s*([0-9]+)", detail)
    if limit_match:
        limit = int(limit_match.group(1))
    return {"metric": metric, "limit": limit}


def text_from_rel_path(rel_path: str | None, max_chars: int | None = None) -> str:
    if not rel_path:
        return ""
    text = (ROOT / rel_path).read_text(encoding="utf-8").strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars] + "\n[TRUNCATED]"
    return text


def call_gemini_text_json(prompt: str, model: str, api_key: str, *, max_output_tokens: int) -> tuple[Any | None, dict[str, Any], str]:
    generation_config = {
        "temperature": 0,
        "maxOutputTokens": max_output_tokens,
        "responseMimeType": "application/json",
    }
    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
        "generationConfig": generation_config,
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    raw = ""
    finish_reasons: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "180"))) as response:
            raw = response.read().decode("utf-8")
        elapsed = round(time.time() - started, 3)
        payload = json.loads(raw)
        finish_reasons = [cand.get("finishReason") for cand in payload.get("candidates", []) if cand.get("finishReason")]
        text = ""
        thought_part_count = 0
        for cand in payload.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("thought"):
                    thought_part_count += 1
                    continue
                text += part.get("text", "")
        parsed = parse_json_response(text)
        return parsed, {
            "ok": True,
            "model": model,
            "elapsed_seconds": elapsed,
            "finish_reasons": finish_reasons,
            "thought_part_count": thought_part_count,
            "response_hash": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }, raw
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        detail = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            try:
                raw = exc.read().decode("utf-8")
                detail = raw[:1000]
            except Exception:
                detail = str(exc)
        return None, {
            "ok": False,
            "model": model,
            "error_type": type(exc).__name__,
            "error": detail[:1000],
            "finish_reasons": finish_reasons,
            "elapsed_seconds": round(time.time() - started, 3),
        }, raw


def sc_verification_direction(row: dict[str, Any]) -> str:
    return f"{str(row['source_language_variant']).lower()}-to-{str(row['target_language_variant']).lower()}"


def count_sc_verification_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "same_information_likely_rows": sum(1 for row in rows if row.get("same_information_likely")),
        "needs_manual_review_rows": sum(1 for row in rows if row.get("needs_manual_review")),
        "bleed_likely_rows": sum(1 for row in rows if row.get("bleed_likely")),
        "relation_counts": dict(Counter(row.get("content_relation") for row in rows)),
    }


def build_sc_verification_pair_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_pair.setdefault(str(row["pair_id"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for pair_id, pair_rows in sorted(rows_by_pair.items(), key=lambda item: str(item[1][0].get("citation_key") or item[0])):
        by_direction = {
            "hin-to-en": [row for row in pair_rows if sc_verification_direction(row) == "hin-to-en"],
            "en-to-hin": [row for row in pair_rows if sc_verification_direction(row) == "en-to-hin"],
        }
        hin_rows = by_direction["hin-to-en"]
        en_rows = by_direction["en-to-hin"]
        first = pair_rows[0]
        summary = {
            "pair_id": pair_id,
            "citation_key": first.get("citation_key"),
            "hindi_page_count": max([int(row.get("source_page_count") or 0) for row in hin_rows] or [0]),
            "english_page_count": max([int(row.get("source_page_count") or 0) for row in en_rows] or [0]),
            "row_count": len(pair_rows),
            "pair_has_bleed_likely": any(row.get("bleed_likely") for row in pair_rows),
            "pair_needs_manual_review": any(row.get("needs_manual_review") for row in pair_rows),
            "pair_has_same_information_likely_pages": any(row.get("same_information_likely") for row in pair_rows),
            "overall": count_sc_verification_rows(pair_rows),
            "directions": {direction: count_sc_verification_rows(direction_rows) for direction, direction_rows in by_direction.items()},
        }
        summaries.append(summary)
    return summaries


def sc_verification_review_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "verification_id": row.get("verification_id"),
        "pair_id": row.get("pair_id"),
        "citation_key": row.get("citation_key"),
        "direction": sc_verification_direction(row),
        "source_language_variant": row.get("source_language_variant"),
        "target_language_variant": row.get("target_language_variant"),
        "source_page_index": row.get("source_page_index"),
        "source_page_count": row.get("source_page_count"),
        "target_page_count": row.get("target_page_count"),
        "predicted_target_page_indices": row.get("predicted_target_page_indices") or [],
        "content_relation": row.get("content_relation"),
        "same_information_likely": row.get("same_information_likely"),
        "bleed_likely": row.get("bleed_likely"),
        "needs_manual_review": row.get("needs_manual_review"),
        "confidence": row.get("confidence"),
        "top_target_page_index": row.get("top_target_page_index"),
        "top_source_overlap_ratio": row.get("top_source_overlap_ratio"),
        "top_target_overlap_ratio": row.get("top_target_overlap_ratio"),
        "failure_reasons": row.get("failure_reasons") or [],
        "source_text_path": row.get("source_text_path"),
        "source_text_sha256": row.get("source_text_sha256"),
    }


def write_sc_verification_report(
    paths: dict[str, Path],
    *,
    report_name: str,
    pair_summary_name: str,
    review_manifest_name: str,
    max_examples: int,
) -> dict[str, Any]:
    rows = jsonl_read(paths["reports"] / "sc_page_content_verification.jsonl")
    summary = read_json(paths["reports"] / "sc_page_content_verification_summary.json", {})
    if not rows:
        raise RuntimeError("Missing sc_page_content_verification.jsonl; run verify-sc-alignment first.")
    pair_summaries = build_sc_verification_pair_summaries(rows)
    review_rows = [sc_verification_review_row(row) for row in rows if row.get("needs_manual_review")]
    review_rows.sort(key=lambda row: (
        str(row.get("citation_key") or ""),
        str(row.get("direction") or ""),
        int(row.get("source_page_index") or 0),
    ))

    report_path = paths["reports"] / report_name
    pair_summary_path = paths["reports"] / pair_summary_name
    review_manifest_path = paths["reports"] / review_manifest_name
    jsonl_write(pair_summary_path, pair_summaries)
    jsonl_write(review_manifest_path, review_rows)

    overall_counts = count_sc_verification_rows(rows)
    direction_rows = {
        "hin-to-en": [row for row in rows if sc_verification_direction(row) == "hin-to-en"],
        "en-to-hin": [row for row in rows if sc_verification_direction(row) == "en-to-hin"],
    }
    direction_counts = {direction: count_sc_verification_rows(direction_rows[direction]) for direction in direction_rows}
    reason_counts = Counter(reason for row in rows for reason in (row.get("failure_reasons") or []))
    pair_status_counts = {
        "pairs_total": len(pair_summaries),
        "pairs_with_bleed_likely": sum(1 for row in pair_summaries if row.get("pair_has_bleed_likely")),
        "pairs_needing_manual_review": sum(1 for row in pair_summaries if row.get("pair_needs_manual_review")),
        "pairs_with_any_same_page_likely": sum(1 for row in pair_summaries if row.get("pair_has_same_information_likely_pages")),
    }
    relation_labels = {
        "same_page_likely": "same page index, same page count, and both extracted texts passed quality checks",
        "source_page_spans_multiple_target_pages": "source page overlaps more than one target page by cumulative text flow",
        "page_count_mismatch_single_target_page": "best target is one page, but PDF page counts differ",
        "same_count_shift_or_bleed": "PDF page counts match, but text-flow best target is not the same page index",
        "same_page_low_text_quality": "same page index, but source or target text quality failed",
        "unmatched_or_empty": "source page is empty/too short or no target overlap was found",
    }

    lines: list[str] = []
    lines.append("# SC Full Deterministic Page-Flow Verification Report")
    lines.append("")
    lines.append(f"Generated: {utc_now()}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- This is a full deterministic report over the available Supreme Court Hindi-English PDF pairs.")
    lines.append("- It uses native extracted page text and cumulative text-flow overlap; it does not call Gemini.")
    lines.append("- It checks both directions: Hindi page to English page candidates and English page to Hindi page candidates.")
    lines.append("- Rows marked `same_page_likely` are safe same-page candidates for the pipeline.")
    lines.append("- Rows marked review-needed indicate probable bleed, page-count mismatch, low text quality, or insufficient text and are excluded from automatic SC export.")
    lines.append("")
    lines.append("Command used:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        ".venv/bin/python scripts/indic_ocr_v1_pipeline.py --output artifacts/page_only_v1 "
        "report-sc-verification"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Corpus Summary")
    lines.append("")
    lines.append(f"- Paired SC PDFs: {summary.get('pair_count', pair_status_counts['pairs_total'])}")
    lines.append(f"- Bidirectional page rows checked: {overall_counts['row_count']}")
    lines.append(f"- Same-information-likely rows: {overall_counts['same_information_likely_rows']}")
    lines.append(f"- Bleed-likely rows: {overall_counts['bleed_likely_rows']}")
    lines.append(f"- Review/excluded rows: {overall_counts['needs_manual_review_rows']}")
    lines.append(f"- Pairs with at least one same-page-likely row: {pair_status_counts['pairs_with_any_same_page_likely']}")
    lines.append(f"- Pairs with at least one bleed-likely row: {pair_status_counts['pairs_with_bleed_likely']}")
    lines.append(f"- Pairs with any review/excluded row: {pair_status_counts['pairs_needing_manual_review']}")
    lines.append("")
    lines.append("## Direction Summary")
    lines.append("")
    lines.append("| Direction | Rows | Same-page likely | Bleed likely | Needs review |")
    lines.append("|---|---:|---:|---:|---:|")
    for direction in ["hin-to-en", "en-to-hin"]:
        counts = direction_counts[direction]
        lines.append(
            f"| `{direction}` | {counts['row_count']} | {counts['same_information_likely_rows']} | "
            f"{counts['bleed_likely_rows']} | {counts['needs_manual_review_rows']} |"
        )
    lines.append("")
    lines.append("## Relation Counts")
    lines.append("")
    lines.append("| Relation | Rows | Meaning |")
    lines.append("|---|---:|---|")
    for relation, count in sorted(overall_counts["relation_counts"].items()):
        lines.append(f"| `{relation}` | {count} | {relation_labels.get(str(relation), '')} |")
    lines.append("")
    if reason_counts:
        lines.append("## Failure/Review Reasons")
        lines.append("")
        for reason, count in reason_counts.most_common():
            lines.append(f"- `{reason}`: {count}")
        lines.append("")
    lines.append("## Review Examples")
    lines.append("")
    lines.append(f"First {min(max_examples, len(review_rows))} review-needed rows, sorted by citation and direction:")
    lines.append("")
    lines.append("| Citation key | Direction | Source page | Predicted target pages | Relation | Top target | Source overlap | Target overlap | Reasons |")
    lines.append("|---|---|---:|---|---|---:|---:|---:|---|")
    for row in review_rows[:max_examples]:
        predicted = ", ".join(str(page) for page in row.get("predicted_target_page_indices") or [])
        reasons = ", ".join(str(reason) for reason in row.get("failure_reasons") or [])
        src_overlap = row.get("top_source_overlap_ratio")
        tgt_overlap = row.get("top_target_overlap_ratio")
        src_overlap_text = f"{src_overlap:.3f}" if isinstance(src_overlap, (int, float)) else ""
        tgt_overlap_text = f"{tgt_overlap:.3f}" if isinstance(tgt_overlap, (int, float)) else ""
        lines.append(
            f"| `{row.get('citation_key')}` | `{row.get('direction')}` | {row.get('source_page_index')} | "
            f"{predicted} | `{row.get('content_relation')}` | {row.get('top_target_page_index') or ''} | "
            f"{src_overlap_text} | {tgt_overlap_text} | {reasons} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `reports/sc_page_content_verification.jsonl`: authoritative page-level deterministic rows for every checked page/direction.")
    lines.append("- `reports/sc_page_content_verification_summary.json`: aggregate deterministic summary.")
    lines.append(f"- `reports/{pair_summary_path.name}`: one summary row per PDF pair.")
    lines.append(f"- `reports/{review_manifest_path.name}`: compact manifest of rows excluded from automatic SC export.")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report_path": str(report_path),
        "pair_summary_path": str(pair_summary_path),
        "review_manifest_path": str(review_manifest_path),
        "row_count": overall_counts["row_count"],
        "pair_count": pair_status_counts["pairs_total"],
        "same_information_likely_rows": overall_counts["same_information_likely_rows"],
        "bleed_likely_rows": overall_counts["bleed_likely_rows"],
        "needs_manual_review_rows": overall_counts["needs_manual_review_rows"],
        "relation_counts": overall_counts["relation_counts"],
    }


def quota_cooldown_remaining_seconds(rows: list[dict[str, Any]]) -> float:
    quota_rows = [row for row in rows if is_quota_status(row) and row.get("created_at")]
    if not quota_rows:
        return 0.0
    latest_quota = quota_rows[-1]
    try:
        created_at = datetime.fromisoformat(str(latest_quota["created_at"]))
    except ValueError:
        return 0.0
    cooldown = float(os.getenv("GEMINI_MIN_QUOTA_COOLDOWN_SECONDS", "0"))
    detail = quota_detail_fields(latest_quota)
    short_limit_threshold = int(os.getenv("GEMINI_SHORT_QUOTA_LIMIT_THRESHOLD", "100"))
    if detail.get("limit") is not None and int(detail["limit"]) <= short_limit_threshold:
        cooldown = retry_after_seconds(latest_quota)
    if cooldown <= 0:
        return 0.0
    age_seconds = (datetime.now(timezone.utc) - created_at).total_seconds()
    return max(0.0, round(cooldown - age_seconds, 1))


def coerce_teacher_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and any(field in item for field in GEMINI_REQUIRED_FIELDS):
                return item
    return {}


def validate_teacher_payload(payload: Any, min_confidence: float) -> tuple[bool, list[str], dict[str, Any]]:
    payload = coerce_teacher_payload(payload)
    errors = [f"missing:{field}" for field in sorted(GEMINI_REQUIRED_FIELDS) if field not in payload]
    normalized = {
        "plain_text": str(payload.get("plain_text") or "").strip(),
        "markdown": str(payload.get("markdown") or payload.get("plain_text") or "").strip(),
        "blocks": payload.get("blocks") if isinstance(payload.get("blocks"), list) else [],
        "key_values": payload.get("key_values") if isinstance(payload.get("key_values"), list) else [],
        "tables": payload.get("tables") if isinstance(payload.get("tables"), list) else [],
        "quality_flags": (
            payload.get("quality_flags")
            if isinstance(payload.get("quality_flags"), list)
            else [f"{key}:{value}" for key, value in payload.get("quality_flags", {}).items()]
            if isinstance(payload.get("quality_flags"), dict)
            else []
        ),
        "confidence": payload.get("confidence"),
    }
    try:
        normalized["confidence"] = float(normalized["confidence"])
        if 1.0 < normalized["confidence"] <= 100.0:
            normalized["confidence"] = normalized["confidence"] / 100.0
    except (TypeError, ValueError):
        normalized["confidence"] = 0.0
        errors.append("invalid:confidence")
    quality_flag_text = " ".join(str(flag).lower() for flag in normalized["quality_flags"])
    structurally_blank = (
        not normalized["plain_text"]
        and not normalized["markdown"]
        and not normalized["blocks"]
        and not normalized["key_values"]
        and not normalized["tables"]
    )
    high_confidence_blank_page = (
        not normalized["plain_text"]
        and normalized["confidence"] >= min_confidence
        and (
            "blank_page" in quality_flag_text
            or "is_blank:true" in quality_flag_text
            or "is_blank: true" in quality_flag_text
            or "no_text_found:true" in quality_flag_text
            or "no_text_found: true" in quality_flag_text
            or structurally_blank
        )
    )
    if high_confidence_blank_page and not any("blank" in str(flag).lower() for flag in normalized["quality_flags"]):
        normalized["quality_flags"].append("blank_page")
    if not normalized["plain_text"] and not high_confidence_blank_page:
        errors.append("empty:plain_text")
    if not isinstance(payload.get("blocks"), list):
        errors.append("invalid:blocks")
    if normalized["confidence"] < min_confidence:
        errors.append("low_confidence")
    return not errors, errors, normalized


def build_fir_gita_gold(page: dict[str, Any], teacher: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    language = page.get("language_hint") or "und"
    blocks = teacher["blocks"] or [{
        "block_id": f"{page['page_id']}:b0001",
        "type": "text",
        "order": 1,
        "text": teacher["plain_text"],
        "bbox": [0, 0, 1000, 1000],
        "language": language,
        "confidence": teacher["confidence"],
    }]
    return {
        "schema_version": GOLD_SCHEMA_VERSION,
        "record_type": "golden_page",
        "record_id": f"gold:fir_gita:{page['page_id']}",
        "source_page_id": page["page_id"],
        "source_document_id": page["document_id"],
        "split": {"name": page.get("split"), "group_id": page["split_group_id"], "seed": SPLIT_SEED},
        "source": {
            "corpus": page["corpus"],
            "source_path": page["source_path"],
            "render_path": page["render_path"],
            "page_index": page["page_index"],
            "image_sha256": page["image_sha256"],
            "release_scope": "internal_unredacted",
        },
        "document_metadata": {"doc_type": page["doc_type"], "page_count": page["page_count"]},
        "language": {"primary": language, "all": [language], "scripts": scripts_from_language_hint(language), "code_mixed": "+" in language},
        "page": {
            "page_id": page["page_id"],
            "page_index": page["page_index"],
            "image_metadata": {
                "width_px": page["image_width_px"],
                "height_px": page["image_height_px"],
                "dpi": page["render_dpi"],
                "bbox_coord_space": "normalized_1000",
                "render_source": "full_page",
                "visual_token_budget": page["visual_token_budget"],
                "is_crop": False,
            },
            "quality_flags": sorted(set(page["quality_flags"] + [str(flag) for flag in teacher["quality_flags"]])),
            "transcription": {"plain_text": teacher["plain_text"], "markdown": teacher["markdown"], "blocks": blocks},
            "tables": teacher["tables"],
            "key_values": teacher["key_values"],
        },
        "pii": {"contains_pii": page["corpus"] == "fir", "mode": "internal_unredacted", "entities": []},
        "annotation": {
            "gold_source": "gemini_teacher",
            "teacher_status": status,
            "created_at": utc_now(),
            "confidence": teacher["confidence"],
            "review": {"status": "unreviewed", "reviewer_id": None},
        },
        "translation_targets": [],
        "validation": {"ok": True, "errors": []},
    }


def fir_gita_gold_teacher_model(row: dict[str, Any]) -> str | None:
    return row.get("annotation", {}).get("teacher_status", {}).get("model")


def current_teacher_fir_gita_gold(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row.get("validation", {}).get("ok")
        and fir_gita_gold_teacher_model(row) == model
    ]


def annotate_gemini_page(
    page: dict[str, Any],
    model: str,
    api_key: str,
    min_confidence: float,
    request_lock: threading.Lock,
    last_request_at: dict[str, float],
    min_request_interval: float,
) -> dict[str, Any]:
    raw_rows: list[dict[str, Any]] = []
    final_status: dict[str, Any] = {}
    accepted: dict[str, Any] | None = None
    invalid_attempts = 0
    total_attempts = 0
    max_total_attempts = int(os.getenv("GEMINI_MAX_TOTAL_ATTEMPTS", "12"))

    while invalid_attempts < 3 and total_attempts < max_total_attempts:
        with request_lock:
            wait = min_request_interval - (time.time() - last_request_at["value"])
            if wait > 0:
                time.sleep(wait)
            last_request_at["value"] = time.time()
        total_attempts += 1
        attempt = total_attempts
        parsed, status, raw = call_gemini(page, model, api_key)
        raw_rows.append({
            "page_id": page["page_id"],
            "attempt": attempt,
            "model": model,
            "ok": status.get("ok", False),
            "raw_response": raw,
            "created_at": utc_now(),
        })
        if parsed is None:
            if is_quota_status(status):
                delay = retry_after_seconds(status)
                final_status = {**status, "attempt": attempt, "status": "quota_backoff", "retry_after_seconds": delay}
                print(f"Gemini quota backoff {delay:.1f}s for {page['page_id']}", flush=True)
                time.sleep(delay)
                continue
            invalid_attempts += 1
            final_status = {**status, "attempt": attempt, "status": "call_failed"}
            time.sleep(float(os.getenv("GEMINI_RETRY_SLEEP_SECONDS", "1.0")))
            continue

        ok, errors, normalized = validate_teacher_payload(parsed, min_confidence)
        if not ok:
            invalid_attempts += 1
        final_status = {
            **status,
            "attempt": attempt,
            "status": "accepted" if ok else "invalid_json_or_quality",
            "validation_errors": errors,
            "confidence": normalized["confidence"],
        }
        if ok:
            accepted = normalized
            break
        time.sleep(float(os.getenv("GEMINI_RETRY_SLEEP_SECONDS", "1.0")))

    run_row = {
        "page_id": page["page_id"],
        "source_path": page["source_path"],
        "corpus": page["corpus"],
        "attempts": final_status.get("attempt", 0),
        "model": model,
        "status": final_status.get("status", "failed"),
        "ok": accepted is not None,
        "confidence": final_status.get("confidence"),
        "validation_errors": final_status.get("validation_errors", []),
        "error_type": final_status.get("error_type"),
        "error": final_status.get("error"),
        "finish_reasons": final_status.get("finish_reasons", []),
        "elapsed_seconds": final_status.get("elapsed_seconds"),
        "thought_part_count": final_status.get("thought_part_count"),
        "retry_after_seconds": final_status.get("retry_after_seconds"),
        "created_at": utc_now(),
    }
    failure_row = None
    gold = None
    if accepted:
        gold = build_fir_gita_gold(page, accepted, final_status)
    else:
        failure_row = {
            "page_id": page["page_id"],
            "source_path": page["source_path"],
            "corpus": page["corpus"],
            "attempts": final_status.get("attempt", 0),
            "model": model,
            "status": final_status.get("status", "failed"),
            "validation_errors": final_status.get("validation_errors", []),
            "error_type": final_status.get("error_type"),
            "finish_reasons": final_status.get("finish_reasons", []),
            "elapsed_seconds": final_status.get("elapsed_seconds"),
            "thought_part_count": final_status.get("thought_part_count"),
            "retry_after_seconds": final_status.get("retry_after_seconds"),
            "created_at": utc_now(),
            "error": final_status.get("error"),
        }
    return {"raw_rows": raw_rows, "run_row": run_row, "failure_row": failure_row, "gold": gold}


def extract_gemini_app_browser_cookies() -> str:
    import gemini_app_browser_smoke as browser_smoke

    browser_args = browser_smoke.parse_args([])
    gemini_url = f"https://gemini.google.com/app?authuser={browser_args.authuser}"
    browser_smoke.launch_browser(browser_args, gemini_url)
    browser_smoke.wait_for_cdp(browser_args.cdp_port, browser_args.cdp_timeout)
    browser_smoke.open_gemini(browser_args)
    cookie_env, cookie_metadata = browser_smoke.cookie_env_from_browser(browser_args)
    os.environ.update(cookie_env)
    print(
        "Extracted required Gemini cookie names from browser: "
        f"{', '.join(cookie_metadata['found_cookie_names'])}.",
        flush=True,
    )
    account_status = browser_smoke.verify_account(browser_args)
    if account_status["status"] == "confirmed":
        print("Confirmed Google account selection for Gemini App cookies.", flush=True)
    else:
        print("Gemini App browser cookie account check was not confirmed; continuing with extracted cookies.", flush=True)
    browser_smoke.open_gemini(browser_args)
    return "browser"


async def annotate_gemini_app_page(
    page: dict[str, Any],
    client: Any,
    selected_model: Any,
    model: str,
    model_info: dict[str, Any],
    credential_source: str,
    min_confidence: float,
    request_lock: asyncio.Lock,
    last_request_at: dict[str, float],
    min_request_interval: float,
    request_timeout: float,
) -> dict[str, Any]:
    import gemini_app_client as app_client

    raw_rows: list[dict[str, Any]] = []
    final_status: dict[str, Any] = {}
    accepted: dict[str, Any] | None = None
    invalid_attempts = 0
    total_attempts = 0
    max_total_attempts = int(os.getenv("GEMINI_MAX_TOTAL_ATTEMPTS", "12"))

    while invalid_attempts < 3 and total_attempts < max_total_attempts:
        async with request_lock:
            wait_seconds = min_request_interval - (time.time() - last_request_at["value"])
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            last_request_at["value"] = time.time()
        total_attempts += 1
        attempt = total_attempts
        parsed, status, raw = await app_client.call_gemini_app_page(
            page=page,
            client=client,
            selected_model=selected_model,
            model_key=model,
            model_info=model_info,
            request_timeout=request_timeout,
        )
        status = {
            **status,
            "model": model,
            "model_info": model_info,
            "transport": "gemini_webapi",
            "credential_source": credential_source,
        }
        raw_rows.append({
            "page_id": page["page_id"],
            "attempt": attempt,
            "model": model,
            "model_info": model_info,
            "transport": "gemini_webapi",
            "ok": status.get("ok", False),
            "raw_response": raw,
            "created_at": utc_now(),
        })
        if parsed is None:
            if is_quota_status(status):
                delay = retry_after_seconds(status)
                final_status = {**status, "attempt": attempt, "status": "quota_backoff", "retry_after_seconds": delay}
                print(f"Gemini App quota backoff {delay:.1f}s for {page['page_id']}", flush=True)
                await asyncio.sleep(delay)
                continue
            invalid_attempts += 1
            final_status = {**status, "attempt": attempt, "status": "call_failed"}
            await asyncio.sleep(float(os.getenv("GEMINI_RETRY_SLEEP_SECONDS", "1.0")))
            continue

        ok, errors, normalized = validate_teacher_payload(parsed, min_confidence)
        if not ok:
            invalid_attempts += 1
        final_status = {
            **status,
            "attempt": attempt,
            "status": "accepted" if ok else "invalid_json_or_quality",
            "validation_errors": errors,
            "confidence": normalized["confidence"],
        }
        if ok:
            accepted = normalized
            break
        await asyncio.sleep(float(os.getenv("GEMINI_RETRY_SLEEP_SECONDS", "1.0")))

    run_row = {
        "page_id": page["page_id"],
        "source_path": page["source_path"],
        "corpus": page["corpus"],
        "attempts": final_status.get("attempt", 0),
        "model": model,
        "model_info": model_info,
        "transport": "gemini_webapi",
        "credential_source": credential_source,
        "status": final_status.get("status", "failed"),
        "ok": accepted is not None,
        "confidence": final_status.get("confidence"),
        "validation_errors": final_status.get("validation_errors", []),
        "error_type": final_status.get("error_type"),
        "error": final_status.get("error"),
        "finish_reasons": final_status.get("finish_reasons", []),
        "elapsed_seconds": final_status.get("elapsed_seconds"),
        "thought_part_count": final_status.get("thought_part_count"),
        "retry_after_seconds": final_status.get("retry_after_seconds"),
        "created_at": utc_now(),
    }
    failure_row = None
    gold = None
    if accepted:
        gold = build_fir_gita_gold(page, accepted, final_status)
    else:
        failure_row = {
            "page_id": page["page_id"],
            "source_path": page["source_path"],
            "corpus": page["corpus"],
            "attempts": final_status.get("attempt", 0),
            "model": model,
            "model_info": model_info,
            "transport": "gemini_webapi",
            "credential_source": credential_source,
            "status": final_status.get("status", "failed"),
            "validation_errors": final_status.get("validation_errors", []),
            "error_type": final_status.get("error_type"),
            "finish_reasons": final_status.get("finish_reasons", []),
            "elapsed_seconds": final_status.get("elapsed_seconds"),
            "thought_part_count": final_status.get("thought_part_count"),
            "retry_after_seconds": final_status.get("retry_after_seconds"),
            "created_at": utc_now(),
            "error": final_status.get("error"),
        }
    return {"raw_rows": raw_rows, "run_row": run_row, "failure_row": failure_row, "gold": gold}


async def run_gemini_app_annotations(
    paths: dict[str, Path],
    pending_pages: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    gold_by_page: dict[str, dict[str, Any]],
    *,
    model: str,
    app_model: str,
    min_confidence: float,
    min_request_interval: float,
    options: argparse.Namespace | None,
) -> list[dict[str, Any]]:
    import gemini_app_client as app_client

    gold_path = paths["gold"] / "fir_gita_golden_documents.jsonl"
    run_path = paths["reports"] / "gemini_teacher_runs.jsonl"
    failure_path = paths["reports"] / "gemini_failures.jsonl"
    raw_path = paths["reports"] / "gemini_raw_responses.jsonl"
    credential_source = "environment"
    if bool(getattr(options, "browser_extract_cookies", False)):
        credential_source = extract_gemini_app_browser_cookies()
    psid, psidts, loaded_source = app_client.load_webapi_credentials()
    if credential_source == "environment":
        credential_source = loaded_source

    previous_cookie_path = os.environ.get("GEMINI_COOKIE_PATH")
    temporary_cookie_cache: tempfile.TemporaryDirectory[str] | None = None
    if previous_cookie_path is None:
        temporary_cookie_cache = tempfile.TemporaryDirectory(prefix="gemini_webapi_cookie_cache_")
        os.environ["GEMINI_COOKIE_PATH"] = temporary_cookie_cache.name

    client = None
    try:
        client = await app_client.init_app_client(
            psid,
            psidts,
            proxy=getattr(options, "gemini_app_proxy", None),
            init_timeout=float(getattr(options, "gemini_app_init_timeout", 60) or 60),
            verbose=env_flag("GEMINI_APP_VERBOSE", False),
        )
        selected_model, model_info = app_client.resolve_requested_model(client, app_model)
        request_lock = asyncio.Lock()
        last_request_at = {"value": 0.0}
        concurrency = max(1, int(os.getenv("GEMINI_CONCURRENCY", "4")))
        quota_stop_count = max(0, int(os.getenv("GEMINI_STOP_ON_QUOTA_COUNT", "3")))
        quota_streak = 0
        completed = 0
        total_to_attempt = len(pending_pages)
        request_timeout = float(getattr(options, "gemini_app_request_timeout", 300) or 300)
        stop_submitting = False
        next_page_index = 0

        while next_page_index < total_to_attempt and not stop_submitting:
            batch = pending_pages[next_page_index:next_page_index + concurrency]
            next_page_index += len(batch)
            results = await asyncio.gather(*[
                annotate_gemini_app_page(
                    page,
                    client,
                    selected_model,
                    model,
                    model_info,
                    credential_source,
                    min_confidence,
                    request_lock,
                    last_request_at,
                    min_request_interval,
                    request_timeout,
                )
                for page in batch
            ])
            for result in results:
                for raw_row in result["raw_rows"]:
                    jsonl_append(raw_path, raw_row)
                jsonl_append(run_path, result["run_row"])
                if result["gold"]:
                    gold_rows.append(result["gold"])
                    gold_by_page[result["gold"]["source_page_id"]] = result["gold"]
                    jsonl_append(gold_path, result["gold"])
                elif result["failure_row"]:
                    jsonl_append(failure_path, result["failure_row"])
                completed += 1

                if is_quota_status(result["run_row"]):
                    quota_streak += 1
                    if quota_stop_count and quota_streak >= quota_stop_count and not stop_submitting:
                        stop_submitting = True
                        print(
                            "Gemini App quota early stop after "
                            f"{quota_streak} consecutive quota backoffs "
                            f"({completed}/{total_to_attempt} pages finished)",
                            flush=True,
                        )
                else:
                    quota_streak = 0

                if completed % 10 == 0:
                    print(f"Gemini App FIR/Gita attempts {completed}/{total_to_attempt} pending ({len(gold_rows)} accepted total)", flush=True)
    finally:
        if client is not None:
            await client.close()
        if temporary_cookie_cache:
            temporary_cookie_cache.cleanup()
        if previous_cookie_path is None:
            os.environ.pop("GEMINI_COOKIE_PATH", None)
        else:
            os.environ["GEMINI_COOKIE_PATH"] = previous_cookie_path

    return gold_rows


def gemini_annotate_fir_gita(
    paths: dict[str, Path],
    pages: list[dict[str, Any]],
    *,
    max_pages: int | None,
    skip_gemini: bool,
    options: argparse.Namespace | None = None,
) -> list[dict[str, Any]]:
    target_pages = [p for p in pages if p["corpus"] in {"fir", "gita"}]
    transport = gemini_transport_from_args(options)
    api_key = os.getenv("GEMINI_API_KEY", "")
    app_model = gemini_app_model_from_args(options)
    model = active_teacher_model(options)
    min_confidence = float(os.getenv("GEMINI_MIN_CONFIDENCE", "0.60"))
    gold_path = paths["gold"] / "fir_gita_golden_documents.jsonl"
    run_path = paths["reports"] / "gemini_teacher_runs.jsonl"
    failure_path = paths["reports"] / "gemini_failures.jsonl"
    raw_path = paths["reports"] / "gemini_raw_responses.jsonl"
    existing_gold = current_teacher_fir_gita_gold(jsonl_read(gold_path), model)
    gold_by_page = {row["source_page_id"]: row for row in existing_gold}
    existing_runs = jsonl_read(run_path)
    current_model_runs = [row for row in existing_runs if row.get("model") == model]
    current_model_failed_attempts: dict[str, int] = {}
    max_page_attempts = max(1, int(os.getenv("GEMINI_MAX_PAGE_ATTEMPTS", "3")))
    for row in current_model_runs:
        if row.get("ok") or is_quota_status(row) or is_retry_neutral_transient_failure(row):
            continue
        current_model_failed_attempts[row["page_id"]] = current_model_failed_attempts.get(row["page_id"], 0) + int(row.get("attempts") or 0)
    gold_rows: list[dict[str, Any]] = list(gold_by_page.values())
    pending_pages: list[dict[str, Any]] = []
    min_request_interval = float(os.getenv("GEMINI_MIN_REQUEST_INTERVAL_SECONDS", "3.2"))
    quota_cooldown_remaining = quota_cooldown_remaining_seconds(current_model_runs)
    app_credentials_present = (
        bool(os.getenv("GEMINI_WEBAPI_SECURE_1PSID") and os.getenv("GEMINI_WEBAPI_SECURE_1PSIDTS"))
        or bool(getattr(options, "browser_extract_cookies", False))
    )
    can_attempt = not skip_gemini and (
        (transport == "official" and bool(api_key))
        or (transport == "app" and app_credentials_present)
    )
    if quota_cooldown_remaining > 0 and can_attempt:
        print(
            f"Gemini {transport} quota cooldown active; skipping API calls for "
            f"{quota_cooldown_remaining:.1f}s",
            flush=True,
        )
        return gold_rows

    for page in target_pages:
        if page["page_id"] in gold_by_page:
            continue
        if current_model_failed_attempts.get(page["page_id"], 0) >= max_page_attempts:
            continue
        if not can_attempt:
            missing_status = "not_attempted_gemini_disabled"
            if not skip_gemini and transport == "official":
                missing_status = "not_attempted_missing_api_key"
            elif not skip_gemini and transport == "app":
                missing_status = "not_attempted_missing_app_credentials"
            row = {
                "page_id": page["page_id"],
                "status": missing_status,
                "attempts": 0,
                "model": model,
                "transport": "gemini_webapi" if transport == "app" else "official_api",
                "created_at": utc_now(),
            }
            jsonl_append(run_path, row)
            jsonl_append(failure_path, row)
            continue
        pending_pages.append(page)

    pending_pages.sort(key=lambda page: (
        current_model_failed_attempts.get(page["page_id"], 0),
        (ROOT / page["render_path"]).stat().st_size,
        page["page_id"],
    ))
    if max_pages is not None and len(pending_pages) > max_pages:
        pending_pages = pending_pages[:max_pages]
    if not pending_pages:
        return gold_rows

    if transport == "app":
        return asyncio.run(run_gemini_app_annotations(
            paths,
            pending_pages,
            gold_rows,
            gold_by_page,
            model=model,
            app_model=app_model,
            min_confidence=min_confidence,
            min_request_interval=min_request_interval,
            options=options,
        ))

    request_lock = threading.Lock()
    last_request_at = {"value": 0.0}
    concurrency = max(1, int(os.getenv("GEMINI_CONCURRENCY", "4")))
    quota_stop_count = max(0, int(os.getenv("GEMINI_STOP_ON_QUOTA_COUNT", "3")))
    quota_streak = 0
    completed = 0
    total_to_attempt = len(pending_pages)
    next_page_index = 0
    stop_submitting = False

    def submit_available(executor: ThreadPoolExecutor, futures: dict[Any, dict[str, Any]]) -> None:
        nonlocal next_page_index
        while not stop_submitting and len(futures) < concurrency and next_page_index < total_to_attempt:
            page = pending_pages[next_page_index]
            next_page_index += 1
            futures[
                executor.submit(
                    annotate_gemini_page,
                    page,
                    model,
                    api_key,
                    min_confidence,
                    request_lock,
                    last_request_at,
                    min_request_interval,
                )
            ] = page

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[Any, dict[str, Any]] = {}
        submit_available(executor, futures)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                for raw_row in result["raw_rows"]:
                    jsonl_append(raw_path, raw_row)
                jsonl_append(run_path, result["run_row"])
                if result["gold"]:
                    gold_rows.append(result["gold"])
                    gold_by_page[result["gold"]["source_page_id"]] = result["gold"]
                    jsonl_append(gold_path, result["gold"])
                elif result["failure_row"]:
                    jsonl_append(failure_path, result["failure_row"])
                completed += 1

                if is_quota_status(result["run_row"]):
                    quota_streak += 1
                    if quota_stop_count and quota_streak >= quota_stop_count and not stop_submitting:
                        stop_submitting = True
                        print(
                            "Gemini quota early stop after "
                            f"{quota_streak} consecutive quota backoffs "
                            f"({completed}/{total_to_attempt} pages finished)",
                            flush=True,
                        )
                else:
                    quota_streak = 0

                if completed % 10 == 0:
                    print(f"Gemini FIR/Gita attempts {completed}/{total_to_attempt} pending ({len(gold_rows)} accepted total)", flush=True)
            submit_available(executor, futures)

    return gold_rows


def build_sc_gold(
    paths: dict[str, Path],
    pages: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    page_by_id = {page["page_id"]: page for page in pages}
    rows: list[dict[str, Any]] = []
    for align in alignment_rows:
        if not align.get("alignment_verified"):
            continue
        page = page_by_id.get(align["hindi_page_id"])
        if not page or not align.get("english_text_path"):
            continue
        english_text = (ROOT / align["english_text_path"]).read_text(encoding="utf-8").strip()
        if not english_text:
            continue
        rows.append({
            "schema_version": GOLD_SCHEMA_VERSION,
            "record_type": "golden_sc_translation_page",
            "record_id": f"gold:sc_hi_en:{align['alignment_id']}",
            "source_page_id": page["page_id"],
            "source_document_id": page["document_id"],
            "split": {"name": page.get("split"), "group_id": page["split_group_id"], "seed": SPLIT_SEED},
            "source": {
                "corpus": "sc",
                "source_path": page["source_path"],
                "render_path": page["render_path"],
                "page_index": page["page_index"],
                "image_sha256": page["image_sha256"],
                "release_scope": "internal_unredacted",
            },
            "document_metadata": {"doc_type": "supreme_court_report", "page_count": page["page_count"], "citation_key": align["citation_key"]},
            "language": {"primary": "hi-Deva", "all": ["hi-Deva", "en-Latn"], "scripts": ["Deva", "Latn"], "code_mixed": False},
            "page": {
                "page_id": page["page_id"],
                "page_index": page["page_index"],
                "image_metadata": {
                    "width_px": page["image_width_px"],
                    "height_px": page["image_height_px"],
                    "dpi": page["render_dpi"],
                    "bbox_coord_space": "normalized_1000",
                    "render_source": "full_page",
                    "visual_token_budget": page["visual_token_budget"],
                    "is_crop": False,
                },
                "quality_flags": page["quality_flags"],
                "transcription": {"plain_text": "", "markdown": "", "blocks": []},
                "tables": [],
                "key_values": [],
            },
            "pii": {"contains_pii": True, "mode": "internal_unredacted", "entities": []},
            "annotation": {
                "gold_source": "native_english_pdf_text_extraction",
                "created_at": utc_now(),
                "confidence": align["alignment_confidence"],
                "review": {"status": "unreviewed", "reviewer_id": None},
            },
            "translation_targets": [{
                "language": "en-Latn",
                "text": english_text,
                "text_sha256": hashlib.sha256(english_text.encode("utf-8")).hexdigest(),
                "source": "paired_native_english_pdf",
                "english_source_path": align["english_source_path"],
                "english_page_index": align["english_page_index"],
                "alignment_id": align["alignment_id"],
                "alignment_method": align["alignment_method"],
                "alignment_verified": True,
                "gemini_used_for_translation": False,
            }],
            "validation": {"ok": True, "errors": []},
        })
    jsonl_write(paths["gold"] / "sc_hindi_to_english_golden_documents.jsonl", rows)
    return rows


def make_fir_gita_ft_rows(gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold in gold_rows:
        target = {
            "plain_text": gold["page"]["transcription"]["plain_text"],
            "markdown": gold["page"]["transcription"]["markdown"],
            "blocks": gold["page"]["transcription"]["blocks"],
            "key_values": gold["page"]["key_values"],
            "tables": gold["page"]["tables"],
            "quality_flags": gold["page"]["quality_flags"],
            "confidence": gold["annotation"]["confidence"],
        }
        rows.append({
            "schema_version": FT_SCHEMA_VERSION,
            "example_id": "ft:ocr_full_json:" + gold["record_id"],
            "source_record_id": gold["record_id"],
            "source_page_ids": [gold["source_page_id"]],
            "split": gold["split"],
            "task": "ocr_full_json",
            "input": {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "You are an OCR expert for Indian legal and Indic documents. Return valid JSON only."}]},
                    {"role": "user", "content": [
                        {"type": "image", "path": gold["source"]["render_path"]},
                        {"type": "text", "text": "Extract this full page as JSON. Do not translate."},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False, sort_keys=True)}]},
                ],
                "document_context": {
                    "doc_type": gold["document_metadata"]["doc_type"],
                    "page_index": gold["source"]["page_index"],
                    "language_hint": gold["language"]["primary"],
                    "visual_token_budget": gold["page"]["image_metadata"]["visual_token_budget"],
                },
            },
            "target": {"format": "json", "json": target},
            "training": {"loss_on": "assistant_target_only", "sample_weight": 1.0, "pii_mode": "internal_unredacted"},
            "provenance": {"generated_from": GOLD_SCHEMA_VERSION},
            "is_crop": False,
        })
    return rows


def make_sc_ft_rows(gold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold in gold_rows:
        target_info = gold["translation_targets"][0]
        target = {
            "language": "en-Latn",
            "text": target_info["text"],
            "alignment": {
                "alignment_id": target_info["alignment_id"],
                "alignment_method": target_info["alignment_method"],
                "alignment_verified": True,
                "native_english_target": True,
                "gemini_used_for_translation": False,
            },
            "quality_flags": gold["page"]["quality_flags"],
        }
        rows.append({
            "schema_version": FT_SCHEMA_VERSION,
            "example_id": "ft:ocr_translate_en_json:" + gold["record_id"],
            "source_record_id": gold["record_id"],
            "source_page_ids": [gold["source_page_id"]],
            "split": gold["split"],
            "task": "ocr_translate_en_json",
            "input": {
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "You convert a Hindi Supreme Court page image into aligned English text from the paired official report. Return valid JSON only."}]},
                    {"role": "user", "content": [
                        {"type": "image", "path": gold["source"]["render_path"]},
                        {"type": "text", "text": "Read this Hindi page image and return the aligned English text target as JSON."},
                    ]},
                    {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target, ensure_ascii=False, sort_keys=True)}]},
                ],
                "document_context": {
                    "doc_type": "supreme_court_report",
                    "page_index": gold["source"]["page_index"],
                    "language_hint": "hi-Deva",
                    "visual_token_budget": gold["page"]["image_metadata"]["visual_token_budget"],
                },
            },
            "target": {"format": "json", "json": target},
            "training": {"loss_on": "assistant_target_only", "sample_weight": 1.0, "pii_mode": "internal_unredacted"},
            "provenance": {"generated_from": GOLD_SCHEMA_VERSION},
            "is_crop": False,
        })
    return rows


def make_ft_exports(paths: dict[str, Path], fir_gita_gold: list[dict[str, Any]], sc_gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fir_gita_rows = make_fir_gita_ft_rows(fir_gita_gold)
    sc_rows = make_sc_ft_rows(sc_gold)
    all_rows = fir_gita_rows + sc_rows
    jsonl_write(paths["exports"] / "ft_examples_fir_gita_ocr.jsonl", fir_gita_rows)
    jsonl_write(paths["exports"] / "ft_examples_sc_hi_to_en.jsonl", sc_rows)
    jsonl_write(paths["exports"] / "ft_examples_all.jsonl", all_rows)
    return all_rows


def make_fir_gita_only_exports(paths: dict[str, Path], fir_gita_gold: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fir_gita_rows = make_fir_gita_ft_rows(fir_gita_gold)
    jsonl_write(paths["exports"] / "ft_examples_fir_gita_ocr.jsonl", fir_gita_rows)
    jsonl_write(paths["exports"] / "ft_examples_sc_hi_to_en.jsonl", [])
    jsonl_write(paths["exports"] / "ft_examples_all.jsonl", fir_gita_rows)
    return fir_gita_rows


def verify_chat_template(paths: dict[str, Path], ft_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": False,
        "checked_rows": 0,
        "dependency": "transformers",
        "model": os.getenv("GEMMA4_TARGET_MODEL", "google/gemma-4-E2B-it"),
        "validator": "AutoProcessor.apply_chat_template",
    }
    try:
        from transformers import AutoProcessor  # type: ignore
    except Exception as exc:
        result.update({"error": f"transformers_unavailable:{type(exc).__name__}:{exc}"})
        write_json(paths["reports"] / "chat_template_check.json", result)
        return result
    processor = None
    load_errors: list[str] = []
    load_attempts = max(1, int(os.getenv("CHAT_TEMPLATE_PROCESSOR_LOAD_ATTEMPTS", "3")))
    load_sleep = float(os.getenv("CHAT_TEMPLATE_PROCESSOR_LOAD_SLEEP_SECONDS", "5"))
    for attempt in range(1, load_attempts + 1):
        try:
            processor = AutoProcessor.from_pretrained(result["model"], padding_side="left")
            result["processor_load_attempts"] = attempt
            break
        except Exception as exc:
            load_errors.append(f"attempt_{attempt}:{type(exc).__name__}:{exc}"[:1000])
            if attempt < load_attempts:
                time.sleep(load_sleep)
    if processor is None:
        result.update({
            "error": "processor_load_failed:" + load_errors[-1],
            "processor_load_attempts": load_attempts,
            "processor_load_errors": load_errors,
        })
        write_json(paths["reports"] / "chat_template_check.json", result)
        return result
    failures = []
    for row in ft_rows:
        messages = row["input"]["messages"][:-1]
        try:
            rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if not isinstance(rendered, str) or not rendered:
                failures.append({"example_id": row["example_id"], "error": "empty_template"})
        except Exception as exc:
            failures.append({"example_id": row["example_id"], "error": f"{type(exc).__name__}:{exc}"})
    result.update({"ok": not failures, "checked_rows": len(ft_rows), "failures": failures[:100]})
    write_json(paths["reports"] / "chat_template_check.json", result)
    return result


def make_calibration(paths: dict[str, Path], ft_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_rows = [row for row in ft_rows if row.get("split", {}).get("name") == "train" and not row.get("is_crop")]
    train_rows = sorted(train_rows, key=lambda r: hashlib.sha256((SPLIT_SEED + r["example_id"]).encode()).hexdigest())
    selected = train_rows[:256]
    rows = [{
        "calibration_id": "cal:" + row["example_id"],
        "example_id": row["example_id"],
        "source_page_ids": row["source_page_ids"],
        "split": "train",
        "task": row["task"],
        "render_path": next(
            part["path"]
            for msg in row["input"]["messages"]
            for part in msg.get("content", [])
            if part.get("type") == "image"
        ),
        "visual_token_budget": row["input"]["document_context"]["visual_token_budget"],
        "is_crop": False,
    } for row in selected]
    jsonl_write(paths["calibration"] / "pruning_calibration_256.jsonl", rows)
    return rows


def evaluate(
    paths: dict[str, Path],
    pages: list[dict[str, Any]],
    fir_gita_gold: list[dict[str, Any]],
    sc_gold: list[dict[str, Any]],
    ft_rows: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
) -> dict[str, Any]:
    split_by_group: dict[str, set[str]] = {}
    for page in pages:
        split_by_group.setdefault(page["split_group_id"], set()).add(page.get("split"))
    leakage = [group for group, splits in split_by_group.items() if len(splits) > 1]
    report = {
        "created_at": utc_now(),
        "page_count": len(pages),
        "fir_gita_gold_count": len(fir_gita_gold),
        "sc_gold_count": len(sc_gold),
        "ft_count": len(ft_rows),
        "calibration_count": len(calibration),
        "ft_task_counts": {
            "ocr_full_json": sum(1 for row in ft_rows if row["task"] == "ocr_full_json"),
            "ocr_translate_en_json": sum(1 for row in ft_rows if row["task"] == "ocr_translate_en_json"),
        },
        "crop_level_rows": sum(1 for row in ft_rows if row.get("is_crop") or any("crop" in pid for pid in row.get("source_page_ids", []))),
        "split_leakage_failures": leakage,
        "sc_targets_from_gemini": sum(
            1
            for row in ft_rows
            if row["task"] == "ocr_translate_en_json"
            and row["target"]["json"]["alignment"].get("gemini_used_for_translation")
        ),
        "json_validity_rate": sum(1 for r in fir_gita_gold + sc_gold if r.get("validation", {}).get("ok")) / max(1, len(fir_gita_gold) + len(sc_gold)),
        "quality_bucket_counts": quality_bucket_counts(pages),
    }
    write_json(paths["evaluation"] / "smoke_eval.json", report)
    return report


def quality_bucket_counts(pages: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page in pages:
        for flag in page.get("quality_flags", []):
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def is_fir_gita_tracking_row(row: dict[str, Any]) -> bool:
    return row.get("corpus") in FIR_GITA_CORPORA or str(row.get("page_id", "")).startswith(("fir:", "gita:"))


def verify_completion(paths: dict[str, Path], options: argparse.Namespace | None = None) -> dict[str, Any]:
    transport = gemini_transport_from_args(options)
    teacher_model = active_teacher_model(options)
    ingestion = jsonl_read(paths["manifests"] / "ingestion_manifest.jsonl")
    pages = jsonl_read(paths["manifests"] / "page_manifest.jsonl")
    fir_gita_gold_all = jsonl_read(paths["gold"] / "fir_gita_golden_documents.jsonl")
    fir_gita_gold = current_teacher_fir_gita_gold(fir_gita_gold_all, teacher_model)
    sc_gold = jsonl_read(paths["gold"] / "sc_hindi_to_english_golden_documents.jsonl")
    ft_all = jsonl_read(paths["exports"] / "ft_examples_all.jsonl")
    ft_fir_gita = jsonl_read(paths["exports"] / "ft_examples_fir_gita_ocr.jsonl")
    ft_sc = jsonl_read(paths["exports"] / "ft_examples_sc_hi_to_en.jsonl")
    calibration = jsonl_read(paths["calibration"] / "pruning_calibration_256.jsonl")
    gemini_runs = jsonl_read(paths["reports"] / "gemini_teacher_runs.jsonl")
    teacher_runs = [row for row in gemini_runs if row.get("model") == teacher_model]
    gemini_failures = jsonl_read(paths["reports"] / "gemini_failures.jsonl")
    teacher_failures = [row for row in gemini_failures if row.get("model") == teacher_model]
    sc_pairs = jsonl_read(paths["manifests"] / "sc_pdf_pairs.jsonl")
    sc_align = jsonl_read(paths["manifests"] / "sc_page_alignment.jsonl")
    text_manifest = jsonl_read(paths["manifests"] / "sc_text_extraction_manifest.jsonl")
    content_verification = jsonl_read(paths["reports"] / "sc_page_content_verification.jsonl")
    content_verification_summary = read_json(paths["reports"] / "sc_page_content_verification_summary.json", {})
    chat_report = read_json(paths["reports"] / "chat_template_check.json", {})
    eval_report = read_json(paths["evaluation"] / "smoke_eval.json", {})
    org_summary = read_json(paths["manifests"] / "organization_summary.json", {})

    valid_sources = [row for row in ingestion if row.get("valid_source")]
    valid_pages_expected = sum(int(row.get("page_count") or 0) for row in valid_sources)
    missing_render_paths = [page["page_id"] for page in pages if not (ROOT / page["render_path"]).exists()]
    fir_gita_pages = [page for page in pages if page["corpus"] in {"fir", "gita"}]
    attempted_page_ids = {row["page_id"] for row in teacher_runs if row.get("attempts", 0) > 0}
    accepted_fir_gita_ids = {row["source_page_id"] for row in fir_gita_gold if row.get("validation", {}).get("ok")}
    non_current_fir_gita_gold_ids = [
        row.get("source_page_id")
        for row in fir_gita_gold_all
        if row.get("validation", {}).get("ok")
        and fir_gita_gold_teacher_model(row) != teacher_model
    ]
    fir_gita_acceptance_rate = len(accepted_fir_gita_ids) / max(1, len(fir_gita_pages))
    sc_hindi_sources = [row for row in valid_sources if row["corpus"] == "sc" and row.get("language_variant") == "HIN"]
    unpaired_sc_hindi = [row for row in sc_pairs if row.get("hindi_source_path") and not row.get("english_source_path")]
    sc_rows_verified = [
        row for row in ft_sc
        if row["target"]["json"]["alignment"].get("alignment_verified")
        and not row["target"]["json"]["alignment"].get("gemini_used_for_translation")
    ]
    split_by_group: dict[str, set[str]] = {}
    for row in ft_all:
        group = row.get("split", {}).get("group_id")
        split = row.get("split", {}).get("name")
        if group:
            split_by_group.setdefault(group, set()).add(split)
    split_leakage = [group for group, splits in split_by_group.items() if len(splits) > 1]
    crop_rows = [row["example_id"] for row in ft_all if row.get("is_crop") or any("crop" in pid for pid in row.get("source_page_ids", []))]
    verified_sc_alignments = [row for row in sc_align if row.get("alignment_verified")]

    checks = {
        "external_pdf_directories_organized": all(org_summary.get("counts", {}).get(key, 0) > 0 for key in ["fir", "sc_english", "sc_hindi"]),
        "valid_sources_have_manifest_and_renders": len(pages) == valid_pages_expected and not missing_render_paths,
        "every_valid_fir_gita_page_attempted_through_gemini": len(fir_gita_pages) > 0 and {p["page_id"] for p in fir_gita_pages}.issubset(attempted_page_ids),
        "fir_gita_acceptance_rate_at_least_95_percent": bool(fir_gita_pages) and fir_gita_acceptance_rate >= 0.95,
        "every_accepted_fir_gita_record_has_valid_json": bool(fir_gita_gold) and all(r.get("validation", {}).get("ok") for r in fir_gita_gold),
        "fir_gita_gold_uses_active_teacher_model": not non_current_fir_gita_gold_ids,
        "gemini_failures_logged_with_retry_status": all("attempts" in row and "status" in row for row in teacher_failures),
        "every_sc_hindi_pdf_has_paired_english_pdf": len(sc_hindi_sources) > 0 and not unpaired_sc_hindi,
        "every_sc_english_target_from_native_pdf_extraction": bool(ft_sc) and all(
            row["target"]["json"]["alignment"].get("native_english_target") for row in ft_sc
        ),
        "every_sc_row_has_verified_alignment_metadata": bool(ft_sc) and len(sc_rows_verified) == len(ft_sc),
        "no_sc_row_from_unverified_alignment": all(row.get("alignment_verified") for row in sc_align if row.get("hindi_page_id") in {pid for ft in ft_sc for pid in ft["source_page_ids"]}),
        "sc_page_content_verification_present": bool(content_verification) and bool(content_verification_summary),
        "sc_verified_alignment_rows_are_content_flow_verified": bool(verified_sc_alignments) and all(
            row.get("content_relation") == "same_page_likely"
            and row.get("content_verification_id")
            and row.get("alignment_method") == "same_page_content_flow_verified_v1"
            for row in verified_sc_alignments
        ),
        "no_sc_target_produced_by_gemini_translation": eval_report.get("sc_targets_from_gemini", 1) == 0,
        "no_crop_level_rows_exist": not crop_rows and bool(ft_all),
        "split_document_level_and_leakage_free": bool(ft_all) and not split_leakage and not eval_report.get("split_leakage_failures"),
        "gemma4_chat_template_validation_passes": chat_report.get("ok") is True and chat_report.get("checked_rows") == len(ft_all),
        "calibration_pack_256_train_only_full_page_examples": len(calibration) == 256 and all(row.get("split") == "train" and row.get("is_crop") is False for row in calibration),
        "sc_text_extraction_manifest_present": bool(text_manifest),
        "required_named_exports_present": bool(ft_all) and len(ft_all) == len(ft_fir_gita) + len(ft_sc),
    }
    audit = {
        "created_at": utc_now(),
        "checks": checks,
        "all_complete": all(checks.values()),
        "evidence": {
            "ingestion_rows": len(ingestion),
            "valid_sources": len(valid_sources),
            "expected_pages_from_valid_sources": valid_pages_expected,
            "page_manifest_rows": len(pages),
            "missing_render_paths": missing_render_paths[:20],
            "gemini_transport": transport,
            "teacher_model": teacher_model,
            "teacher_run_rows": len(teacher_runs),
            "historical_run_rows": len(gemini_runs),
            "fir_gita_pages": len(fir_gita_pages),
            "gemini_attempted_pages": len(attempted_page_ids),
            "fir_gita_gold_rows": len(fir_gita_gold),
            "non_current_fir_gita_gold_rows": len(non_current_fir_gita_gold_ids),
            "non_current_fir_gita_gold_page_ids": non_current_fir_gita_gold_ids[:20],
            "accepted_fir_gita_page_ids": len(accepted_fir_gita_ids),
            "fir_gita_acceptance_rate": round(fir_gita_acceptance_rate, 6),
            "gemini_failure_rows": len(teacher_failures),
            "sc_hindi_sources": len(sc_hindi_sources),
            "sc_pair_rows": len(sc_pairs),
            "unpaired_sc_hindi": unpaired_sc_hindi[:20],
            "sc_alignment_rows": len(sc_align),
            "sc_verified_alignment_rows": sum(1 for row in sc_align if row.get("alignment_verified")),
            "sc_page_content_verification_rows": len(content_verification),
            "sc_page_content_verification_summary": content_verification_summary,
            "sc_gold_rows": len(sc_gold),
            "ft_rows_all": len(ft_all),
            "ft_rows_fir_gita": len(ft_fir_gita),
            "ft_rows_sc": len(ft_sc),
            "crop_rows": crop_rows[:20],
            "split_leakage_failures": split_leakage[:20],
            "chat_template_report": chat_report,
            "calibration_rows": len(calibration),
        },
    }
    write_json(paths["reports"] / "completion_audit.json", audit)
    return audit


def verify_fir_gita_completion(paths: dict[str, Path], options: argparse.Namespace | None = None) -> dict[str, Any]:
    transport = gemini_transport_from_args(options)
    teacher_model = active_teacher_model(options)
    ingestion = [row for row in jsonl_read(paths["manifests"] / "ingestion_manifest.jsonl") if row.get("corpus") in FIR_GITA_CORPORA]
    pages = [page for page in jsonl_read(paths["manifests"] / "page_manifest.jsonl") if page.get("corpus") in FIR_GITA_CORPORA]
    fir_gita_gold_all = jsonl_read(paths["gold"] / "fir_gita_golden_documents.jsonl")
    fir_gita_gold = current_teacher_fir_gita_gold(fir_gita_gold_all, teacher_model)
    ft_all_path = paths["exports"] / "ft_examples_all.jsonl"
    ft_fir_gita_path = paths["exports"] / "ft_examples_fir_gita_ocr.jsonl"
    ft_sc_path = paths["exports"] / "ft_examples_sc_hi_to_en.jsonl"
    ft_all = jsonl_read(ft_all_path)
    ft_fir_gita = jsonl_read(ft_fir_gita_path)
    ft_sc = jsonl_read(ft_sc_path)
    calibration = jsonl_read(paths["calibration"] / "pruning_calibration_256.jsonl")
    gemini_runs = jsonl_read(paths["reports"] / "gemini_teacher_runs.jsonl")
    teacher_runs = [
        row for row in gemini_runs
        if row.get("model") == teacher_model and is_fir_gita_tracking_row(row)
    ]
    gemini_failures = jsonl_read(paths["reports"] / "gemini_failures.jsonl")
    teacher_failures = [
        row for row in gemini_failures
        if row.get("model") == teacher_model and is_fir_gita_tracking_row(row)
    ]
    chat_report = read_json(paths["reports"] / "chat_template_check.json", {})
    eval_report = read_json(paths["evaluation"] / "smoke_eval.json", {})

    valid_sources = [row for row in ingestion if row.get("valid_source")]
    valid_pages_expected = sum(int(row.get("page_count") or 0) for row in valid_sources)
    missing_render_paths = [page["page_id"] for page in pages if not (ROOT / page["render_path"]).exists()]
    attempted_page_ids = {row["page_id"] for row in teacher_runs if int(row.get("attempts") or 0) > 0}
    accepted_fir_gita_ids = {row["source_page_id"] for row in fir_gita_gold if row.get("validation", {}).get("ok")}
    non_current_fir_gita_gold_ids = [
        row.get("source_page_id")
        for row in fir_gita_gold_all
        if row.get("validation", {}).get("ok")
        and fir_gita_gold_teacher_model(row) != teacher_model
    ]
    fir_gita_acceptance_rate = len(accepted_fir_gita_ids) / max(1, len(pages))
    split_by_group: dict[str, set[str]] = {}
    for row in ft_all:
        group = row.get("split", {}).get("group_id")
        split = row.get("split", {}).get("name")
        if group:
            split_by_group.setdefault(group, set()).add(split)
    split_leakage = [group for group, splits in split_by_group.items() if len(splits) > 1]
    crop_rows = [row["example_id"] for row in ft_all if row.get("is_crop") or any("crop" in pid for pid in row.get("source_page_ids", []))]
    train_full_page_rows = [row for row in ft_all if row.get("split", {}).get("name") == "train" and not row.get("is_crop")]
    expected_calibration_rows = min(256, len(train_full_page_rows))

    checks = {
        "fir_gita_sources_present": bool(valid_sources) and bool(pages),
        "valid_sources_have_manifest_and_renders": len(pages) == valid_pages_expected and not missing_render_paths,
        "every_valid_fir_gita_page_attempted_through_gemini": bool(pages) and {p["page_id"] for p in pages}.issubset(attempted_page_ids),
        "fir_gita_acceptance_rate_at_least_95_percent": bool(pages) and fir_gita_acceptance_rate >= 0.95,
        "every_accepted_fir_gita_record_has_valid_json": bool(fir_gita_gold) and all(r.get("validation", {}).get("ok") for r in fir_gita_gold),
        "fir_gita_gold_uses_active_teacher_model": not non_current_fir_gita_gold_ids,
        "gemini_failures_logged_with_retry_status": all("attempts" in row and "status" in row for row in teacher_failures),
        "fir_gita_export_matches_current_gold": bool(ft_fir_gita) and len(ft_fir_gita) == len(fir_gita_gold),
        "standalone_exports_do_not_include_sc_rows": ft_sc_path.exists() and not ft_sc and all(row.get("task") == "ocr_full_json" for row in ft_all),
        "no_crop_level_rows_exist": not crop_rows and bool(ft_all),
        "split_document_level_and_leakage_free": bool(ft_all) and not split_leakage and not eval_report.get("split_leakage_failures"),
        "gemma4_chat_template_validation_passes": chat_report.get("ok") is True and chat_report.get("checked_rows") == len(ft_all),
        "calibration_pack_train_only_full_page_examples": (
            bool(ft_all)
            and len(calibration) == expected_calibration_rows
            and all(row.get("split") == "train" and row.get("is_crop") is False for row in calibration)
        ),
        "required_named_exports_present": ft_all_path.exists() and ft_fir_gita_path.exists() and ft_sc_path.exists() and len(ft_all) == len(ft_fir_gita),
    }
    audit = {
        "created_at": utc_now(),
        "scope": "fir_gita_only",
        "checks": checks,
        "all_complete": all(checks.values()),
        "evidence": {
            "ingestion_rows": len(ingestion),
            "valid_sources": len(valid_sources),
            "expected_pages_from_valid_sources": valid_pages_expected,
            "page_manifest_rows": len(pages),
            "missing_render_paths": missing_render_paths[:20],
            "gemini_transport": transport,
            "teacher_model": teacher_model,
            "teacher_run_rows": len(teacher_runs),
            "historical_run_rows": len(gemini_runs),
            "fir_gita_pages": len(pages),
            "gemini_attempted_pages": len(attempted_page_ids),
            "fir_gita_gold_rows": len(fir_gita_gold),
            "non_current_fir_gita_gold_rows": len(non_current_fir_gita_gold_ids),
            "non_current_fir_gita_gold_page_ids": non_current_fir_gita_gold_ids[:20],
            "accepted_fir_gita_page_ids": len(accepted_fir_gita_ids),
            "fir_gita_acceptance_rate": round(fir_gita_acceptance_rate, 6),
            "gemini_failure_rows": len(teacher_failures),
            "ft_rows_all": len(ft_all),
            "ft_rows_fir_gita": len(ft_fir_gita),
            "ft_rows_sc": len(ft_sc),
            "crop_rows": crop_rows[:20],
            "split_leakage_failures": split_leakage[:20],
            "chat_template_report": chat_report,
            "train_full_page_rows": len(train_full_page_rows),
            "calibration_rows": len(calibration),
            "expected_calibration_rows": expected_calibration_rows,
        },
    }
    write_json(paths["reports"] / "fir_gita_completion_audit.json", audit)
    return audit


def output_from_args(args: argparse.Namespace) -> Path:
    output = Path(args.output or os.getenv("PIPELINE_OUTPUT_DIR", str(DEFAULT_OUTPUT)))
    if not output.is_absolute():
        output = ROOT / output
    return output


def load_pages_with_existing_splits(paths: dict[str, Path]) -> list[dict[str, Any]]:
    pages = jsonl_read(paths["manifests"] / "page_manifest.jsonl")
    split_rows = jsonl_read(paths["splits"] / "split_assignments.jsonl")
    assignments = {row["split_group_id"]: row["split"] for row in split_rows}
    for page in pages:
        page["split"] = page.get("split") or assignments.get(page["split_group_id"])
    return pages


def rebuild_sc_alignment_from_existing(paths: dict[str, Path], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_rows = jsonl_read(paths["manifests"] / "sc_pdf_pairs.jsonl")
    if not pair_rows:
        raise RuntimeError("Missing sc_pdf_pairs.jsonl; run the full pipeline once before rebuilding SC alignment.")
    text_index = sc_text_index_from_manifest(paths)
    if not text_index:
        raise RuntimeError("Missing sc_text_extraction_manifest.jsonl; run the full pipeline once before rebuilding SC alignment.")
    return build_sc_alignment_from_pairs(paths, pair_rows, pages, text_index)


def run_fir_gita(args: argparse.Namespace) -> int:
    load_dotenv()
    ensure_tools()
    paths = mkdirs(output_from_args(args))
    organize_external_sources(paths)
    smoke_paths = fir_gita_smoke_allowed_paths(args.smoke_fir_docs, args.smoke_gita_pages) if args.smoke else None
    rows, duplicate_groups = discover_files(smoke_paths)
    rows, duplicate_groups = filter_rows_by_corpus(rows, duplicate_groups, FIR_GITA_CORPORA)
    valid_rows = [row for row in rows if row.get("valid_source")]
    if not valid_rows:
        raise RuntimeError("No valid FIR/Gita inputs found. Add FIR PDFs and/or Gita images before running run-fir-gita.")
    if args.smoke:
        duplicate_groups = {}
    write_ingestion_artifacts(paths, rows, duplicate_groups)
    pages = render_sources(paths, rows)
    assignments = assign_splits(paths, rows, pages)
    for page in pages:
        page["split"] = assignments.get(page["split_group_id"])
    jsonl_write(paths["manifests"] / "page_manifest.jsonl", pages)
    fir_gita_gold = gemini_annotate_fir_gita(paths, pages, max_pages=args.gemini_max_pages, skip_gemini=args.skip_gemini, options=args)
    ft_rows = make_fir_gita_only_exports(paths, fir_gita_gold)
    calibration = make_calibration(paths, ft_rows)
    verify_chat_template(paths, ft_rows)
    evaluate(paths, pages, fir_gita_gold, [], ft_rows, calibration)
    audit = verify_fir_gita_completion(paths, args)
    print(json.dumps({"output": str(paths["output"]), "all_complete": audit["all_complete"], "checks": audit["checks"]}, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def run_all(args: argparse.Namespace) -> int:
    load_dotenv()
    ensure_tools()
    paths = mkdirs(output_from_args(args))
    organize_external_sources(paths)
    smoke_paths = smoke_allowed_paths(args.smoke_fir_docs, args.smoke_gita_pages, args.smoke_sc_pairs) if args.smoke else None
    rows, duplicate_groups = discover_files(smoke_paths)
    if args.smoke:
        duplicate_groups = {}
    write_ingestion_artifacts(paths, rows, duplicate_groups)
    pages = render_sources(paths, rows)
    assignments = assign_splits(paths, rows, pages)
    for page in pages:
        page["split"] = assignments.get(page["split_group_id"])
    jsonl_write(paths["manifests"] / "page_manifest.jsonl", pages)
    text_index = extract_sc_text(paths, rows)
    _, alignments = build_sc_pairs_and_alignment(paths, rows, pages, text_index)
    fir_gita_gold = gemini_annotate_fir_gita(paths, pages, max_pages=args.gemini_max_pages, skip_gemini=args.skip_gemini, options=args)
    sc_gold = build_sc_gold(paths, pages, alignments)
    ft_rows = make_ft_exports(paths, fir_gita_gold, sc_gold)
    calibration = make_calibration(paths, ft_rows)
    verify_chat_template(paths, ft_rows)
    evaluate(paths, pages, fir_gita_gold, sc_gold, ft_rows, calibration)
    audit = verify_completion(paths, args)
    print(json.dumps({"output": str(paths["output"]), "all_complete": audit["all_complete"], "checks": audit["checks"]}, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def run_organize(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    summary = organize_external_sources(paths)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not summary.get("conflicts") else 2


def run_verify(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    audit = verify_completion(paths, args)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def run_verify_fir_gita(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    audit = verify_fir_gita_completion(paths, args)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def run_status(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    teacher_model = active_teacher_model(args)
    audit = verify_completion(paths, args)
    pages = jsonl_read(paths["manifests"] / "page_manifest.jsonl")
    runs = jsonl_read(paths["reports"] / "gemini_teacher_runs.jsonl")
    teacher_runs = [row for row in runs if row.get("model") == teacher_model]
    gold = current_teacher_fir_gita_gold(jsonl_read(paths["gold"] / "fir_gita_golden_documents.jsonl"), teacher_model)
    sc_content_summary = read_json(paths["reports"] / "sc_page_content_verification_summary.json", {})
    fir_gita_pages = [page for page in pages if page.get("corpus") in {"fir", "gita"}]
    accepted_page_ids = {row.get("source_page_id") for row in gold if row.get("validation", {}).get("ok")}
    attempted_page_ids = {row.get("page_id") for row in teacher_runs if int(row.get("attempts") or 0) > 0}
    target_95 = int(len(fir_gita_pages) * 0.95)
    if target_95 / max(1, len(fir_gita_pages)) < 0.95:
        target_95 += 1
    recent = teacher_runs[-int(args.latest):] if args.latest else teacher_runs
    quota_rows = [row for row in teacher_runs if is_quota_status(row)]
    latest_quota = quota_rows[-1] if quota_rows else {}
    latest_quota_age_seconds = None
    latest_quota_retry_after = None
    latest_quota_wait_recommended_seconds = None
    if latest_quota.get("created_at"):
        try:
            created_at = datetime.fromisoformat(str(latest_quota["created_at"]))
            latest_quota_age_seconds = round((datetime.now(timezone.utc) - created_at).total_seconds(), 1)
        except ValueError:
            latest_quota_age_seconds = None
    if latest_quota:
        latest_quota_retry_after = latest_quota.get("retry_after_seconds") or retry_after_seconds(latest_quota)
        if latest_quota_age_seconds is not None:
            latest_quota_wait_recommended_seconds = max(0.0, round(float(latest_quota_retry_after) - latest_quota_age_seconds, 1))
    latest_quota_detail = quota_detail_fields(latest_quota) if latest_quota else {}
    status = {
        "output": str(paths["output"]),
        "gemini_transport": gemini_transport_from_args(args),
        "all_complete": audit.get("all_complete"),
        "failed_checks": [key for key, ok in audit.get("checks", {}).items() if not ok],
        "fir_gita_pages": len(fir_gita_pages),
        "accepted_fir_gita_pages": len(accepted_page_ids),
        "fir_gita_acceptance_rate": round(len(accepted_page_ids) / max(1, len(fir_gita_pages)), 6),
        "accepted_needed_for_95_percent": max(0, target_95 - len(accepted_page_ids)),
        "unattempted_fir_gita_pages": len({page["page_id"] for page in fir_gita_pages} - attempted_page_ids),
        "gemini_attempted_pages": len(attempted_page_ids),
        "recent_rows": len(recent),
        "recent_statuses": dict(Counter(str(row.get("status") or "missing") for row in recent)),
        "recent_error_types": dict(Counter(str(row.get("error_type") or "missing") for row in recent if not row.get("ok"))),
        "teacher_model": teacher_model,
        "teacher_run_rows": len(teacher_runs),
        "historical_run_rows": len(runs),
        "quota_early_stop_count": int(os.getenv("GEMINI_STOP_ON_QUOTA_COUNT", "3")),
        "quota_cooldown_seconds": float(os.getenv("GEMINI_MIN_QUOTA_COOLDOWN_SECONDS", "0")),
        "quota_cooldown_remaining_seconds": quota_cooldown_remaining_seconds(teacher_runs),
        "latest_quota_created_at": latest_quota.get("created_at"),
        "latest_quota_age_seconds": latest_quota_age_seconds,
        "latest_quota_retry_after_seconds": latest_quota_retry_after,
        "latest_quota_wait_recommended_seconds": latest_quota_wait_recommended_seconds,
        "latest_quota_metric": latest_quota_detail.get("metric"),
        "latest_quota_limit": latest_quota_detail.get("limit"),
        "sc_page_content_verification": {
            "row_count": sc_content_summary.get("row_count"),
            "same_information_likely_rows": sc_content_summary.get("same_information_likely_rows"),
            "needs_manual_review_rows": sc_content_summary.get("needs_manual_review_rows"),
            "relation_counts": sc_content_summary.get("relation_counts"),
        },
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["all_complete"] else 2


def run_status_fir_gita(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    teacher_model = active_teacher_model(args)
    audit = verify_fir_gita_completion(paths, args)
    pages = [page for page in jsonl_read(paths["manifests"] / "page_manifest.jsonl") if page.get("corpus") in FIR_GITA_CORPORA]
    runs = jsonl_read(paths["reports"] / "gemini_teacher_runs.jsonl")
    teacher_runs = [row for row in runs if row.get("model") == teacher_model and is_fir_gita_tracking_row(row)]
    gold = current_teacher_fir_gita_gold(jsonl_read(paths["gold"] / "fir_gita_golden_documents.jsonl"), teacher_model)
    accepted_page_ids = {row.get("source_page_id") for row in gold if row.get("validation", {}).get("ok")}
    attempted_page_ids = {row.get("page_id") for row in teacher_runs if int(row.get("attempts") or 0) > 0}
    target_95 = int(len(pages) * 0.95)
    if target_95 / max(1, len(pages)) < 0.95:
        target_95 += 1
    recent = teacher_runs[-int(args.latest):] if args.latest else teacher_runs
    quota_rows = [row for row in teacher_runs if is_quota_status(row)]
    latest_quota = quota_rows[-1] if quota_rows else {}
    latest_quota_age_seconds = None
    latest_quota_retry_after = None
    latest_quota_wait_recommended_seconds = None
    if latest_quota.get("created_at"):
        try:
            created_at = datetime.fromisoformat(str(latest_quota["created_at"]))
            latest_quota_age_seconds = round((datetime.now(timezone.utc) - created_at).total_seconds(), 1)
        except ValueError:
            latest_quota_age_seconds = None
    if latest_quota:
        latest_quota_retry_after = latest_quota.get("retry_after_seconds") or retry_after_seconds(latest_quota)
        if latest_quota_age_seconds is not None:
            latest_quota_wait_recommended_seconds = max(0.0, round(float(latest_quota_retry_after) - latest_quota_age_seconds, 1))
    latest_quota_detail = quota_detail_fields(latest_quota) if latest_quota else {}
    status = {
        "output": str(paths["output"]),
        "scope": "fir_gita_only",
        "gemini_transport": gemini_transport_from_args(args),
        "all_complete": audit.get("all_complete"),
        "failed_checks": [key for key, ok in audit.get("checks", {}).items() if not ok],
        "fir_gita_pages": len(pages),
        "accepted_fir_gita_pages": len(accepted_page_ids),
        "fir_gita_acceptance_rate": round(len(accepted_page_ids) / max(1, len(pages)), 6),
        "accepted_needed_for_95_percent": max(0, target_95 - len(accepted_page_ids)),
        "unattempted_fir_gita_pages": len({page["page_id"] for page in pages} - attempted_page_ids),
        "gemini_attempted_pages": len(attempted_page_ids),
        "recent_rows": len(recent),
        "recent_statuses": dict(Counter(str(row.get("status") or "missing") for row in recent)),
        "recent_error_types": dict(Counter(str(row.get("error_type") or "missing") for row in recent if not row.get("ok"))),
        "teacher_model": teacher_model,
        "teacher_run_rows": len(teacher_runs),
        "historical_run_rows": len(runs),
        "quota_early_stop_count": int(os.getenv("GEMINI_STOP_ON_QUOTA_COUNT", "3")),
        "quota_cooldown_seconds": float(os.getenv("GEMINI_MIN_QUOTA_COOLDOWN_SECONDS", "0")),
        "quota_cooldown_remaining_seconds": quota_cooldown_remaining_seconds(teacher_runs),
        "latest_quota_created_at": latest_quota.get("created_at"),
        "latest_quota_age_seconds": latest_quota_age_seconds,
        "latest_quota_retry_after_seconds": latest_quota_retry_after,
        "latest_quota_wait_recommended_seconds": latest_quota_wait_recommended_seconds,
        "latest_quota_metric": latest_quota_detail.get("metric"),
        "latest_quota_limit": latest_quota_detail.get("limit"),
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["all_complete"] else 2


def run_verify_sc_alignment(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    pages = jsonl_read(paths["manifests"] / "page_manifest.jsonl")
    if not pages:
        raise RuntimeError("Missing page_manifest.jsonl; run the full pipeline once before verify-sc-alignment.")
    alignments = rebuild_sc_alignment_from_existing(paths, pages)
    summary = read_json(paths["reports"] / "sc_page_content_verification_summary.json", {})
    result = {
        "output": str(paths["output"]),
        "alignment_rows": len(alignments),
        "verified_alignment_rows": sum(1 for row in alignments if row.get("alignment_verified")),
        "content_verification_summary": summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_resume_fir_gita(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    pages = load_pages_with_existing_splits(paths)
    if not pages:
        raise RuntimeError("Missing page_manifest.jsonl; run run-fir-gita once before resume-fir-gita.")
    fir_gita_pages = [page for page in pages if page.get("corpus") in FIR_GITA_CORPORA]
    if not fir_gita_pages:
        raise RuntimeError("No FIR/Gita pages found in page_manifest.jsonl.")
    fir_gita_gold = gemini_annotate_fir_gita(paths, fir_gita_pages, max_pages=args.gemini_max_pages, skip_gemini=args.skip_gemini, options=args)
    ft_rows = make_fir_gita_only_exports(paths, fir_gita_gold)
    calibration = make_calibration(paths, ft_rows)
    verify_chat_template(paths, ft_rows)
    evaluate(paths, fir_gita_pages, fir_gita_gold, [], ft_rows, calibration)
    audit = verify_fir_gita_completion(paths, args)
    print(json.dumps({"output": str(paths["output"]), "all_complete": audit["all_complete"], "checks": audit["checks"]}, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def run_resume_gemini(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    pages = jsonl_read(paths["manifests"] / "page_manifest.jsonl")
    if not pages:
        raise RuntimeError("Missing page_manifest.jsonl; run the full pipeline once before resume-gemini.")
    split_rows = jsonl_read(paths["splits"] / "split_assignments.jsonl")
    assignments = {row["split_group_id"]: row["split"] for row in split_rows}
    for page in pages:
        page["split"] = page.get("split") or assignments.get(page["split_group_id"])
    alignments = rebuild_sc_alignment_from_existing(paths, pages)
    fir_gita_gold = gemini_annotate_fir_gita(paths, pages, max_pages=args.gemini_max_pages, skip_gemini=args.skip_gemini, options=args)
    sc_gold = build_sc_gold(paths, pages, alignments)
    ft_rows = make_ft_exports(paths, fir_gita_gold, sc_gold)
    calibration = make_calibration(paths, ft_rows)
    verify_chat_template(paths, ft_rows)
    evaluate(paths, pages, fir_gita_gold, sc_gold, ft_rows, calibration)
    audit = verify_completion(paths, args)
    print(json.dumps({"output": str(paths["output"]), "all_complete": audit["all_complete"], "checks": audit["checks"]}, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def run_finalize_fir_gita(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    teacher_model = active_teacher_model(args)
    pages = load_pages_with_existing_splits(paths)
    if not pages:
        raise RuntimeError("Missing page_manifest.jsonl; run run-fir-gita once before finalize-fir-gita.")
    fir_gita_pages = [page for page in pages if page.get("corpus") in FIR_GITA_CORPORA]
    if not fir_gita_pages:
        raise RuntimeError("No FIR/Gita pages found in page_manifest.jsonl.")
    fir_gita_gold = current_teacher_fir_gita_gold(jsonl_read(paths["gold"] / "fir_gita_golden_documents.jsonl"), teacher_model)
    ft_rows = make_fir_gita_only_exports(paths, fir_gita_gold)
    calibration = make_calibration(paths, ft_rows)
    verify_chat_template(paths, ft_rows)
    evaluate(paths, fir_gita_pages, fir_gita_gold, [], ft_rows, calibration)
    audit = verify_fir_gita_completion(paths, args)
    print(json.dumps({"output": str(paths["output"]), "all_complete": audit["all_complete"], "checks": audit["checks"]}, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def run_finalize_existing(args: argparse.Namespace) -> int:
    load_dotenv()
    paths = mkdirs(output_from_args(args))
    teacher_model = active_teacher_model(args)
    pages = jsonl_read(paths["manifests"] / "page_manifest.jsonl")
    if not pages:
        raise RuntimeError("Missing page_manifest.jsonl; run the full pipeline once before finalize-existing.")
    split_rows = jsonl_read(paths["splits"] / "split_assignments.jsonl")
    assignments = {row["split_group_id"]: row["split"] for row in split_rows}
    for page in pages:
        page["split"] = page.get("split") or assignments.get(page["split_group_id"])
    alignments = rebuild_sc_alignment_from_existing(paths, pages)
    fir_gita_gold = current_teacher_fir_gita_gold(jsonl_read(paths["gold"] / "fir_gita_golden_documents.jsonl"), teacher_model)
    sc_gold = build_sc_gold(paths, pages, alignments)
    ft_rows = make_ft_exports(paths, fir_gita_gold, sc_gold)
    calibration = make_calibration(paths, ft_rows)
    verify_chat_template(paths, ft_rows)
    evaluate(paths, pages, fir_gita_gold, sc_gold, ft_rows, calibration)
    audit = verify_completion(paths, args)
    print(json.dumps({"output": str(paths["output"]), "all_complete": audit["all_complete"], "checks": audit["checks"]}, indent=2, sort_keys=True))
    return 0 if audit["all_complete"] else 2


def add_gemini_transport_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gemini-transport",
        choices=["app", "official"],
        default=None,
        help="Gemini transport for FIR/Gita OCR. Defaults to GEMINI_TRANSPORT or app.",
    )
    parser.add_argument(
        "--gemini-app-model",
        default=None,
        help="Gemini App model selector. Defaults to GEMINI_APP_MODEL or auto-flash.",
    )
    parser.add_argument("--gemini-app-init-timeout", type=float, default=60)
    parser.add_argument("--gemini-app-request-timeout", type=float, default=300)
    parser.add_argument("--gemini-app-proxy", default=None)
    parser.add_argument(
        "--browser-extract-cookies",
        action="store_true",
        help="Extract Gemini App cookies from the local CDP browser before annotation.",
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Output artifact directory. Defaults to PIPELINE_OUTPUT_DIR or artifacts/page_only_v1.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("organize")
    all_p = sub.add_parser("all")
    add_gemini_transport_args(all_p)
    all_p.add_argument("--smoke", action="store_true", help="Run the small plan smoke subset instead of the full corpus.")
    all_p.add_argument("--smoke-fir-docs", type=int, default=2)
    all_p.add_argument("--smoke-gita-pages", type=int, default=2)
    all_p.add_argument("--smoke-sc-pairs", type=int, default=2)
    all_p.add_argument("--gemini-max-pages", type=int, default=None, help="Cap FIR/Gita Gemini attempts for controlled tests.")
    all_p.add_argument("--skip-gemini", action="store_true", help="Do not call Gemini; audit will remain incomplete.")
    fir_gita_p = sub.add_parser("run-fir-gita", help="Run only the FIR/Gita synthetic OCR pipeline.")
    add_gemini_transport_args(fir_gita_p)
    fir_gita_p.add_argument("--smoke", action="store_true", help="Run a small FIR/Gita-only smoke subset.")
    fir_gita_p.add_argument("--smoke-fir-docs", type=int, default=2)
    fir_gita_p.add_argument("--smoke-gita-pages", type=int, default=2)
    fir_gita_p.add_argument("--gemini-max-pages", type=int, default=None, help="Cap FIR/Gita Gemini attempts for controlled tests.")
    fir_gita_p.add_argument("--skip-gemini", action="store_true", help="Do not call Gemini; audit will remain incomplete.")
    verify_p = sub.add_parser("verify")
    add_gemini_transport_args(verify_p)
    verify_fir_gita_p = sub.add_parser("verify-fir-gita", help="Verify a FIR/Gita-only output directory.")
    add_gemini_transport_args(verify_fir_gita_p)
    sub.add_parser("verify-sc-alignment")
    report_verify_p = sub.add_parser("report-sc-verification")
    report_verify_p.add_argument("--report-name", default="sc_page_content_verification_report.md", help="Full deterministic SC verification report filename under reports.")
    report_verify_p.add_argument("--pair-summary-name", default="sc_page_content_verification_pair_summary.jsonl", help="Pair summary filename under reports.")
    report_verify_p.add_argument("--review-manifest-name", default="sc_page_content_verification_review_manifest.jsonl", help="Review manifest filename under reports.")
    report_verify_p.add_argument("--max-examples", type=int, default=25, help="Number of review-needed rows to show in the Markdown report.")
    status_p = sub.add_parser("status")
    add_gemini_transport_args(status_p)
    status_p.add_argument("--latest", type=int, default=50, help="Number of latest Gemini run rows to summarize.")
    status_fir_gita_p = sub.add_parser("status-fir-gita", help="Summarize a FIR/Gita-only run.")
    add_gemini_transport_args(status_fir_gita_p)
    status_fir_gita_p.add_argument("--latest", type=int, default=50, help="Number of latest Gemini run rows to summarize.")
    resume_p = sub.add_parser("resume-gemini")
    add_gemini_transport_args(resume_p)
    resume_p.add_argument("--gemini-max-pages", type=int, default=None, help="Cap FIR/Gita Gemini attempts for controlled tests.")
    resume_p.add_argument("--skip-gemini", action="store_true", help="Do not call Gemini; audit will remain incomplete.")
    resume_fir_gita_p = sub.add_parser("resume-fir-gita", help="Resume only FIR/Gita Gemini OCR annotation.")
    add_gemini_transport_args(resume_fir_gita_p)
    resume_fir_gita_p.add_argument("--gemini-max-pages", type=int, default=None, help="Cap FIR/Gita Gemini attempts for controlled tests.")
    resume_fir_gita_p.add_argument("--skip-gemini", action="store_true", help="Do not call Gemini; audit will remain incomplete.")
    finalize_existing_p = sub.add_parser("finalize-existing")
    add_gemini_transport_args(finalize_existing_p)
    finalize_fir_gita_p = sub.add_parser("finalize-fir-gita", help="Rebuild only FIR/Gita exports from existing Gemini gold.")
    add_gemini_transport_args(finalize_fir_gita_p)
    args = parser.parse_args(argv)
    if args.command == "organize":
        return run_organize(args)
    if args.command == "all":
        return run_all(args)
    if args.command == "run-fir-gita":
        return run_fir_gita(args)
    if args.command == "verify":
        return run_verify(args)
    if args.command == "verify-fir-gita":
        return run_verify_fir_gita(args)
    if args.command == "verify-sc-alignment":
        return run_verify_sc_alignment(args)
    if args.command == "report-sc-verification":
        return run_report_sc_verification(args)
    if args.command == "status":
        return run_status(args)
    if args.command == "status-fir-gita":
        return run_status_fir_gita(args)
    if args.command == "resume-gemini":
        return run_resume_gemini(args)
    if args.command == "resume-fir-gita":
        return run_resume_fir_gita(args)
    if args.command == "finalize-existing":
        return run_finalize_existing(args)
    if args.command == "finalize-fir-gita":
        return run_finalize_fir_gita(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
