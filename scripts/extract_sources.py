#!/usr/bin/env python3
"""Extract text from source files listed in inputs/intake.json into extracted/.

Supports: PDF (pypdf), DOCX (zip + XML), TXT/MD/TEX/PY.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INTAKE_PATH = ROOT / "inputs" / "intake.json"
OUT_DIR = ROOT / "extracted"


def _safe_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def extract_docx(path: Path) -> str:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    # Replace paragraph and line breaks with newlines
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    # Drop remaining tags
    text = re.sub(r"<[^>]+>", "", xml)
    text = html.unescape(text)
    # Normalize whitespace a bit
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "pypdf is required for PDF extraction. Install into .venv_extract and rerun."
        ) from exc

    reader = PdfReader(str(path))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    combined = "\n\n".join(chunks)
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined.strip()


def extract_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def extract_file(path: Path) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf", extract_pdf(path)
    if suffix == ".docx":
        return "docx", extract_docx(path)
    if suffix in {".txt", ".md", ".tex", ".py"}:
        return suffix.lstrip("."), extract_plain(path)
    raise ValueError(f"Unsupported extension: {suffix}")


def main() -> int:
    if not INTAKE_PATH.exists():
        print(f"Missing intake file: {INTAKE_PATH}")
        return 1
    data = json.loads(INTAKE_PATH.read_text())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = []
    for _, files in data.get("files", {}).items():
        sources.extend(files)

    seen = set()
    results = []
    for rel in sources:
        if rel in seen:
            continue
        seen.add(rel)
        path = ROOT / rel
        if not path.exists():
            print(f"SKIP missing: {rel}")
            continue
        try:
            file_type, text = extract_file(path)
        except ValueError:
            print(f"SKIP unsupported: {rel}")
            continue
        except Exception as exc:
            print(f"ERROR {rel}: {exc}")
            continue

        if not text:
            print(f"WARN empty extraction: {rel}")
            continue

        out_name = _safe_slug(path.name) + ".txt"
        out_path = OUT_DIR / out_name
        header = f"Source: {path.name}\nType: {file_type}\n\n"
        out_path.write_text(header + text, encoding="utf-8")
        results.append({"source": rel, "output": str(out_path.relative_to(ROOT))})
        print(f"OK {rel} -> {out_path.relative_to(ROOT)}")

    index_path = OUT_DIR / "index.json"
    index_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote index: {index_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
