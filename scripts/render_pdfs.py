#!/usr/bin/env python3
"""Render simple PDFs from markdown files using fpdf2.

Usage:
  python scripts/render_pdfs.py cv_tailored.md cover_letter.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF  # type: ignore

ROOT = Path(__file__).resolve().parents[1]


class SimplePDF(FPDF):
    def header(self):
        return

    def footer(self):
        return


def sanitize_text(text: str) -> str:
    # Replace unsupported unicode bullets/dashes for core fonts
    return (
        text.replace("\u2022", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2011", "-")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", "\"")
        .replace("\u201d", "\"")
        .replace("\u00a0", " ")
    )


def parse_markdown(text: str):
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = sanitize_text(lines[i].rstrip())
        if not line.strip():
            i += 1
            continue
        if line.startswith("#"):
            m = re.match(r"^(#+)\s+(.*)$", line)
            if m:
                level = len(m.group(1))
                blocks.append(("heading", level, m.group(2).strip()))
                i += 1
                continue
        if line.lstrip().startswith("- "):
            bullets = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                bullets.append(sanitize_text(lines[i].lstrip()[2:].strip()))
                i += 1
            blocks.append(("bullets", bullets))
            continue
        # paragraph
        para = [line.strip()]
        i += 1
        while i < len(lines):
            nxt = sanitize_text(lines[i].rstrip())
            if not nxt.strip() or nxt.startswith("#") or nxt.lstrip().startswith("- "):
                break
            para.append(nxt.strip())
            i += 1
        blocks.append(("para", " ".join(para)))
    return blocks


def render_pdf(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding="utf-8")
    blocks = parse_markdown(text)

    pdf = SimplePDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    for block in blocks:
        kind = block[0]
        if kind == "heading":
            _, level, content = block
            if level == 1:
                size = 16
            elif level == 2:
                size = 13
            else:
                size = 12
            pdf.set_font("Helvetica", "B", size)
            pdf.multi_cell(0, 6, content)
            pdf.ln(1)
        elif kind == "bullets":
            _, bullets = block
            pdf.set_font("Helvetica", "", 11)
            for item in bullets:
                pdf.set_x(pdf.l_margin)
                # bullet indent
                pdf.cell(4, 5, "-")
                pdf.multi_cell(0, 5, item)
            pdf.ln(1)
        else:
            _, content = block
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 5, content)
            pdf.ln(1)

    pdf.output(str(pdf_path))


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/render_pdfs.py <file1.md> [file2.md ...]")
        return 1
    for md in argv[1:]:
        md_path = (ROOT / md).resolve()
        if not md_path.exists():
            print(f"Missing markdown file: {md}")
            return 1
        pdf_path = md_path.with_suffix(".pdf")
        render_pdf(md_path, pdf_path)
        print(f"Rendered {pdf_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
