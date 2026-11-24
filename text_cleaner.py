from __future__ import annotations

import io
import re
from typing import Any

from docx import Document
from pypdf import PdfReader


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str,
    fix_hyphens: bool = True,
    remove_page_nums: bool = True,
    fix_lines: bool = True,
    clean_spaces: bool = True,
) -> str:
    """
    Clean extracted document text before chunking.
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")

    if fix_hyphens:
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    if remove_page_nums:
        text = re.sub(
            r"(?m)^\s*(?:Page\s*\d+(?:\s*of\s*\d+)?|\d{1,3})\s*$",
            "",
            text,
        )

    if fix_lines:
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    if clean_spaces:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# FILE EXTRACTORS (PDF, TXT, MD, DOCX)
# ============================================================

def extract_pdf(uploaded_file, cleaning_options: dict[str, bool]) -> list[dict[str, Any]]:
    """
    Extract one text record per PDF page so page citations survive.
    """
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(
            page.extract_text() or "",
            fix_hyphens=cleaning_options.get("fix_hyphens", True),
            remove_page_nums=cleaning_options.get("remove_page_nums", True),
            fix_lines=cleaning_options.get("fix_lines", True),
            clean_spaces=cleaning_options.get("clean_spaces", True),
        )
        if text:
            records.append(
                {
                    "source": uploaded_file.name,
                    "page": page_number,
                    "text": text,
                }
            )

    return records


def extract_text_file(uploaded_file, cleaning_options: dict[str, bool]) -> list[dict[str, Any]]:
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = str(raw)

    text = clean_text(
        text,
        fix_hyphens=cleaning_options.get("fix_hyphens", True),
        remove_page_nums=cleaning_options.get("remove_page_nums", True),
        fix_lines=cleaning_options.get("fix_lines", True),
        clean_spaces=cleaning_options.get("clean_spaces", True),
    )

    return [
        {
            "source": uploaded_file.name,
            "page": None,
            "text": text,
        }
    ] if text else []


def extract_docx(uploaded_file, cleaning_options: dict[str, bool]) -> list[dict[str, Any]]:
    uploaded_file.seek(0)
    doc = Document(io.BytesIO(uploaded_file.read()))

    text = "\n".join(
        paragraph.text
        for paragraph in doc.paragraphs
        if paragraph.text.strip()
    )

    text = clean_text(
        text,
        fix_hyphens=cleaning_options.get("fix_hyphens", True),
        remove_page_nums=cleaning_options.get("remove_page_nums", True),
        fix_lines=cleaning_options.get("fix_lines", True),
        clean_spaces=cleaning_options.get("clean_spaces", True),
    )

    return [
        {
            "source": uploaded_file.name,
            "page": None,
            "text": text,
        }
    ] if text else []


def extract_uploaded_file(uploaded_file, cleaning_options: dict[str, bool]) -> list[dict[str, Any]]:
    extension = uploaded_file.name.rsplit(".", 1).pop(-1).lower()

    if extension == "pdf":
        return extract_pdf(uploaded_file, cleaning_options)

    if extension in {"txt", "md"}:
        return extract_text_file(uploaded_file, cleaning_options)

    if extension == "docx":
        return extract_docx(uploaded_file, cleaning_options)

    raise ValueError(f"Unsupported file type: {extension}")
