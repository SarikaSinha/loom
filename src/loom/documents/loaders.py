from __future__ import annotations

import os
import stat
import unicodedata
import zipfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

from loom.contracts import (
    Diagnostic,
    DOCXLocation,
    Location,
    MarkdownLocation,
    PDFLocation,
    ToolFailure,
)


def normalize(text: str) -> str:
    return unicodedata.normalize(
        "NFC",
        "\n".join(
            line.rstrip()
            for line in text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("\t", " ")
            .split("\n")
        ),
    )


def source_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise ToolFailure("source_symlink")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ToolFailure("source_not_regular")
        if info.st_size > 25 * 1024**2:
            raise ToolFailure("source_too_large")
        data = stream.read(25 * 1024**2 + 1)
        if len(data) > 25 * 1024**2:
            raise ToolFailure("source_too_large")
        return data


def blocks(
    path: Path,
    diagnostics: list[Diagnostic],
    *,
    data: bytes | None = None,
) -> Iterator[tuple[str, Location]]:
    data = source_bytes(path) if data is None else data
    if len(data) > 25 * 1024**2:
        raise ToolFailure("source_too_large")
    suffix = path.suffix.lower()
    if suffix == ".md":
        yield from markdown(data)
    elif suffix == ".pdf":
        from pypdf.errors import PyPdfError

        try:
            yield from pdf(data, diagnostics)
        except PyPdfError:
            raise ToolFailure("pdf_parse_failed") from None
    elif suffix == ".docx":
        from docx.opc.exceptions import PackageNotFoundError

        try:
            yield from docx(data, diagnostics)
        except (zipfile.BadZipFile, PackageNotFoundError, SyntaxError, KeyError):
            raise ToolFailure("docx_parse_failed") from None
    else:
        raise ToolFailure("unsupported_format")


def markdown(data: bytes) -> Iterator[tuple[str, Location]]:
    from markdown_it import MarkdownIt

    text = data.decode("utf-8")
    tokens = MarkdownIt("commonmark", {"html": False}).parse(text)
    headings: dict[int, str] = {}
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag[1:])
            headings = {depth: heading for depth, heading in headings.items() if depth < level}
            headings[level] = tokens[index + 1].content
        if token.type not in {"inline", "fence", "code_block"}:
            continue
        content = token.content
        if token.type == "inline" and token.children:
            content = "".join(
                child.content
                if child.type in {"text", "code_inline"}
                else "\n"
                if child.type in {"softbreak", "hardbreak"}
                else ""
                for child in token.children
            )
        location = MarkdownLocation(
            section=" / ".join(headings.values()) or None,
            line_start=token.map[0] + 1 if token.map else 1,
        )
        yield normalize(content), location


def pdf(data: bytes, diagnostics: list[Diagnostic]) -> Iterator[tuple[str, Location]]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    reader = PdfReader(BytesIO(data))
    if reader.is_encrypted:
        raise ToolFailure("encrypted_pdf")
    if len(reader.pages) > 2000:
        raise ToolFailure("pdf_page_limit")
    extracted = False
    for page_number, page in enumerate(reader.pages, 1):
        try:
            contents = page.get_contents()
            if contents is not None and len(contents.get_data()) > 16 * 1024**2:
                raise ToolFailure("pdf_stream_limit")
            text = normalize(page.extract_text())
        except (PyPdfError, ValueError, TypeError, KeyError):
            diagnostics.append(
                Diagnostic(
                    code="page_extraction_failed",
                    message="A PDF page could not be extracted.",
                    origin="index",
                )
            )
            continue
        if text.strip():
            extracted = True
            yield text, PDFLocation(page=page_number)
        else:
            diagnostics.append(
                Diagnostic(
                    code="ocr_required",
                    message="A PDF page has no extractable text.",
                    origin="index",
                )
            )
    if not extracted:
        raise ToolFailure("ocr_required")


def docx(data: bytes, diagnostics: list[Diagnostic]) -> Iterator[tuple[str, Location]]:
    from docx import Document
    from docx.table import Table

    with zipfile.ZipFile(BytesIO(data)) as archive:
        entries = archive.infolist()
        if sum(entry.file_size for entry in entries) > 100 * 1024**2:
            raise ToolFailure("docx_uncompressed_limit")
        if any(
            entry.filename.startswith(
                ("word/media/", "word/embeddings/", "word/header", "word/footer")
            )
            or "vbaProject" in entry.filename
            for entry in entries
        ):
            diagnostics.append(
                Diagnostic(
                    code="non_text_content_excluded",
                    message="Images, embedded content or headers/footers are not indexed.",
                    origin="index",
                )
            )
    document = Document(BytesIO(data))
    section: str | None = None
    for number, item in enumerate(document.iter_inner_content(), 1):
        if isinstance(item, Table):
            content = "\n".join("\t".join(cell.text for cell in row.cells) for row in item.rows)
        else:
            content = item.text
            if (
                item.style is not None
                and item.style.name is not None
                and item.style.name.startswith("Heading")
            ):
                section = content
        yield normalize(content), DOCXLocation(section=section, block=number)
