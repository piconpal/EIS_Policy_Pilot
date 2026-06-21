"""
chunker.py — Step 3 (Hybrid Strategy)
Splits extracted page text using a two-stage hybrid approach:

  Stage 1 — Section-aware splitting:
      Detects section headers via regex and splits the document into
      logical sections first. Preserves section_header as metadata.

  Stage 2 — Recursive character fallback:
      Any section that exceeds chunk_size is further split using
      LangChain's RecursiveCharacterTextSplitter.

Output: one JSON file per PDF saved to data/processed/<filename>_chunks.json
"""

import re
import json
import uuid
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ── Section header regex ───────────────────────────────────────────────────────
# Tightened rules to avoid picking up long body sentences as headers:
#   - "Chapter/Section/Part N" (no trailing body text)
#   - Numbered headers: "1.", "1.1", "2.3.1" followed by at most 5 words
#   - Exact keyword-only headers (must be the entire line, up to 60 chars)
_MAX_HEADER_LEN = 60

_SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:chapter|section|part)\s+\d+(?:\.\d+)*"                   # Chapter/Section/Part N or N.N
    r"|(?:\d+\.)+\d*\s+(?:\w+\s*){1,5}"                           # 1. / 1.1 / 2.3.1 + 1-5 words
    r"|\d+\.\s+[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,4}"              # 1. TitleCase + up to 4 more words
    r"|(?:introduction|overview|summary|conclusion|purpose"
    r"|policy|framework|background|scope|objectives?"
    r"|requirements?|guidelines?|appendix|references?|definitions?)"
    r"(?:\s+[A-Za-z]+){0,3}"                                       # keyword + up to 3 extra words
    r")$",                                                          # must consume the whole line
    re.IGNORECASE,
)


def _load_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

_config: dict = _load_config()


# ── Stage 1: Section-aware splitting ──────────────────────────────────────────

def _split_into_sections(text: str) -> list[dict]:
    """
    Split a page's text into logical sections by detecting headers.

    Returns:
        List of {"header": str, "body": str} dicts.
        Pages with no detectable headers are returned as a single section
        with header "".
    """
    lines = text.splitlines()
    sections: list[dict] = []
    current_header = ""
    current_body_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Accept as a header only if regex matches AND line is within max length
        if _SECTION_HEADER_RE.match(stripped) and len(stripped) <= _MAX_HEADER_LEN:
            # Flush previous section
            body = "\n".join(current_body_lines).strip()
            if body:
                sections.append({"header": current_header, "body": body})
            current_header = stripped
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Flush the last section
    body = "\n".join(current_body_lines).strip()
    if body:
        sections.append({"header": current_header, "body": body})

    # If nothing was detected, treat the whole page as one unnamed section
    if not sections:
        sections = [{"header": "", "body": text.strip()}]

    return sections


# ── Stage 2: Recursive character fallback ─────────────────────────────────────

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int,
                     separators: list[str]) -> list[str]:
    """
    Pure-Python recursive character splitter — no external dependencies.
    Tries each separator in order; recurses with the next separator when
    a piece still exceeds chunk_size.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            parts = text.split(sep) if sep != "" else list(text)
            chunks: list[str] = []
            current = ""

            for part in parts:
                candidate = current + (sep if current else "") + part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    if len(part) > chunk_size and i + 1 < len(separators):
                        chunks.extend(
                            _recursive_split(part, chunk_size, chunk_overlap, separators[i + 1:])
                        )
                        current = ""
                    else:
                        current = part

            if current.strip():
                chunks.append(current.strip())

            # Apply overlap between adjacent chunks
            if chunk_overlap > 0 and len(chunks) > 1:
                overlapped = [chunks[0]]
                for j in range(1, len(chunks)):
                    tail = chunks[j - 1][-chunk_overlap:]
                    overlapped.append(tail + " " + chunks[j])
                chunks = overlapped

            return chunks

    return [text]


# ── Sentence boundary cleaner ─────────────────────────────────────────────────

def _trim_to_sentence_boundary(text: str) -> str:
    """
    Strip any leading partial sentence from a chunk produced by recursive
    splitting. A chunk is considered to start mid-sentence if it does not
    begin with an uppercase letter or a digit after the overlap tail is
    prepended.

    Strategy:
      - Find the first occurrence of ". " or "\n" in the first 60 chars.
      - If found, trim everything up to and including that boundary.
      - If the trimmed result is too short (< 20 chars), keep the original
        to avoid losing tiny but valid chunks.
    """
    # Already starts cleanly — capital letter, digit, or quote
    if re.match(r'^[A-Z0-9\"\']', text):
        return text

    # Search for the first sentence/line boundary within the opening 60 chars
    window = text[:60]
    match = re.search(r'[.\n]\s+', window)
    if match:
        trimmed = text[match.end():].strip()
        if len(trimmed) >= 20:
            return trimmed

    return text


# ── Chunk ID builder ───────────────────────────────────────────────────────────

def _make_chunk_id(source_file: str, page_number: int,
                   section_header: str, chunk_index: int,
                   is_table: bool = False) -> str:
    """
    Build a deterministic, human-readable chunk ID.
    Format: <stem>_p<page>_s<section_slug>_c<index>  (body text)
            <stem>_p<page>_t<section_slug>_c<index>  (table content)

    Falls back to a UUID suffix for uniqueness when the section slug
    would be empty (headerless sections).
    """
    stem   = Path(source_file).stem                          # e.g. "iam_rbac_policy"
    slug   = re.sub(r"\W+", "_", section_header.lower())[:30] if section_header else str(uuid.uuid4())[:8]
    marker = "t" if is_table else "s"
    return f"{stem}_p{page_number}_{marker}{slug}_c{chunk_index}"


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Apply hybrid chunking to a list of page dicts from pdf_loader.

    Pipeline per page:
      1. Split into sections (header detection).
      2. If a section body > chunk_size → apply RecursiveCharacterTextSplitter.
      3. Attach full metadata to every chunk.
      4. Save per-PDF JSON to data/processed/<filename>_chunks.json.

    Args:
        pages:         Output of pdf_loader.load_pdfs().
        chunk_size:    Overrides config.yaml chunk_size.
        chunk_overlap: Overrides config.yaml chunk_overlap.

    Returns:
        Flat list of all chunk dicts across all pages and all PDFs.

    Raises:
        ValueError: If pages is empty.
    """
    chunk_size    = chunk_size    if chunk_size    is not None else _config["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else _config["chunk_overlap"]

    if not pages:
        raise ValueError("No pages provided. Run pdf_loader.load_pdfs() first.")

    logger.info(
        f"Hybrid chunking {len(pages)} page(s) | "
        f"chunk_size={chunk_size} | chunk_overlap={chunk_overlap}"
    )

    ingestion_timestamp = datetime.now(timezone.utc).isoformat()

    all_chunks: list[dict] = []

    # Group pages by source_file so we can save one JSON per PDF
    pages_by_file: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        pages_by_file[page["source_file"]].append(page)

    for source_file, file_pages in pages_by_file.items():
        file_chunks: list[dict] = []

        for page in file_pages:
            page_number = page["page_number"]
            text        = page["text"]
            has_table   = page.get("has_table", False)

            # Stage 1 — section detection
            sections = _split_into_sections(text)

            chunk_index = 0  # global index within this page

            for section in sections:
                header = section["header"]
                body   = section["body"]

                if not body:
                    continue

                # Stage 2 — fallback split if section exceeds chunk_size
                if len(body) > chunk_size:
                    sub_texts = _recursive_split(body, chunk_size, chunk_overlap, _SEPARATORS)
                else:
                    sub_texts = [body]

                for sub_text in sub_texts:
                    cleaned = _trim_to_sentence_boundary(sub_text.strip())
                    if not cleaned:
                        continue

                    chunk = {
                        "chunk_id":            _make_chunk_id(source_file, page_number, header, chunk_index, is_table=has_table),
                        "source_file":         source_file,
                        "page_number":         page_number,
                        "section_header":      header,
                        "has_table":           has_table,
                        "text":                cleaned,
                        "chunk_size_actual":   len(cleaned),
                        "ingestion_timestamp": ingestion_timestamp,
                    }
                    file_chunks.append(chunk)
                    chunk_index += 1

        logger.info(f"  {source_file} → {len(file_chunks)} chunks")
        _save_chunks(file_chunks, source_file, _config)
        all_chunks.extend(file_chunks)

    logger.info(f"Total chunks across all files: {len(all_chunks)}")
    return all_chunks


# ── Persistence ────────────────────────────────────────────────────────────────

def _save_chunks(chunks: list[dict], source_file: str, config: dict) -> None:
    """
    Save chunks for a single PDF to data/processed/<stem>_chunks.json.

    Args:
        chunks:      Chunk list for this PDF.
        source_file: Original PDF filename (e.g. "iam_rbac_policy.pdf").
        config:      Loaded config dict.
    """
    project_root  = Path(__file__).resolve().parents[2]
    processed_dir = (project_root / config["processed_data_path"]).resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    stem        = Path(source_file).stem          # "iam_rbac_policy"
    output_path = processed_dir / f"{stem}_chunks.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"  Saved → {output_path}")


# ── Smoke-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_pages = [
        {
            "text": (
                "Introduction\n"
                "Identity and Access Management (IAM) is a framework of policies and "
                "technologies ensuring the right people access the right resources.\n\n"
                "1. Role-Based Access Control\n"
                "RBAC assigns permissions to roles rather than individuals. "
                "Users inherit permissions by being assigned to roles. "
                "This model scales well across large organisations with hundreds of users "
                "and thousands of resources, reducing administrative overhead significantly.\n\n"
                "2. Attribute-Based Access Control\n"
                "ABAC uses policies combining user attributes, resource attributes, and "
                "environmental conditions to make fine-grained, dynamic access decisions."
            ),
            "page_number": 1,
            "source_file": "iam_rbac_policy.pdf",
        },
        {
            "text": (
                "Policy\n"
                "All privileged accounts must be managed through a PAM solution. "
                "Just-in-time access provisioning eliminates standing privileges.\n\n"
                "Framework\n"
                "Session recording and credential vaulting are mandatory controls "
                "for all administrative access to production systems."
            ),
            "page_number": 2,
            "source_file": "iam_rbac_policy.pdf",
        },
    ]

    chunks = chunk_pages(sample_pages)
    print(f"\nTotal chunks: {len(chunks)}\n")
    for c in chunks:
        print(
            f"[{c['chunk_id']}] "
            f"Page {c['page_number']} | "
            f"Section: '{c['section_header']}' | "
            f"{c['chunk_size_actual']} chars"
        )
        print(f"  {c['text'][:100]}")
        print()
