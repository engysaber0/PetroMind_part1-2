"""
  1. RE_SECTION now matches appendix sections (B-1, B-2, etc.) which were
     silently missed before — fixing the "tool care" retrieval failure.
  2. Handles split-line section headings (e.g. "12-1." on one line,
     title on the next) — fixes chunk-0132 having no section_id.
  3. chunk-0000 and chunk-0001 (TOC/front-matter noise) are dropped via
     stricter front-matter detection.
  4. MAX_CHUNK_CHARS tightened to 1200 for better retrieval granularity.
  5. Tables that exceed 3000 chars are now split by subsection headers
     (equipment sub-groups) to improve retrieval precision on large tables.
  6. Tiny chunks (<150 chars) are dropped unless they are standalone
     one-liner sections that contain meaningful content.
  7. Metadata enrichment improved: appendix section headings are extracted
     from text and added to enriched_text for better embedding signal.
"""

import re
import fitz  

# ── Regex patterns

# "3-1. Purpose"  "2-8. Electrical safety"  "B-2. Tool care and usage"
# Also handles split-line: "12-1." alone (title on next line)
RE_SECTION = re.compile(
    r"^([A-Z]?\d{1,2}-\d{1,2}|[A-Z]-\d+)\.?\s{0,4}(.*)$", re.IGNORECASE
)
RE_SECTION_STRICT = re.compile(
    r"^([A-Z]?\d{1,2}-\d{1,2}|[A-Z]-\d+)\.\s+\S", re.IGNORECASE
)
RE_SECTION_ALONE = re.compile(
    r"^([A-Z]?\d{1,2}-\d{1,2}|[A-Z]-\d+)\.$", re.IGNORECASE
)

# "Table 3-1. Diesel engine – standby mode"
# MUST NOT match lines that contain "(continued)"
RE_TABLE_NEW = re.compile(
    r"^Table\s+(\d{1,2}-\d{1,2}(?:\.\d)?)\.\s+(.+)$", re.IGNORECASE
)
RE_CONTINUED = re.compile(r"\(continued\)", re.IGNORECASE)

# "CHAPTER 7." / "APPENDIX B"
RE_CHAPTER = re.compile(r"^(CHAPTER\s+\d+|APPENDIX\s+[A-Z])\b", re.IGNORECASE)

# Running page header / section-local footer  — noise
RE_PAGE_HEADER = re.compile(r"^TM\s+5-692-1\s*$", re.IGNORECASE)
RE_PAGE_FOOTER = re.compile(r"^\d{1,2}-\d{1,2}$|^[A-Z]-\d+$")

# TOC noise lines: pure page number or short roman numeral lines
RE_TOC_NOISE = re.compile(r"^(i{1,4}|v|vi{1,3}|ix|x{1,3}|\d{1,3})$")

# Subparagraph boundaries inside narrative text
RE_SUBPARA = re.compile(r"^[a-z]\.\s+\S")           # a.  b.  c. …
RE_NUMITEM = re.compile(r"^\(\d+\)\s+\S")            # (1) (2) (3) …

# Table sub-group headers (ALL CAPS lines inside tables, e.g. "Cooling Tower", "Fans")
RE_TABLE_SUBGROUP = re.compile(r"^[A-Z][A-Z\s\(\)&/–-]{8,}$")

# Army sign-off boilerplate that leaks into the last appendix chunk
BOILERPLATE_MARKERS = [
    "The proponent agency of this publication",
    "By Order of the Secretary",
    "PETER J. SCHOOMAKER",
    "Distribution:",
    "PIN:",
]

# Frequency tokens used in maintenance tables
FREQUENCY_TOKENS = {
    "hr", "8 hrs", "day", "week", "mo", "3 mos", "6 mos", "yr",
    "250 hrs", "500 hrs", "1k hrs", "2k hrs", "5k hrs", "10k hrs",
    "as required", "at startup", "shift", "per mfg",
}

MAX_CHUNK_CHARS   = 1200   # tightened from 1400 for better retrieval granularity
MAX_TABLE_CHARS   = 3000   # tables larger than this get split by sub-group
SUB_OVERLAP_CHARS = 120    # overlap tail carried into next sub-chunk


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_frequency_tags(text: str) -> list:
    text_lower = text.lower()
    return [t for t in FREQUENCY_TOKENS
            if re.search(r"\b" + re.escape(t) + r"\b", text_lower)]


def _is_noise(line: str) -> bool:
    if not line:
        return True
    if RE_PAGE_HEADER.match(line):
        return True
    if RE_PAGE_FOOTER.match(line):
        return True
    if RE_TOC_NOISE.match(line):
        return True
    return False


def _make_label(chapter, section_id, section_title, table_id, table_title, ctype):
    if ctype == "table" and table_id:
        return f"[{chapter}] Table {table_id}: {table_title}"
    if ctype == "narrative" and section_id:
        return f"[{chapter}] §{section_id} {section_title or ''}"
    return f"[{chapter}] {ctype}"


def _trim_boilerplate(text: str) -> str:
    """Remove Army sign-off boilerplate that leaks into final chunk."""
    for marker in BOILERPLATE_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
    return text


def _is_toc_chunk(text: str) -> bool:
    """Detect chunks that are pure table-of-contents or list-of-tables noise."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    # If >60% of lines are "Table X-Y  some title" or "Figure X-Y  title" patterns
    toc_line = re.compile(r"^(Table|Figure)\s+\d{1,2}-\d{1,2}", re.IGNORECASE)
    toc_count = sum(1 for l in lines if toc_line.match(l) or RE_TOC_NOISE.match(l))
    return toc_count / len(lines) > 0.35


# ── Step 1: Extract clean lines per page ──────────────────────────────────────

def _extract_pages(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        raw_lines = page.get_text("text").splitlines()
        clean = [l.strip() for l in raw_lines if not _is_noise(l.strip())]
        pages.append({"page_num": page_num, "lines": clean})
    doc.close()
    return pages


# ── Step 2: Walk lines and emit raw structural chunks ─────────────────────────

def _build_raw_chunks(pages: list) -> list:
    chunks = []

    current_chapter       = "TM 5-692-1"
    current_section_id    = None
    current_section_title = None
    current_table_id      = None
    current_table_title   = None
    content_type          = "front_matter"
    buffer_lines          = []
    buffer_start_page     = 1
    pending_section_id    = None   # for split-line headings
    past_toc              = False  # guard: only capture chapters after TOC is over

    def flush(end_page: int):
        nonlocal buffer_lines, buffer_start_page
        text = "\n".join(buffer_lines).strip()
        if not text:
            buffer_lines = []
            return
        chunk = {
            "text": text,
            "metadata": {
                "chapter":       current_chapter,
                "section_id":    current_section_id,
                "section_title": current_section_title,
                "table_id":      current_table_id,
                "table_title":   current_table_title,
                "content_type":  content_type,
                "page_start":    buffer_start_page,
                "page_end":      end_page,
                "label": _make_label(
                    current_chapter, current_section_id, current_section_title,
                    current_table_id, current_table_title, content_type,
                ),
            },
        }
        if content_type == "table":
            chunk["metadata"]["frequency_tags"] = _extract_frequency_tags(text)
        chunks.append(chunk)
        buffer_lines = []

    for page in pages:
        page_num = page["page_num"]

        for line in page["lines"]:

            # ── Chapter / Appendix ───────────────────────────────────────────
            if RE_CHAPTER.match(line) and past_toc:
                flush(page_num)
                current_chapter       = line.strip()
                current_section_id    = None
                current_section_title = None
                current_table_id      = None
                current_table_title   = None
                pending_section_id    = None
                content_type          = "narrative"
                buffer_start_page     = page_num
                continue

            # ── Table heading — check (continued) FIRST ──────────────────────
            m_tbl = RE_TABLE_NEW.match(line)
            if m_tbl:
                detected_id = m_tbl.group(1)
                if RE_CONTINUED.search(line):
                    continue
                elif detected_id == current_table_id:
                    # Same table reprinted on new page — skip repeated title
                    continue
                else:
                    flush(page_num)
                    current_table_id    = detected_id
                    current_table_title = m_tbl.group(2).strip()
                    current_section_id  = None
                    current_section_title = None
                    pending_section_id  = None
                    content_type        = "table"
                    buffer_start_page   = page_num
                    buffer_lines.append(line)
                    continue

            # ── Handle pending split-line section heading ────────────────────
            # e.g. previous line was "12-1." alone; this line is the title
            if pending_section_id is not None:
                flush(page_num)
                current_section_id    = pending_section_id
                current_section_title = line.strip()
                current_table_id      = None
                current_table_title   = None
                content_type          = "narrative"
                buffer_start_page     = page_num
                buffer_lines.append(f"{current_section_id}.  {current_section_title}")
                pending_section_id    = None
                continue

            # ── Numbered section heading (only outside tables) ───────────────
            if content_type != "table":
                # Check for split-line: "12-1." alone on a line
                m_alone = RE_SECTION_ALONE.match(line)
                if m_alone:
                    past_toc = True
                    pending_section_id = m_alone.group(1)
                    continue

                # Normal section heading: "2-8. Electrical safety"
                m_sec = RE_SECTION_STRICT.match(line)
                if m_sec:
                    past_toc = True
                    # Extract id and title
                    sec_id = re.match(r"^([A-Z]?\d{1,2}-\d{1,2}|[A-Z]-\d+)", line, re.IGNORECASE).group(1)
                    sec_title = re.sub(r"^[A-Z]?\d{1,2}-\d{1,2}\.\s*|^[A-Z]-\d+\.\s*", "", line, flags=re.IGNORECASE).strip()
                    flush(page_num)
                    current_section_id    = sec_id
                    current_section_title = sec_title
                    current_table_id      = None
                    current_table_title   = None
                    content_type          = "narrative"
                    buffer_start_page     = page_num
                    buffer_lines.append(line)
                    continue

            buffer_lines.append(line)

    flush(pages[-1]["page_num"] if pages else 1)
    return chunks


# ── Step 3: Split large tables by sub-group headers ───────────────────────────

def _split_table(chunk: dict) -> list:
    """
    Split large tables (>MAX_TABLE_CHARS) at equipment sub-group boundaries.
    Sub-groups are ALL CAPS header lines like 'Cooling Tower', 'Fans', 'Pumps'.
    Keeps the table title as context in each sub-chunk.
    """
    text  = chunk["text"]
    lines = text.splitlines()
    if not lines:
        return [chunk]

    title_line = lines[0]   # "Table X-Y.  Title"

    # Find sub-group split points (skip first 3 lines — table header area)
    boundaries = [0]
    for i, line in enumerate(lines[3:], start=3):
        if RE_TABLE_SUBGROUP.match(line) and len(line) < 60:
            boundaries.append(i)
    boundaries.append(len(lines))

    if len(boundaries) <= 2:
        return [chunk]   # no meaningful sub-groups found

    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        seg_lines = lines[start:end]
        # Always prepend table title for context
        if start > 0:
            seg_lines = [title_line, ""] + seg_lines
        seg_text = "\n".join(seg_lines).strip()
        if len(seg_text) < 100:
            continue
        sc = {
            "text": seg_text,
            "metadata": dict(chunk["metadata"]),
        }
        result.append(sc)

    return result if result else [chunk]


# ── Step 4: Sub-split long narrative chunks ───────────────────────────────────

def _split_narrative(text: str) -> list:
    """
    Split at subparagraph letter boundaries (a. b. c.) first.
    If any piece is still > MAX_CHUNK_CHARS, split at numbered items (1)(2)(3).
    Overlap tail of SUB_OVERLAP_CHARS is prepended to the next piece.
    """
    lines = text.splitlines()

    boundaries = [0]
    for i, line in enumerate(lines[1:], start=1):
        if RE_SUBPARA.match(line):
            boundaries.append(i)
    boundaries.append(len(lines))

    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        seg = "\n".join(lines[start:end]).strip()
        if not seg:
            continue
        if len(seg) > MAX_CHUNK_CHARS:
            seg_lines = seg.splitlines()
            sub_bounds = [0]
            for i, l in enumerate(seg_lines[1:], start=1):
                if RE_NUMITEM.match(l):
                    sub_bounds.append(i)
            sub_bounds.append(len(seg_lines))
            for sb, se in zip(sub_bounds, sub_bounds[1:]):
                piece = "\n".join(seg_lines[sb:se]).strip()
                if piece:
                    segments.append(piece)
        else:
            segments.append(seg)

    result       = []
    current      = ""
    overlap_tail = ""

    for seg in segments:
        if current and len(current) + len(seg) + 1 > MAX_CHUNK_CHARS:
            result.append((overlap_tail + current).strip())
            overlap_tail = current[-SUB_OVERLAP_CHARS:].lstrip() + "\n" if current else ""
            current = seg
        else:
            current = (current + "\n" + seg).strip() if current else seg

    if current:
        result.append((overlap_tail + current).strip())

    return [r for r in result if r]


# ── Step 5: Finalize — filter noise, split, assign IDs ────────────────────────

def _finalize(raw_chunks: list) -> list:
    final    = []
    chunk_id = 0

    for chunk in raw_chunks:
        ctype = chunk["metadata"]["content_type"]

        # Drop front matter entirely
        if ctype == "front_matter":
            continue

        text = chunk["text"]

        # Drop TOC/list-of-tables noise chunks
        if _is_toc_chunk(text):
            continue

        # Drop tiny structureless chunks
        if (len(text) < 150
                and chunk["metadata"]["section_id"] is None
                and chunk["metadata"]["table_id"] is None):
            continue

        # Drop tiny chunks that are just a section heading with no body
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(text) < 150 and len(lines) <= 2:
            continue

        # Trim Army boilerplate from narrative chunks
        if ctype == "narrative":
            text = _trim_boilerplate(text)
            chunk["text"] = text
            if not text:
                continue

        # ── Large tables: split by sub-group ──────────────────────────────────
        if ctype == "table" and len(text) > MAX_TABLE_CHARS:
            sub_chunks = _split_table(chunk)
            if len(sub_chunks) > 1:
                for i, sc in enumerate(sub_chunks):
                    sc["id"] = f"chunk-{chunk_id:04d}"
                    sc["metadata"]["sub_index"] = i
                    sc["metadata"]["sub_total"] = len(sub_chunks)
                    sc["metadata"]["label"] = (
                        chunk["metadata"]["label"] + f" (part {i+1}/{len(sub_chunks)})"
                    )
                    chunk_id += 1
                    final.append(sc)
                continue

        # ── Tables and short narratives → one chunk as-is ─────────────────────
        if ctype == "table" or len(text) <= MAX_CHUNK_CHARS:
            chunk["id"] = f"chunk-{chunk_id:04d}"
            chunk_id += 1
            final.append(chunk)
            continue

        # ── Long narratives → sub-split ───────────────────────────────────────
        sub_texts = _split_narrative(text)
        if len(sub_texts) <= 1:
            chunk["id"] = f"chunk-{chunk_id:04d}"
            chunk_id += 1
            final.append(chunk)
            continue

        for i, sub in enumerate(sub_texts):
            sc = {
                "id":   f"chunk-{chunk_id:04d}",
                "text": sub,
                "metadata": dict(chunk["metadata"]),
            }
            sc["metadata"]["sub_index"] = i
            sc["metadata"]["sub_total"] = len(sub_texts)
            sc["metadata"]["label"] = (
                chunk["metadata"]["label"] + f" (part {i+1}/{len(sub_texts)})"
            )
            chunk_id += 1
            final.append(sc)

    return final


# ── Step 6: Enrich text for embedding ─────────────────────────────────────────

def _enrich(chunk: dict) -> str:
    """
    Prepend structured context so the embedding captures chapter + section +
    table identity — not just raw content.
    For appendix chunks without a section_id, extract B-N. headings from
    the text itself so the embedding captures the topic.
    """
    m     = chunk["metadata"]
    parts = [f"TM 5-692-1 | {m['chapter']}"]

    if m.get("section_id"):
        parts.append(f"Section {m['section_id']}: {m.get('section_title', '')}")
    if m.get("table_id"):
        parts.append(f"Table {m['table_id']}: {m.get('table_title', '')}")
    if m.get("frequency_tags"):
        parts.append(f"Maintenance frequencies: {', '.join(m['frequency_tags'])}")

    # For appendix/narrative chunks with no section_id, extract appendix headings
    # like "B-2. Tool care and usage" from the text body as topic signal
    if not m.get("section_id") and not m.get("table_id"):
        appendix_headings = re.findall(
            r"^[A-Z]-\d+\.\s+.+$", chunk["text"], re.MULTILINE
        )
        if appendix_headings:
            parts.append("Topics: " + " | ".join(h.strip() for h in appendix_headings[:5]))

    header = " | ".join(parts)

    # Boost short but important chunks (e.g. B-2 Tool care) by repeating the
    # section title so the embedding does not get diluted by brevity.
    boost = ""
    if len(chunk["text"]) < 600 and m.get("section_id"):
        title = m.get("section_title", "")
        if title:
            boost = f"\nKeywords: {title} | {title}\n"

    return header + boost + "\n\n" + chunk["text"]


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_document(pdf_path: str) -> list:
    """
    Returns a list of chunk dicts, each with:
      id             — unique string  e.g. 'chunk-0042'
      text           — raw text shown to the user / LLM as context
      enriched_text  — text sent to the embedding model (has metadata header)
      metadata       — chapter, section_id, section_title, table_id,
                       table_title, content_type, page_start, page_end,
                       label, frequency_tags (tables only),
                       sub_index / sub_total (split chunks only)
    """
    pages  = _extract_pages(pdf_path)
    raw    = _build_raw_chunks(pages)
    chunks = _finalize(raw)
    for chunk in chunks:
        chunk["enriched_text"] = _enrich(chunk)
    return chunks


# ── CLI diagnostic ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, statistics
    from collections import Counter

    pdf    = sys.argv[1] if len(sys.argv) > 1 else "tm_5_692_1.pdf"
    chunks = parse_document(pdf)

    tables    = [c for c in chunks if c["metadata"]["content_type"] == "table"]
    narrative = [c for c in chunks if c["metadata"]["content_type"] == "narrative"]
    sizes     = [len(c["text"]) for c in chunks]

    print(f"\n{'='*60}")
    print(f"  Total chunks     : {len(chunks)}")
    print(f"  Table chunks     : {len(tables)}")
    print(f"  Narrative chunks : {len(narrative)}")
    print(f"  Size  min  : {min(sizes)}")
    print(f"        max  : {max(sizes)}")
    print(f"        avg  : {statistics.mean(sizes):.0f}")
    print(f"        med  : {statistics.median(sizes):.0f}")
    print(f"{'='*60}")

    # Check no section/table id chunks
    no_meta = [c for c in chunks if not c['metadata'].get('section_id') and not c['metadata'].get('table_id')]
    print(f"\nChunks with no section_id/table_id: {len(no_meta)}")
    for c in no_meta:
        print(f"  {c['id']} | {c['metadata']['label']} | {len(c['text'])}ch")

    # Oversized
    big = [c for c in chunks if len(c["text"]) > 3000]
    print(f"\nOversized >3000 chars: {len(big)}")
    for c in big:
        print(f"  {c['id']} | {len(c['text'])} chars | {c['metadata']['label']}")

    # Tiny
    tiny = [c for c in chunks if len(c['text']) < 150]
    print(f"\nTiny <150 chars: {len(tiny)}")
    for c in tiny:
        print(f"  {c['id']} | {len(c['text'])}ch | {repr(c['text'][:80])}")

    # Tool care check
    tool = [c for c in chunks if "tool" in c["text"].lower() and
            any(w in c["text"].lower() for w in ["monthly", "inspect", "care", "storage"])]
    print(f"\nTool-related chunks: {len(tool)}")
    for t in tool:
        print(f"  {t['id']} | {t['metadata']['label']}")
        print(f"  section_id={t['metadata'].get('section_id')} | enriched preview: {t['enriched_text'][:120]}")
