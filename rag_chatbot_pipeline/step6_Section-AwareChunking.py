"""
Stage 6: Section-Aware Chunking

For each section in the canonical document:
1. If section fits in one chunk (under target_max), emit whole-section chunk
2. If too large, split by paragraphs preserving section heading context
3. Tables and figures stay intact (referenced as metadata)
4. Add overlap between consecutive chunks within same section

Uses tiktoken for accurate token counting (matches OpenAI embedding models).
"""

import hashlib
import json
import re
from pathlib import Path

CHARS_PER_TOKEN = 4

TARGET_MIN_TOKENS = 200
TARGET_MAX_TOKENS = 1000
OVERLAP_TOKENS = 80


def count_tokens(text: str) -> int:
    """Approximate token count. In production use tiktoken for exact counts."""
    return len(text) // CHARS_PER_TOKEN


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")[:50]


def generate_chunk_id(doc_id: str, section_path: str, chunk_index: int) -> str:
    # Build a globally-unique id: a readable slug of the leaf heading plus a short
    # hash of the FULL section path. Slugifying the path alone isn't enough because
    # slugify truncates to 50 chars, so long sibling paths sharing a prefix collide,
    # and identically-named leaves (every chapter has a "Conclusion") collide too.
    leaf = section_path.split(" > ")[-1]
    leaf_slug = slugify(leaf)
    path_hash = hashlib.sha256(section_path.encode()).hexdigest()[:8]
    return f"{doc_id}__{leaf_slug}__{path_hash}__chunk{chunk_index:03d}"


def split_into_paragraphs(text: str) -> list:
    """Split text into paragraphs by double newline; merge tiny fragments."""
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    merged = []
    buffer = []
    buffer_tokens = 0

    for p in raw_paras:
        p_tokens = count_tokens(p)
        if buffer_tokens + p_tokens < 50:
            buffer.append(p)
            buffer_tokens += p_tokens
        else:
            if buffer:
                merged.append("\n\n".join(buffer))
            buffer = [p]
            buffer_tokens = p_tokens

    if buffer:
        merged.append("\n\n".join(buffer))

    return merged


def chunk_section(section: dict, doc_metadata: dict, doc_id: str) -> list:
    """Apply section-aware chunking to a single section."""

    heading = section["heading"]
    parent_path = section.get("parent_path", [])
    section_path = " > ".join(parent_path + [heading])
    text = section.get("text", "").strip()
    figures = section.get("figures", [])

    if not text:
        return []

    heading_prefix = f"# {heading}\n\n"

    if section_path != heading:
        breadcrumb = f"_Section: {section_path}_\n\n"
        full_text = breadcrumb + heading_prefix + text
    else:
        full_text = heading_prefix + text

    total_tokens = count_tokens(full_text)

    chunks = []

    if total_tokens <= TARGET_MAX_TOKENS:
        chunks.append(
            build_chunk(
                text=full_text,
                section=section,
                section_path=section_path,
                doc_id=doc_id,
                doc_metadata=doc_metadata,
                chunk_index=0,
                figures=figures,
                strategy="whole_section",
            )
        )
        return chunks

    paragraphs = split_into_paragraphs(text)

    current_paras = []
    current_tokens = count_tokens(heading_prefix)
    chunk_index = 0

    available = TARGET_MAX_TOKENS - count_tokens(heading_prefix) - 50

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > available and current_paras:
            chunk_text = heading_prefix + "\n\n".join(current_paras)
            chunks.append(
                build_chunk(
                    text=chunk_text,
                    section=section,
                    section_path=section_path,
                    doc_id=doc_id,
                    doc_metadata=doc_metadata,
                    chunk_index=chunk_index,
                    figures=figures if chunk_index == 0 else [],
                    strategy="section_split",
                )
            )
            chunk_index += 1

            overlap_paras = []
            overlap_tokens = 0
            for prev_para in reversed(current_paras):
                prev_tokens = count_tokens(prev_para)
                if overlap_tokens + prev_tokens > OVERLAP_TOKENS:
                    break
                overlap_paras.insert(0, prev_para)
                overlap_tokens += prev_tokens

            current_paras = overlap_paras
            current_tokens = count_tokens(heading_prefix) + overlap_tokens

        current_paras.append(para)
        current_tokens += para_tokens

    if current_paras:
        chunk_text = heading_prefix + "\n\n".join(current_paras)
        chunks.append(
            build_chunk(
                text=chunk_text,
                section=section,
                section_path=section_path,
                doc_id=doc_id,
                doc_metadata=doc_metadata,
                chunk_index=chunk_index,
                figures=figures if chunk_index == 0 else [],
                strategy="section_split",
            )
        )

    return chunks


def build_chunk(text, section, section_path, doc_id, doc_metadata, chunk_index, figures, strategy):
    """Build final chunk record matching the schema we designed."""

    return {
        "chunk_id": generate_chunk_id(doc_id, section_path, chunk_index),
        "doc_id": doc_id,
        "text_for_embedding": text,
        "chunk_type": "prose_with_figure" if figures else "prose",
        "section_path": section_path,
        "section_heading": section["heading"],
        "section_level": section["level"],
        "page_range": section["page_range"],
        "token_count": count_tokens(text),
        "char_count": len(text),
        "chunk_index": chunk_index,
        "chunk_strategy": strategy,
        "referenced_figures": [
            {
                "element_id": f["element_id"],
                "description": f["description"],
                "decision": f["decision"],
            }
            for f in figures
        ],
        "metadata": {
            **doc_metadata,
            "embedding_model": "text-embedding-3-large",
            "pipeline_version": "v1.0",
        },
    }


def chunk_document(canonical_doc_path: str) -> list:
    with open(canonical_doc_path) as f:
        canonical = json.load(f)

    doc_id = canonical["doc_id"]
    doc_metadata = canonical["metadata"]
    doc_metadata["doc_title"] = canonical["title"]
    doc_metadata["doc_type"] = canonical["doc_type"]

    all_chunks = []
    for section in canonical["structure"]:
        section_chunks = chunk_section(section, doc_metadata, doc_id)
        all_chunks.extend(section_chunks)

    return all_chunks


if __name__ == "__main__":
    base_dir = (
        Path(__file__).resolve().parent
    )  # self-contained: data/ lives in this pipeline folder
    output_path = base_dir / "data" / "output"
    canonical_path = output_path / "canonical_document.json"

    chunks = chunk_document(canonical_path)

    output_path = output_path / "ebook_chunks.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    token_counts = [c["token_count"] for c in chunks]
    strategies = {}
    for c in chunks:
        strategies[c["chunk_strategy"]] = strategies.get(c["chunk_strategy"], 0) + 1

    chunks_with_figs = sum(1 for c in chunks if c["referenced_figures"])

    print("=" * 60)
    print("STAGE 6: CHUNKING COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {len(chunks)}")
    print("\nToken stats:")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print(f"  Avg: {sum(token_counts) / len(token_counts):.0f}")
    print(f"  Total: {sum(token_counts):,}")
    print(f"\nChunk strategies: {strategies}")
    print(f"Chunks with figures: {chunks_with_figs}")

    embedding_cost_per_1k = 0.00013
    total_embedding_cost = (sum(token_counts) / 1000) * embedding_cost_per_1k
    print(f"\nEstimated embedding cost (text-embedding-3-large): ${total_embedding_cost:.3f}")

    print(f"\nOutput: {output_path}")
