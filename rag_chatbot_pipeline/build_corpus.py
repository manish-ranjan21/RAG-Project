#!/usr/bin/env python3
"""
build_corpus.py - multi-PDF ingestion orchestrator.

The original pipeline (steps 1-8) was hardcoded to a single PDF: every step
read/wrote fixed filenames and the doc_id was hardcoded to "ebook", so running
it on a second PDF would overwrite the first and collide chunk IDs.

This orchestrator drives the SAME step functions for EVERY PDF in docs/:

    for each docs/*.pdf:
        step2  extract_layout_elemnts      -> <doc>/layout_analysis.json
        step3  process_all_image_regions   -> <doc>/image_routing.json
        step4_5 build_canonical_document    -> <doc>/canonical_document.json
        step6  chunk_document               -> chunks (accumulated)
    step7  embed_pipeline   (upsert by chunk_id, checkpointed)  -> corpus_embeddings.jsonl
    step8  load_pipeline    (ON CONFLICT upsert)                -> pgvector

Each document gets a unique doc_id derived from its filename, so chunk IDs
(`{doc_id}__{section}__{hash}__chunkNNN`) never collide across books.

Intermediates live under data/output/_corpus/<doc_id>/ and are reused on
re-runs (skip the slow pdfplumber passes) unless --force is given.

Usage:
    cd rag_chatbot_pipeline
    export PGPASSWORD=REDACTED-DEV-PASSWORD          # or rely on the dev default below
    python build_corpus.py                      # all PDFs (except already-loaded)
    python build_corpus.py --only "AI Engineering"   # filename substring filter
    python build_corpus.py --chunks-only        # build chunks, no embedding/DB
    python build_corpus.py --force              # re-run layout/canonical from scratch
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DOCS = BASE / "docs"
CORPUS = BASE / "data" / "output" / "_corpus"

# Already in the DB under doc_id "ebook" from the original single-doc run.
# Skip so we don't load the same book twice under two different doc_ids.
ALREADY_LOADED = {
    "building-machine-learning-powered-applications-going-from-idea-to-product",
}


def load_module(filename: str, modname: str):
    """Import a step script by file path (their names contain '-'/'.' so a
    normal `import` won't work). Their heavy logic is under __main__ guards,
    so importing only defines the functions we call."""
    spec = importlib.util.spec_from_file_location(modname, BASE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def slugify_doc_id(stem: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", stem.lower()).strip("_")
    return re.sub(r"_+", "_", s)


def pretty_title(stem: str) -> str:
    t = re.sub(r"[-_]+", " ", stem).strip()
    return t[:1].upper() + t[1:] if t else stem


def build_canonical_fallback(pdf_path: Path, doc_id: str, step6, target_tokens: int = 850):
    """No-TOC fallback: many PDFs (e.g. AIMA, Deep Learning) ship with zero
    bookmark outline, so step4_5's TOC-driven build returns None. Here we read
    text page-by-page and group consecutive pages into ~target_tokens pseudo-
    sections, each carrying an accurate page_range. chunk_document then splits
    them into token-bounded chunks (with overlap) exactly like real sections,
    so page citations stay correct without any bookmarks."""
    import pdfplumber

    def mk(start, end, text):
        return {
            "element_id": f"p{start:04d}",
            "type": "section",
            "heading": f"Pages {start}-{end}",
            "level": 1,
            "parent_path": [],
            "page_range": [start, end],
            "text": text,
            "figures": [],
        }

    sections = []
    buf, buf_start, buf_tokens, last = [], None, 0, 0
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            last = i
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            if buf_start is None:
                buf_start = i
            buf.append(text)
            buf_tokens += step6.count_tokens(text)
            if buf_tokens >= target_tokens:
                sections.append(mk(buf_start, i, "\n\n".join(buf)))
                buf, buf_start, buf_tokens = [], None, 0
    if buf:
        sections.append(mk(buf_start, last, "\n\n".join(buf)))

    if not sections:
        return None
    return {
        "doc_id": doc_id,
        "doc_type": "ebook",
        "title": pretty_title(pdf_path.stem),
        "metadata": {"access_classification": "internal"},
        "structure": sections,
    }


def process_pdf(pdf_path: Path, modules: dict, force: bool, no_images: bool,
                use_fallback: bool) -> list | None:
    """Run steps 2-6 for one PDF. Returns the chunk list, or None if the PDF
    has no extractable TOC (build_canonical_document returns None)."""
    step2, step3, step45, step6 = (
        modules["step2"], modules["step3"], modules["step45"], modules["step6"]
    )
    doc_id = slugify_doc_id(pdf_path.stem)
    workdir = CORPUS / doc_id
    workdir.mkdir(parents=True, exist_ok=True)
    canonical_path = workdir / "canonical_document.json"

    # Fast path: reuse the canonical doc from a previous run (skip slow pdfplumber).
    if canonical_path.exists() and not force:
        print(f"    reusing cached canonical_document.json")
        return step6.chunk_document(str(canonical_path))

    routing_path = workdir / "image_routing.json"
    if no_images:
        # step4_5 builds the canonical doc from the PDF's TOC + text directly;
        # steps 2 & 3 only exist to populate `[Figure on page N]` placeholders,
        # which are irrelevant to text retrieval. Skip them and feed an empty
        # routing file. On image-heavy books (e.g. AIMA's 4,705 figures) this
        # turns a ~40-80 min step3 into nothing.
        print(f"    skipping layout/image-routing (--no-images)", flush=True)
        routing_path.write_text(json.dumps({"routing_decisions": [], "summary": {}}))
    else:
        # step 2: layout analysis
        print(f"    step2 layout analysis...", flush=True)
        layout = step2.extract_layout_elemnts(str(pdf_path))
        layout_path = workdir / "layout_analysis.json"
        layout_path.write_text(json.dumps(layout))

        # step 3: image region routing
        print(f"    step3 image routing...", flush=True)
        results, counts = step3.process_all_image_regions(str(layout_path), str(pdf_path))
        routing_path.write_text(json.dumps({"routing_decisions": results, "summary": counts}))

    # step 4-5: canonical document (TOC-driven). Returns None if no bookmarks.
    print(f"    step4_5 canonical document...", flush=True)
    canonical = step45.build_canonical_document(str(pdf_path), doc_id, str(routing_path))
    if canonical is None:
        if not use_fallback:
            return None
        print(f"    no TOC/bookmarks -> page-based fallback chunking...", flush=True)
        canonical = build_canonical_fallback(pdf_path, doc_id, step6)
        if canonical is None:
            return None

    # The step hardcodes title/author/page_count for the original book — override
    # with this PDF's real values so the `documents` table is correct.
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        meta = reader.metadata or {}
        author = (getattr(meta, "author", None) or "").strip() or "Unknown"
    except Exception:
        page_count, author = None, "Unknown"

    canonical["title"] = pretty_title(pdf_path.stem)
    canonical["doc_type"] = "ebook"
    canonical["metadata"]["author"] = author
    canonical["metadata"]["page_count"] = page_count
    canonical["metadata"]["source_path"] = str(pdf_path)
    canonical_path.write_text(json.dumps(canonical))

    # step 6: section-aware chunking
    print(f"    step6 chunking...", flush=True)
    return step6.chunk_document(str(canonical_path))


def main():
    parser = argparse.ArgumentParser(description="Multi-PDF RAG corpus builder")
    parser.add_argument("--only", default=None,
                        help="Only process PDFs whose filename contains this substring")
    parser.add_argument("--chunks-only", action="store_true",
                        help="Build chunks only; skip embedding (step7) and DB load (step8)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run layout/canonical extraction even if cached")
    parser.add_argument("--no-images", action="store_true",
                        help="Skip layout (step2) + image routing (step3). Much faster on "
                             "image-heavy PDFs; figure placeholders are omitted (no effect on "
                             "text retrieval, since images are never embedded).")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable the page-based fallback for PDFs with no bookmark "
                             "outline. By default such PDFs are chunked by page text; with "
                             "this flag they are skipped (original step4_5 behaviour).")
    parser.add_argument("--include-loaded", action="store_true",
                        help="Also (re)process PDFs already loaded under another doc_id")
    args = parser.parse_args()

    CORPUS.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DOCS.glob("*.pdf"))
    if args.only:
        pdfs = [p for p in pdfs if args.only.lower() in p.name.lower()]
    if not args.include_loaded:
        pdfs = [p for p in pdfs if p.stem not in ALREADY_LOADED]

    if not pdfs:
        print("No PDFs to process (after filters).")
        return

    print(f"Processing {len(pdfs)} PDF(s):")
    for p in pdfs:
        print(f"  - {p.name}")

    modules = {
        "step2": load_module("step2_layout_analysis.py", "step2_layout"),
        "step3": load_module("step3_ImageRegionRouting.py", "step3_routing"),
        "step45": load_module("step4_5_CanonicalDocumentStructure.py", "step45_canonical"),
        "step6": load_module("step6_Section-AwareChunking.py", "step6_chunk"),
    }

    all_chunks = []
    summary = []
    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf.name}", flush=True)
        t0 = time.time()
        try:
            chunks = process_pdf(pdf, modules, args.force, args.no_images,
                                 use_fallback=not args.no_fallback)
            if chunks is None:
                print(f"    SKIPPED: no TOC/bookmarks (build_canonical_document returned None)")
                summary.append((pdf.name, "skipped (no TOC)", 0))
                continue
            all_chunks.extend(chunks)
            print(f"    OK: {len(chunks)} chunks in {time.time()-t0:.0f}s")
            summary.append((pdf.name, "ok", len(chunks)))
        except Exception as e:
            print(f"    ERROR: {e}")
            summary.append((pdf.name, f"error: {e}", 0))

    # Combined chunks file (source for embedding).
    chunks_path = CORPUS / "corpus_chunks.jsonl"
    with open(chunks_path, "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)
    for name, status, n in summary:
        print(f"  {n:>5}  {status:<22}  {name}")
    print(f"\n  TOTAL chunks: {len(all_chunks)}  ->  {chunks_path}")

    if not all_chunks:
        print("\nNo chunks produced; nothing to embed/load.")
        return
    if args.chunks_only:
        print("\n--chunks-only set: stopping before embedding/DB load.")
        return

    # step 7: embed (upsert by chunk_id, checkpointed across runs)
    print("\n" + "=" * 70)
    print("EMBEDDING (step7)")
    print("=" * 70, flush=True)
    step7 = load_module("step7_Production-GradeEmbedding.py", "step7_embed")
    embeddings_path = CORPUS / "corpus_embeddings.jsonl"
    embedder = step7.OpenAIEmbedder()
    step7.embed_pipeline(
        chunks_path=str(chunks_path),
        output_path=str(embeddings_path),
        checkpoint_path=str(CORPUS / ".checkpoint.json"),
        embedder=embedder,
        batch_size=50,
    )

    # step 8: load into pgvector (ON CONFLICT upsert -> accumulates)
    print("\n" + "=" * 70)
    print("LOADING TO PGVECTOR (step8)")
    print("=" * 70, flush=True)
    os.environ.setdefault("PGPASSWORD", "REDACTED-DEV-PASSWORD")
    step8 = load_module("step8_LoadVectors_into_pgvector.py", "step8_load")
    db_config = step8.DatabaseConfig(
        host="localhost", port=5432, database="ragchatbot", user="ragchatbot_loader",
    )
    step8.load_pipeline(
        embeddings_path=str(embeddings_path),
        db_config=db_config,
        pipeline_version="v1.1-multidoc",
        batch_size=100,
        dead_letter_path=str(CORPUS / "dead_letter.jsonl"),
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
