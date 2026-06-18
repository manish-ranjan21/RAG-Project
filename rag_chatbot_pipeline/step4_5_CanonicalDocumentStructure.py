"""
Stage 4 & 5: Build Canonical Document Structure

Use pypdf to extract the Table of Contents (TOC/bookmarks) which gives us
the AUTHORITATIVE section structure. Combine with extracted text per page.
This is more reliable than inferring sections from font sizes.
"""
import json
from pathlib import Path
from pypdf import PdfReader
import pdfplumber


def extract_toc(pdf_path: str) -> list:
    """Extract bookmarks/outline as our section structure."""
    reader = PdfReader(pdf_path)
    
    def walk_outline(items, level=1, parent_path=None):
        sections = []
        parent_path = parent_path or []
        
        for item in items:
            if isinstance(item, list):
                if sections:
                    sections.extend(walk_outline(item, level + 1, 
                                                  parent_path + [sections[-1]["title"]]))
            else:
                try:
                    page_num = reader.get_destination_page_number(item) + 1
                    sections.append({
                        "title": item.title,
                        "level": level,
                        "page_start": page_num,
                        "parent_path": parent_path.copy()
                    })
                except Exception:
                    continue
        return sections
    
    sections = walk_outline(reader.outline)
    
    for i in range(len(sections) - 1):
        sections[i]["page_end"] = sections[i + 1]["page_start"] - 1
    if sections:
        sections[-1]["page_end"] = len(reader.pages)
    
    return sections


def extract_section_text(pdf_path: str, page_start: int, page_end: int) -> str:
    """Extract clean text for a page range."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(page_start, min(page_end + 1, len(pdf.pages) + 1)):
            page = pdf.pages[page_num - 1]
            text = page.extract_text() or ""
            parts.append(text)
    return "\n\n".join(parts)


def build_canonical_document(pdf_path: str, doc_id: str, image_routing_path: str):
    """Assemble unified document representation with sections and elements."""
    
    sections = extract_toc(pdf_path)
    
    if not sections:
        return None
    
    with open(image_routing_path) as f:
        routing_data = json.load(f)
    
    image_by_page = {}
    for img_record in routing_data["routing_decisions"]:
        if "routing" not in img_record:
            continue
        if img_record["routing"]["decision"] != "skip_decorative":
            page = img_record["page"]
            image_by_page.setdefault(page, []).append({
                "element_id": img_record["element_id"],
                "bbox": img_record["bbox"],
                "decision": img_record["routing"]["decision"],
                "description": f"[Figure on page {page}: would be described by VLM in production]"
            })
    
    canonical = {
        "doc_id": doc_id,
        "doc_type": "ebook",
        "title": "Building Machine Learning Powered Applications",
        "metadata": {
            "author": "Emmanuel Ameisen",
            "page_count": 308,
            "access_classification": "internal",
            "source_path": f"{pdf_path}"
        },
        "structure": []
    }
    
    for section in sections:
        section_text = extract_section_text(pdf_path, section["page_start"], 
                                             section["page_end"])
        
        section_images = []
        for p in range(section["page_start"], section["page_end"] + 1):
            if p in image_by_page:
                section_images.extend(image_by_page[p])
        
        canonical["structure"].append({
            "element_id": f"sec_{section['page_start']:03d}",
            "type": "section",
            "heading": section["title"].strip(),
            "level": section["level"],
            "parent_path": section["parent_path"],
            "page_range": [section["page_start"], section["page_end"]],
            "text": section_text,
            "figures": section_images
        })
    
    return canonical


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent  # self-contained: docs/ & data/ live in this pipeline folder
    pdf_path = base_dir/ "docs" / "building-machine-learning-powered-applications-going-from-idea-to-product.pdf"
    routing_path = base_dir/ "data" / "output" /"image_routing.json"
    
    canonical = build_canonical_document(pdf_path, "ebook", routing_path)
    
    if canonical is None:
        print("No TOC found - would fall back to font-based section detection")
    else:
        output_path = base_dir/ "data" / "output" / "canonical_document.json"
        with open(output_path, "w") as f:
            json.dump(canonical, f, indent=2)
        
        print("=" * 60)
        print("STAGE 4-5: CANONICAL DOCUMENT BUILT")
        print("=" * 60)
        print(f"Doc: {canonical['title']}")
        print(f"Total sections: {len(canonical['structure'])}")
        print(f"\nFirst 12 sections:")
        for s in canonical["structure"][:12]:
            indent = "  " * (s["level"] - 1)
            text_len = len(s["text"])
            fig_count = len(s["figures"])
            print(f"  {indent}- L{s['level']} [{s['page_range'][0]:3d}-{s['page_range'][1]:3d}] "
                  f"{s['heading'][:50]:<50} ({text_len:>5} chars, {fig_count} figs)")