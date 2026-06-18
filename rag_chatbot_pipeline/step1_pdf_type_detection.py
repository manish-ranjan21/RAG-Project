import json
from pathlib import Path
import pdfplumber
from pypdf import PdfReader


def detect_pdf_type(pdf_path: str) -> dict:
    """
    Analyze PDF characteristics to determine procerssing strategy
    """

    reader = PdfReader(pdf_path)
    page_count = len(reader.pages)

    total_chars = 0
    pages_without_text = 0
    pages_with_images = 0
    total_images = 0


    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            total_chars += len(text.strip())


            if len(text.strip()) < 50:
                pages_without_text += 1

            if page.images:
                pages_with_images += 1
                total_images += len(page.images)

    print(f"total_page - {page_count}\ntotal_chars - {total_chars}\npages_without_text - {pages_without_text}\npages_with_images - {pages_with_images}\ntotal_images - {total_images}")

    avg_chars_per_page = total_chars/page_count if page_count > 0 else 0

    if pages_without_text > page_count * 0.7:
        primary_type = "scanned"
    elif total_images > 0 and avg_chars_per_page > 200:
        primary_type = "mixed_content"
    elif avg_chars_per_page > 200:
        primary_type = "native_text"
    else:
        primary_type = "sparse_text"

    return {
        "doc_id": Path(pdf_path).stem,
        "page_count": page_count,
        "total_text_chars": total_chars,
        "avg_chars_per_page": avg_chars_per_page,
        "pages_with_images":pages_with_images,
        "total_images":total_images,
        "scanned_pages": pages_without_text,
        "primary_type": primary_type,
        "processing_strategy": "full_layout_analysis" if primary_type == "mixed_content" else "text_extraction"
    }


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent  # self-contained: docs/ & data/ live in this pipeline folder
    pdf_path = base_dir / "docs" / "building-machine-learning-powered-applications-going-from-idea-to-product.pdf"
    pdf_detection = detect_pdf_type(pdf_path)

    output_path = base_dir / "data" / "output" / "pdf_detection.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(pdf_detection, f , indent=2)

    print("=" * 60)
    print("Step 1: Type Detection")
    print("=" * 60)
    print(json.dumps(pdf_detection, indent=2))