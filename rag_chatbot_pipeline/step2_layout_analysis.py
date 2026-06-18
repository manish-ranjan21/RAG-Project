"""
Step 2: Layout Analysis (Open sourece alternative to textract)
- Text blocks with positions (bounding boxes)
- Tables (with row/column structure)
- Images (with bounding boxes)
- Font information (for heading detection)
"""

import json
from pathlib import Path
import pdfplumber
from pypdf import PdfReader



def extract_layout_elemnts(pdf_path: str, max_pages: int = None) -> dict:
    """
    Extract structured layout from PDF.
    Returns a textract(AWS)-equivalent structure with typed elemenbts per page.
    """

    pages_data = []

    #max_pages = 100
    with pdfplumber.open(pdf_path) as pdf:

        page_iter = pdf.pages[:max_pages] if max_pages else pdf.pages

        for page_num, page in enumerate(page_iter, 1):
            #print(f"{page_num}\n{page}")
            elements = []

            tables = page.extract_tables() or []
            table_bboxes = []

            for table_obj in page.find_tables():
                table_bboxes.append(table_obj.bbox)


            for idx, (table_data, bbox) in enumerate(zip(tables, table_bboxes)):
                if table_data and len(table_data) > 1:
                    elements.append({
                        "id": f"p{page_num}_t{idx}",
                        "type": "TABLE",
                        "bbox": list(bbox),
                        "headers": table_data[0] if table_data else [],
                        "rows": table_data[1:] if len(table_data) > 1 else [],
                        "confidence": 90.0

                    })
            words = page.extract_words(extra_attrs=["fontname","size"])
            text_blocks = group_words_into_blocks(words, table_bboxes)


            for idx, block in enumerate(text_blocks):
                element_type = classify_text_block(block)
                elements.append({
                    "idx": f"p{page_num}_text{idx}",
                    "type": element_type,
                    "text": block["text"],
                    "bbox": block["bbox"],
                    "font_size": block.get("font_size"),
                    "is_bold": block.get("is_bold", False)
                })


            for idx, img in enumerate(page.images):
                elements.append({
                    "id": f"p{page_num}_img{idx}",
                    "type": "FIGURE",
                    "bbox": [img["x0"], img["top"], img["x1"], img["bottom"]],
                    "width": img.get("width", 0),
                    "height": img.get("height", 0),
                    "image_ref": f"page{page_num}_img{idx}"
                })

            elements.sort(key=lambda e: (e["bbox"][1], e["bbox"][0]))

            pages_data.append({
                "page_number": page_num,
                "width": page.width,
                "height": page.height,
                "elements": elements
            })

    return {"pages": pages_data}


def group_words_into_blocks(words, exclude_bboxes):
    """
    Group consecutive words into text blocks based on vertical proximity.
    Skip words that fall within table bounding boxes.
    """
    if not words:
        return []
    
    filtered_words = []
    for w in words:
        word_box = (w["x0"], w["top"], w["x1"], w["bottom"])

        if not any(bbox_overlaps(word_box, tb) for tb in exclude_bboxes):
            filtered_words.append(w)

    if not filtered_words:
        return []
    blocks = []
    current_block = {"words": [filtered_words[0]], "font_sizes": [filtered_words[0].get("size", 10)]}


    for word in filtered_words[1:]:
        last_word = current_block["words"][-1]
        vertical_gap = word["top"] - last_word["bottom"]
        font_size = word.get("size", 10)

        same_block = (
            abs(word["top"] - last_word["top"]) < 5 or 
            (vertical_gap < font_size * 1.5 and abs(font_size - last_word.get("size", 10)) < 2)
        )

        if same_block:
            current_block["words"].append(word)
            current_block["font_sizes"].append(font_size)
        else:
            blocks.append(finalize_block(current_block))
            current_block = {"words": [word], "font_sizes": [font_size]}

    blocks.append(finalize_block(current_block))

    return [b for b in blocks if len(b["text"].strip()) > 0]



def finalize_block(block):
    """
    Convert a word group into a block with bbox and text.
    """

    words = block["words"]

    return {
        "text": " ".join(w["text"] for w in words),
        "bbox": [
            min(w["x0"] for w in words),
            min(w["top"] for w in words),
            max(w["x1"] for w in words),
            max(w["bottom"] for w in words)
        ],
        "font_size": max(block["font_sizes"]),
        "is_bold": any("Bold" in w.get("fontname", "") for w in words)
    }




def classify_text_block(block) -> str:
    """Classify block as TITLE, HEADING, or TEXT based on font characteristics"""

    font_size = block.get("font_size", 10)
    is_bold = block.get("is_bold", False)
    text_len = len(block["text"])

    if font_size >= 18:
        return "TITLE"
    if font_size >= 14 or (is_bold and text_len < 100):
        return "SECTION_HEADER"
    if font_size >= 12 and is_bold:
        return "SUBSECTION_HEAADER"
    
    return "TEXT"



def bbox_overlaps(box1, box2):
    """Check if two bounding boxes overlap"""
    return not (box1[2] < box2[0] or box1[0] > box2[2] or 
                box1[3] < box2[1] or box1[1] > box2[3])



if __name__ == "__main__":
    
    #pdf_path = 'docs/building-machine-learning-powered-applications-going-from-idea-to-product.pdf'
    base_dir = Path(__file__).resolve().parent  # self-contained: docs/ & data/ live in this pipeline folder
    pdf_path = base_dir / "docs" / "building-machine-learning-powered-applications-going-from-idea-to-product.pdf"

    print(f"Running layout analysis on {pdf_path.name}")
    result = extract_layout_elemnts(pdf_path=pdf_path)


    output_path = base_dir/ "data" / "output" / "layout_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    total_elements = sum(len(p["elements"]) for p in result["pages"])
    element_types = {}

    for page in result["pages"]:
        for elem in page["elements"]:
            element_types[elem["type"]] = element_types.get(elem["type"], 0) + 1

    print("=" * 60)
    print("STAGE 2: LAYOUT ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Pages processed: {len(result['pages'])}")
    print(f"Total elements: {total_elements}")
    print(f"Element types: {json.dumps(element_types, indent=2)}")
    print(f"Output: {output_path}")