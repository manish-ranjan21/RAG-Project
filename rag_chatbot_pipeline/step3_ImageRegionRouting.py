"""
Step 3: Image Region Routing
Apply cascading cheap filters to decide what to do with each image:
- SKIP_DECORATIVE: logos, dividers, tiny icons
- PROCESS_AS_CHART: charts, diagrams (would go to VLM in production)
- PROCESS_AS_SCAN: text-heavy images (would go to OCR)
- PROCESS_AS_SCREENSHOT: UI captures
 
All filters here are FREE (no API calls). Only the final VLM/OCR step costs money.
"""

import json
from pathlib import Path
from enum import Enum
import io

import pdfplumber
from pypdf import PdfReader
from PIL import Image
import numpy as np





class ImageDecision(Enum):
    SKIP_DECORATIVE = "skip_decorative"
    PROCESS_AS_CHART = "process_as_chart"
    PROCESS_AS_DIAGRAM = "process_as_diagram"
    PROCESS_AS_SCAN = "process_as_scan"
    PROCESS_AS_SCREENSHOT = "process_as_screenshot"



def extract_image_from_page(pdf_path: str, page_num: int, bbox: list) -> bytes:
    """Extract image bytes from a PDF page region"""

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        cropped = page.crop(bbox)
        img = cropped.to_image(resolution=150)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    

def get_size_info(image_bytes: bytes, page_width: float, page_height: float, bbox: list) -> dict:
    """Compute size-related metrics."""
    img = Image.open(io.BytesIO(image_bytes))
    width_px, height_px = img.size
    
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    bbox_area = bbox_width * bbox_height
    page_area = page_width * page_height
    
    return {
        "width_px": width_px,
        "height_px": height_px,
        "min_dimension_px": min(width_px, height_px),
        "bbox_area": bbox_area,
        "area_ratio": bbox_area / page_area if page_area > 0 else 0,
        "aspect_ratio": width_px / height_px if height_px > 0 else 1.0
    }

def get_position_zone(bbox: list, page_width: float, page_height: float) -> str:
    """Determine where on the page the image sits."""
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    rel_x = center_x / page_width
    rel_y = center_y / page_height
    
    if rel_y < 0.10:
        if rel_x < 0.15 or rel_x > 0.85:
            return "header_corner"
        return "header"
    if rel_y > 0.90:
        return "footer"
    if rel_x < 0.10 or rel_x > 0.90:
        return "margin"
    return "body"



def analyze_visual_properties(image_bytes: bytes) -> dict:
    """Cheap pixel-level analysis to detect logos vs information-rich images."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    
    quantized = (arr // 32) * 32
    flat = quantized.reshape(-1, 3)
    unique = np.unique(flat, axis=0)
    unique_color_count = len(unique)
    
    if len(unique) > 1:
        _, counts = np.unique(flat, axis=0, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        top_5_share = sorted_counts[:5].sum() / counts.sum()
    else:
        top_5_share = 1.0
    
    gray = np.mean(arr, axis=2)
    h_edges = np.abs(np.diff(gray, axis=1)).mean()
    v_edges = np.abs(np.diff(gray, axis=0)).mean()
    edge_density = (h_edges + v_edges) / 2
    
    white_pixels = np.sum(np.all(arr > 240, axis=2))
    total_pixels = arr.shape[0] * arr.shape[1]
    whitespace_ratio = white_pixels / total_pixels
    
    return {
        "unique_colors": int(unique_color_count),
        "top_5_color_share": float(top_5_share),
        "edge_density": float(edge_density),
        "whitespace_ratio": float(whitespace_ratio),
        "pixel_variance": float(arr.var())
    }




def route_image(image_bytes: bytes, bbox: list, page_width: float, page_height: float) -> dict:
    """
    Run cascading filters to decide what to do with this image.
    Returns decision + reasoning + which filters fired.
    """
    filters_evaluated = []
    
    size = get_size_info(image_bytes, page_width, page_height, bbox)
    filters_evaluated.append("size")
    
    if size["min_dimension_px"] < 80 or size["area_ratio"] < 0.01:
        return {
            "decision": ImageDecision.SKIP_DECORATIVE.value,
            "reason": f"too_small: {size['width_px']}x{size['height_px']}, area={size['area_ratio']:.4f}",
            "confidence": 0.95,
            "filters_evaluated": filters_evaluated,
            "metrics": {"size": size}
        }
    
    zone = get_position_zone(bbox, page_width, page_height)
    filters_evaluated.append("position")
    
    if zone in ("header_corner", "footer") and size["min_dimension_px"] < 150:
        return {
            "decision": ImageDecision.SKIP_DECORATIVE.value,
            "reason": f"position_decorative: zone={zone}",
            "confidence": 0.85,
            "filters_evaluated": filters_evaluated,
            "metrics": {"size": size, "zone": zone}
        }
    
    visual = analyze_visual_properties(image_bytes)
    filters_evaluated.append("visual")
    
    if visual["unique_colors"] < 8 and size["min_dimension_px"] < 200:
        return {
            "decision": ImageDecision.SKIP_DECORATIVE.value,
            "reason": f"few_colors_and_small: colors={visual['unique_colors']}, dim={size['min_dimension_px']}",
            "confidence": 0.80,
            "filters_evaluated": filters_evaluated,
            "metrics": {"size": size, "zone": zone, "visual": visual}
        }
    
    if visual["top_5_color_share"] > 0.95 and size["min_dimension_px"] < 300:
        return {
            "decision": ImageDecision.SKIP_DECORATIVE.value,
            "reason": f"uniform_colors: top_5_share={visual['top_5_color_share']:.2f}",
            "confidence": 0.75,
            "filters_evaluated": filters_evaluated,
            "metrics": {"size": size, "zone": zone, "visual": visual}
        }
    
    if visual["edge_density"] > 15 and visual["whitespace_ratio"] > 0.4:
        return {
            "decision": ImageDecision.PROCESS_AS_SCAN.value,
            "reason": "high_text_density_pattern",
            "confidence": 0.75,
            "filters_evaluated": filters_evaluated,
            "metrics": {"size": size, "zone": zone, "visual": visual}
        }
    
    if size["aspect_ratio"] > 1.3 and size["area_ratio"] > 0.1:
        return {
            "decision": ImageDecision.PROCESS_AS_DIAGRAM.value,
            "reason": "large_wide_image_in_body",
            "confidence": 0.70,
            "filters_evaluated": filters_evaluated,
            "metrics": {"size": size, "zone": zone, "visual": visual}
        }
    
    return {
        "decision": ImageDecision.PROCESS_AS_CHART.value,
        "reason": "body_image_passed_all_filters",
        "confidence": 0.60,
        "filters_evaluated": filters_evaluated,
        "metrics": {"size": size, "zone": zone, "visual": visual}
    }


def process_all_image_regions(layout_output_path: str, pdf_path: str):
    """Walk through all image regions and route each one."""
    with open(layout_output_path) as f:
        layout = json.load(f)
    
    results = []
    counts = {d.value: 0 for d in ImageDecision}
    
    image_elements = []
    for page in layout["pages"]:
        for elem in page["elements"]:
            if elem["type"] == "FIGURE":
                image_elements.append({
                    "page_number": page["page_number"],
                    "page_width": page["width"],
                    "page_height": page["height"],
                    "element": elem
                })
    
    print(f"Total image regions to route: {len(image_elements)}")
    
    for i, item in enumerate(image_elements):
        try:
            image_bytes = extract_image_from_page(
                pdf_path, item["page_number"], item["element"]["bbox"]
            )
            
            routing = route_image(
                image_bytes,
                item["element"]["bbox"],
                item["page_width"],
                item["page_height"]
            )
            
            counts[routing["decision"]] += 1
            
            results.append({
                "element_id": item["element"]["id"],
                "page": item["page_number"],
                "bbox": item["element"]["bbox"],
                "routing": routing
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Routed {i+1}/{len(image_elements)} images...")
        except Exception as e:
            results.append({
                "element_id": item["element"]["id"],
                "page": item["page_number"],
                "error": str(e)
            })
    
    return results, counts


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent  # self-contained: docs/ & data/ live in this pipeline folder
    pdf_path = base_dir / "docs" / "building-machine-learning-powered-applications-going-from-idea-to-product.pdf"
    layout_path  = base_dir/ "data" / "output" / "layout_analysis.json"
    
    results, counts = process_all_image_regions(layout_path, pdf_path)
    
    output_path =  base_dir/ "data" / "output" /"image_routing.json"
    with open(output_path, "w") as f:
        json.dump({"routing_decisions": results, "summary": counts}, f, indent=2)
    
    print("=" * 60)
    print("STAGE 3: IMAGE ROUTING COMPLETE")
    print("=" * 60)
    print(f"\nRouting summary:")
    for decision, count in counts.items():
        if count > 0:
            print(f"  {decision}: {count}")
    
    cost_savings = counts["skip_decorative"] * 0.01
    process_cost = (counts["process_as_chart"] + counts["process_as_diagram"]) * 0.01
    scan_cost = counts["process_as_scan"] * 0.001
    print(f"\nEstimated VLM cost saved by filtering: ${cost_savings:.3f}")
    print(f"Estimated VLM cost to process charts: ${process_cost:.3f}")
    print(f"Estimated OCR cost for scans: ${scan_cost:.4f}")

