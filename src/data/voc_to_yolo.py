"""
voc_to_yolo.py - Pascal VOC XML to YOLO Annotation Converter
-------------------------------------------------------------
Converts VOC bounding boxes to normalized YOLO format (class x_center y_center w h),
maps class labels ('hat' -> 0, 'person'/'no_helmet' -> 1),
handles background images, and verifies lossless conversion with an IoU round-trip test.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple
import shutil
from tqdm import tqdm


# Label mapping: Note that 'person' in SHWD specifically denotes a bare human head.
# We map it to 'no_helmet' for domain clarity across our system.
CLASS_MAPPING: Dict[str, int] = {
    "hat": 0,
    "person": 1,        # Original VOC tag for bare head
    "no_helmet": 1,     # Target domain name
}

ID_TO_CLASS: Dict[int, str] = {
    0: "hat",
    1: "no_helmet"
}


def parse_voc_xml(xml_path: Path) -> Tuple[int, int, List[Tuple[int, float, float, float, float]]]:
    """
    Parses a VOC XML annotation file.
    
    Returns:
        width (int), height (int), list of boxes as (class_id, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size_elem = root.find("size")
    if size_elem is None:
        raise ValueError(f"Missing <size> tag in {xml_path}")
        
    width = int(float(size_elem.find("width").text))
    height = int(float(size_elem.find("height").text))
    
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_MAPPING:
            continue
            
        class_id = CLASS_MAPPING[name]
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        boxes.append((class_id, xmin, ymin, xmax, ymax))
        
    return width, height, boxes


def voc_to_yolo_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Converts absolute VOC coordinates to normalized YOLO format.
    """
    # Clamp coordinates to image boundaries
    xmin = max(0.0, min(float(img_w), xmin))
    ymin = max(0.0, min(float(img_h), ymin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymax = max(0.0, min(float(img_h), ymax))
    
    bw = xmax - xmin
    bh = ymax - ymin
    xc = xmin + (bw / 2.0)
    yc = ymin + (bh / 2.0)
    
    return xc / img_w, yc / img_h, bw / img_w, bh / img_h


def yolo_to_voc_bbox(
    xc: float, yc: float, bw: float, bh: float, img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Converts normalized YOLO coordinates back to absolute VOC coordinates.
    """
    abs_xc = xc * img_w
    abs_yc = yc * img_h
    abs_w = bw * img_w
    abs_h = bh * img_h
    
    xmin = abs_xc - (abs_w / 2.0)
    ymin = abs_yc - (abs_h / 2.0)
    xmax = abs_xc + (abs_w / 2.0)
    ymax = abs_yc + (abs_h / 2.0)
    return xmin, ymin, xmax, ymax


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def round_trip_check(xmin: float, ymin: float, xmax: float, ymax: float, img_w: int, img_h: int) -> float:
    """
    Verifies that VOC -> YOLO -> VOC round-trip is lossless within numerical precision.
    """
    xc, yc, bw, bh = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
    r_xmin, r_ymin, r_xmax, r_ymax = yolo_to_voc_bbox(xc, yc, bw, bh, img_w, img_h)
    return compute_iou((xmin, ymin, xmax, ymax), (r_xmin, r_ymin, r_xmax, r_ymax))


def convert_dataset(
    voc_root: Path,
    output_dir: Path = Path("data/processed"),
    verify_roundtrip: bool = True
) -> Dict[str, int]:
    """
    Converts full VOC dataset to YOLO structure:
    data/processed/images/
    data/processed/labels/
    """
    img_dir = voc_root / "JPEGImages" if (voc_root / "JPEGImages").is_dir() else voc_root / "images"
    ann_dir = voc_root / "Annotations"
    
    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(ann_dir.glob("*.xml"))
    stats = {"total_images": 0, "total_boxes": 0, "hat_boxes": 0, "no_helmet_boxes": 0, "empty_images": 0}
    
    print(f"[INFO] Converting {len(xml_files)} annotations from {ann_dir} to {output_dir}...")
    min_roundtrip_iou = 1.0
    
    for xml_file in tqdm(xml_files, desc="Converting VOC to YOLO"):
        stem = xml_file.stem
        img_candidates = list(img_dir.glob(f"{stem}.*"))
        if not img_candidates:
            continue
            
        src_img = img_candidates[0]
        dst_img = out_images / src_img.name
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)
            
        w, h, boxes = parse_voc_xml(xml_file)
        stats["total_images"] += 1
        
        label_file = out_labels / f"{stem}.txt"
        yolo_lines = []
        
        if not boxes:
            stats["empty_images"] += 1
            label_file.touch(exist_ok=True)
            continue
            
        for cid, xmin, ymin, xmax, ymax in boxes:
            if xmax <= xmin or ymax <= ymin:
                # Discard degenerate zero/negative area boxes
                continue
                
            xc, yc, bw, bh = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, w, h)
            
            # Lossless validation
            if verify_roundtrip:
                iou = round_trip_check(xmin, ymin, xmax, ymax, w, h)
                min_roundtrip_iou = min(min_roundtrip_iou, iou)
                assert iou > 0.995, f"Round-trip failed on {xml_file.name}: IoU={iou:.5f}"
                
            yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            stats["total_boxes"] += 1
            if cid == 0:
                stats["hat_boxes"] += 1
            else:
                stats["no_helmet_boxes"] += 1
                
        with open(label_file, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            
    print(f"\n[SUCCESS] Conversion complete. Min round-trip IoU across dataset: {min_roundtrip_iou:.6f}")
    print(f"Stats: {stats}")
    return stats


if __name__ == "__main__":
    voc_path = Path("data/raw/VOC2028")
    if voc_path.exists():
        convert_dataset(voc_path)
    else:
        print(f"VOC path {voc_path} not found. Run download.py first.")
