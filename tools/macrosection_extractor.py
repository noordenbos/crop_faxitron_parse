import argparse
import os
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw
try:
    # Optional deps for auto segmentation
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import pytesseract
    HAVE_TESSERACT = True
except Exception:
    HAVE_TESSERACT = False


def load_image_and_dpi(image_path: str) -> Tuple[np.ndarray, Tuple[float, float]]:
    img = Image.open(image_path).convert('RGB')
    dpi = img.info.get('dpi', (72.0, 72.0))
    if isinstance(dpi, tuple):
        dpi_xy = (float(dpi[0]), float(dpi[1]))
    else:
        dpi_xy = (float(dpi), float(dpi))
    return np.array(img).astype(np.uint8), dpi_xy


def load_mask(mask_path: str) -> np.ndarray:
    m = Image.open(mask_path).convert('1').convert('L')
    return np.array(m).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray, crop_margin: float = 0.1, size_th: int = 10) -> Tuple[int, int, int, int]:
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw, bh = x2 - x1, y2 - y1
    bw = int((1.0 + crop_margin) * bw)
    bh = int((1.0 + crop_margin) * bh)
    s = max(bw, bh)
    x1, y1 = max(0, cx - s // 2), max(0, cy - s // 2)
    x2, y2 = min(w, cx + s // 2), min(h, cy + s // 2)
    if (x2 - x1) <= size_th or (y2 - y1) <= size_th:
        return 0, 0, w, h
    return x1, y1, x2, y2


def rescale_bbox_between_dpi(xyxy: Tuple[int, int, int, int], src_dpi: Tuple[float, float], trg_dpi: Tuple[float, float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    if src_dpi is None or trg_dpi is None:
        return x1, y1, x2, y2
    sx, sy = float(src_dpi[0]), float(src_dpi[1])
    tx, ty = float(trg_dpi[0]), float(trg_dpi[1])
    x1 = int(round(x1 / sx * tx))
    x2 = int(round(x2 / sx * tx))
    y1 = int(round(y1 / sy * ty))
    y2 = int(round(y2 / sy * ty))
    return x1, y1, x2, y2


def crop_macrosection(faxitron_rgb: np.ndarray, bbox_trg: Tuple[int, int, int, int]) -> np.ndarray:
    h, w, _ = faxitron_rgb.shape
    x1, y1, x2, y2 = bbox_trg
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return faxitron_rgb[y1:y2, x1:x2]


def save_image(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path)


def visualize_bbox_on_faxitron(faxitron_rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]], out_path: str) -> None:
    img = Image.fromarray(faxitron_rgb.copy())
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 165, 0), (128, 0, 128)]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    img.save(out_path)


def detect_greyscale_slabs(img_gray: np.ndarray, variation_threshold: float = 0.05) -> List[Tuple[int, int, int, int]]:
    """Detect tissue slabs based on greyscale variation."""
    print("Detecting greyscale slabs...")
    
    # Remove right 5% of image (boilerplate metadata)
    right_cutoff = int(img_gray.shape[1] * 0.95)
    img_gray = img_gray[:, :right_cutoff]
    
    # Remove full white pixels
    img_gray[img_gray == 255] = 0
    
    # Calculate coefficient of variation for each pixel's neighborhood
    kernel_size = 15
    pad = kernel_size // 2
    img_padded = cv2.copyMakeBorder(img_gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    
    variation_map = np.zeros_like(img_gray, dtype=np.float32)
    for y in range(img_gray.shape[0]):
        for x in range(img_gray.shape[1]):
            if img_gray[y, x] > 0:
                neighborhood = img_padded[y:y+kernel_size, x:x+kernel_size]
                if np.std(neighborhood) > 0:
                    cv = np.std(neighborhood) / np.mean(neighborhood)
                    variation_map[y, x] = cv
    
    # Threshold based on variation
    slab_mask = variation_map > variation_threshold
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    slab_mask = cv2.morphologyEx(slab_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    slab_mask = cv2.morphologyEx(slab_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slab_mask, connectivity=8)
    
    # Filter components by area and aspect ratio
    slabs = []
    areas = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = h / w if w > 0 else 0
        area_ratio = area / (img_gray.shape[0] * img_gray.shape[1])
        
        # Very aggressive filtering for slabs
        if (aspect_ratio > 5.0 and area_ratio < 0.0001) or area_ratio < 0.0001:
            continue
            
        if area > 100000:  # Size-based filtering
            slabs.append((x, y, x + w, y + h))
            areas.append(area)
    
    # If we found too few slabs, use alternative detection
    if len(slabs) < 5:
        print(f"Warning: Only found {len(slabs)} slabs, using alternative detection...")
        # Simple thresholding approach
        _, binary = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        slabs = []
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100000:  # Large areas only
                x, y, w, h = cv2.boundingRect(contour)
                slabs.append((x, y, x + w, y + h))
                areas.append(area)
    
    print(f"Detected {len(slabs)} slabs")
    return slabs

def detect_grey_numbers(img_gray: np.ndarray, slabs: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Detect grey numbers near the detected slabs."""
    print("Detecting grey numbers...")
    
    # Remove right 5% of image
    right_cutoff = int(img_gray.shape[1] * 0.95)
    img_gray = img_gray[:, :right_cutoff]
    
    # Remove full white pixels
    img_gray[img_gray == 255] = 0
    
    # Multi-strategy approach for number detection
    numbers = []
    
    # Strategy 1: Look above each slab
    for slab_x1, slab_y1, slab_x2, slab_y2 in slabs:
        # Search region above the slab
        search_y1 = max(0, slab_y1 - 100)
        search_y2 = slab_y1
        search_x1 = max(0, slab_x1 - 50)
        search_x2 = min(img_gray.shape[1], slab_x2 + 50)
        
        if search_y2 > search_y1:
            search_region = img_gray[search_y1:search_y2, search_x1:search_x2]
            if search_region.size > 0:
                # Find connected components in search region
                _, binary = cv2.threshold(search_region, 1, 255, cv2.THRESH_BINARY)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                
                for i in range(1, num_labels):
                    x, y, w, h, area = stats[i]
                    if 1000 <= area <= 1200:  # Tight size range based on our analysis
                        # Convert back to full image coordinates
                        full_x = x + search_x1
                        full_y = y + search_y1
                        numbers.append((full_x, full_y, full_x + w, full_y + h))
    
    # Strategy 2: Global scan with multiple thresholds
    for threshold in [1, 5, 10]:
        _, binary = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if 1000 <= area <= 1200:  # Tight size range
                numbers.append((x, y, x + w, y + h))
    
    # Remove duplicates
    unique_numbers = []
    for num in numbers:
        is_duplicate = False
        for existing in unique_numbers:
            # Check if boxes overlap significantly
            overlap_x = max(0, min(num[2], existing[2]) - max(num[0], existing[0]))
            overlap_y = max(0, min(num[3], existing[3]) - max(num[1], existing[1]))
            if overlap_x > 0 and overlap_y > 0:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_numbers.append(num)
    
    print(f"Detected {len(unique_numbers)} unique numbers")
    return unique_numbers

def parse_number_from_region_tesseract(img_gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    """Parse number from a region using Tesseract OCR."""
    if not HAVE_TESSERACT:
        return "?"
    
    x1, y1, x2, y2 = bbox
    region = img_gray[y1:y2, x1:x2]
    
    # Preprocess for better OCR
    region = cv2.resize(region, (region.shape[1]*2, region.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    _, region = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR with Tesseract
    try:
        text = pytesseract.image_to_string(region, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789')
        text = text.strip()
        
        # Get confidence
        data = pytesseract.image_to_data(region, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789', output_type=pytesseract.Output.DICT)
        if data['conf']:
            confidence = max(data['conf'])
            if confidence > 10:  # Very low threshold to catch all numbers
                return text if text else "?"
        
        return "?"
    except Exception as e:
        print(f"OCR error: {e}")
        return "?"

def match_numbers_to_slabs_corner_distance(numbers: List[Tuple[int, int, int, int]], 
                                         slabs: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int]]:
    """Match numbers to slabs using corner distance method."""
    print("Matching numbers to slabs using corner distance...")
    
    matches = []
    used_slabs = set()
    
    for num_idx, (num_x1, num_y1, num_x2, num_y2) in enumerate(numbers):
        num_centroid = ((num_x1 + num_x2) // 2, (num_y1 + num_y2) // 2)
        
        # Calculate distances to all slab corners
        slab_distances = []
        for slab_idx, (slab_x1, slab_y1, slab_x2, slab_y2) in enumerate(slabs):
            if slab_idx in used_slabs:
                continue
                
            # Distance to upper-left corner
            dist_ul = np.sqrt((num_centroid[0] - slab_x1)**2 + (num_centroid[1] - slab_y1)**2)
            # Distance to upper-right corner  
            dist_ur = np.sqrt((num_centroid[0] - slab_x2)**2 + (num_centroid[1] - slab_y1)**2)
            
            # Use minimum distance
            min_dist = min(dist_ul, dist_ur)
            slab_distances.append((slab_idx, min_dist))
        
        if slab_distances:
            # Sort by distance and pick closest
            slab_distances.sort(key=lambda x: x[1])
            best_slab_idx, best_distance = slab_distances[0]
            
            matches.append((num_idx, best_slab_idx))
            used_slabs.add(best_slab_idx)
            
            print(f"  Number {num_idx} -> Slab {best_slab_idx} (distance: {best_distance:.1f})")
    
    return matches

def visualize_size_debug(img_gray: np.ndarray, slabs: List[Tuple[int, int, int, int]], 
                         numbers: List[Tuple[int, int, int, int]], output_path: str):
    """Create debug visualization showing all components color-coded by size."""
    print("Creating size debug visualization...")
    
    # Create RGB image
    vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    
    # Draw slabs in blue
    for i, (x1, y1, x2, y2) in enumerate(slabs):
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(vis_img, f"SLAB_{i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Draw numbers in green
    for i, (x1, y1, x2, y2) in enumerate(numbers):
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(vis_img, f"NUM_{i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Find and color-code all connected components by size
    _, binary = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100000:  # Large - blue (already drawn as slabs)
            continue
        elif 1000 <= area <= 1200:  # Medium - green (already drawn as numbers)
            continue
        elif 100 < area <= 1000:  # Small - yellow
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        else:  # Very small - red
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    
    cv2.imwrite(output_path, vis_img)
    print(f"Size debug visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract per-slab macrosections from a Faxitron using source masks with DPI alignment.')
    parser.add_argument('--faxitron', type=str, required=True, help='Path to Faxitron target image')
    parser.add_argument('--faxitron-mask', type=str, default=None, help='Optional Faxitron mask for visualization/filtering (not used for bbox)')
    parser.add_argument('--sources', type=str, nargs='+', default=None, help='List of source (segment) image paths (for DPI reference)')
    parser.add_argument('--source-masks', type=str, nargs='+', default=None, help='List of source mask paths (same order as sources)')
    parser.add_argument('--auto', action='store_true', help='Faxitron-only mode: auto-detect slabs from Faxitron image (cv2 required)')
    parser.add_argument('--min-area', type=int, default=5000, help='Minimum area (pixels) for auto-detected component to be considered a slab')
    parser.add_argument('--color-sat-th', type=int, default=60, help='Saturation threshold to consider a pixel colored (HSV S>=th)')
    parser.add_argument('--color-val-th', type=int, default=60, help='Value threshold to consider a colored pixel non-dark (HSV V>=th)')
    parser.add_argument('--white-th', type=int, default=240, help='Intensity threshold to consider a pixel near-white (removed as text/labels)')
    parser.add_argument('--keep-color-annotations', action='store_true', help='Do not remove colored annotations before segmentation')
    parser.add_argument('--keep-white-text', action='store_true', help='Do not remove near-white text/labels before segmentation')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory to save crops and overlays')
    parser.add_argument('--crop-margin', type=float, default=0.1, help='Relative margin around mask bbox')
    parser.add_argument('--viz', action='store_true', help='Also save an overlay visualization of all boxes on the Faxitron')
    parser.add_argument('--greyscale-detection', action='store_true', help='Detect greyscale slabs and numbers automatically')
    parser.add_argument('--debug-size', action='store_true', help='Generate debug visualization showing all components color-coded by size')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fax_rgb, fax_dpi = load_image_and_dpi(args.faxitron)
    if args.faxitron_mask is not None and os.path.isfile(args.faxitron_mask):
        fax_mask = load_mask(args.faxitron_mask)
    else:
        fax_mask = None

    overlay_boxes = []

    if args.greyscale_detection:
        if not HAVE_CV2:
            raise RuntimeError('Greyscale detection mode requires OpenCV (cv2). Please install opencv-python.')
        if not HAVE_TESSERACT:
            print("Warning: Tesseract not available. Number parsing will be limited.")
        
        print("Running greyscale detection mode...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(fax_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect slabs
        slabs = detect_greyscale_slabs(gray)
        
        # Detect numbers
        numbers = detect_grey_numbers(gray, slabs)
        
        # Parse numbers using OCR
        parsed_numbers = []
        for i, num_bbox in enumerate(numbers):
            parsed_num = parse_number_from_region_tesseract(gray, num_bbox)
            parsed_numbers.append(parsed_num)
            print(f"Number {i}: {parsed_num}")
        
        # Match numbers to slabs
        matches = match_numbers_to_slabs_corner_distance(numbers, slabs)
        
        # Save number-to-slab mapping
        with open(os.path.join(args.out_dir, 'number_slab_matches.txt'), 'w') as f:
            f.write("Number to Slab Mapping:\n")
            for num_idx, slab_idx in matches:
                f.write(f"Number {parsed_numbers[num_idx]} -> Slab {slab_idx}\n")
        
        # Save preprocessed image
        cv2.imwrite(os.path.join(args.out_dir, 'preprocessed_greyscale.png'), gray)
        
        # Create overlay visualization
        overlay_path = os.path.join(args.out_dir, 'greyscale_detection_overlay.png')
        visualize_bbox_on_faxitron(fax_rgb, slabs + numbers, overlay_path)
        
        # Save individual slab crops
        for idx, (x1, y1, x2, y2) in enumerate(slabs):
            crop = crop_macrosection(fax_rgb, (x1, y1, x2, y2))
            crop_name = f'greyscale_slab_{idx:03d}.png'
            save_image(crop, os.path.join(args.out_dir, crop_name))
            overlay_boxes.append((x1, y1, x2, y2))
        
        # Create size debug visualization if requested
        if args.debug_size:
            debug_path = os.path.join(args.out_dir, 'size_debug_visualization.png')
            visualize_size_debug(gray, slabs, numbers, debug_path)
        
        print(f"Greyscale detection complete. Found {len(slabs)} slabs and {len(numbers)} numbers.")
        
    elif args.auto:
        if not HAVE_CV2:
            raise RuntimeError('Auto mode requires OpenCV (cv2). Please install opencv-python.')
        img_clean = fax_rgb.copy()
        # Remove colored annotations using HSV saturation/value
        if not args.keep_color_annotations:
            hsv = cv2.cvtColor(img_clean, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            color_mask = (s >= args.color_sat_th) & (v >= args.color_val_th)
            img_clean[color_mask] = 0
        # Remove near-white text/labels
        if not args.keep_white_text:
            white_mask = (img_clean[:, :, 0] >= args.white_th) & (img_clean[:, :, 1] >= args.white_th) & (img_clean[:, :, 2] >= args.white_th)
            img_clean[white_mask] = 0

        gray = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)
        # Otsu thresholding to segment slabs from background
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Ensure slabs are white blobs; invert if needed
        if np.mean(th) < 127:
            th = 255 - th
        # Morphology to clean noise and close gaps
        kernel = np.ones((9, 9), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((th > 0).astype(np.uint8), connectivity=8)
        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < args.min_area:  # skip small artifacts
                continue
            # Expand to square with margin
            cx, cy = x + w // 2, y + h // 2
            s = int(max(w, h) * (1.0 + args.crop_margin))
            x1 = max(0, cx - s // 2)
            y1 = max(0, cy - s // 2)
            x2 = min(fax_rgb.shape[1], cx + s // 2)
            y2 = min(fax_rgb.shape[0], cy + s // 2)
            boxes.append(((x1, y1, x2, y2), cx))
        # Sort left-to-right by centroid x
        boxes.sort(key=lambda b: b[1])
        for idx, (bbox, _) in enumerate(boxes):
            crop = crop_macrosection(fax_rgb, bbox)
            crop_name = f'macrosection_{idx:03d}.png'
            save_image(crop, os.path.join(args.out_dir, crop_name))
            overlay_boxes.append(bbox)
    else:
        if args.sources is None or args.source_masks is None:
            raise ValueError('Provide --sources and --source-masks, or use --auto for Faxitron-only mode')
        if len(args.sources) != len(args.source_masks):
            raise ValueError('sources and source-masks must have the same length')
        for idx, (src_path, msk_path) in enumerate(zip(args.sources, args.source_masks)):
            _, src_dpi = load_image_and_dpi(src_path)
            mask = load_mask(msk_path)
            bbox_src = bbox_from_mask(mask, crop_margin=args.crop_margin)
            bbox_trg = rescale_bbox_between_dpi(bbox_src, src_dpi, fax_dpi)
            crop = crop_macrosection(fax_rgb, bbox_trg)
            crop_name = f'macrosection_{idx:03d}.png'
            save_image(crop, os.path.join(args.out_dir, crop_name))
            overlay_boxes.append(bbox_trg)

    if args.viz:
        overlay_path = os.path.join(args.out_dir, 'faxitron_overlay_boxes.png')
        visualize_bbox_on_faxitron(fax_rgb, overlay_boxes, overlay_path)


if __name__ == '__main__':
    main()


