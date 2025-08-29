#!/usr/bin/env python3
"""
Fast macrosection extractor with optimized greyscale detection.
"""

import argparse
import os
import datetime
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw
try:
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


def visualize_bbox_on_faxitron(faxitron_rgb: np.ndarray, boxes: List[Tuple[int, int, int, int]], 
                              labels: List[str] = None, out_path: str = None) -> None:
    img = Image.fromarray(faxitron_rgb.copy())
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 165, 0), (128, 0, 128)]
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Add label if provided
        if labels and i < len(labels):
            label = labels[i]
            # Position label above the box
            draw.text((x1, y1-20), label, fill=color, stroke_width=2)
    
    if out_path:
        img.save(out_path)
    return img


def detect_greyscale_slabs_fast(img_gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect tissue slabs using fast binary thresholding."""
    print("Detecting greyscale slabs (fast mode)...")
    
    # Remove right 5% of image (boilerplate metadata)
    right_cutoff = int(img_gray.shape[1] * 0.95)
    img_gray = img_gray[:, :right_cutoff]
    
    # Remove near-white pixels (metadata, text, labels)
    img_gray[img_gray >= 220] = 0
    
    # FAST APPROACH: Use simple thresholding + morphological operations
    # This is much faster than pixel-by-pixel variation calculation
    
    # Create binary mask from non-zero pixels
    _, binary = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up noise
    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter components by area - keep large components (slabs)
    slabs = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100000:  # Size-based filtering for slabs
            slabs.append((x, y, x + w, y + h))
    
    # If we found too few slabs, relax the filtering
    if len(slabs) < 5:
        print(f"Warning: Only found {len(slabs)} slabs, relaxing filtering...")
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 50000:  # Lower threshold
                slabs.append((x, y, x + w, y + h))
    
    print(f"Detected {len(slabs)} slabs")
    return slabs


def detect_grey_numbers_fast(img_gray: np.ndarray, slabs: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Detect grey numbers using fast size-based filtering."""
    print("Detecting grey numbers (fast mode)...")
    
    # Remove right 5% of image
    right_cutoff = int(img_gray.shape[1] * 0.95)
    img_gray = img_gray[:, :right_cutoff]
    
    # Remove near-white pixels (metadata, text, labels)
    img_gray[img_gray >= 220] = 0
    
    # Create binary mask
    _, binary = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter by size: numbers should be around 1000-1200 pixels based on our analysis
    numbers = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if 500 <= area <= 2000:  # Much wider size range to catch all numbers
            numbers.append((x, y, x + w, y + h))
    
    print(f"Detected {len(numbers)} potential numbers")
    return numbers


def parse_number_from_region_tesseract(img_gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
    """Parse number from a region using Tesseract OCR with enhanced preprocessing."""
    if not HAVE_TESSERACT:
        return "?"
    
    x1, y1, x2, y2 = bbox
    region = img_gray[y1:y2, x1:x2]
    
    # Enhanced preprocessing for better OCR
    # 1. Upscale for better resolution
    region = cv2.resize(region, (region.shape[1]*3, region.shape[0]*3), interpolation=cv2.INTER_CUBIC)
    
    # 2. Apply Gaussian blur to reduce noise
    region = cv2.GaussianBlur(region, (3, 3), 0)
    
    # 3. Try multiple binarization methods
    best_text = "?"
    best_confidence = 0
    
    # Method 1: Otsu thresholding
    _, binary1 = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 2: Adaptive thresholding
    binary2 = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Method 3: Simple threshold with manual value
    _, binary3 = cv2.threshold(region, 127, 255, cv2.THRESH_BINARY)
    
    # Test all binarization methods with multiple PSM modes
    for binary_img, method_name in [(binary1, "Otsu"), (binary2, "Adaptive"), (binary3, "Simple")]:
        for psm in [7, 8, 13]:  # Different Page Segmentation Modes
            try:
                config = f'--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789'
                text = pytesseract.image_to_string(binary_img, config=config)
                text = text.strip()
                
                if text:
                    # Get confidence
                    data = pytesseract.image_to_data(binary_img, config=config, output_type=pytesseract.Output.DICT)
                    if data['conf']:
                        confidence = max(data['conf'])
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_text = text
                            print(f"    {method_name} + PSM{psm}: '{text}' (conf: {confidence})")
            except Exception as e:
                continue
    
    # Return best result if confidence is reasonable
    if best_confidence > 0:
        return best_text if best_text else "?"
    
    return "?"


def merge_adjacent_numbers(numbers: List[Tuple[int, int, int, int]], 
                          parsed_numbers: List[str]) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
    """Merge adjacent number components and combine their parsed values."""
    print("Merging adjacent number components...")
    
    if len(numbers) <= 1:
        return numbers, parsed_numbers
    
    # Sort numbers by x-coordinate (left to right)
    sorted_indices = sorted(range(len(numbers)), key=lambda i: numbers[i][0])
    
    merged_numbers = []
    merged_parsed = []
    used_indices = set()
    
    for i in sorted_indices:
        if i in used_indices:
            continue
            
        current_num = numbers[i]
        current_parsed = parsed_numbers[i]
        merged_components = [current_parsed]
        merged_bbox = list(current_num)
        
        # Look for adjacent numbers to merge
        for j in sorted_indices:
            if j <= i or j in used_indices:
                continue
                
            other_num = numbers[j]
            other_parsed = parsed_numbers[j]
            
            # Check if numbers are adjacent (similar y-coordinate and close x-coordinate)
            y_diff = abs(current_num[1] - other_num[1])
            x_distance = other_num[0] - current_num[2]  # Distance between right edge of current and left edge of other
            
            # Merge if y-coordinates are similar and x-coordinates are close
            if y_diff < 20 and 0 <= x_distance < 30:  # Adjustable thresholds
                print(f"    Merging '{current_parsed}' and '{other_parsed}' (y_diff: {y_diff}, x_dist: {x_distance})")
                merged_components.append(other_parsed)
                # Expand bbox to include both components
                merged_bbox[0] = min(merged_bbox[0], other_num[0])  # x1
                merged_bbox[1] = min(merged_bbox[1], other_num[1])  # y1
                merged_bbox[2] = max(merged_bbox[2], other_num[2])  # x2
                merged_bbox[3] = max(merged_bbox[3], other_num[3])  # y2
                used_indices.add(j)
        
        # Combine parsed numbers (e.g., ['1', '0'] -> '10')
        if len(merged_components) > 1:
            merged_parsed_value = ''.join(merged_components)
            print(f"    Combined: {merged_components} -> '{merged_parsed_value}'")
        else:
            merged_parsed_value = current_parsed
        
        merged_numbers.append(tuple(merged_bbox))
        merged_parsed.append(merged_parsed_value)
        used_indices.add(i)
    
    print(f"Merged {len(numbers)} components into {len(merged_numbers)} numbers")
    return merged_numbers, merged_parsed


def predict_missing_numbers(parsed_numbers: List[str], expected_count: int = 10) -> List[str]:
    """Predict missing numbers based on sequence pattern (L->R, T->B)."""
    print("Predicting missing numbers...")
    
    # Convert parsed numbers to integers, filtering out non-numeric values
    numeric_numbers = []
    for num in parsed_numbers:
        try:
            numeric_numbers.append(int(num))
        except ValueError:
            continue
    
    if not numeric_numbers:
        print("    No numeric numbers found for prediction")
        return parsed_numbers
    
    # Sort by value to find gaps
    numeric_numbers.sort()
    print(f"    Found numbers: {numeric_numbers}")
    
    # Find missing numbers
    missing = []
    for i in range(1, expected_count + 1):
        if i not in numeric_numbers:
            missing.append(i)
            print(f"    Missing number: {i}")
    
    # Add predictions to the list
    predicted_numbers = parsed_numbers.copy()
    for missing_num in missing:
        predicted_numbers.append(str(missing_num))
        print(f"    Predicted missing number: {missing_num}")
    
    return predicted_numbers


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
            dist_ur = np.sqrt((num_centroid[0] - slab_x2)**2 + (num_centroid[1] - slab_y2)**2)
            
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
    parser = argparse.ArgumentParser(description='Fast macrosection extractor with greyscale detection.')
    parser.add_argument('--faxitron', type=str, required=True, help='Path to Faxitron target image')
    parser.add_argument('--out-dir', type=str, required=True, help='Output directory to save crops and overlays')
    parser.add_argument('--greyscale-detection', action='store_true', help='Detect greyscale slabs and numbers automatically')
    parser.add_argument('--debug-size', action='store_true', help='Generate debug visualization showing all components color-coded by size')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.greyscale_detection:
        if not HAVE_CV2:
            raise RuntimeError('Greyscale detection mode requires OpenCV (cv2). Please install opencv-python.')
        if not HAVE_TESSERACT:
            print("Warning: Tesseract not available. Number parsing will be limited.")
        
        print("Running fast greyscale detection mode...")
        
        # Load image
        fax_rgb, _ = load_image_and_dpi(args.faxitron)
        
        # Convert to grayscale
        gray = cv2.cvtColor(fax_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect slabs (fast)
        slabs = detect_greyscale_slabs_fast(gray)
        
        # Detect numbers (fast)
        numbers = detect_grey_numbers_fast(gray, slabs)
        
        # Parse numbers using OCR
        parsed_numbers = []
        for i, num_bbox in enumerate(numbers):
            parsed_num = parse_number_from_region_tesseract(gray, num_bbox)
            parsed_numbers.append(parsed_num)
            print(f"Number {i}: {parsed_num}")
        
        # Merge adjacent number components (e.g., "1" + "0" -> "10")
        merged_numbers, merged_parsed = merge_adjacent_numbers(numbers, parsed_numbers)
        
        # Predict missing numbers based on sequence pattern
        final_parsed = predict_missing_numbers(merged_parsed, expected_count=10)
        
        # Match merged numbers to slabs
        matches = match_numbers_to_slabs_corner_distance(merged_numbers, slabs)
        
        # Create complete mapping from slab_idx to final number (parsed or predicted)
        slab_to_number = {}
        slab_to_source = {}
        
        # First, add parsed numbers
        for num_idx, slab_idx in matches:
            number_value = merged_parsed[num_idx]
            slab_to_number[slab_idx] = number_value
            slab_to_source[slab_idx] = "parsed"
        
        # Then, add predicted numbers for missing slabs
        predicted_numbers = [n for n in final_parsed if n not in [merged_parsed[idx] for idx, _ in matches]]
        
        # Find slabs that don't have good number assignments (like OCR failures or unassigned)
        available_slabs = []
        for slab_idx in range(len(slabs)):
            if slab_idx not in slab_to_number or slab_to_number[slab_idx] in ["?", "1?"]:
                available_slabs.append(slab_idx)
        
        # Assign predicted numbers to available slabs
        for i, predicted_num in enumerate(predicted_numbers):
            if i < len(available_slabs):
                slab_idx = available_slabs[i]
                slab_to_number[slab_idx] = predicted_num
                slab_to_source[slab_idx] = "predicted"
        
        # Save simplified core metrics (PNG filename -> final parsed/predicted number)
        with open(os.path.join(args.out_dir, 'core_mapping.txt'), 'w') as f:
            f.write("Core Mapping: PNG_Filename -> Final_Number\n")
            f.write("=" * 40 + "\n")
            
            # Output core mapping using the consolidated mapping
            for slab_idx in range(len(slabs)):
                png_filename = f"greyscale_slab_{slab_idx:03d}.png"
                final_number = slab_to_number.get(slab_idx, "UNASSIGNED")
                f.write(f"{png_filename} -> {final_number}\n")
        
        # Write CSV with 3 columns: filename, slab_number, source
        csv_path = os.path.join(args.out_dir, 'slab_mapping.csv')
        with open(csv_path, 'w') as f:
            f.write("filename,slab_number,source\n")
            for slab_idx in range(len(slabs)):
                png_filename = f"greyscale_slab_{slab_idx:03d}.png"
                slab_number = slab_to_number.get(slab_idx, "UNASSIGNED")
                source = slab_to_source.get(slab_idx, "unknown")
                f.write(f"{png_filename},{slab_number},{source}\n")
        
        print(f"Simplified CSV mapping saved to: {csv_path}")
        
        # Save comprehensive JSON export with all details
        json_path = os.path.join(args.out_dir, 'comprehensive_annotations.json')
        
        # Prepare comprehensive data structure
        annotations_data = {
            "metadata": {
                "source_image": args.faxitron,
                "processing_timestamp": str(datetime.datetime.now()),
                "total_slabs": len(slabs),
                "total_numbers_detected": len(merged_numbers),
                "numbers_successfully_parsed": len([n for n in merged_parsed if n != "?"]),
                "numbers_with_slab_matches": len(matches),
                "predicted_numbers": len([n for n in final_parsed if n not in [merged_parsed[idx] for idx, _ in matches]])
            },
            "slabs": [],
            "numbers": [],
            "matches": [],
            "predictions": []
        }
        
        # Add slab information with coordinates
        for i, (x1, y1, x2, y2) in enumerate(slabs):
            slab_info = {
                "slab_id": i,
                "png_filename": f"greyscale_slab_{i:03d}.png",
                "coordinates": {
                    "x1": int(x1), "y1": int(y1), 
                    "x2": int(x2), "y2": int(y2),
                    "width": int(x2 - x1), "height": int(y2 - y1)
                },
                "centroid": {
                    "x": int((x1 + x2) // 2), 
                    "y": int((y1 + y2) // 2)
                }
            }
            annotations_data["slabs"].append(slab_info)
        
        # Add number information with coordinates and parsing details
        for i, (x1, y1, x2, y2) in enumerate(merged_numbers):
            number_info = {
                "number_id": i,
                "parsed_value": merged_parsed[i],
                "coordinates": {
                    "x1": int(x1), "y1": int(y1), 
                    "x2": int(x2), "y2": int(y2),
                    "width": int(x2 - x1), "height": int(y2 - y1)
                },
                "centroid": {
                    "x": int((x1 + x2) // 2), 
                    "y": int((y1 + y2) // 2)
                },
                "status": "parsed" if merged_parsed[i] != "?" else "ocr_failed"
            }
            annotations_data["numbers"].append(number_info)
        
        # Add matching information
        for num_idx, slab_idx in matches:
            match_info = {
                "number_id": int(num_idx),
                "slab_id": int(slab_idx),
                "parsed_value": merged_parsed[num_idx],
                "png_filename": f"greyscale_slab_{slab_idx:03d}.png"
            }
            annotations_data["matches"].append(match_info)
        
        # Add prediction information
        for num in final_parsed:
            if num not in [merged_parsed[idx] for idx, _ in matches]:
                prediction_info = {
                    "predicted_number": num,
                    "status": "predicted_no_slab_match"
                }
                annotations_data["predictions"].append(prediction_info)
        
        # Save JSON
        import json
        with open(json_path, 'w') as f:
            json.dump(annotations_data, f, indent=2)
        
        print(f"Comprehensive JSON annotations saved to: {json_path}")
        
        # Save preprocessed image
        cv2.imwrite(os.path.join(args.out_dir, 'preprocessed_greyscale.png'), gray)
        
        # Create detailed debug overlay with labels
        overlay_path = os.path.join(args.out_dir, 'greyscale_detection_overlay.png')
        
        # Create labels: slabs get "SLAB_X", numbers get their parsed value or "?"
        labels = []
        for i in range(len(slabs)):
            labels.append(f"SLAB_{i}")
        for i, parsed_num in enumerate(merged_parsed):
            labels.append(f"NUM_{parsed_num}")
        
        # Create overlay with labels
        visualize_bbox_on_faxitron(fax_rgb, slabs + merged_numbers, labels, overlay_path)
        
        # Also create a detailed debug overlay showing all detected components
        debug_overlay_path = os.path.join(args.out_dir, 'detailed_debug_overlay.png')
        
        # Find all connected components and label them
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        num_labels, labels_debug, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Create detailed visualization showing all components
        vis_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Draw slabs in blue with ID
        for i, (x1, y1, x2, y2) in enumerate(slabs):
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(vis_img, f"SLAB_{i}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Draw numbers in green with parsed value
        for i, (x1, y1, x2, y2) in enumerate(merged_numbers):
            color = (0, 255, 0) if merged_parsed[i] != "?" else (0, 255, 255)  # Green if parsed, yellow if not
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
            label = f"NUM_{merged_parsed[i]}" if merged_parsed[i] != "?" else "NUM_?"
            cv2.putText(vis_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw all other components in red with size info
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Skip if this is already drawn as slab or number
            is_drawn = False
            for slab in slabs:
                if x == slab[0] and y == slab[1] and x+w == slab[2] and y+h == slab[3]:
                    is_drawn = True
                    break
            for num in numbers:
                if x == num[0] and y == num[1] and x+w == num[2] and y+h == num[3]:
                    is_drawn = True
                    break
            
            if not is_drawn and area > 100:  # Only show size for components > 100 pixels
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(vis_img, f"size_{area}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.imwrite(debug_overlay_path, vis_img)
        print(f"Detailed debug overlay saved to: {debug_overlay_path}")
        
        # Save individual slab crops
        overlay_boxes = []
        for idx, (x1, y1, x2, y2) in enumerate(slabs):
            crop = crop_macrosection(fax_rgb, (x1, y1, x2, y2))
            crop_name = f'greyscale_slab_{idx:03d}.png'
            save_image(crop, os.path.join(args.out_dir, crop_name))
            overlay_boxes.append((x1, y1, x2, y2))
        
        # Create size debug visualization if requested
        if args.debug_size:
            debug_path = os.path.join(args.out_dir, 'size_debug_visualization.png')
            visualize_size_debug(gray, slabs, numbers, debug_path)
        
        print(f"Fast greyscale detection complete. Found {len(slabs)} slabs and {len(merged_numbers)} merged numbers.")
        print(f"Final parsed numbers: {final_parsed}")
    else:
        print("Please use --greyscale-detection flag for automatic detection.")


if __name__ == '__main__':
    main()

