#!/usr/bin/env python3
"""
Debug script to analyze size distribution of connected components in the preprocessed image.
"""

import cv2
import numpy as np
import os

def analyze_size_distribution(image_path):
    """Analyze the size distribution of all connected components in an image."""
    print(f"Analyzing size distribution for: {image_path}")
    
    # Load the preprocessed image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min/max values: {img.min()}/{img.max()}")
    
    # Find all connected components
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    print(f"\nTotal connected components: {num_labels}")
    print("Component | Area | Width | Height | Aspect Ratio | Type")
    print("-" * 70)
    
    # Analyze each component
    areas = []
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        aspect_ratio = h / w if w > 0 else 0
        areas.append(area)
        
        # Classify by size
        if area > 100000:  # Large
            comp_type = 'SLAB'
        elif 1000 < area <= 100000:  # Medium
            comp_type = 'MEDIUM'
        elif 100 < area <= 1000:  # Small
            comp_type = 'SMALL'
        else:  # Very small
            comp_type = 'NOISE'
        
        print(f'{i:9d} | {area:5d} | {w:5d} | {h:5d} | {aspect_ratio:11.2f} | {comp_type}')
    
    # Calculate percentiles
    areas_sorted = sorted(areas)
    print(f'\nArea percentiles:')
    print(f'Min: {min(areas)}')
    print(f'25%: {areas_sorted[int(len(areas_sorted)*0.25)]}')
    print(f'50%: {areas_sorted[int(len(areas_sorted)*0.50)]}')
    print(f'75%: {areas_sorted[int(len(areas_sorted)*0.75)]}')
    print(f'Max: {max(areas)}')
    
    # Count by category
    slabs = sum(1 for area in areas if area > 100000)
    medium = sum(1 for area in areas if 1000 < area <= 100000)
    small = sum(1 for area in areas if 100 < area <= 1000)
    noise = sum(1 for area in areas if area <= 100)
    
    print(f'\nSize category counts:')
    print(f'Slabs (>100k): {slabs}')
    print(f'Medium (1k-100k): {medium}')
    print(f'Small (100-1k): {small}')
    print(f'Noise (≤100): {noise}')
    
    # Look for distinct size clusters
    print(f'\nSize clusters (gaps > 10x):')
    for i in range(len(areas_sorted)-1):
        ratio = areas_sorted[i+1] / areas_sorted[i]
        if ratio > 10:
            print(f'Gap between {areas_sorted[i]} and {areas_sorted[i+1]} (ratio: {ratio:.1f}x)')
    
    # Look for potential number sizes (assuming they should be similar)
    if medium > 0:
        medium_areas = [area for area in areas if 1000 < area <= 100000]
        medium_areas_sorted = sorted(medium_areas)
        print(f'\nMedium-sized components (potential numbers):')
        print(f'Count: {len(medium_areas)}')
        print(f'Areas: {medium_areas_sorted}')
        if len(medium_areas) > 1:
            print(f'Size variation: min={min(medium_areas)}, max={max(medium_areas)}')
            print(f'Size ratio: {max(medium_areas)/min(medium_areas):.1f}x')

if __name__ == '__main__':
    # Check if preprocessed image exists
    preprocessed_path = 'preprocessed_greyscale.png'
    if os.path.exists(preprocessed_path):
        analyze_size_distribution(preprocessed_path)
    else:
        print(f"Preprocessed image not found: {preprocessed_path}")
        print("Please run the greyscale detection first to generate this file.")

