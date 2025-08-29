#!/usr/bin/env python3
"""
Visualize the medium-sized components that are likely the numbers.
"""

import cv2
import numpy as np
import os

def visualize_potential_numbers(image_path):
    """Visualize the medium-sized components that are likely numbers."""
    print(f"Visualizing potential numbers in: {image_path}")
    
    # Load the preprocessed image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Find all connected components
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create RGB image for visualization
    vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Find medium-sized components (potential numbers: 1000-1200 pixels)
    potential_numbers = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if 1000 <= area <= 1200:  # Tight range based on our analysis
            potential_numbers.append((i, x, y, w, h, area))
    
    print(f"Found {len(potential_numbers)} potential numbers:")
    for idx, (comp_id, x, y, w, h, area) in enumerate(potential_numbers):
        print(f"  Number {idx+1}: Component {comp_id}, Area: {area}, Position: ({x}, {y}), Size: {w}x{h}")
        
        # Draw bounding box in green
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Add label
        cv2.putText(vis_img, f"NUM_{area}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Also highlight slabs for reference (in blue)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 100000:  # Slabs
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(vis_img, f"SLAB_{area//1000}k", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Save visualization
    output_path = 'potential_numbers_visualization.png'
    cv2.imwrite(output_path, vis_img)
    print(f"\nVisualization saved to: {output_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total components: {num_labels}")
    print(f"  Potential numbers (1000-1200): {len(potential_numbers)}")
    print(f"  Slabs (>100k): {sum(1 for i in range(1, num_labels) if stats[i][4] > 100000)}")
    print(f"  Expected: 10 slabs + 10 numbers")
    
    if len(potential_numbers) == 6:
        print(f"  ✓ Found 6 potential numbers (close to expected 10)")
        print(f"  ✓ Numbers are very similar in size (ratio: {max(p[5] for p in potential_numbers)/min(p[5] for p in potential_numbers):.1f}x)")
    else:
        print(f"  ⚠ Found {len(potential_numbers)} potential numbers (expected 10)")

if __name__ == '__main__':
    # Check if preprocessed image exists
    preprocessed_path = 'preprocessed_greyscale.png'
    if os.path.exists(preprocessed_path):
        visualize_potential_numbers(preprocessed_path)
    else:
        print(f"Preprocessed image not found: {preprocessed_path}")
        print("Please run the greyscale detection first to generate this file.")

