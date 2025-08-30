# Getting Up to Speed: Enhanced Macrosection Extractor

## 🎯 **Project Overview**

This repository contains an enhanced version of the PViT-AIR macrosection extraction tool, specifically designed to convert Faxitron images into individual tissue slab crops with intelligent number detection and labeling.

## 🚀 **Key Features**

### **1. Intelligent OCR & Number Detection**
- **Compound number detection**: Automatically merges adjacent components (e.g., "2" + "3" → "23")
- **Confidence filtering**: Only keeps high-confidence OCR results (>50%) to eliminate noise
- **Multi-strategy OCR**: Uses Tesseract with multiple binarization methods and PSM modes
- **Smart merging**: Combines adjacent number components before OCR for better accuracy

### **2. Advanced Slab Detection**
- **Fast binary thresholding**: Much faster than pixel-by-pixel variation calculation
- **Morphological operations**: Clean noise removal and slab isolation
- **Size-based filtering**: Configurable thresholds for slab detection
- **No over-detection**: Removed relaxation logic that created false positives

### **3. Perfect Mask-Based Cropping**
- **Exact component masks**: Uses connected component segmentation for pixel-perfect isolation
- **No contamination**: Each crop contains only pixels from its specific slab
- **Consistent coordinates**: All operations use the same coordinate space
- **Clean output**: Background pixels set to black, tissue preserved

### **4. Automatic Sequential Labeling**
- **Fallback labeling**: When no numbers detected, automatically assigns sequential labels
- **L→R, T→B ordering**: Top to bottom, then left to right
- **Starting from 1**: Upper left corner gets label "1"
- **Source tracking**: All auto-assigned numbers marked as "sequential_auto"

### **5. Quality Assessment & Flags**
- **Batch processing flags**: Identifies potential issues for manual review
- **Overlap detection**: Warns when slabs significantly overlap
- **OCR failure tracking**: Monitors and reports parsing success rates
- **Manual cropping recommendations**: Suggests when manual intervention needed

## 🛠 **Technical Architecture**

### **Core Functions**
- `detect_greyscale_slabs_fast()`: Fast slab detection using binary thresholding
- `detect_grey_numbers_fast()`: Number detection with smart component merging
- `parse_number_from_region_tesseract()`: Enhanced OCR with multiple strategies
- `merge_number_components()`: Pre-OCR component merging for compound numbers
- `match_numbers_to_slabs_optimal()`: Hungarian algorithm for optimal number-slab matching

### **Workflow**
1. **Image preprocessing**: Remove metadata, near-white pixels
2. **Slab detection**: Binary thresholding + morphological operations
3. **Number detection**: Component detection + smart merging
4. **OCR processing**: Multiple binarization methods + confidence filtering
5. **Number-slab matching**: Optimal assignment using distance minimization
6. **Mask-based cropping**: Exact component isolation + individual crops
7. **Quality assessment**: Flags and recommendations for manual review

## 📁 **File Structure**

```
tools/
├── macrosection_extractor_fast.py    # Main extraction tool
├── macrosection_extractor.py         # Original version (legacy)
└── ...

output_*/                              # Test outputs
├── slab_mapping.csv                  # CSV with filename, slab_number, source
├── core_mapping.txt                  # Simple PNG -> number mapping
├── comprehensive_annotations.json    # Full metadata and coordinates
├── enhanced_matching_overlay.png     # Visual number-slab matching
├── detailed_debug_overlay.png        # All detected components
├── size_debug_visualization.png      # Size-based component analysis
├── greyscale_slab_XXX.png           # Original rectangular crops
└── slab_XXX_source_masked.png       # Clean masked crops
```

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install opencv-python pillow pytesseract numpy scipy
```

### **2. Install Tesseract**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### **3. Run Extraction**
```bash
python tools/macrosection_extractor_fast.py \
    --faxitron '/path/to/your/faxitron_image.JPEG' \
    --greyscale-detection \
    --debug-size \
    --out-dir output_results
```

## 📊 **Output Files Explained**

### **Core Outputs**
- **`slab_mapping.csv`**: Main mapping file with filename, slab_number, source
- **`core_mapping.txt`**: Simple PNG filename → number mapping
- **`comprehensive_annotations.json`**: Full metadata, coordinates, and processing details

### **Visual Debug Files**
- **`enhanced_matching_overlay.png`**: Shows number-to-slab assignments with color coding
- **`detailed_debug_overlay.png`**: All detected components with labels and sizes
- **`size_debug_visualization.png`**: Components color-coded by size category

### **Individual Crops**
- **`greyscale_slab_XXX.png`**: Original rectangular crops (for reference)
- **`slab_XXX_source_masked.png`**: Clean, masked crops with exact slab isolation

## 🔧 **Configuration Options**

### **Detection Thresholds**
- **Slab detection**: `area > 100000` pixels (configurable in code)
- **Number detection**: `500 <= area <= 2000` pixels
- **OCR confidence**: `> 50%` minimum threshold

### **Merging Parameters**
- **Y-coordinate tolerance**: `y_diff < 30` pixels
- **X-distance tolerance**: `x_distance < 50` pixels
- **Component adjacency**: Configurable thresholds for number merging

## 🎯 **Use Cases**

### **1. Standard Faxitron Images**
- Images with visible number annotations
- OCR detects and matches numbers to slabs
- High-quality individual crops with proper labeling

### **2. Images Without Numbers**
- Automatic sequential labeling (1, 2, 3, ...)
- L→R, T→B ordering from upper left corner
- Perfect for batch processing or research workflows

### **3. Challenging Images**
- Quality flags identify potential issues
- Manual cropping recommendations
- Overlap detection for complex cases

## 🚨 **Quality Flags & Recommendations**

### **Warning Flags**
- **`LOW_SLAB_COUNT`**: Very few slabs detected (< 3)
- **`HIGH_OCR_FAILURE_RATE`**: Many numbers couldn't be parsed
- **`HIGH_PREDICTION_RATE`**: Many numbers had to be predicted
- **`OVERLAPPING_SLABS`**: Slabs overlap significantly

### **Manual Review Triggers**
- Overlapping slabs detected
- Very low slab count
- High prediction rates
- Unusual image characteristics

## 🔄 **Development Workflow**

### **Current Branch**
- **Branch**: `tnoorden/macrosection-extractor-enhanced`
- **Remote**: `my-fork` → `https://github.com/noordenbos/crop_faxitron_parse.git`

### **Making Changes**
```bash
# Make your changes
git add tools/macrosection_extractor_fast.py
git commit -m "Your change description"
git push my-fork tnoorden/macrosection-extractor-enhanced
```

### **Updating from Upstream**
```bash
# Pull latest changes from original repo
git fetch origin
git merge origin/main

# Resolve conflicts if any
git push my-fork tnoorden/macrosection-extractor-enhanced
```

## 🧪 **Testing & Validation**

### **Test Images Used**
- **SBS-19-00098_E_Image.JPEG**: Standard 10-slab image
- **SBS-19-00260_A_Image_010.JPEG**: 3-slab image with compound numbers
- **SBS-19-00260_A_Image_012.JPEG**: Complex overlapping slabs
- **SBS-19-00260_D_Image_003.JPEG**: 2-slab image with high-quality OCR
- **SBS-19-00507_C_Image.JPEG**: 13-slab image without numbers (sequential labeling)

### **Validation Metrics**
- **Slab detection accuracy**: Correct number of slabs identified
- **Number parsing success**: OCR confidence and accuracy
- **Masking quality**: Clean isolation without contamination
- **Processing speed**: Fast detection and cropping

## 🚀 **Next Steps & Improvements**

### **Potential Enhancements**
1. **Batch processing**: Process multiple images simultaneously
2. **GUI interface**: User-friendly graphical interface
3. **Advanced filtering**: More sophisticated noise removal
4. **Export formats**: Additional output formats (TIFF, DICOM)
5. **Performance optimization**: GPU acceleration for large images

### **Known Limitations**
1. **Image quality dependency**: Works best with high-contrast images
2. **Number visibility**: Requires visible number annotations for OCR
3. **Slab separation**: May struggle with heavily overlapping tissue
4. **Processing time**: Large images may take several seconds

## 📞 **Support & Troubleshooting**

### **Common Issues**
1. **Tesseract not found**: Install Tesseract OCR engine
2. **Low detection accuracy**: Adjust threshold parameters
3. **Poor masking**: Check image quality and contrast
4. **Slow processing**: Consider image resizing for large files

### **Debug Mode**
- Use `--debug-size` flag for detailed visualizations
- Check `detailed_debug_overlay.png` for component detection
- Review `comprehensive_annotations.json` for processing details

---

## 🎉 **Summary**

This enhanced macrosection extractor provides:
- **Intelligent number detection** with compound number support
- **Perfect mask-based cropping** for clean individual slabs
- **Automatic sequential labeling** when numbers aren't visible
- **Quality assessment** and recommendations for manual review
- **Fast processing** with reliable results across different image types

The tool is production-ready and handles both standard Faxitron images with numbers and challenging cases without visible annotations. All improvements are committed and pushed to your development branch for continued development and collaboration.
