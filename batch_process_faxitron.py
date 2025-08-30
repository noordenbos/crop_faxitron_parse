#!/usr/bin/env python3
"""
Batch Processing Script for Faxitron Images
Processes all *.JPEG images (excluding *[a].JPEG and *.jpg) in parallel using 6 cores.
"""

import os
import sys
import subprocess
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse
from datetime import datetime

def find_faxitron_images(root_dir):
    """Find all valid Faxitron images in the directory tree."""
    print(f"🔍 Scanning for Faxitron images in: {root_dir}")
    
    valid_images = []
    root_path = Path(root_dir)
    
    # Find all *.JPEG files, excluding *[a].JPEG and *.jpg
    for jpeg_file in root_path.rglob("*.JPEG"):
        filename = jpeg_file.name
        if not filename.endswith("[a].JPEG") and not filename.endswith(".jpg"):
            valid_images.append(str(jpeg_file))
    
    print(f"✅ Found {len(valid_images)} valid images to process")
    return valid_images

def process_single_image(args_tuple):
    """Process a single Faxitron image."""
    image_path, output_base_dir, script_path = args_tuple
    
    try:
        # Extract case ID from path (e.g., "37 complete" -> "37_complete")
        path_parts = Path(image_path).parts
        case_folder = None
        for i, part in enumerate(path_parts):
            if part.endswith("complete") or part.isdigit():
                case_folder = part
                # Also include parent folder if it's a number
                if i > 0 and path_parts[i-1].isdigit():
                    case_folder = f"{path_parts[i-1]}_{case_folder}"
                break
        
        if not case_folder:
            case_folder = "unknown_case"
        
        # Create output directory for this case
        case_output_dir = os.path.join(output_base_dir, case_folder)
        os.makedirs(case_output_dir, exist_ok=True)
        
        # Extract image name without extension for output subdirectory
        image_name = Path(image_path).stem
        image_output_dir = os.path.join(case_output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Run the macrosection extractor
        cmd = [
            sys.executable, script_path,
            "--faxitron", image_path,
            "--greyscale-detection",
            "--debug-size",
            "--out-dir", image_output_dir
        ]
        
        # Execute the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per image
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "image": image_path,
                "output_dir": image_output_dir,
                "case": case_folder
            }
        else:
            return {
                "status": "error",
                "image": image_path,
                "output_dir": image_output_dir,
                "case": case_folder,
                "error": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "image": image_path,
            "case": case_folder,
            "error": "Processing timed out after 5 minutes"
        }
    except Exception as e:
        return {
            "status": "exception",
            "image": image_path,
            "case": case_folder,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Batch process Faxitron images using parallel processing")
    parser.add_argument("--root-dir", 
                       default="/Users/tnoorden/Library/CloudStorage/Box-Box/Neoadjuvant Imaging Study (Rusu)",
                       help="Root directory containing Faxitron images")
    parser.add_argument("--output-dir", 
                       default="./batch_output",
                       help="Base output directory for all processed images")
    parser.add_argument("--script-path", 
                       default="./tools/macrosection_extractor_fast.py",
                       help="Path to the macrosection extractor script")
    parser.add_argument("--max-workers", 
                       type=int, 
                       default=6,
                       help="Number of parallel workers (default: 6)")
    parser.add_argument("--dry-run", 
                       action="store_true",
                       help="Show what would be processed without actually processing")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.root_dir):
        print(f"❌ Root directory does not exist: {args.root_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.script_path):
        print(f"❌ Script does not exist: {args.script_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all images
    images = find_faxitron_images(args.root_dir)
    
    if not images:
        print("❌ No valid images found to process")
        sys.exit(1)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Images that would be processed:")
        for i, img in enumerate(images[:10]):  # Show first 10
            print(f"  {i+1:3d}. {img}")
        if len(images) > 10:
            print(f"  ... and {len(images) - 10} more images")
        print(f"\nTotal: {len(images)} images")
        return
    
    # Start processing
    print(f"\n🚀 Starting batch processing with {args.max_workers} workers...")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Prepare arguments for parallel processing
    args_list = [(img, args.output_dir, args.script_path) for img in images]
    
    # Process images in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_image = {executor.submit(process_single_image, args_tuple): args_tuple[0] 
                          for args_tuple in args_list}
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_image):
            result = future.result()
            results.append(result)
            completed += 1
            
            # Progress update
            if completed % 10 == 0 or completed == len(images):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(images) - completed) / rate if rate > 0 else 0
                
                print(f"📊 Progress: {completed}/{len(images)} ({completed/len(images)*100:.1f}%) "
                      f"- Rate: {rate:.1f} img/min - ETA: {eta/60:.1f} min")
    
    # Summary
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n🎉 Batch processing completed!")
    print(f"⏰ Total time: {total_time/60:.1f} minutes")
    print(f"📊 Processed: {len(images)} images")
    print(f"⚡ Average rate: {len(images)/total_time*60:.1f} images/minute")
    
    # Results summary
    success_count = len([r for r in results if r["status"] == "success"])
    error_count = len([r for r in results if r["status"] == "error"])
    timeout_count = len([r for r in results if r["status"] == "timeout"])
    exception_count = len([r for r in results if r["status"] == "exception"])
    
    print(f"\n📈 Results Summary:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Errors: {error_count}")
    print(f"  ⏰ Timeouts: {timeout_count}")
    print(f"  💥 Exceptions: {exception_count}")
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, "batch_processing_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Batch Processing Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total images: {len(images)}\n")
        f.write(f"Total time: {total_time/60:.1f} minutes\n")
        f.write(f"Success rate: {success_count/len(images)*100:.1f}%\n\n")
        
        f.write("SUCCESSFUL PROCESSING:\n")
        f.write("-" * 30 + "\n")
        for r in results:
            if r["status"] == "success":
                f.write(f"✅ {r['image']} -> {r['output_dir']}\n")
        
        f.write("\nFAILED PROCESSING:\n")
        f.write("-" * 30 + "\n")
        for r in results:
            if r["status"] != "success":
                f.write(f"❌ {r['image']} ({r['status']}): {r.get('error', 'Unknown error')}\n")
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Show some example errors if any
    if error_count > 0 or timeout_count > 0 or exception_count > 0:
        print(f"\n⚠️  Example issues:")
        for r in results:
            if r["status"] != "success":
                print(f"  ❌ {Path(r['image']).name}: {r['status']} - {r.get('error', 'Unknown')[:100]}...")
                break

if __name__ == "__main__":
    main()
