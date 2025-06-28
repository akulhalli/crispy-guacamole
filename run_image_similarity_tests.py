#!/usr/bin/env python3
"""
Comprehensive test runner for the image similarity engine.
Runs both unit tests and real image comparison tests.
"""

import os
import sys
import traceback
import random
import time
import logging
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import packages with error handling
try:
    from PIL import Image
    import numpy as np
    from src.image_similarity import ImageSimilarityEngine
    HAS_REQUIRED_PACKAGES = True
except ImportError as e:
    print(f"[!] Missing required packages: {e}")
    HAS_REQUIRED_PACKAGES = False

# Try to import tabulate for nice formatting
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

def setup_logging():
    """Setup logging with timestamped filename in yymmdd-hh-mm format"""
    timestamp = datetime.now().strftime('%y%m%d-%H-%M')
    log_filename = f"comprehensive_test_{timestamp}.log"
    html_filename = f"comprehensive_test_{timestamp}.html"
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_filename, html_filename, timestamp

def log_print(message):
    """Print and log message"""
    print(message)
    logging.info(message)

def run_unit_tests():
    """Run the unit tests from test_image_similarity.py"""
    log_print("=" * 60)
    log_print("RUNNING UNIT TESTS")
    log_print("=" * 60)
    
    # Import test functions
    try:
        from tests.test_image_similarity import (
            test_tc01_identical_images,
            test_tc02_layer1_early_exit,
            test_tc03_layer2a_early_exit,
            test_tc04_layer2b_early_exit,
            test_tc05_layer3_texture,
            test_edge_different_sizes,
            test_edge_all_black_white
        )
        
        # List of test functions
        test_functions = [
            ("TC01: Identical Images", test_tc01_identical_images),
            ("TC02: Layer 1 Early Exit", test_tc02_layer1_early_exit),
            ("TC03: Layer 2A Early Exit", test_tc03_layer2a_early_exit),
            ("TC04: Layer 2B Early Exit", test_tc04_layer2b_early_exit),
            ("TC05: Layer 3 Texture", test_tc05_layer3_texture),
            ("Edge: Different Sizes", test_edge_different_sizes),
            ("Edge: Black vs White", test_edge_all_black_white)
        ]
        
        results = []
        
        for test_name, test_func in test_functions:
            try:
                test_func()  # This will raise AssertionError if test fails
                results.append([test_name, "[+] PASS", ""])
                log_print(f"[+] {test_name}: PASS")
            except AssertionError as e:
                results.append([test_name, "[!] FAIL", str(e)])
                log_print(f"[!] {test_name}: FAIL - {e}")
            except Exception as e:
                results.append([test_name, "[!] ERROR", str(e)])
                log_print(f"[!] {test_name}: ERROR - {e}")
        
        # Print summary table
        log_print("\n" + "=" * 60)
        log_print("UNIT TEST RESULTS SUMMARY")
        log_print("=" * 60)
        
        if HAS_TABULATE:
            log_print(tabulate(results, headers=["Test Case", "Status", "Details"], tablefmt="grid"))
        else:
            # Fallback formatting
            log_print(f"{'Test Case':<30} {'Status':<15} {'Details'}")
            log_print("-" * 80)
            for row in results:
                log_print(f"{row[0]:<30} {row[1]:<15} {row[2]}")
        
        return results
        
    except ImportError as e:
        log_print(f"[!] Failed to import test modules: {e}")
        return []

def get_sample_images(image_dir, max_images=10):
    """Get a sample of images from the specified directory"""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        log_print(f"[!] Image directory not found: {image_dir}")
        return []
    
    # Get all PNG files
    png_files = list(image_dir.glob("*.png"))
    
    if not png_files:
        log_print(f"[!] No PNG files found in: {image_dir}")
        return []
    
    # Take a diverse sample - get images from different product categories
    categories = {}
    for file in png_files:
        # Extract category from filename (everything before the first dash)
        category = file.name.split('-')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(file)
    
    # Get a few images from each category
    sample_images = []
    images_per_category = max(1, max_images // len(categories))
    
    for category, files in categories.items():
        sample_images.extend(files[:images_per_category])
        if len(sample_images) >= max_images:
            break
    
    return sample_images[:max_images]

def test_with_real_images():
    """Test the similarity engine with real images from the scraper directory"""
    log_print("\n" + "=" * 60)
    log_print("TESTING WITH REAL IMAGES")
    log_print("=" * 60)
    
    # Get sample images
    image_dir = "scraper/uro"
    sample_images = get_sample_images(image_dir, max_images=8)
    
    if len(sample_images) < 2:
        log_print("[!] Need at least 2 images to run comparison tests")
        return []
    
    log_print(f"[+] Found {len(sample_images)} sample images")
    for img in sample_images:
        log_print(f"   - {img.name}")
    
    # Initialize similarity engine
    engine = ImageSimilarityEngine()
    
    # Test combinations
    results = []
    
    log_print(f"\n[*] Running similarity comparisons...")
    
    # Test identical image (same file compared to itself)
    log_print("Testing identical images...")
    try:
        img_path = sample_images[0]
        img = Image.open(img_path)
        
        # Get processed images and convert to base64
        processed_img1, processed_img2 = engine.get_processed_images(img, img.copy())
        img1_base64 = image_to_base64(processed_img1)
        img2_base64 = image_to_base64(processed_img2)
        
        result = engine.calculate_similarity(img, img.copy())
        results.append([
            f"{img_path.name}",
            f"{img_path.name}",
            "IDENTICAL",
            f"{result['L1_score']:.1f}",
            f"{result['L2_score']:.1f}",
            f"{result['L3_score']:.1f}",
            f"{result['Final_score']:.1f}",
            img1_base64,
            img2_base64
        ])
    except Exception as e:
        log_print(f"[!] Error testing identical images: {e}")
    
    # Test same category images (likely similar)
    log_print("Testing same category images...")
    same_category_pairs = []
    for img1 in sample_images:
        for img2 in sample_images:
            if img1 != img2:
                cat1 = img1.name.split('-')[0]
                cat2 = img2.name.split('-')[0]
                if cat1 == cat2:
                    same_category_pairs.append((img1, img2))
                    break
        if same_category_pairs:
            break
    
    for img1_path, img2_path in same_category_pairs[:2]:  # Limit to 2 pairs
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            # Get processed images and convert to base64
            processed_img1, processed_img2 = engine.get_processed_images(img1, img2)
            img1_base64 = image_to_base64(processed_img1)
            img2_base64 = image_to_base64(processed_img2)
            
            result = engine.calculate_similarity(img1, img2)
            results.append([
                f"{img1_path.name}",
                f"{img2_path.name}",
                "SAME CATEGORY",
                f"{result['L1_score']:.1f}",
                f"{result['L2_score']:.1f}",
                f"{result['L3_score']:.1f}",
                f"{result['Final_score']:.1f}",
                img1_base64,
                img2_base64
            ])
        except Exception as e:
            log_print(f"[!] Error comparing {img1_path.name} vs {img2_path.name}: {e}")
    
    # Test different category images (likely different)
    log_print("Testing different category images...")
    different_category_pairs = []
    for img1 in sample_images:
        for img2 in sample_images:
            if img1 != img2:
                cat1 = img1.name.split('-')[0]
                cat2 = img2.name.split('-')[0]
                if cat1 != cat2:
                    different_category_pairs.append((img1, img2))
                    break
        if different_category_pairs:
            break
    
    for img1_path, img2_path in different_category_pairs[:3]:  # Limit to 3 pairs
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            # Get processed images and convert to base64
            processed_img1, processed_img2 = engine.get_processed_images(img1, img2)
            img1_base64 = image_to_base64(processed_img1)
            img2_base64 = image_to_base64(processed_img2)
            
            result = engine.calculate_similarity(img1, img2)
            results.append([
                f"{img1_path.name}",
                f"{img2_path.name}",
                "DIFFERENT CATEGORY",
                f"{result['L1_score']:.1f}",
                f"{result['L2_score']:.1f}",
                f"{result['L3_score']:.1f}",
                f"{result['Final_score']:.1f}",
                img1_base64,
                img2_base64
            ])
        except Exception as e:
            log_print(f"[!] Error comparing {img1_path.name} vs {img2_path.name}: {e}")
    
    # Print results table
    log_print("\n" + "=" * 80)
    log_print("REAL IMAGE SIMILARITY TEST RESULTS")
    log_print("=" * 80)
    
    headers = ["Image 1", "Image 2", "Relationship", "L1 Score", "L2 Score", "L3 Score", "Final Score"]
    
    if HAS_TABULATE:
        log_print(tabulate(results, headers=headers, tablefmt="grid"))
    else:
        # Fallback formatting
        log_print(f"{'Image 1':<25} {'Image 2':<25} {'Relationship':<15} {'L1':<6} {'L2':<6} {'L3':<6} {'Final':<6}")
        log_print("-" * 95)
        for row in results:
            log_print(f"{row[0]:<25} {row[1]:<25} {row[2]:<15} {row[3]:<6} {row[4]:<6} {row[5]:<6} {row[6]:<6}")
    
    return results

def analyze_score_patterns(results):
    """Analyze the patterns in the similarity scores"""
    log_print("\n" + "=" * 60)
    log_print("SCORE ANALYSIS")
    log_print("=" * 60)
    
    if not results:
        log_print("[!] No results to analyze")
        return
    
    # Group by relationship type
    identical = [r for r in results if r[2] == "IDENTICAL"]
    same_cat = [r for r in results if r[2] == "SAME CATEGORY"]
    diff_cat = [r for r in results if r[2] == "DIFFERENT CATEGORY"]
    
    analysis = []
    
    for group_name, group_data in [("Identical Images", identical), 
                                   ("Same Category", same_cat), 
                                   ("Different Category", diff_cat)]:
        if group_data:
            # Extract scores (skip first 3 columns which are names and relationship)
            l1_scores = [float(r[3]) for r in group_data]
            l2_scores = [float(r[4]) for r in group_data]
            l3_scores = [float(r[5]) for r in group_data]
            final_scores = [float(r[6]) for r in group_data]
            
            analysis.append([
                group_name,
                f"{np.mean(l1_scores):.1f} ± {np.std(l1_scores):.1f}",
                f"{np.mean(l2_scores):.1f} ± {np.std(l2_scores):.1f}",
                f"{np.mean(l3_scores):.1f} ± {np.std(l3_scores):.1f}",
                f"{np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}",
                len(group_data)
            ])
    
    headers = ["Group", "L1 Score (μ±σ)", "L2 Score (μ±σ)", "L3 Score (μ±σ)", "Final Score (μ±σ)", "Count"]
    log_print(tabulate(analysis, headers=headers, tablefmt="grid"))

def image_to_base64(image, format='PNG'):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def create_html_report(results, html_filename, timestamp):
    """Create comprehensive HTML report with embedded base64 images"""
    reports_dir = Path("reports") / f"comprehensive_test_{timestamp}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / html_filename
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Image Similarity Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .score-high { color: #28a745; font-weight: bold; }
        .score-medium { color: #ffc107; font-weight: bold; }
        .score-low { color: #dc3545; font-weight: bold; }
        .image-cell { text-align: center; min-width: 150px; }
        .image-cell img { max-width: 120px; max-height: 120px; border: 1px solid #ccc; margin: 3px; }
        .filename { font-family: monospace; font-size: 0.8em; color: #666; }
        .summary-stats { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .relation-section { margin: 30px 0; padding: 20px; border: 1px solid #dee2e6; border-radius: 5px; }
        .relation-header { background-color: #f8f9fa; margin: -20px -20px 20px -20px; padding: 15px 20px; border-bottom: 1px solid #dee2e6; }
    </style>
</head>
<body>
""")
        
        f.write(f"<h1>Comprehensive Image Similarity Test Report</h1>\n")
        f.write(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        f.write(f"<p><strong>Test Type:</strong> Comprehensive suite with unit tests and real image comparisons</p>\n")
        f.write(f"<p><strong>Total Real Image Comparisons:</strong> {len(results)}</p>\n")
        
        # Group results by relationship type
        identical = [r for r in results if r[2] == "IDENTICAL"]
        same_cat = [r for r in results if r[2] == "SAME CATEGORY"]
        diff_cat = [r for r in results if r[2] == "DIFFERENT CATEGORY"]
        
        if results:
            f.write("<h2>Summary by Relationship Type</h2>\n")
            
            for group_name, group_data in [("Identical Images", identical), 
                                          ("Same Category", same_cat), 
                                          ("Different Category", diff_cat)]:
                if group_data:
                    # Extract scores (skip first 3 columns which are names and relationship)
                    l1_scores = [float(r[3]) if len(r) > 3 else 0 for r in group_data]
                    l2_scores = [float(r[4]) if len(r) > 4 else 0 for r in group_data]
                    l3_scores = [float(r[5]) if len(r) > 5 else 0 for r in group_data]
                    final_scores = [float(r[6]) if len(r) > 6 else 0 for r in group_data]
                    
                    f.write(f"<div class='relation-section'>\n")
                    f.write(f"<div class='relation-header'>\n")
                    f.write(f"<h3>{group_name}</h3>\n")
                    f.write(f"</div>\n")
                    
                    f.write("<ul>\n")
                    f.write(f"<li><strong>Count:</strong> {len(group_data)}</li>\n")
                    if final_scores:
                        f.write(f"<li><strong>L1 Score:</strong> {np.mean(l1_scores):.1f} ± {np.std(l1_scores):.1f}</li>\n")
                        f.write(f"<li><strong>L2 Score:</strong> {np.mean(l2_scores):.1f} ± {np.std(l2_scores):.1f}</li>\n")
                        f.write(f"<li><strong>L3 Score:</strong> {np.mean(l3_scores):.1f} ± {np.std(l3_scores):.1f}</li>\n")
                        f.write(f"<li><strong>Final Score:</strong> {np.mean(final_scores):.1f} ± {np.std(final_scores):.1f}</li>\n")
                    f.write("</ul>\n")
                    f.write("</div>\n")
        
        # Detailed Results Table
        f.write("<h2>Detailed Test Results</h2>\n")
        f.write("<table>\n")
        f.write("<thead>\n")
        f.write("<tr><th>Image 1</th><th>Image 2</th><th>Relationship</th><th>L1 Score</th><th>L2 Score</th><th>L3 Score</th><th>Final Score</th></tr>\n")
        f.write("</thead>\n")
        f.write("<tbody>\n")
        
        for row in results:
            if len(row) >= 7:
                # Determine score color class
                final_score = float(row[6])
                score_class = "score-high" if final_score >= 65 else "score-medium" if final_score >= 40 else "score-low"
                
                f.write(f"<tr>\n")
                # Check if we have processed image base64 data (enhanced results)
                if len(row) >= 9:  # Enhanced format with base64 images
                    f.write(f"<td class='image-cell'>\n")
                    f.write(f"<img src='{row[7]}' alt='{row[0]}' title='{row[0]}'>\n")
                    f.write(f"<div class='filename'>{row[0]}</div>\n")
                    f.write(f"</td>\n")
                    f.write(f"<td class='image-cell'>\n")
                    f.write(f"<img src='{row[8]}' alt='{row[1]}' title='{row[1]}'>\n")
                    f.write(f"<div class='filename'>{row[1]}</div>\n")
                    f.write(f"</td>\n")
                else:  # Basic format without images
                    f.write(f"<td class='filename'>{row[0]}</td>\n")
                    f.write(f"<td class='filename'>{row[1]}</td>\n")
                
                f.write(f"<td>{row[2]}</td>\n")
                f.write(f"<td>{row[3]}</td>\n")
                f.write(f"<td>{row[4]}</td>\n")
                f.write(f"<td>{row[5]}</td>\n")
                f.write(f"<td class='{score_class}'>{row[6]}</td>\n")
                f.write(f"</tr>\n")
        
        f.write("</tbody>\n")
        f.write("</table>\n")
        
        # Score Explanations
        f.write("<h2>Score Interpretation</h2>\n")
        f.write("<ul>\n")
        f.write("<li><strong>85-100:</strong> Near-identical images</li>\n")
        f.write("<li><strong>65-84:</strong> High similarity (same material, different texture/pattern)</li>\n")
        f.write("<li><strong>40-64:</strong> Moderate similarity (related materials or patterns)</li>\n")
        f.write("<li><strong>20-39:</strong> Some similarity (same color family or basic features)</li>\n")
        f.write("<li><strong>0-19:</strong> Very different images</li>\n")
        f.write("</ul>\n")
        
        # Algorithm Details
        f.write("<h2>Algorithm Details</h2>\n")
        f.write("<h3>Three-Layer Processing</h3>\n")
        f.write("<ol>\n")
        f.write("<li><strong>Layer 1 (L1):</strong> Spatially-aware color filtering with plain image detection</li>\n")
        f.write("<li><strong>Layer 2 (L2):</strong> Structural similarity using perceptual hashing</li>\n")
        f.write("<li><strong>Layer 3 (L3):</strong> Fine-grained texture analysis using Local Binary Patterns</li>\n")
        f.write("</ol>\n")
        f.write("<h3>Test Coverage</h3>\n")
        f.write("<ul>\n")
        f.write("<li><strong>Unit Tests:</strong> Validation of individual algorithm components</li>\n")
        f.write("<li><strong>Real Image Tests:</strong> Performance on actual URO veneer dataset</li>\n")
        f.write("<li><strong>Multiple Categories:</strong> Cross-validation across different material types</li>\n")
        f.write("</ul>\n")
        
        f.write("</body>\n</html>")
    
    return html_path

def main():
    """Main test runner function"""
    log_filename, html_filename, timestamp = setup_logging()
    log_print("[*] IMAGE SIMILARITY ENGINE TEST SUITE")
    log_print("Testing image_similarity.py with unit tests and real images")
    log_print(f"Logging to: {log_filename}")
    log_print("=" * 60)
    
    # Check if required packages are installed
    if not HAS_REQUIRED_PACKAGES:
        log_print("[!] Missing required packages")
        log_print("Please install: pip install pillow numpy scikit-image imagehash")
        return 1
    else:
        log_print("[+] All required packages found")
    
    # Check if tabulate is available for nice tables
    try:
        import tabulate
    except ImportError:
        log_print("[!] 'tabulate' not found - installing for better table formatting")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
            import tabulate
        except:
            log_print("[!] Could not install tabulate - using basic formatting")
    
    # Run unit tests
    unit_test_results = run_unit_tests()
    
    # Run real image tests
    real_image_results = test_with_real_images()
    
    # Analyze patterns
    if real_image_results:
        analyze_score_patterns(real_image_results)
    
    # Final summary
    log_print("\n" + "=" * 60)
    log_print("TEST SUITE COMPLETE")
    log_print("=" * 60)
    
    unit_passed = sum(1 for r in unit_test_results if "PASS" in r[1])
    unit_total = len(unit_test_results)
    real_tests = len(real_image_results)
    
    log_print(f"[*] Unit Tests: {unit_passed}/{unit_total} passed")
    log_print(f"[*] Real Image Tests: {real_tests} comparisons completed")
    
    if unit_passed == unit_total and real_tests > 0:
        log_print("[+] All tests completed successfully!")
        log_print(f"Results logged to: {log_filename}")
    
        # Create HTML report
        html_path = create_html_report(real_image_results, html_filename, timestamp)
        log_print(f"HTML report generated: {html_path}")
    
        return 0
    else:
        log_print("[!] Some tests failed or incomplete")
        log_print(f"Results logged to: {log_filename}")
        return 1

if __name__ == "__main__":
    exit(main()) 