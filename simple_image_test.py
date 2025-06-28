#!/usr/bin/env python3
"""
Simple focused test for the image similarity engine with real images.
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.image_similarity import ImageSimilarityEngine

def setup_logging():
    """Setup logging with timestamped filename in yymmdd-hh-mm format"""
    timestamp = datetime.now().strftime('%y%m%d-%H-%M')
    log_filename = f"simple_test_{timestamp}.log"
    html_filename = f"simple_test_{timestamp}.html"
    
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

def image_to_base64(image, format='PNG'):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def create_html_report(results, html_filename, timestamp, successful):
    """Create comprehensive HTML report with embedded base64 images"""
    reports_dir = Path("reports") / f"simple_test_{timestamp}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / html_filename
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Image Similarity Test Report</title>
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
        .image-cell { text-align: center; min-width: 200px; }
        .image-cell img { max-width: 150px; max-height: 150px; border: 1px solid #ccc; margin: 5px; }
        .filename { font-family: monospace; font-size: 0.9em; color: #666; }
        .summary-stats { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .efficiency-list { list-style-type: none; padding: 0; }
        .efficiency-list li { background: #f8f9fa; margin: 5px 0; padding: 8px; border-left: 4px solid #007bff; }
    </style>
</head>
<body>
""")
        
        f.write(f"<h1>Simple Image Similarity Test Report</h1>\n")
        f.write(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        f.write(f"<p><strong>Test Type:</strong> Quick validation with {len(results)} image pairs</p>\n")
        
        # Summary Statistics
        if successful:
            scores = [r['Final'] for r in successful]
            times = [r['total_ms'] for r in successful]
            
            f.write("<div class='summary-stats'>\n")
            f.write("<h2>Summary Statistics</h2>\n")
            f.write(f"<ul>\n")
            f.write(f"<li><strong>Tests Completed:</strong> {len(successful)}/{len(results)}</li>\n")
            f.write(f"<li><strong>Score Range:</strong> {min(scores):.1f} - {max(scores):.1f}</li>\n")
            f.write(f"<li><strong>Average Score:</strong> {sum(scores)/len(scores):.1f}</li>\n")
            f.write(f"<li><strong>Average Processing Time:</strong> {sum(times)/len(times):.1f}ms</li>\n")
            f.write(f"</ul>\n")
            
            # Layer usage stats
            layer_counts = {}
            for r in successful:
                layers = r['layers']
                layer_counts[layers] = layer_counts.get(layers, 0) + 1
            
            f.write("<h3>Processing Efficiency</h3>\n")
            f.write("<ul class='efficiency-list'>\n")
            for layers, count in layer_counts.items():
                f.write(f"<li><strong>{layers}:</strong> {count} comparisons</li>\n")
            f.write("</ul>\n")
            f.write("</div>\n")
        
        # Detailed Results Table
        f.write("<h2>Detailed Results</h2>\n")
        f.write("<table>\n")
        f.write("<thead>\n")
        f.write("<tr><th>#</th><th>Image 1</th><th>Image 2</th><th>Type</th><th>L1</th><th>L2</th><th>L3</th><th>Final</th><th>Layers</th><th>Time (ms)</th></tr>\n")
        f.write("</thead>\n")
        f.write("<tbody>\n")
        
        for r in results:
            # Determine score color class
            score_class = "score-high" if r['Final'] >= 65 else "score-medium" if r['Final'] >= 40 else "score-low"
            
            f.write(f"<tr>\n")
            f.write(f"<td>{r['pair']}</td>\n")
            f.write(f"<td class='image-cell'>\n")
            f.write(f"<img src='{r['img1_base64']}' alt='{r['img1']}' title='{r['img1']}'>\n")
            f.write(f"<div class='filename'>{r['img1']}</div>\n")
            f.write(f"</td>\n")
            f.write(f"<td class='image-cell'>\n")
            f.write(f"<img src='{r['img2_base64']}' alt='{r['img2']}' title='{r['img2']}'>\n")
            f.write(f"<div class='filename'>{r['img2']}</div>\n")
            f.write(f"</td>\n")
            f.write(f"<td>{r['type']}</td>\n")
            f.write(f"<td>{r['L1']:.1f}</td>\n")
            f.write(f"<td>{r['L2']:.1f}</td>\n")
            f.write(f"<td>{r['L3']:.1f}</td>\n")
            f.write(f"<td class='{score_class}'>{r['Final']:.1f}</td>\n")
            f.write(f"<td>{r['layers']}</td>\n")
            f.write(f"<td>{r['total_ms']:.1f}</td>\n")
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
        
        # Performance Analysis
        f.write("<h2>Performance Analysis</h2>\n")
        f.write("<table>\n")
        f.write("<thead>\n")
        f.write("<tr><th>Test</th><th>Preprocessing</th><th>Layer 1</th><th>Layer 2</th><th>Layer 3</th><th>Total</th></tr>\n")
        f.write("</thead>\n")
        f.write("<tbody>\n")
        
        for r in results:
            f.write(f"<tr>\n")
            f.write(f"<td>{r['pair']}</td>\n")
            f.write(f"<td>{r['preprocess_ms']:.2f}ms</td>\n")
            f.write(f"<td>{r['l1_ms']:.2f}ms</td>\n")
            f.write(f"<td>{r['l2_ms']:.2f}ms</td>\n")
            f.write(f"<td>{r['l3_ms']:.2f}ms</td>\n")
            f.write(f"<td><strong>{r['total_ms']:.2f}ms</strong></td>\n")
            f.write(f"</tr>\n")
        
        f.write("</tbody>\n")
        f.write("</table>\n")
        
        # Algorithm Details
        f.write("<h2>Algorithm Details</h2>\n")
        f.write("<h3>Three-Layer Processing</h3>\n")
        f.write("<ol>\n")
        f.write("<li><strong>Layer 1 (L1):</strong> Spatially-aware color filtering with plain image detection</li>\n")
        f.write("<li><strong>Layer 2 (L2):</strong> Structural similarity using perceptual hashing</li>\n")
        f.write("<li><strong>Layer 3 (L3):</strong> Fine-grained texture analysis using Local Binary Patterns</li>\n")
        f.write("</ol>\n")
        f.write("<h3>Early Exit Strategy</h3>\n")
        f.write("<ul>\n")
        f.write("<li>Images with L1 < 65 skip L2/L3 processing for efficiency</li>\n")
        f.write("<li>L2 has additional early exit conditions to prevent unnecessary L3 computation</li>\n")
        f.write("<li>This results in faster processing for clearly different images</li>\n")
        f.write("</ul>\n")
        
        f.write("</body>\n</html>")
    
    return html_path

def main():
    log_filename, html_filename, timestamp = setup_logging()
    log_print("[*] IMAGE SIMILARITY ENGINE TEST")
    log_print("Quick test with 5 image pairs")
    log_print(f"Logging to: {log_filename}")
    log_print("=" * 50)
    
    # Find image files
    image_dir = Path("scraper/uro")
    if not image_dir.exists():
        log_print("[!] Error: scraper/uro directory not found")
        return 1
    
    image_files = list(image_dir.glob("*.png"))
    if len(image_files) < 2:
        log_print("[!] Error: Need at least 2 images for testing")
        return 1
    
    # Use first 3 images for quick testing
    images = image_files[:3]
    
    log_print(f"[+] Using {len(images)} images:")
    for img in images:
        log_print(f"   - {img.name}")
    
    # Create test pairs
    test_pairs = []
    
    # Test 1: Identical (same image with itself)
    test_pairs.append((images[0], images[0], "IDENTICAL"))
    
    # Test 2-3: Sequential pairs (likely same category)
    for i in range(len(images) - 1):
        test_pairs.append((images[i], images[i + 1], "SEQUENTIAL"))
    
    # Test 4: Cross comparison (first and last)
    if len(images) > 2:
        test_pairs.append((images[0], images[-1], "CROSS"))
    
    log_print(f"\n[*] Running {len(test_pairs)} similarity tests...")
    
    # Initialize similarity engine
    engine = ImageSimilarityEngine()
    results = []
    
    for i, (img1_path, img2_path, pair_type) in enumerate(test_pairs, 1):
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            # Get processed images (same as used in similarity calculation)
            processed_img1, processed_img2 = engine.get_processed_images(img1, img2)
            
            # Convert processed images to base64
            img1_base64 = image_to_base64(processed_img1)
            img2_base64 = image_to_base64(processed_img2)
            
            start_time = time.perf_counter()
            result = engine.calculate_similarity(img1, img2)
            total_time = time.perf_counter() - start_time
            
            # Calculate individual timings
            preprocess_ms = result['preprocess_time'] * 1000
            l1_ms = result['L1_time'] * 1000
            l2_ms = result['L2_time'] * 1000
            l3_ms = result['L3_time'] * 1000
            total_ms = total_time * 1000
            
            results.append({
                'pair': i,
                'img1': img1_path.name,
                'img2': img2_path.name,
                'img1_base64': img1_base64,
                'img2_base64': img2_base64,
                'type': pair_type,
                'L1': result['L1_score'],
                'L2': result['L2_score'],
                'L3': result['L3_score'],
                'Final': result['Final_score'],
                'layers': ', '.join(result['layers_computed']),
                'preprocess_ms': preprocess_ms,
                'l1_ms': l1_ms,
                'l2_ms': l2_ms,
                'l3_ms': l3_ms,
                'total_ms': total_ms
            })
            
            log_print(f"[+] Test {i}: {result['Final_score']:.1f} ({', '.join(result['layers_computed'])})")
            
        except Exception as e:
            log_print(f"[!] Test {i} failed: {e}")
            results.append({
                'pair': i, 'img1': img1_path.name, 'img2': img2_path.name,
                'img1_base64': '', 'img2_base64': '',
                'type': pair_type, 'L1': 0, 'L2': 0, 'L3': 0, 'Final': 0,
                'layers': 'ERROR', 'preprocess_ms': 0, 'l1_ms': 0, 'l2_ms': 0,
                'l3_ms': 0, 'total_ms': 0
            })
    
    # Summary table
    log_print("\n" + "=" * 100)
    log_print("SIMILARITY SCORES")
    log_print("=" * 100)
    log_print(f"{'#':<3} {'Image 1':<25} {'Image 2':<25} {'Type':<10} {'L1':<6} {'L2':<6} {'L3':<6} {'Final':<7} {'Layers':<12}")
    log_print("-" * 100)
    
    for r in results:
        log_print(f"{r['pair']:<3} {r['img1'][:24]:<25} {r['img2'][:24]:<25} {r['type']:<10} "
              f"{r['L1']:<6.1f} {r['L2']:<6.1f} {r['L3']:<6.1f} {r['Final']:<7.1f} {r['layers']:<12}")
    
    # Timing table
    log_print("\n" + "=" * 90)
    log_print("PERFORMANCE TIMING")
    log_print("=" * 90)
    log_print(f"{'#':<3} {'Comparison':<35} {'Pre(ms)':<8} {'L1(ms)':<7} {'L2(ms)':<7} {'L3(ms)':<7} {'Total(ms)':<9}")
    log_print("-" * 90)
    
    for r in results:
        comparison = f"{r['img1'][:15]}...{r['img2'][:15]}"
        log_print(f"{r['pair']:<3} {comparison:<35} {r['preprocess_ms']:<8.2f} {r['l1_ms']:<7.2f} "
              f"{r['l2_ms']:<7.2f} {r['l3_ms']:<7.2f} {r['total_ms']:<9.2f}")
    
    # Final summary
    successful = [r for r in results if r['Final'] > 0]
    if successful:
        scores = [r['Final'] for r in successful]
        times = [r['total_ms'] for r in successful]
        
        log_print(f"\n[*] SUMMARY:")
        log_print(f"   Tests completed: {len(successful)}/{len(results)}")
        log_print(f"   Score range: {min(scores):.1f} - {max(scores):.1f}")
        log_print(f"   Average score: {sum(scores)/len(scores):.1f}")
        log_print(f"   Average time: {sum(times)/len(times):.1f}ms")
        
        # Layer usage stats
        layer_counts = {}
        for r in successful:
            layers = r['layers']
            layer_counts[layers] = layer_counts.get(layers, 0) + 1
        
        log_print(f"   Layer usage:")
        for layers, count in layer_counts.items():
            log_print(f"     {layers}: {count}")
    
    # Generate HTML report
    html_path = create_html_report(results, html_filename, timestamp, successful)
    
    log_print(f"\n[+] TEST COMPLETE!")
    log_print(f"   Results logged to: {log_filename}")
    log_print(f"   HTML report: {html_path}")
    return 0

if __name__ == "__main__":
    exit(main()) 