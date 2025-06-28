#!/usr/bin/env python3
"""
Type-based comprehensive test - 15 pairs from each image type prefix.
"""

import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from PIL import Image
from collections import defaultdict
import os
import base64
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent))
from src.image_similarity import ImageSimilarityEngine

def setup_logging():
    """Setup logging with timestamped filename in yymmdd-hh-mm format"""
    timestamp = datetime.now().strftime('%y%m%d-%H-%M')
    log_filename = f"type_based_test_{timestamp}.log"
    html_filename = f"type_based_test_{timestamp}.html"
    
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

def create_html_report(results, html_filename, timestamp, valid_types):
    """Create comprehensive HTML report with embedded base64 images grouped by type"""
    reports_dir = Path("reports") / f"type_based_test_{timestamp}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / html_filename
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Type-Based Image Similarity Test Report</title>
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
        .efficiency-list { list-style-type: none; padding: 0; }
        .efficiency-list li { background: #f8f9fa; margin: 5px 0; padding: 8px; border-left: 4px solid #007bff; }
        .type-section { margin: 30px 0; padding: 20px; border: 1px solid #dee2e6; border-radius: 5px; }
        .type-header { background-color: #f8f9fa; margin: -20px -20px 20px -20px; padding: 15px 20px; border-bottom: 1px solid #dee2e6; }
    </style>
</head>
<body>
""")
        
        f.write(f"<h1>Type-Based Image Similarity Test Report</h1>\n")
        f.write(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        f.write(f"<p><strong>Test Type:</strong> Comprehensive analysis across {len(valid_types)} image types</p>\n")
        f.write(f"<p><strong>Total Comparisons:</strong> {len(results)}</p>\n")
        
        # Overall Statistics
        successful = [r for r in results if r['Final_score'] > 0]
        if successful:
            all_scores = [r['Final_score'] for r in successful]
            all_times = [r['total_time'] * 1000 for r in successful]
            
            f.write("<div class='summary-stats'>\n")
            f.write("<h2>Overall Statistics</h2>\n")
            f.write("<ul>\n")
            f.write(f"<li><strong>Successful Comparisons:</strong> {len(successful)}/{len(results)}</li>\n")
            f.write(f"<li><strong>Score Range:</strong> {min(all_scores):.1f} - {max(all_scores):.1f}</li>\n")
            f.write(f"<li><strong>Average Score:</strong> {sum(all_scores)/len(all_scores):.1f}</li>\n")
            f.write(f"<li><strong>Average Processing Time:</strong> {sum(all_times)/len(all_times):.1f}ms</li>\n")
            f.write("</ul>\n")
            
            # Global layer distribution
            total_layer_dist = {'L1 Only': 0, 'L1+L2': 0, 'L1+L2+L3': 0}
            for r in successful:
                layers = r['layers_computed']
                if layers == 'L1': total_layer_dist['L1 Only'] += 1
                elif layers == 'L1, L2': total_layer_dist['L1+L2'] += 1
                elif layers == 'L1, L2, L3': total_layer_dist['L1+L2+L3'] += 1
            
            f.write("<h3>Global Processing Efficiency</h3>\n")
            f.write("<ul class='efficiency-list'>\n")
            for comp, count in total_layer_dist.items():
                pct = (count / len(successful)) * 100
                f.write(f"<li><strong>{comp}:</strong> {count} ({pct:.1f}%)</li>\n")
            f.write("</ul>\n")
            f.write("</div>\n")
        
        # Analysis by Image Type
        type_analysis = defaultdict(list)
        for result in successful:
            type_analysis[result['image_type']].append(result)
        
        f.write("<h2>Analysis by Image Type</h2>\n")
        for image_type in sorted(type_analysis.keys()):
            type_results = type_analysis[image_type]
            scores = [r['Final_score'] for r in type_results]
            times = [r['total_time'] * 1000 for r in type_results]
            
            f.write(f"<div class='type-section'>\n")
            f.write(f"<div class='type-header'>\n")
            f.write(f"<h3>{image_type.upper()}</h3>\n")
            f.write(f"</div>\n")
            
            f.write("<ul>\n")
            f.write(f"<li><strong>Pairs Tested:</strong> {len(type_results)}</li>\n")
            f.write(f"<li><strong>Score Range:</strong> {min(scores):.1f} - {max(scores):.1f}</li>\n")
            f.write(f"<li><strong>Average Score:</strong> {sum(scores)/len(scores):.1f}</li>\n")
            f.write(f"<li><strong>Average Time:</strong> {sum(times)/len(times):.1f}ms</li>\n")
            f.write("</ul>\n")
            
            # Layer distribution for this type
            layer_dist = {'L1': 0, 'L1+L2': 0, 'L1+L2+L3': 0}
            for r in type_results:
                layers = r['layers_computed']
                if layers == 'L1': layer_dist['L1'] += 1
                elif layers == 'L1, L2': layer_dist['L1+L2'] += 1
                elif layers == 'L1, L2, L3': layer_dist['L1+L2+L3'] += 1
            
            f.write("<h4>Layer Usage:</h4>\n")
            f.write("<ul>\n")
            for layer, count in layer_dist.items():
                f.write(f"<li>{layer}: {count}</li>\n")
            f.write("</ul>\n")
            f.write("</div>\n")
        
        # Detailed Results Table - Top performers per type
        f.write("<h2>Top Similarity Matches by Type</h2>\n")
        f.write("<p><em>Showing highest scoring pairs for each image type</em></p>\n")
        
        for image_type in sorted(type_analysis.keys()):
            type_results = type_analysis[image_type]
            # Get top 3 highest scoring pairs for this type
            top_results = sorted(type_results, key=lambda x: x['Final_score'], reverse=True)[:3]
            
            f.write(f"<h3>{image_type.upper()} - Top Matches</h3>\n")
            f.write("<table>\n")
            f.write("<thead>\n")
            f.write("<tr><th>#</th><th>Image 1</th><th>Image 2</th><th>Score</th><th>Layers</th><th>Time (ms)</th></tr>\n")
            f.write("</thead>\n")
            f.write("<tbody>\n")
            
            for r in top_results:
                score_class = "score-high" if r['Final_score'] >= 65 else "score-medium" if r['Final_score'] >= 40 else "score-low"
                
                f.write(f"<tr>\n")
                f.write(f"<td>{r['pair_num']}</td>\n")
                f.write(f"<td class='image-cell'>\n")
                f.write(f"<img src='{r['img1_base64']}' alt='{r['img1_name']}' title='{r['img1_name']}'>\n")
                f.write(f"<div class='filename'>{r['img1_name']}</div>\n")
                f.write(f"</td>\n")
                f.write(f"<td class='image-cell'>\n")
                f.write(f"<img src='{r['img2_base64']}' alt='{r['img2_name']}' title='{r['img2_name']}'>\n")
                f.write(f"<div class='filename'>{r['img2_name']}</div>\n")
                f.write(f"</td>\n")
                f.write(f"<td class='{score_class}'>{r['Final_score']:.1f}</td>\n")
                f.write(f"<td>{r['layers_computed']}</td>\n")
                f.write(f"<td>{r['total_time']*1000:.1f}</td>\n")
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
        f.write("<h3>Early Exit Strategy</h3>\n")
        f.write("<ul>\n")
        f.write("<li>Images with L1 < 65 skip L2/L3 processing for efficiency</li>\n")
        f.write("<li>L2 has additional early exit conditions to prevent unnecessary L3 computation</li>\n")
        f.write("<li>This results in faster processing for clearly different images</li>\n")
        f.write("</ul>\n")
        
        f.write("</body>\n</html>")
    
    return html_path

def get_score_explanation(score):
    if score >= 85: return "Near-identical"
    elif score >= 65: return "High similarity" 
    elif score >= 40: return "Moderate similarity"
    elif score >= 20: return "Some similarity"
    else: return "Very different"

def get_image_type(filename):
    """Extract image type from filename (everything before first hyphen)"""
    return filename.split('-')[0]

def create_pairs_for_type(images, type_name, target_pairs=15):
    """Create test pairs for a specific image type"""
    if len(images) < 2:
        return []
    
    pairs = []
    
    # Add identical pair if we have images
    if len(images) > 0:
        pairs.append((images[0], images[0], f"{type_name}-IDENTICAL"))
    
    # Add sequential pairs (adjacent images)
    for i in range(min(len(images) - 1, target_pairs - 1)):
        pairs.append((images[i], images[i + 1], f"{type_name}-SEQUENTIAL"))
        if len(pairs) >= target_pairs:
            break
    
    # Add cross pairs (first half with second half) if we need more
    if len(pairs) < target_pairs and len(images) > 2:
        mid_point = len(images) // 2
        for i in range(min(mid_point, target_pairs - len(pairs))):
            j = len(images) - 1 - i
            if i != j and j >= mid_point:
                pairs.append((images[i], images[j], f"{type_name}-CROSS"))
                if len(pairs) >= target_pairs:
                    break
    
    return pairs[:target_pairs]

def main():
    log_filename, html_filename, timestamp = setup_logging()
    log_print("IMAGE SIMILARITY ENGINE - TYPE-BASED COMPREHENSIVE TEST")
    log_print("Testing 15 pairs from each image type prefix")
    log_print("")
    
    # Initialize similarity engine
    engine = ImageSimilarityEngine()
    
    # Get all images
    image_dir = Path("scraper/uro")
    if not image_dir.exists():
        log_print("[!] Error: scraper/uro directory not found")
        return 1
    
    all_images = list(image_dir.glob("*.png"))
    if len(all_images) < 10:
        log_print(f"[!] Error: Need at least 10 images, found {len(all_images)}")
        return 1
    
    log_print(f"[+] Found {len(all_images)} total images")
    
    # Group images by type (prefix before first hyphen)
    type_groups = {}
    for img_path in all_images:
        img_type = img_path.stem.split('-')[0]
        if img_type not in type_groups:
            type_groups[img_type] = []
        type_groups[img_type].append(img_path)
    
    # Filter to types with at least 3 images (to make pairs)
    valid_types = {k: v for k, v in type_groups.items() if len(v) >= 3}
    log_print(f"[+] Found {len(valid_types)} image types with 3+ images each")
    
    # Create 15 pairs per type
    all_pairs = []
    pair_counter = 0
    
    for img_type, type_images in valid_types.items():
        type_images.sort()  # Consistent ordering
        type_pairs = []
        
        # Generate pairs for this type
        for i in range(min(15, len(type_images) * (len(type_images) - 1) // 2)):
            # Cycle through different pairing strategies
            if i < len(type_images) - 1:
                # Sequential pairs
                img1, img2 = type_images[i], type_images[i + 1]
                pair_type = f"{img_type}-sequential"
            else:
                # Cross pairs and random pairs
                idx1 = i % len(type_images)
                idx2 = (i + len(type_images) // 2) % len(type_images)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % len(type_images)
                img1, img2 = type_images[idx1], type_images[idx2]
                pair_type = f"{img_type}-cross"
            
            type_pairs.append((img1, img2, pair_type))
            if len(type_pairs) >= 15:
                break
        
        all_pairs.extend(type_pairs)
        log_print(f"   {img_type}: {len(type_pairs)} pairs")
    
    log_print(f"\n[*] Testing {len(all_pairs)} total pairs...")
    log_print("")
    
    # Run similarity tests
    results = []
    
    for i, (img1_path, img2_path, pair_type) in enumerate(all_pairs, 1):
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            # Get processed images (same as used in similarity calculation)
            processed_img1, processed_img2 = engine.get_processed_images(img1, img2)
            
            # Convert processed images to base64
            img1_base64 = image_to_base64(processed_img1)
            img2_base64 = image_to_base64(processed_img2)
            
            result = engine.calculate_similarity(img1, img2)
            
            total_time = (result['preprocess_time'] + result['L1_time'] + 
                         result['L2_time'] + result['L3_time'])
            
            # Extract base type and relationship
            image_type = pair_type.split('-')[0]
            relationship = '-'.join(pair_type.split('-')[1:])
            
            results.append({
                'pair_num': i,
                'img1_name': img1_path.name,
                'img2_name': img2_path.name,
                'img1_base64': img1_base64,
                'img2_base64': img2_base64,
                'image_type': image_type,
                'relationship': relationship,
                'L1_score': result['L1_score'],
                'L2_score': result['L2_score'],
                'L3_score': result['L3_score'],
                'Final_score': result['Final_score'],
                'layers_computed': ', '.join(result['layers_computed']),
                'preprocess_time': result['preprocess_time'],
                'L1_time': result['L1_time'],
                'L2_time': result['L2_time'],
                'L3_time': result['L3_time'],
                'total_time': total_time
            })
            
            if i % 10 == 0:
                log_print(f"Processed {i}/{len(all_pairs)} pairs...")
                
        except Exception as e:
            log_print(f"FAIL Pair {i}: {e}")
            results.append({
                'pair_num': i, 'img1_name': img1_path.name, 'img2_name': img2_path.name,
                'img1_base64': '', 'img2_base64': '',
                'image_type': 'ERROR', 'relationship': 'ERROR',
                'L1_score': 0.0, 'L2_score': 0.0, 'L3_score': 0.0, 'Final_score': 0.0,
                'layers_computed': 'ERROR', 'preprocess_time': 0.0, 'L1_time': 0.0,
                'L2_time': 0.0, 'L3_time': 0.0, 'total_time': 0.0
            })
    
    # Sort by image type, then by Final_score
    results.sort(key=lambda x: (x['image_type'], -x['Final_score']))
    
    # Summary statistics
    successful_results = [r for r in results if r['Final_score'] > 0]
    
    if successful_results:
        scores = [r['Final_score'] for r in successful_results]
        times = [r['total_time'] * 1000 for r in successful_results]  # Convert to ms
        
        log_print("\n" + "=" * 80)
        log_print("SUMMARY STATISTICS")
        log_print("=" * 80)
        log_print(f"Total comparisons: {len(results)}")
        log_print(f"Successful: {len(successful_results)}")
        log_print(f"Failed: {len(results) - len(successful_results)}")
        log_print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
        log_print(f"Average score: {sum(scores)/len(scores):.1f}")
        log_print(f"Average processing time: {sum(times)/len(times):.1f}ms")
        
        # Layer distribution
        layer_distribution = {}
        for r in successful_results:
            layers = r['layers_computed']
            layer_distribution[layers] = layer_distribution.get(layers, 0) + 1
        
        log_print(f"\nProcessing efficiency:")
        for layers, count in layer_distribution.items():
            percentage = (count / len(successful_results)) * 100
            if layers == 'L1':
                log_print(f"  L1 only: {count} ({percentage:.1f}%)")
            elif layers == 'L1, L2':
                log_print(f"  L1+L2: {count} ({percentage:.1f}%)")
            elif layers == 'L1, L2, L3':
                log_print(f"  L1+L2+L3: {count} ({percentage:.1f}%)")
        
        # Show some high-scoring pairs
        log_print(f"\nTop 10 highest similarity pairs:")
        top_pairs = sorted(successful_results, key=lambda x: x['Final_score'], reverse=True)[:10]
        for i, pair in enumerate(top_pairs, 1):
            log_print(f"  {i:2d}. {pair['img1_name']} vs {pair['img2_name']} = {pair['Final_score']:.1f} ({pair['layers_computed']})")
    
    log_print(f"\nTEST COMPLETE! Processed {len(all_pairs)} pairs across {len(valid_types)} image types.")
    log_print(f"Results logged to: {log_filename}")
    
    # Generate HTML report
    html_path = create_html_report(results, html_filename, timestamp, valid_types)
    log_print(f"HTML report generated to: {html_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 