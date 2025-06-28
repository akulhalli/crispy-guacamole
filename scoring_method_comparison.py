#!/usr/bin/env python3
"""
Scoring Method Comparison Script

This script compares three different scoring methods:
1. Original (with early exits)
2. Weighted Geometric Mean
3. Adaptive Threshold

It selects 10 image pairs from scraper/uro/ directory:
- 5 pairs with high similarity (80-100)
- 5 pairs with low similarity (20-50)

Generates an HTML report comparing all five methods.
"""

import os
import sys
import time
import base64
import random
from pathlib import Path
from PIL import Image
from datetime import datetime
from itertools import combinations

# Add src directory to path for imports
sys.path.append('src')
from image_similarity import ImageSimilarityEngine

def image_to_base64(image_path):
    """Convert image file to base64 string for HTML embedding."""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            return f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
    except Exception:
        return ""

def load_images_from_directory(directory_path):
    """Load all PNG images from the specified directory."""
    image_files = []
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return []
    
    for file_path in directory.glob("*.png"):
        try:
            # Test if image can be loaded
            with Image.open(file_path) as img:
                # Basic validation
                if img.size[0] > 0 and img.size[1] > 0:
                    image_files.append(file_path)
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
    
    return image_files

def generate_diverse_pairs(image_files, num_pairs=300):
    """Generate diverse image pairs avoiding too many sequential comparisons."""
    if len(image_files) < 2:
        return []
    
    pairs = []
    
    # Generate comprehensive random pairs
    random.shuffle(image_files)
    
    # Strategy 1: Random consecutive pairs
    for i in range(0, min(len(image_files) - 1, num_pairs // 4)):
        pairs.append((image_files[i], image_files[i + 1]))
    
    # Strategy 2: Random non-consecutive pairs 
    for i in range(min(len(image_files) // 2, num_pairs // 4)):
        j = (i + random.randint(2, min(10, len(image_files) - 1))) % len(image_files)
        pairs.append((image_files[i], image_files[j]))
    
    # Strategy 3: Material-based pairing
    materials = {}
    for img_file in image_files:
        material = img_file.name.split('-')[0]  # e.g., "zero_by_euro", "woodlark_hybrid_veneers"
        if material not in materials:
            materials[material] = []
        materials[material].append(img_file)
    
    # Add same-material pairs (likely high similarity)
    for material, files in materials.items():
        if len(files) >= 2:
            # Add multiple pairs within same material
            for i in range(min(10, len(files) - 1)):
                if len(pairs) >= num_pairs:
                    break
                for j in range(i + 1, min(i + 4, len(files))):
                    if len(pairs) >= num_pairs:
                        break
                    pairs.append((files[i], files[j]))
    
    # Strategy 4: Cross-material pairs (likely low similarity)
    material_names = list(materials.keys())
    for i, mat1 in enumerate(material_names):
        for mat2 in material_names[i+1:]:
            if len(pairs) >= num_pairs:
                break
            # Add several pairs between different materials
            for j in range(min(8, len(materials[mat1]), len(materials[mat2]))):
                if len(pairs) >= num_pairs:
                    break
                pairs.append((materials[mat1][j % len(materials[mat1])], 
                            materials[mat2][j % len(materials[mat2])]))
    
    # Strategy 5: Random wide-spread pairs
    while len(pairs) < num_pairs and len(image_files) >= 2:
        idx1 = random.randint(0, len(image_files) - 1)
        idx2 = random.randint(0, len(image_files) - 1)
        if idx1 != idx2:
            pair = (image_files[idx1], image_files[idx2])
            if pair not in pairs and (image_files[idx2], image_files[idx1]) not in pairs:
                pairs.append(pair)
    
    # Remove duplicates and return
    unique_pairs = []
    seen = set()
    for pair in pairs:
        pair_key = tuple(sorted([str(pair[0]), str(pair[1])]))
        if pair_key not in seen:
            seen.add(pair_key)
            unique_pairs.append(pair)
    
    return unique_pairs[:num_pairs]

def generate_material_based_pairs(image_files):
    """
    Generate image pairs based on material types for targeted similarity testing.
    
    Strategy:
    - Same material pairs -> likely high similarity (80-100)
    - Different material pairs -> likely low similarity (20-40)
    
    Returns (same_material_pairs, cross_material_pairs)
    """
    # Group images by material type (prefix before first hyphen)
    materials = {}
    for img_file in image_files:
        # Extract material name (everything before first hyphen)
        material = img_file.name.split('-')[0]
        if material not in materials:
            materials[material] = []
        materials[material].append(img_file)
    
    print(f"Found {len(materials)} different material types:")
    for material, files in materials.items():
        print(f"  {material}: {len(files)} images")
    
    # Generate same-material pairs (for high similarity candidates)
    same_material_pairs = []
    for material, files in materials.items():
        if len(files) >= 2:
            # Add all possible pairs within this material
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    same_material_pairs.append((files[i], files[j]))
    
    # Generate cross-material pairs (for low similarity candidates)
    cross_material_pairs = []
    material_names = list(materials.keys())
    for i, mat1 in enumerate(material_names):
        for mat2 in material_names[i+1:]:
            # Add pairs between different materials
            for file1 in materials[mat1][:10]:  # Limit to first 10 files per material
                for file2 in materials[mat2][:10]:
                    cross_material_pairs.append((file1, file2))
    
    print(f"Generated {len(same_material_pairs)} same-material pairs")
    print(f"Generated {len(cross_material_pairs)} cross-material pairs")
    
    return same_material_pairs, cross_material_pairs

def find_targeted_pairs(same_material_pairs, cross_material_pairs, target_high=5, target_low=5):
    """
    Find exactly the target number of high and low similarity pairs.
    Limited to testing 250 pairs maximum.
    
    Args:
        same_material_pairs: Pairs from same material (likely high similarity)
        cross_material_pairs: Pairs from different materials (likely low similarity)
        target_high: Number of high-similarity pairs needed (80-100 range)
        target_low: Number of low-similarity pairs needed (20-40 range)
    
    Returns:
        (high_pairs, low_pairs) - each containing exactly target_high/target_low pairs
    """
    engine = ImageSimilarityEngine(scoring_method='dynamic_plain')
    
    high_pairs = []
    low_pairs = []
    all_tested_pairs = []  # Store all tested pairs with scores
    max_pairs_to_test = 250
    
    # Combine all pairs for testing (don't restrict to same material)
    all_pairs = same_material_pairs + cross_material_pairs
    random.shuffle(all_pairs)
    
    print(f"\nTesting up to {max_pairs_to_test} pairs for similarity...")
    print(f"Target: {target_high} high-similarity pairs (80-100), {target_low} low-similarity pairs (20-40)")
    
    # Test pairs up to the maximum limit
    for i, (img1_path, img2_path) in enumerate(all_pairs):
        if i >= max_pairs_to_test:
            print(f"\nReached maximum of {max_pairs_to_test} pairs tested.")
            break
            
        if len(high_pairs) >= target_high and len(low_pairs) >= target_low:
            print(f"\nFound target pairs early! Stopping after {i+1} tests.")
            break
            
        try:
            print(f"  Testing pair {i+1}/{max_pairs_to_test}: {img1_path.name} vs {img2_path.name}")
            
            with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
                result = engine.calculate_similarity(img1, img2)
                final_score = result['Final_score']
                
                pair_info = {
                    'img1_path': img1_path,
                    'img2_path': img2_path,
                    'score': final_score,
                    'result': result
                }
                all_tested_pairs.append(pair_info)
                
                # Check for high similarity (80-100)
                if 80 <= final_score <= 100 and len(high_pairs) < target_high:
                    high_pairs.append(pair_info)
                    print(f"    [+] HIGH similarity found: {final_score:.1f}")
                # Check for low similarity (20-40)
                elif 20 <= final_score <= 40 and len(low_pairs) < target_low:
                    low_pairs.append(pair_info)
                    print(f"    [+] LOW similarity found: {final_score:.1f}")
                else:
                    print(f"    -> Score: {final_score:.1f}")
                    
        except Exception as e:
            print(f"    -> Error: {e}")
    
    # If we don't have enough high similarity pairs, use the highest scoring ones
    if len(high_pairs) < target_high:
        print(f"\nOnly found {len(high_pairs)} high-similarity pairs in 80-100 range.")
        print("Selecting highest-scoring pairs from all tested...")
        
        # Sort all pairs by score (highest first) and take the top ones
        all_tested_pairs.sort(key=lambda x: x['score'], reverse=True)
        needed_high = target_high - len(high_pairs)
        
        for pair in all_tested_pairs:
            if len(high_pairs) >= target_high:
                break
            # Don't double-add pairs that are already in high_pairs
            if pair not in high_pairs:
                high_pairs.append(pair)
                print(f"    [+] Added high pair: {pair['score']:.1f}")
    
    # If we don't have enough low similarity pairs, use the lowest scoring ones
    if len(low_pairs) < target_low:
        print(f"\nOnly found {len(low_pairs)} low-similarity pairs in 20-40 range.")
        print("Selecting lowest-scoring pairs from all tested...")
        
        # Sort all pairs by score (lowest first) and take the bottom ones
        all_tested_pairs.sort(key=lambda x: x['score'])
        needed_low = target_low - len(low_pairs)
        
        for pair in all_tested_pairs:
            if len(low_pairs) >= target_low:
                break
            # Don't double-add pairs that are already in low_pairs or high_pairs
            if pair not in low_pairs and pair not in high_pairs:
                low_pairs.append(pair)
                print(f"    [+] Added low pair: {pair['score']:.1f}")
    
    print(f"\nFinal results after testing {len(all_tested_pairs)} pairs:")
    print(f"  High-similarity pairs: {len(high_pairs)} (scores: {[f'{p['score']:.1f}' for p in high_pairs]})")
    print(f"  Low-similarity pairs: {len(low_pairs)} (scores: {[f'{p['score']:.1f}' for p in low_pairs]})")
    
    return high_pairs[:target_high], low_pairs[:target_low]

def compare_scoring_methods(image_pairs):
    """Compare all five scoring methods on the selected image pairs."""
    methods = {
        'dynamic_plain': 'Dynamic Plain (Default)',
        'original_early_exit': 'Original Early Exit',
        'dynamic_texture_moderate': 'Dynamic + Texture Moderate (0.15/0.35/0.50)',
        'dynamic_texture_strong': 'Dynamic + Texture Strong (0.10/0.30/0.60)'
    }
    
    results = []
    
    print(f"\nComparing scoring methods on {len(image_pairs)} selected pairs...")
    
    for i, pair_info in enumerate(image_pairs):
        img1_path = pair_info['img1_path']
        img2_path = pair_info['img2_path']
        
        print(f"  Pair {i+1}: {img1_path.name} vs {img2_path.name}")
        
        try:
            with Image.open(img1_path) as img1, Image.open(img2_path) as img2:
                pair_result = {
                    'pair_num': i + 1,
                    'img1_name': img1_path.name,
                    'img2_name': img2_path.name,
                    'img1_base64': image_to_base64(img1_path),
                    'img2_base64': image_to_base64(img2_path),
                    'similarity_type': 'HIGH' if pair_info['score'] >= 70 else 'LOW'
                }
                
                # Test each scoring method
                for method_key, method_name in methods.items():
                    engine = ImageSimilarityEngine(scoring_method=method_key)
                    result = engine.calculate_similarity(img1, img2)
                    
                    pair_result[f'{method_key}_L1'] = result['L1_score']
                    pair_result[f'{method_key}_L2'] = result['L2_score'] 
                    pair_result[f'{method_key}_L3'] = result['L3_score']
                    pair_result[f'{method_key}_final'] = result['Final_score']
                    pair_result[f'{method_key}_layers'] = ', '.join(result['layers_computed'])
                
                results.append(pair_result)
                
        except Exception as e:
            print(f"    -> Error: {e}")
    
    return results

def create_html_report(results, html_filename, timestamp):
    """Create comprehensive HTML report comparing all five scoring methods."""
    
    html_path = Path("reports") / html_filename
    html_path.parent.mkdir(exist_ok=True)
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Scoring Method Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .summary {{ background-color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
        th {{ background-color: #34495e; color: white; font-weight: bold; }}
        .image-cell {{ width: 120px; padding: 8px; }}
        .image-cell img {{ max-width: 100px; max-height: 100px; border-radius: 5px; cursor: pointer; transition: transform 0.2s; }}
        .image-cell img:hover {{ transform: scale(1.05); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        .filename {{ font-size: 10px; margin-top: 5px; word-break: break-all; }}
        .score-high {{ background-color: #d4edda; color: #155724; font-weight: bold; }}
        .score-medium {{ background-color: #fff3cd; color: #856404; font-weight: bold; }}
        .score-low {{ background-color: #f8d7da; color: #721c24; font-weight: bold; }}
        .method-header {{ background-color: #3498db; color: white; }}
        .layer-score {{ font-size: 11px; color: #666; }}
        .comparison-highlight {{ background-color: #e8f4f8; }}
        .similarity-type {{ font-weight: bold; padding: 8px; border-radius: 4px; }}
        .type-high {{ background-color: #d1ecf1; color: #0c5460; }}
        .type-low {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Scoring Method Comparison Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Comparison:</strong> Dynamic Plain (Default) vs Original Early Exit vs Dynamic + Texture Moderate vs Dynamic + Texture Strong</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Pairs Tested:</strong> {len(results)}</p>
        <p><strong>High Similarity Pairs:</strong> {len([r for r in results if r['similarity_type'] == 'HIGH'])}</p>
        <p><strong>Low Similarity Pairs:</strong> {len([r for r in results if r['similarity_type'] == 'LOW'])}</p>
    </div>

    <h2>Detailed Comparison</h2>
    <table>
        <thead>
            <tr>
                <th rowspan="2">#</th>
                <th rowspan="2">Image 1</th>
                <th rowspan="2">Image 2</th>
                <th rowspan="2">Type</th>
                <th colspan="4" class="method-header">Dynamic Plain (Default)</th>
                <th colspan="4" class="method-header">Original Early Exit</th>
                <th colspan="4" class="method-header">Dynamic + Texture Moderate</th>
                <th colspan="4" class="method-header">Dynamic + Texture Strong</th>
            </tr>
            <tr>
                <th>Color</th><th>Structure</th><th>Texture</th><th>Final</th>
                <th>Color</th><th>Structure</th><th>Texture</th><th>Final</th>
                <th>Color</th><th>Structure</th><th>Texture</th><th>Final</th>
                <th>Color</th><th>Structure</th><th>Texture</th><th>Final</th>
            </tr>
        </thead>
        <tbody>
""")
        
        for r in results:
            # Determine score classes for final scores
            dynamic_class = "score-high" if r['dynamic_plain_final'] >= 70 else "score-medium" if r['dynamic_plain_final'] >= 40 else "score-low"
            orig_early_class = "score-high" if r['original_early_exit_final'] >= 70 else "score-medium" if r['original_early_exit_final'] >= 40 else "score-low"
            texture_mod_class = "score-high" if r['dynamic_texture_moderate_final'] >= 70 else "score-medium" if r['dynamic_texture_moderate_final'] >= 40 else "score-low"
            texture_strong_class = "score-high" if r['dynamic_texture_strong_final'] >= 70 else "score-medium" if r['dynamic_texture_strong_final'] >= 40 else "score-low"
            
            type_class = "type-high" if r['similarity_type'] == 'HIGH' else "type-low"
            
            f.write(f"""
            <tr>
                <td>{r['pair_num']}</td>
                <td class='image-cell'>
                    <a href='{r['img1_base64']}' target='_blank' title='Click to open {r['img1_name']} in new tab'>
                        <img src='{r['img1_base64']}' alt='{r['img1_name']}' title='{r['img1_name']}'>
                    </a>
                    <div class='filename'>{r['img1_name']}</div>
                </td>
                <td class='image-cell'>
                    <a href='{r['img2_base64']}' target='_blank' title='Click to open {r['img2_name']} in new tab'>
                        <img src='{r['img2_base64']}' alt='{r['img2_name']}' title='{r['img2_name']}'>
                    </a>
                    <div class='filename'>{r['img2_name']}</div>
                </td>
                <td class='similarity-type {type_class}'>{r['similarity_type']}</td>
                
                <!-- Dynamic Plain (Default) -->
                <td class='layer-score'>{r['dynamic_plain_L1']:.1f}</td>
                <td class='layer-score'>{r['dynamic_plain_L2']:.1f}</td>
                <td class='layer-score'>{r['dynamic_plain_L3']:.1f}</td>
                <td class='{dynamic_class}'>{r['dynamic_plain_final']:.1f}</td>
                
                <!-- Original Early Exit -->
                <td class='layer-score'>{r['original_early_exit_L1']:.1f}</td>
                <td class='layer-score'>{r['original_early_exit_L2']:.1f}</td>
                <td class='layer-score'>{r['original_early_exit_L3']:.1f}</td>
                <td class='{orig_early_class}'>{r['original_early_exit_final']:.1f}</td>
                
                <!-- Dynamic + Texture Moderate -->
                <td class='layer-score'>{r['dynamic_texture_moderate_L1']:.1f}</td>
                <td class='layer-score'>{r['dynamic_texture_moderate_L2']:.1f}</td>
                <td class='layer-score'>{r['dynamic_texture_moderate_L3']:.1f}</td>
                <td class='{texture_mod_class}'>{r['dynamic_texture_moderate_final']:.1f}</td>
                
                <!-- Dynamic + Texture Strong -->
                <td class='layer-score'>{r['dynamic_texture_strong_L1']:.1f}</td>
                <td class='layer-score'>{r['dynamic_texture_strong_L2']:.1f}</td>
                <td class='layer-score'>{r['dynamic_texture_strong_L3']:.1f}</td>
                <td class='{texture_strong_class}'>{r['dynamic_texture_strong_final']:.1f}</td>
            </tr>
            """)
        
        f.write("""
        </tbody>
    </table>

    <div class="summary">
        <h2>Score Analysis</h2>
        <h3>Scoring Methods Explained</h3>
        <ul>
            <li><strong>Dynamic Plain (Default):</strong> Always computes all layers. Allows negative Structure/Texture scores. Uses dynamic weighting with L2 dampening to prevent misleading scores when layers disagree.</li>
            <li><strong>Original Early Exit:</strong> Uses early exits for efficiency. Color < 65 stops processing. Structure issues can stop processing early.</li>
            <li><strong>Dynamic + Texture Moderate:</strong> Moderate texture emphasis. Reduces color influence, boosts texture impact (0.15×Color + 0.35×Structure + 0.50×Texture).</li>
            <li><strong>Dynamic + Texture Strong:</strong> Strong texture emphasis. Minimizes color influence, maximizes texture impact (0.10×Color + 0.30×Structure + 0.60×Texture).</li>
        </ul>
        
        <h3>Score Interpretation</h3>
        <ul>
            <li><strong>70-100:</strong> High similarity (similar materials/textures)</li>
            <li><strong>40-69:</strong> Moderate similarity (related features)</li>
            <li><strong>0-39:</strong> Low similarity (different materials)</li>
        </ul>
        
        <h3>Key Differences</h3>
        <ul>
            <li><strong>Efficiency:</strong> Original method is fastest (early exits), others always compute all layers</li>
            <li><strong>Penalty Behavior:</strong> Geometric and Adaptive methods penalize low individual layer scores more severely</li>
            <li><strong>Consistency:</strong> New methods provide more nuanced scoring when layers disagree</li>
        </ul>
    </div>

</body>
</html>""")
    
    return html_path

def main():
    """Main execution function."""
    print("SCORING METHOD COMPARISON TOOL")
    print("=" * 50)
    
    # Configuration
    image_directory = "scraper/uro"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    html_filename = f"scoring_comparison_{timestamp}.html"
    
    # Load images
    print(f"Loading images from {image_directory}...")
    image_files = load_images_from_directory(image_directory)
    print(f"Found {len(image_files)} valid images")
    
    if len(image_files) < 10:
        print("Need at least 10 images for comparison. Exiting.")
        return 1
    
    # Generate material-based pairs for targeted similarity testing
    print("\nGenerating material-based image pairs for targeted testing...")
    same_material_pairs, cross_material_pairs = generate_material_based_pairs(image_files)
    
    # Find exactly 5 high-similarity and 5 low-similarity pairs
    print(f"\nSearching for exactly 5 high-similarity (80-100) and 5 low-similarity (20-40) pairs...")
    high_pairs, low_pairs = find_targeted_pairs(same_material_pairs, cross_material_pairs, target_high=5, target_low=5)
    
    selected_pairs = high_pairs + low_pairs
    
    print(f"\nFinal selection: {len(high_pairs)} high + {len(low_pairs)} low = {len(selected_pairs)} total pairs")
    
    if len(selected_pairs) < 10:
        print(f"Warning: Only found {len(selected_pairs)} pairs (target was 10)")
        print("Proceeding with available pairs...")
    else:
        print("[+] Successfully found exactly 10 pairs matching criteria!")
    
    # Compare all scoring methods
    comparison_results = compare_scoring_methods(selected_pairs)
    
    # Generate HTML report
    print(f"\nGenerating HTML report...")
    html_path = create_html_report(comparison_results, html_filename, timestamp)
    
    print(f"\n[+] Comparison complete!")
    print(f"[+] HTML report generated: {html_path}")
    print(f"[+] Tested {len(comparison_results)} image pairs")
    
    # Print summary
    if comparison_results:
        print(f"\n[*] Quick Summary:")
        for r in comparison_results:
            print(f"  {r['pair_num']:2d}. {r['img1_name'][:20]}... vs {r['img2_name'][:20]}...")
            print(f"      Dynamic: {r['dynamic_plain_final']:5.1f} | Early Exit: {r['original_early_exit_final']:5.1f} | Dyn+TexMod: {r['dynamic_texture_moderate_final']:5.1f} | Dyn+TexStr: {r['dynamic_texture_strong_final']:5.1f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
