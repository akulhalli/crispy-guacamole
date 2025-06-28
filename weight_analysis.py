#!/usr/bin/env python3
"""
Weight Analysis Script
Shows the impact of different L3 (texture) weighting schemes on final scores.
"""

import math

def original_scoring(l1, l2, l3):
    """Current original method: 0.2*L1 + 0.4*L2 + 0.4*L3"""
    return 0.2 * l1 + 0.4 * l2 + 0.4 * l3

def alternative_1(l1, l2, l3):
    """Moderate L3 emphasis: 0.15*L1 + 0.35*L2 + 0.50*L3"""
    return 0.15 * l1 + 0.35 * l2 + 0.50 * l3

def alternative_2(l1, l2, l3):
    """Strong L3 emphasis: 0.10*L1 + 0.30*L2 + 0.60*L3"""
    return 0.10 * l1 + 0.30 * l2 + 0.60 * l3

def alternative_3(l1, l2, l3):
    """L3-dominant: 0.15*L1 + 0.25*L2 + 0.60*L3"""
    return 0.15 * l1 + 0.25 * l2 + 0.60 * l3

def alternative_4(l1, l2, l3):
    """Exponential L3 impact: 0.15*L1 + 0.35*L2 + 0.50*L3_scaled"""
    # Apply exponential scaling to L3
    if l3 > 50:
        l3_scaled = l3 ** 1.2  # Boost positive texture scores
        l3_scaled = min(l3_scaled, 100)  # Cap at 100
    elif l3 < 50:
        l3_scaled = -(abs(l3) ** 1.5) if l3 < 0 else l3 ** 1.5  # Amplify negative/low scores
        l3_scaled = max(l3_scaled, -100)  # Floor at -100
    else:
        l3_scaled = l3
    
    return 0.15 * l1 + 0.35 * l2 + 0.50 * l3_scaled

def analyze_scenarios():
    """Analyze different scoring scenarios"""
    
    scenarios = [
        {
            'name': 'High Texture Similarity (Same Wood Grain)',
            'description': 'Similar materials with matching texture patterns',
            'scores': (75, 70, 85),
            'expected': 'L3 emphasis should boost this significantly'
        },
        {
            'name': 'Low Texture Similarity (Smooth vs Textured)',
            'description': 'Same color/structure but very different textures',
            'scores': (75, 70, 25),
            'expected': 'L3 emphasis should penalize this more'
        },
        {
            'name': 'Negative Texture (Very Different)',
            'description': 'Completely different texture types',
            'scores': (60, 65, -15),
            'expected': 'L3 emphasis should heavily penalize negative texture'
        },
        {
            'name': 'Mixed: Good Color, Poor Structure, Excellent Texture',
            'description': 'Edge case where texture saves the score',
            'scores': (80, 35, 90),
            'expected': 'L3 emphasis should rescue this score'
        },
        {
            'name': 'Mixed: Poor Color, Good Structure, Poor Texture',
            'description': 'Edge case where texture hurts the score',
            'scores': (40, 75, 20),
            'expected': 'L3 emphasis should lower this score'
        }
    ]
    
    methods = [
        ('Original (0.2/0.4/0.4)', original_scoring),
        ('Alt 1: Moderate L3 (0.15/0.35/0.50)', alternative_1),
        ('Alt 2: Strong L3 (0.10/0.30/0.60)', alternative_2),
        ('Alt 3: L3-Dominant (0.15/0.25/0.60)', alternative_3),
        ('Alt 4: Exponential L3', alternative_4)
    ]
    
    print("TEXTURE-FOCUSED SCORING ANALYSIS")
    print("=" * 80)
    
    for scenario in scenarios:
        l1, l2, l3 = scenario['scores']
        print(f"\nSCENARIO: {scenario['name']}")
        print(f"Layer Scores: L1={l1}, L2={l2}, L3={l3}")
        print(f"Description: {scenario['description']}")
        print(f"Expected: {scenario['expected']}")
        print("-" * 60)
        
        results = []
        for method_name, method_func in methods:
            final_score = method_func(l1, l2, l3)
            results.append((method_name, final_score))
            print(f"{method_name:35s}: {final_score:6.1f}")
        
        # Show differences from original
        original_score = results[0][1]
        print(f"\nDifferences from Original:")
        for i in range(1, len(results)):
            method_name, score = results[i]
            diff = score - original_score
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {method_name:35s}: {diff:+6.1f} {direction}")
        
        print()

def texture_sensitivity_analysis():
    """Analyze how each method responds to texture changes"""
    print("\nTEXTURE SENSITIVITY ANALYSIS")
    print("=" * 80)
    print("Fixed L1=70, L2=70, varying L3 from -20 to 100")
    print()
    
    l1, l2 = 70, 70
    texture_values = [-20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    methods = [
        ('Original', original_scoring),
        ('Alt 1', alternative_1),
        ('Alt 2', alternative_2),
        ('Alt 3', alternative_3),
        ('Alt 4', alternative_4)
    ]
    
    print(f"{'L3':<4}", end="")
    for method_name, _ in methods:
        print(f"{method_name:>12}", end="")
    print()
    print("-" * (4 + 12 * len(methods)))
    
    for l3 in texture_values:
        print(f"{l3:>3d}", end=" ")
        for method_name, method_func in methods:
            score = method_func(l1, l2, l3)
            print(f"{score:>11.1f}", end=" ")
        print()

if __name__ == "__main__":
    analyze_scenarios()
    texture_sensitivity_analysis() 