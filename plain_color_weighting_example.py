#!/usr/bin/env python3
"""
Plain Color Weighting Example
Demonstrates the proposed dynamic weighting system based on plain color detection.
"""

def is_plain_color_similar(plain_score_1, plain_score_2, similarity_threshold=75):
    """Check if plain colors are similar enough to boost texture importance."""
    return min(plain_score_1, plain_score_2) >= similarity_threshold

def get_dynamic_weights(is_plain_1, is_plain_2, l1_score, color_similarity_threshold=75):
    """
    Get dynamic weights based on plain color detection.
    
    Args:
        is_plain_1: Boolean - is image 1 a plain color
        is_plain_2: Boolean - is image 2 a plain color  
        l1_score: Float - L1 (color) similarity score
        color_similarity_threshold: Float - threshold for considering colors "similar"
    
    Returns:
        tuple: (w1, w2, w3, description)
    """
    
    if is_plain_1 and is_plain_2:
        # Both images are plain colors -> prioritize color matching
        return (0.50, 0.35, 0.15, "Both Plain: Prioritize Color")
    
    elif is_plain_1 or is_plain_2:
        # One image is plain
        if l1_score >= color_similarity_threshold:
            # Colors are similar -> boost texture to differentiate
            return (0.15, 0.25, 0.60, "One Plain + Similar Colors: Boost Texture")
        else:
            # Colors are different -> standard weighting
            return (0.20, 0.40, 0.40, "One Plain + Different Colors: Standard")
    
    else:
        # Neither image is plain -> standard weighting
        return (0.20, 0.40, 0.40, "Neither Plain: Standard")

def demonstrate_scenarios():
    """Demonstrate different scenarios with concrete examples."""
    
    print("PLAIN COLOR DYNAMIC WEIGHTING EXAMPLES")
    print("=" * 70)
    
    scenarios = [
        {
            'name': 'Both Plain - Similar Colors (Blue vs Light Blue)',
            'is_plain_1': True,
            'is_plain_2': True,
            'scores': (82, 70, 25),  # L1, L2, L3
            'description': 'Two plain blue walls with slightly different shades'
        },
        {
            'name': 'Both Plain - Different Colors (Blue vs Red)',
            'is_plain_1': True,
            'is_plain_2': True,
            'scores': (15, 75, 30),  # L1, L2, L3
            'description': 'Plain blue wall vs plain red wall'
        },
        {
            'name': 'One Plain - Similar Colors (Blue Wall vs Blue Fabric)',
            'is_plain_1': True,
            'is_plain_2': False,
            'scores': (78, 65, 85),  # L1, L2, L3
            'description': 'Plain blue wall vs textured blue fabric'
        },
        {
            'name': 'One Plain - Different Colors (Blue Wall vs Red Fabric)',
            'is_plain_1': True,
            'is_plain_2': False,
            'scores': (25, 40, 75),  # L1, L2, L3
            'description': 'Plain blue wall vs textured red fabric'
        },
        {
            'name': 'Neither Plain (Wood vs Marble)',
            'is_plain_1': False,
            'is_plain_2': False,
            'scores': (45, 55, 80),  # L1, L2, L3
            'description': 'Textured wood grain vs textured marble'
        }
    ]
    
    for scenario in scenarios:
        l1, l2, l3 = scenario['scores']
        is_plain_1 = scenario['is_plain_1']
        is_plain_2 = scenario['is_plain_2']
        
        # Get dynamic weights
        w1, w2, w3, weight_desc = get_dynamic_weights(is_plain_1, is_plain_2, l1)
        
        # Calculate scores
        original_score = 0.20 * l1 + 0.40 * l2 + 0.40 * l3
        dynamic_score = w1 * l1 + w2 * l2 + w3 * l3
        
        print(f"\nSCENARIO: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Plain Detection: Image1={is_plain_1}, Image2={is_plain_2}")
        print(f"Layer Scores: L1={l1}, L2={l2}, L3={l3}")
        print("-" * 50)
        print(f"Original Weights (0.20/0.40/0.40): {original_score:.1f}")
        print(f"Dynamic Weights  ({w1:.2f}/{w2:.2f}/{w3:.2f}): {dynamic_score:.1f}")
        print(f"Weight Strategy: {weight_desc}")
        print(f"Difference: {dynamic_score - original_score:+.1f}")
        
        # Analysis
        if dynamic_score > original_score:
            impact = "BOOST" 
        elif dynamic_score < original_score:
            impact = "PENALTY"
        else:
            impact = "NO CHANGE"
        print(f"Impact: {impact}")

def texture_importance_analysis():
    """Show how texture importance changes based on plain detection."""
    
    print(f"\n\nTEXTURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print("How L3 (texture) weight changes based on plain color detection:")
    print()
    
    cases = [
        ("Both Plain", 0.15, "Color dominates, texture less important"),
        ("One Plain + Similar Colors", 0.60, "Texture crucial for differentiation"), 
        ("One Plain + Different Colors", 0.40, "Standard texture importance"),
        ("Neither Plain", 0.40, "Standard texture importance")
    ]
    
    print(f"{'Case':<30} {'L3 Weight':<12} {'Rationale'}")
    print("-" * 70)
    for case, weight, rationale in cases:
        print(f"{case:<30} {weight:<12.2f} {rationale}")

if __name__ == "__main__":
    demonstrate_scenarios()
    texture_importance_analysis() 