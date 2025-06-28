Excellent. Here is a detailed Product Requirement Document (PRD) for the image similarity pipeline. This document is designed to be consumed by a developer (e.g., Claude Sonnet) to ensure an accurate and complete implementation.

---

## Product Requirement Document: Multi-Layered Image Similarity Engine

| **Document ID:** | IS-PRD-V1.0 |
| :--------------- | :------------------------------------------------------ |
| **Version:**     | 1.0                                                     |
| **Date:**        | October 26, 2023                                        |
| **Author:**      | Staff Architect                                         |
| **Status:**      | Approved for Implementation                             |

### 1. Introduction & System Goals

#### 1.1. Purpose

This document specifies the functional and non-functional requirements for a multi-layered image similarity engine. The system's primary goal is to efficiently compute a similarity score between two computer-generated images of identical dimensions, lighting, and quality.

#### 1.2. Core Architectural Principle

The system employs a **coarse-to-fine filtering cascade**. It uses a sequence of three layers, each more computationally expensive and analytically specific than the last. This design ensures computational efficiency by performing early exits on obviously dissimilar image pairs, reserving intensive analysis for the strongest candidates.

#### 1.3. Key Requirement: Mandatory Scoring

Regardless of where the pipeline exits, it **MUST** return a complete breakdown of all layer scores and a final combined score. Layers that are skipped due to an early exit **MUST** be assigned a similarity score of `0.0`.

---

### 2. Functional Requirements

#### 2.1. System Interface

The system **SHALL** be encapsulated in a class. This class **MUST** expose a single public method:

```python
calculate_similarity(image1: Image, image2: Image) -> dict
```

-   **Input:** Two `Pillow.Image` objects of identical dimensions.
-   **Output:** A dictionary containing the computed similarity scores.

#### 2.2. Output Format and Final Score Calculation

The output dictionary **MUST** follow this structure:

```json
{
  "L1_Score": <float>, // Similarity score from Layer 1 (0-100)
  "L2_Score": <float>, // Similarity score from Layer 2 (0-100)
  "L3_Score": <float>, // Similarity score from Layer 3 (0-100)
  "Final_Score": <float> // Final combined score (0-100)
}
```

The `Final_Score` **SHALL** be a weighted average of the individual layer scores:

`Final_Score = (0.2 * L1_Score) + (0.4 * L2_Score) + (0.4 * L3_Score)`

#### 2.3. Core Processing Workflow

The system **MUST** follow a strict sequential workflow with early exit conditions.

1.  **Initialize Results:** Create the results dictionary with all scores set to `0.0`.
2.  **Execute Layer 1:**
    *   Compute `L1_Score` and store it.
    *   Check `L1_Score` against its cutoff (`< 65`). If it fails, HALT processing and proceed to step 5.
3.  **Execute Layer 2:**
    *   Execute Layer 2A (Global). If its check (`Hamming > 10`) fails, compute the partial `L2_Score` using `Local_Sim = 0.0`, store it, HALT, and proceed to step 5.
    *   Execute Layer 2B (Local). Compute the full `L2_Score` using both `Global_Sim` and `Local_Sim`. Store it. Check against its cutoff (`Avg. Hamming > 5`). If it fails, HALT and proceed to step 5.
4.  **Execute Layer 3:**
    *   Compute `L3_Score` and store it.
5.  **Calculate Final Score:** Compute the `Final_Score` using the stored `L1`, `L2`, and `L3` values.
6.  **Return** the final results dictionary.

#### 2.4. Layer 1 Implementation Details: Spatially-Aware Color Filter

-   **Goal:** Fast rejection based on color palette and layout.
-   **Metrics & Techniques:**
    -   **Grid Analysis (Absolute Difference):**
        -   The image **SHALL** be divided into a non-overlapping 4x4 grid.
        -   For each of the 16 blocks, the average `(R, G, B)` color **SHALL** be computed, forming a 48-dimensional feature vector.
        -   The two vectors **SHALL** be compared using **Euclidean Distance**. This distance **MUST** be normalized to a similarity score from 0-100 (`sim_grid`).
    -   **Junction Analysis (Proportional Difference):**
        -   Five non-contiguous blocks **SHALL** be defined at the four major interior grid corners and the exact image center.
        -   For each of these 5 blocks, the average `(R, G, B)` color **SHALL** be computed, forming a 15-dimensional feature vector.
        -   The two vectors **SHALL** be compared using **Cosine Similarity**. The result (`sim_overlap`) is naturally in a 0-1 range and should be scaled to 0-100.
-   **Score Combination:** `L1_Score = (0.6 * sim_grid) + (0.4 * sim_overlap)`
-   **Early Exit Cutoff:** HALT if `L1_Score < 65.0`.

#### 2.5. Layer 2 Implementation Details: Structural Similarity Filter

-   **Goal:** Verify structural integrity using perceptual hashing. A **perceptual hash (pHash)** is a compact binary "fingerprint" of an image's low-frequency structure. Hashes are compared using **Hamming Distance**, which counts the number of differing bits.
-   **Two-Step Process:**
    1.  **Step 2A (Global Analysis):**
        -   A single pHash (64-bit) **SHALL** be generated for the entire image.
        -   The Hamming distance between the two global hashes **SHALL** be calculated.
        -   **Early Exit Cutoff:** HALT if `Global Hamming Distance > 10`.
    2.  **Step 2B (Local Analysis):**
        -   The image **SHALL** be divided into a 3x3 non-overlapping grid.
        -   A pHash **SHALL** be generated for each of the 9 blocks.
        -   The average of the 9 block-wise Hamming distances **SHALL** be calculated.
        -   **Early Exit Cutoff:** HALT if `Average Local Hamming Distance > 5`.
-   **Score Combination:** All Hamming distances **MUST** be normalized to a 0-100 similarity scale (e.g., `Sim = (1 - (dist / 64)) * 100`). The final `L2_Score` **SHALL** be:
    `L2_Score = (0.3 * Global_Sim) + (0.7 * Local_Sim)`

#### 2.6. Layer 3 Implementation Details: Fine-Grained Texture Analysis

-   **Goal:** Final arbitration based on surface texture.
-   **Metrics & Techniques:**
    -   The system **SHALL** use **Local Binary Patterns (LBP)** to describe texture. LBP is an operator that generates a label for each pixel based on its local neighborhood, describing patterns like edges, corners, and flat regions. Suggested parameters: 8 neighbors, radius of 1.
    -   A normalized histogram of the LBP labels across the entire grayscale image **SHALL** be generated.
    -   The two LBP histograms **SHALL** be compared using **Chi-Squared Distance**. This distance **MUST** be normalized to a similarity score from 0-100 (`L3_Score`). A score of 0 for identical histograms and an increasing value for more dissimilar histograms. A suggested normalization is `Sim = max(0, 100 * (1 - dist))`.

---

### 3. Non-Functional Requirements

-   **Dependencies:** The implementation will depend on `Pillow`, `numpy`, `scikit-image`, and `imagehash`. These must be properly managed.
-   **Error Handling:** The system **MUST** raise an error if the input images do not have the same dimensions.

---

### 4. Acceptance Criteria & Test Cases

The following test cases **MUST** be passed for the implementation to be considered complete.

| Test ID | Objective                                           | Input A & B Description                                                                              | Expected Workflow & Exit Point                               | Expected Output (approx.)                                          |
| :------ | :---------------------------------------------------- | :--------------------------------------------------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------------- |
| **TC-01** | **Verify Full Pipeline (Happy Path)**                 | Image A and B are identical computer-generated textures.                                               | All layers pass all checks.                                  | `{"L1": 100, "L2": 100, "L3": 100, "Final": 100}`                    |
| **TC-02** | **Verify Layer 1 Early Exit (Color)**               | A: A blue-green gradient. B: A red-yellow gradient.                                                  | Layer 1 fails with score `<65`. **Exits at Layer 1.**           | `{"L1": <50, "L2": 0.0, "L3": 0.0, "Final": <10}`                     |
| **TC-03** | **Verify Layer 2A Early Exit (Global Structure)**   | A: Blue circle left, gray right. B: Gray left, blue circle right.                                     | Layer 1 passes (`>65`). Layer 2A fails (Global Ham. >10). **Exits at Layer 2A.** | `{"L1": ~70, "L2": <25, "L3": 0.0, "Final": <25}`                      |
| **TC-04** | **Verify Layer 2B Early Exit (Local Structure)**    | A: Globally similar gradient to B, but A has noisy local patterns added to each grid block.                | Layer 1 passes. Layer 2A passes (Global Ham. <10). Layer 2B fails (Avg. Local Ham. >5). **Exits at Layer 2B.** | `{"L1": >90, "L2": <70, "L3": 0.0, "Final": <50}` |
| **TC-05** | **Verify Layer 3 Functionality (Texture)**          | A: Perfectly smooth gray square. B: A gray square of same size, but with a rough sandpaper texture. | L1 passes (~100). L2 passes (~100) as pHash ignores fine texture. L3 registers high distance. | `{"L1": ~100, "L2": ~100, "L3": <20, "Final": <70}`                   |