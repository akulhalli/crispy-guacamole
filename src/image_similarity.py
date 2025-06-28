import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from skimage.util import img_as_ubyte
import imagehash
from numpy.linalg import norm
import time

class ImageSimilarityEngine:
    def __init__(self, scoring_method='dynamic_plain'):
        """
        Initialize the ImageSimilarityEngine with a configurable scoring method.
        
        Args:
            scoring_method (str): The scoring method to use. Options:
                - 'dynamic_plain': Dynamic weighting with L2 dampening (default)
                - 'original_early_exit': Uses the original early-exit system  
                - 'dynamic_texture_moderate': Dynamic logic + moderate L3 emphasis (0.15/0.35/0.50)
                - 'dynamic_texture_strong': Dynamic logic + strong L3 emphasis (0.10/0.30/0.60)
        """
        self.scoring_method = scoring_method
        self._dynamic_plain_score = 0.0  # For dynamic_plain method

    def calculate_similarity(self, image1: Image.Image, image2: Image.Image) -> dict:
        # Step 1: Preprocess images to ensure same dimensions (tiling/cropping logic)
        start_preprocess = time.perf_counter()
        image1, image2 = self._make_same_size(image1, image2)
        preprocess_time = time.perf_counter() - start_preprocess

        results = {
            "L1_score": 0.0,
            "L2_score": 0.0,
            "L3_score": 0.0,
            "Final_score": 0.0,
            "preprocess_time": preprocess_time,
            "L1_time": 0.0,
            "L2_time": 0.0,
            "L3_time": 0.0,
            "layers_computed": [],
            "scoring_method": self.scoring_method
        }

        if self.scoring_method == 'original_early_exit':
            # Use the original early-exit system for backward compatibility
            return self._calculate_similarity_original(image1, image2, results)
        else:
            # For all other methods (including new 'original'), always compute all layers
            return self._calculate_similarity_no_exit(image1, image2, results)

    def _calculate_similarity_original(self, image1, image2, results):
        """Original implementation with early exits."""
        # Layer 1: Spatially-Aware Color Filter
        start_l1 = time.perf_counter()
        l1_score = self._layer1(image1, image2)
        l1_time = time.perf_counter() - start_l1
        
        results["L1_score"] = l1_score
        results["L1_time"] = l1_time
        results["layers_computed"].append("L1")
        
        if l1_score < 65.0:
            # Only L1 computed - final score is 100% L1
            results["Final_score"] = l1_score
            return results

        # Layer 2: Structural Similarity Filter
        start_l2 = time.perf_counter()
        l2_score, early_exit = self._layer2(image1, image2)
        l2_time = time.perf_counter() - start_l2
        
        results["L2_score"] = l2_score
        results["L2_time"] = l2_time
        results["layers_computed"].append("L2")
        
        if early_exit:
            # Only L1 and L2 computed - normalize weights: L1=33%, L2=67%
            results["Final_score"] = (l1_score + 2 * l2_score) / 3
            return results

        # Layer 3: Fine-Grained Texture Analysis
        start_l3 = time.perf_counter()
        l3_score = self._layer3(image1, image2)
        l3_time = time.perf_counter() - start_l3
        
        results["L3_score"] = l3_score
        results["L3_time"] = l3_time
        results["layers_computed"].append("L3")

        # All three layers computed - original weights: L1=20%, L2=40%, L3=40%
        results["Final_score"] = 0.2 * l1_score + 0.4 * l2_score + 0.4 * l3_score
        return results

    def _calculate_similarity_no_exit(self, image1, image2, results):
        """Calculate similarity without early exits - always compute all layers."""
        # Layer 1: Spatially-Aware Color Filter
        start_l1 = time.perf_counter()
        l1_score = self._layer1(image1, image2)
        l1_time = time.perf_counter() - start_l1
        
        results["L1_score"] = l1_score
        results["L1_time"] = l1_time
        results["layers_computed"].append("L1")

        # Layer 2: Structural Similarity Filter (no early exits)
        start_l2 = time.perf_counter()
        l2_score = self._layer2_no_exit(image1, image2)
        l2_time = time.perf_counter() - start_l2
        
        results["L2_score"] = l2_score
        results["L2_time"] = l2_time
        results["layers_computed"].append("L2")

        # Layer 3: Fine-Grained Texture Analysis
        start_l3 = time.perf_counter()
        l3_score = self._layer3(image1, image2)
        l3_time = time.perf_counter() - start_l3
        
        results["L3_score"] = l3_score
        results["L3_time"] = l3_time
        results["layers_computed"].append("L3")

        # Handle dynamic scoring methods (all variants with L2 dampening)
        if self.scoring_method in ['dynamic_plain', 'dynamic_texture_moderate', 'dynamic_texture_strong']:
            # Check if images are plain and get dynamic weights
            is_plain_1 = self._is_plain_image(image1)
            is_plain_2 = self._is_plain_image(image2)
            
            # Get base weights depending on method
            if self.scoring_method == 'dynamic_texture_moderate':
                w1, w2, w3 = self._get_dynamic_texture_weights(is_plain_1, is_plain_2, l1_score, 'moderate')
            elif self.scoring_method == 'dynamic_texture_strong':
                w1, w2, w3 = self._get_dynamic_texture_weights(is_plain_1, is_plain_2, l1_score, 'strong')
            else:  # dynamic_plain
                w1, w2, w3 = self._get_dynamic_weights(is_plain_1, is_plain_2, l1_score)
            
            # Calculate base score with dynamic weights
            base_score = w1 * l1_score + w2 * l2_score + w3 * l3_score
            
            # Check for L2 dampening case
            needs_dampening = self._detect_l2_dampening_case(l1_score, l2_score, l3_score)
            
            if needs_dampening:
                dampening_factor = self._calculate_l2_dampening_factor(l1_score, l2_score, l3_score)
                self._dynamic_plain_score = base_score * dampening_factor
                
                # Ensure L2 never increases a low score
                if base_score < 50 and l2_score > (l1_score + l3_score) / 2:
                    # Additional penalty for high L2 when overall score is low
                    self._dynamic_plain_score *= 0.85
            else:
                self._dynamic_plain_score = base_score
        
        # Apply the selected scoring method
        results["Final_score"] = self._compute_final_score(l1_score, l2_score, l3_score)
        
        return results

    def _layer2_no_exit(self, img1, img2):
        """Layer 2 implementation without early exits - always computes both 2A and 2B.
        
        Note: L2 scores can be negative for very dissimilar images.
        """
        # Step 2A: Global pHash
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        global_ham = hash1 - hash2
        # Allow negative scores for L2
        global_sim = (1 - float(global_ham) / 64) * 100
        
        # Step 2B: Local pHash (3x3 grid) - always compute
        local_hams = []
        for i in range(3):
            for j in range(3):
                block1 = self._crop_grid_block(img1, 3, i, j)
                block2 = self._crop_grid_block(img2, 3, i, j)
                h1 = imagehash.phash(block1)
                h2 = imagehash.phash(block2)
                local_hams.append(h1 - h2)
        avg_local_ham = np.mean(local_hams)
        # Allow negative scores for L2
        local_sim = (1 - avg_local_ham / 64) * 100
        
        # Always compute full L2 score
        l2_score = 0.3 * global_sim + 0.7 * local_sim
        return l2_score

    def _is_within_percent(self, value1, value2, percent=10):
        """Check if two values are within a certain percentage of each other."""
        if value1 == 0 and value2 == 0:
            return True
        avg = (abs(value1) + abs(value2)) / 2
        if avg == 0:
            return True
        diff_percent = abs(value1 - value2) / avg * 100
        return diff_percent <= percent

    def _detect_l2_dampening_case(self, l1_score, l2_score, l3_score, similarity_threshold=12, l2_diff_threshold=20):
        """
        Detect if L1 and L3 are similar but L2 is significantly different.
        
        Args:
            l1_score, l2_score, l3_score: Layer scores
            similarity_threshold: % threshold for L1/L3 to be considered "similar" (default 12%)
            l2_diff_threshold: Minimum difference for L2 to be considered "significantly different"
        
        Returns:
            bool: True if dampening should be applied
        """
        # Check if L1 and L3 are similar (within threshold)
        l1_l3_similar = self._is_within_percent(l1_score, l3_score, similarity_threshold)
        
        if not l1_l3_similar:
            return False
        
        # Check if L2 is significantly different from both L1 and L3
        l1_l3_avg = (l1_score + l3_score) / 2
        l2_diff = abs(l2_score - l1_l3_avg)
        
        return l2_diff >= l2_diff_threshold

    def _calculate_l2_dampening_factor(self, l1_score, l2_score, l3_score):
        """
        Calculate dampening factor based on L2 disagreement.
        
        Returns:
            float: Dampening factor (0.5 to 1.0, where lower means more dampening)
        """
        l1_l3_avg = (l1_score + l3_score) / 2
        l2_diff = abs(l2_score - l1_l3_avg)
        
        # More difference = more dampening (reduced by 5% to be less aggressive)
        if l2_diff >= 40:
            return 0.55  # Strong dampening (was 0.50)
        elif l2_diff >= 30:
            return 0.70  # Moderate dampening (was 0.65)
        elif l2_diff >= 20:
            return 0.85  # Mild dampening (was 0.80)
        else:
            return 1.0   # No dampening

    def _get_dynamic_weights(self, is_plain_1, is_plain_2, l1_score, color_similarity_threshold=75):
        """
        Get dynamic weights based on plain color detection.
        
        Args:
            is_plain_1: Boolean - is image 1 a plain color
            is_plain_2: Boolean - is image 2 a plain color  
            l1_score: Float - L1 (color) similarity score
            color_similarity_threshold: Float - threshold for considering colors "similar"
        
        Returns:
            tuple: (w1, w2, w3)
        """
        
        if is_plain_1 and is_plain_2:
            # Both images are plain colors -> prioritize color matching
            return (0.50, 0.35, 0.15)
        
        elif is_plain_1 or is_plain_2:
            # One image is plain
            if l1_score >= color_similarity_threshold:
                # Colors are similar -> boost texture to differentiate
                return (0.15, 0.25, 0.60)
            else:
                # Colors are different -> standard weighting
                return (0.20, 0.40, 0.40)
        
        else:
            # Neither image is plain -> standard weighting
            return (0.20, 0.40, 0.40)

    def _get_dynamic_texture_weights(self, is_plain_1, is_plain_2, l1_score, texture_mode, color_similarity_threshold=75):
        """
        Get dynamic weights based on plain color detection with texture emphasis.
        
        Args:
            is_plain_1: Boolean - is image 1 a plain color
            is_plain_2: Boolean - is image 2 a plain color  
            l1_score: Float - L1 (color) similarity score
            texture_mode: String - 'moderate' or 'strong' for texture emphasis level
            color_similarity_threshold: Float - threshold for considering colors "similar"
        
        Returns:
            tuple: (w1, w2, w3) with texture emphasis applied
        """
        
        # Base texture weights
        if texture_mode == 'moderate':
            texture_weights = (0.15, 0.35, 0.50)
        elif texture_mode == 'strong':
            texture_weights = (0.10, 0.30, 0.60)
        else:
            raise ValueError(f"Unknown texture mode: {texture_mode}")
        
        if is_plain_1 and is_plain_2:
            # Both images are plain colors -> prioritize color matching
            return (0.50, 0.35, 0.15)
        
        elif is_plain_1 or is_plain_2:
            # One image is plain
            if l1_score >= color_similarity_threshold:
                # Colors are similar -> use maximum texture weighting
                return texture_weights
            else:
                # Colors are different -> use texture weights but reduce L1 penalty
                return (texture_weights[0] + 0.05, texture_weights[1], texture_weights[2] - 0.05)
        
        else:
            # Neither image is plain -> use texture weighting
            return texture_weights

    def _compute_final_score(self, l1_score, l2_score, l3_score):
        """Compute final score using the selected scoring method."""
        if self.scoring_method == 'original_early_exit':
            # Original method: simple weighted average with negative L2/L3 allowed
            return 0.2 * l1_score + 0.4 * l2_score + 0.4 * l3_score
        elif self.scoring_method in ['dynamic_plain', 'dynamic_texture_moderate', 'dynamic_texture_strong']:
            # Dynamic weighting with L2 dampening - handled in _calculate_similarity_no_exit
            return self._dynamic_plain_score
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")



    def _make_same_size(self, img1, img2):
        """
        Crop images to same size using center cropping approach.
        
        Logic:
        1. If both width AND height differences are within 10%, crop both to smaller image size
        2. Otherwise, crop larger image to smaller image size
        3. Always use center cropping for equal margins
        """
        w1, h1 = img1.size
        w2, h2 = img2.size
        
        # If already same size, return as-is
        if (w1, h1) == (w2, h2):
            return img1, img2
        
        # Calculate size differences as percentages
        width_diff_pct = abs(w1 - w2) / min(w1, w2) * 100
        height_diff_pct = abs(h1 - h2) / min(h1, h2) * 100
        
        # Always target the minimum dimensions to ensure no upscaling
        target_w = min(w1, w2)
        target_h = min(h1, h2)
        
        def center_crop(img, target_width, target_height):
            """Center crop image to target dimensions"""
            current_w, current_h = img.size
            
            # If target is same size, return as-is
            if current_w == target_width and current_h == target_height:
                return img
            
            # If target is larger than current, we can't crop (this shouldn't happen in our logic)
            if target_width > current_w or target_height > current_h:
                raise ValueError(f"Cannot crop image ({current_w}x{current_h}) to larger size ({target_width}x{target_height})")
            
            # Calculate center crop coordinates
            crop_left = (current_w - target_width) // 2
            crop_top = (current_h - target_height) // 2
            crop_right = crop_left + target_width
            crop_bottom = crop_top + target_height
            
            return img.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # Apply logic based on size differences
        if width_diff_pct <= 10.0 and height_diff_pct <= 10.0:
            # Both differences within 10% - crop both images to minimum dimensions
            cropped_img1 = center_crop(img1, target_w, target_h)
            cropped_img2 = center_crop(img2, target_w, target_h)
        else:
            # Size difference too large - crop both images to minimum dimensions
            # (This ensures no image is larger than the other after processing)
            cropped_img1 = center_crop(img1, target_w, target_h)
            cropped_img2 = center_crop(img2, target_w, target_h)
        
        return cropped_img1, cropped_img2

    def _is_plain_image(self, img):
        """Detect if an image is plain/solid colored by checking color variance"""
        arr = np.array(img.convert('RGB'))
        
        # Calculate standard deviation across all pixels for each channel
        std_r = np.std(arr[:, :, 0])
        std_g = np.std(arr[:, :, 1])
        std_b = np.std(arr[:, :, 2])
        
        # If all channels have low variance, it's a plain image
        # Threshold: std < 15 (out of 255) indicates very uniform color
        threshold = 15.0
        return std_r < threshold and std_g < threshold and std_b < threshold

    def _plain_image_similarity(self, img1, img2):
        """Enhanced color comparison specifically for plain/solid colored images"""
        # Get average colors
        arr1 = np.array(img1.convert('RGB'))
        arr2 = np.array(img2.convert('RGB'))
        
        avg_color1 = np.mean(arr1, axis=(0, 1))  # [R, G, B]
        avg_color2 = np.mean(arr2, axis=(0, 1))  # [R, G, B]
        
        # Calculate Euclidean distance in RGB space
        color_dist = np.sqrt(np.sum((avg_color1 - avg_color2) ** 2))
        
        # Maximum possible distance is sqrt(3 * 255^2) = ~441
        max_color_dist = np.sqrt(3 * 255**2)
        
        # Convert to similarity percentage (more sensitive than original algorithm)
        color_similarity = max(0.0, float(100 * (1 - color_dist / max_color_dist)))
        
        # For plain images, apply stricter thresholds
        # Even small color differences should result in lower scores
        if color_dist > 30:  # Significant color difference (e.g., white vs cream)
            color_similarity *= 0.5  # Reduce score by half
        
        if color_dist > 60:  # Major color difference  
            color_similarity *= 0.3  # Reduce score significantly
        
        # Add perceptual color difference (Delta E approximation)
        # Convert to LAB-like comparison for better perceptual accuracy
        lab_similarity = self._perceptual_color_similarity(avg_color1, avg_color2)
        
        # Combine RGB distance and perceptual similarity
        final_similarity = 0.7 * color_similarity + 0.3 * lab_similarity
        
        return max(0, min(100, final_similarity))

    def _perceptual_color_similarity(self, color1, color2):
        """Calculate perceptual color similarity using simplified Delta E"""
        # Simple RGB to approximate LAB conversion for better perceptual accuracy
        # This is a simplified version - not true LAB but better than pure RGB
        
        # Weight channels by human eye sensitivity (approximate)
        weights = np.array([0.299, 0.587, 0.114])  # Luminance weights
        
        # Calculate weighted difference
        weighted_diff = np.abs(color1 - color2) * weights
        perceptual_dist = np.sum(weighted_diff)
        
        # Maximum weighted difference
        max_perceptual_dist = np.sum(255 * weights)
        
        # Convert to similarity
        similarity = max(0.0, float(100 * (1 - perceptual_dist / max_perceptual_dist)))
        
        return similarity

    def _layer1(self, img1, img2):
        # Check if images are plain/solid colored
        plain1 = self._is_plain_image(img1)
        plain2 = self._is_plain_image(img2)
        
        if plain1 or plain2:
            # Use enhanced color comparison for plain images
            return self._plain_image_similarity(img1, img2)
        
        # Original algorithm for textured images
        # 4x4 grid analysis
        grid_feat1 = self._grid_features(img1, 4)
        grid_feat2 = self._grid_features(img2, 4)
        dist = norm(grid_feat1 - grid_feat2)
        max_dist = np.sqrt(255**2 * 48)  # 48 dims, max diff per channel
        sim_grid = max(0, 100 * (1 - dist / max_dist))
        # Junction analysis
        junction_feat1 = self._junction_features(img1)
        junction_feat2 = self._junction_features(img2)
        cos_sim = np.dot(junction_feat1, junction_feat2) / (norm(junction_feat1) * norm(junction_feat2) + 1e-8)
        sim_overlap = max(0, min(1, cos_sim)) * 100
        # Combine
        return 0.6 * sim_grid + 0.4 * sim_overlap

    def _grid_features(self, img, grid_size):
        arr = np.array(img.convert('RGB'))
        h, w, _ = arr.shape
        feat = []
        for i in range(grid_size):
            for j in range(grid_size):
                y0 = i * h // grid_size
                y1 = (i + 1) * h // grid_size
                x0 = j * w // grid_size
                x1 = (j + 1) * w // grid_size
                block = arr[y0:y1, x0:x1]
                avg = block.mean(axis=(0, 1))
                feat.extend(avg)
        return np.array(feat)

    def _junction_features(self, img):
        arr = np.array(img.convert('RGB'))
        h, w, _ = arr.shape
        grid_h = h // 4
        grid_w = w // 4
        # Four interior corners: (1,1), (1,2), (2,1), (2,2)
        blocks = [
            arr[grid_h:2*grid_h, grid_w:2*grid_w],
            arr[grid_h:2*grid_h, 2*grid_w:3*grid_w],
            arr[2*grid_h:3*grid_h, grid_w:2*grid_w],
            arr[2*grid_h:3*grid_h, 2*grid_w:3*grid_w],
        ]
        # Center block
        ch, cw = h // 2, w // 2
        center_block = arr[ch-grid_h//2:ch+grid_h//2, cw-grid_w//2:cw+grid_w//2]
        blocks.append(center_block)
        feat = [block.mean(axis=(0, 1)) for block in blocks]
        return np.concatenate(feat)

    def _layer2(self, img1, img2):
        # Step 2A: Global pHash
        hash1 = imagehash.phash(img1)
        hash2 = imagehash.phash(img2)
        global_ham = hash1 - hash2
        global_sim = max(0, (1 - global_ham / 64) * 100)
        if global_ham > 10:
            return 0.3 * global_sim, True  # Local_Sim = 0.0, early exit
        # Step 2B: Local pHash (3x3 grid)
        local_hams = []
        for i in range(3):
            for j in range(3):
                block1 = self._crop_grid_block(img1, 3, i, j)
                block2 = self._crop_grid_block(img2, 3, i, j)
                h1 = imagehash.phash(block1)
                h2 = imagehash.phash(block2)
                local_hams.append(h1 - h2)
        avg_local_ham = np.mean(local_hams)
        local_sim = max(0.0, float((1 - avg_local_ham / 64) * 100))
        l2_score = 0.3 * global_sim + 0.7 * local_sim
        if avg_local_ham > 5:
            return l2_score, True
        return l2_score, False

    def _crop_grid_block(self, img, grid_size, i, j):
        arr = np.array(img)
        h, w = arr.shape[:2]
        y0 = i * h // grid_size
        y1 = (i + 1) * h // grid_size
        x0 = j * w // grid_size
        x1 = (j + 1) * w // grid_size
        block = arr[y0:y1, x0:x1]
        return Image.fromarray(block)

    def _layer3(self, img1, img2):
        """LBP histogram comparison for texture analysis.
        
        Note: L3 scores can be negative for very dissimilar textures.
        """
        gray1 = img_as_ubyte(img1.convert('L'))
        gray2 = img_as_ubyte(img2.convert('L'))
        lbp1 = local_binary_pattern(gray1, P=8, R=1, method='uniform')
        lbp2 = local_binary_pattern(gray2, P=8, R=1, method='uniform')
        n_bins = int(lbp1.max() + 1)
        hist1, _ = np.histogram(lbp1, bins=n_bins, range=(0, n_bins), density=True)
        hist2, _ = np.histogram(lbp2, bins=n_bins, range=(0, n_bins), density=True)
        chi2 = 0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-8))
        # Allow negative scores for L3 - remove clamping to 0
        sim_value = 100 * (1 - chi2)
        return sim_value

    def get_processed_images(self, image1: Image.Image, image2: Image.Image):
        """
        Get the processed (cropped/resized) images that would be used in similarity calculation.
        This is useful for generating reports that show the actual images being compared.
        """
        return self._make_same_size(image1, image2) 