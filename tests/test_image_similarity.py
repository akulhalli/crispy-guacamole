import pytest
import numpy as np
from PIL import Image, ImageDraw
from src.image_similarity import ImageSimilarityEngine

def make_gradient_image(color1, color2, size=(128, 128)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for i in range(size[1]):
        ratio = i / (size[1] - 1)
        arr[i, :, :] = np.array(color1) * (1 - ratio) + np.array(color2) * ratio
    return Image.fromarray(arr)

def make_circle_image(bg_color, circle_color, size=(128, 128), left=True):
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    if left:
        bbox = (10, size[1]//4, size[0]//2-10, 3*size[1]//4)
    else:
        bbox = (size[0]//2+10, size[1]//4, size[0]-10, 3*size[1]//4)
    draw.ellipse(bbox, fill=circle_color)
    return img

def make_noisy_blocks_image(base_color, noise_level=30, size=(128, 128), grid=3):
    arr = np.full((size[1], size[0], 3), base_color, dtype=np.uint8)
    block_h = size[1] // grid
    block_w = size[0] // grid
    for i in range(grid):
        for j in range(grid):
            noise = np.random.randint(-noise_level, noise_level+1, (block_h, block_w, 3))
            y0, y1 = i*block_h, (i+1)*block_h
            x0, x1 = j*block_w, (j+1)*block_w
            arr[y0:y1, x0:x1] = np.clip(arr[y0:y1, x0:x1] + noise, 0, 255)
    return Image.fromarray(arr)

def make_texture_image(base_color, texture='smooth', size=(128, 128)):
    arr = np.full((size[1], size[0], 3), base_color, dtype=np.uint8)
    if texture == 'rough':
        noise = np.random.randint(-80, 80, (size[1], size[0], 3))
        arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr)

def test_tc01_identical_images():
    engine = ImageSimilarityEngine()
    img = make_gradient_image((100, 200, 150), (200, 100, 50))
    result = engine.calculate_similarity(img, img.copy())
    assert result['L1_score'] == pytest.approx(100, abs=1)
    assert result['L2_score'] == pytest.approx(100, abs=1)
    assert result['L3_score'] == pytest.approx(100, abs=1)
    assert result['Final_score'] == pytest.approx(100, abs=1)

def test_tc02_layer1_early_exit():
    engine = ImageSimilarityEngine()
    img1 = make_gradient_image((0, 128, 255), (0, 255, 128))  # blue-green
    img2 = make_gradient_image((255, 0, 0), (255, 255, 0))    # red-yellow
    result = engine.calculate_similarity(img1, img2)
    assert result['L1_score'] < 65
    assert result['L2_score'] == 0.0
    assert result['L3_score'] == 0.0
    assert result['Final_score'] < 20

def test_tc03_layer2a_early_exit():
    engine = ImageSimilarityEngine()
    img1 = make_circle_image((128, 128, 128), (0, 0, 255), left=True)
    img2 = make_circle_image((128, 128, 128), (0, 0, 255), left=False)
    result = engine.calculate_similarity(img1, img2)
    assert result['L1_score'] > 65
    assert result['L2_score'] < 40
    assert result['L3_score'] == 0.0
    assert result['Final_score'] < 40

def test_tc04_layer2b_early_exit():
    engine = ImageSimilarityEngine()
    img1 = make_gradient_image((100, 200, 150), (200, 100, 50))
    img2 = make_noisy_blocks_image((150, 150, 150), noise_level=60)
    result = engine.calculate_similarity(img1, img2)
    assert result['L1_score'] > 60
    assert result['L2_score'] < 70
    assert result['L3_score'] == 0.0
    assert result['Final_score'] < 60

def test_tc05_layer3_texture():
    engine = ImageSimilarityEngine()
    img1 = make_texture_image((180, 180, 180), 'smooth')
    img2 = make_texture_image((180, 180, 180), 'rough')
    result = engine.calculate_similarity(img1, img2)
    assert result['L1_score'] > 90
    assert result['L2_score'] > 90
    assert result['L3_score'] < 40
    assert result['Final_score'] < 80

def test_edge_different_sizes():
    engine = ImageSimilarityEngine()
    img1 = make_gradient_image((0, 0, 0), (255, 255, 255), size=(64, 128))
    img2 = make_gradient_image((0, 0, 0), (255, 255, 255), size=(128, 64))
    result = engine.calculate_similarity(img1, img2)
    assert all(0 <= v <= 100 for v in result.values())

def test_edge_all_black_white():
    engine = ImageSimilarityEngine()
    img1 = Image.new('RGB', (128, 128), (0, 0, 0))
    img2 = Image.new('RGB', (128, 128), (255, 255, 255))
    result = engine.calculate_similarity(img1, img2)
    assert all(0 <= v <= 100 for v in result.values()) 