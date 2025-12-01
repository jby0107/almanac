"""Classification utilities for card attributes."""

import cv2
import numpy as np
from .constants import COLOR_FROM_NAME, SHAPE_FROM_NAME, NUMBER_FROM_NAME, FILL_FROM_NAME
from .image_processing import contour_to_mask
from .templates import get_template_moments


def classify_color(card, symbol_contours):
    """Classify the color of symbols on a card."""
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

    hues = []
    for cnt in symbol_contours:
        mask = contour_to_mask(card.shape, cnt, thickness=2)
        # Only consider pixels where mask>0
        symbol_pixels = hsv[mask > 0]
        if len(symbol_pixels) == 0:
            continue
        mean_hue = symbol_pixels[:, 0].mean()
        hues.append(mean_hue)

    if not hues:
        raise Exception("No colors found")

    avg_hue = np.mean(hues)

    # Threshold logic for color classification
    h = avg_hue
    if h > 170 or h < 10:
        color = "purple"
    elif 20 < h < 80:
        color = "green"
    else:
        color = "red"

    return COLOR_FROM_NAME[color]


def classify_filling(card, symbol_contours):
    """Classify the filling style of symbols on a card."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    ratios = []

    for cnt in symbol_contours:
        mask = contour_to_mask(card.shape, cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        symbol_gray = gray[y:y+h, x:x+w]
        symbol_mask = mask[y:y+h, x:x+w]

        inside = symbol_gray[symbol_mask > 0]
        if inside.size == 0:
            continue

        # Threshold to decide "ink" vs background
        ink = (inside < 210).sum()
        total = inside.size
        ratio = ink / total
        ratios.append(ratio)
        break

    if not ratios:
        return FILL_FROM_NAME["outline"]

    r = np.mean(ratios)

    # Heuristic for fill classification
    if r > 0.98:
        fill = "solid"
    elif r < 0.5:
        fill = "outline"
    else:
        fill = "striped"

    return FILL_FROM_NAME[fill]


def classify_shape_geometric(card, symbol_contours):
    """Classify symbol shape using geometric features."""
    if not symbol_contours:
        return SHAPE_FROM_NAME["flower"]
    
    # Use largest contour as the symbol
    cnt = max(symbol_contours, key=cv2.contourArea)
    mask = contour_to_mask(card.shape, cnt, thickness=1)
    edge_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not edge_cnts:
        return SHAPE_FROM_NAME["flower"]
    
    # Get the largest contour from the edge detection
    edge_cnt = max(edge_cnts, key=cv2.contourArea)

    area = cv2.contourArea(edge_cnt)
    perimeter = cv2.arcLength(edge_cnt, True)

    if area <= 0 or perimeter <= 0:
        return SHAPE_FROM_NAME["flower"]

    # Convex hull & solidity
    hull = cv2.convexHull(edge_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Compactness (not heavily used here, but kept for debugging/future tuning)
    compactness = 4 * np.pi * area / (perimeter * perimeter)

    # Polygon approximation (vertex count)
    epsilon = 0.05 * perimeter
    approx = cv2.approxPolyDP(edge_cnt, epsilon, True)
    vertices = len(approx)

    # Convexity defects: number and average depth
    hull_indices = cv2.convexHull(edge_cnt, returnPoints=False)
    num_defects = 0
    mean_defect_depth = 0.0

    if hull_indices is not None and len(hull_indices) > 3:
        defects = cv2.convexityDefects(edge_cnt, hull_indices)
        if defects is not None and len(defects) > 0:
            num_defects = defects.shape[0]
            # depth is stored *256
            mean_defect_depth = float(defects[:, 0, 3].mean()) / 256.0

    # Heuristic classification tuned to three shapes
    print(num_defects, vertices, mean_defect_depth)
    # Sandglass: very few but deep concavities, outline otherwise smooth
    if vertices == 6 or (abs(vertices - 6) < 2 and num_defects <= 10):
        return SHAPE_FROM_NAME["sandglass"]
    # Flower: many shallow indentations (lots of defects & vertices)
    if num_defects >= 10 or (vertices >= 14 and mean_defect_depth < 10):
        return SHAPE_FROM_NAME["flower"]
    # Star: a handful of medium concavities
    return SHAPE_FROM_NAME["star"]


def classify_shape(card, symbol_contours, use_geometric=False):
    """
    Classify shape using Hu moments (default) or geometric features (fallback).
    
    Args:
        use_geometric: If True, use geometric features instead of template matching
    """
    if use_geometric:
        return classify_shape_geometric(card, symbol_contours)
    
    # We'll classify based on the largest symbol (assuming all 1/2/3 symbols are same shape)
    if not symbol_contours:
        return SHAPE_FROM_NAME["flower"]  # default/fallback
    
    template_moments = get_template_moments()
    if not template_moments:
        print("Warning: No template moments loaded, falling back to geometric method")
        return classify_shape_geometric(card, symbol_contours)

    # Use the largest contour (most representative symbol)
    cnt = max(symbol_contours, key=cv2.contourArea)
    
    # Calculate Hu moments from the contour directly
    moments = cv2.moments(cnt)
    if moments['m00'] == 0:
        return SHAPE_FROM_NAME["flower"]  # fallback
    
    hu = cv2.HuMoments(moments)
    
    # Normalize Hu moments to log scale for comparison (more stable)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    
    # Compare to each template
    best_name = None
    best_dist = float("inf")

    for name, tmpl_hu in template_moments.items():
        tmpl_hu_log = -np.sign(tmpl_hu) * np.log10(np.abs(tmpl_hu) + eps)
        # Use Euclidean distance in log space
        d = np.linalg.norm(hu_log - tmpl_hu_log)
        if d < best_dist:
            best_dist = d
            best_name = name

    if best_name is None:
        best_name = "flower"
    
    # Debug output
    print(f"Shape match: {best_name} (distance: {best_dist:.4f})")

    return SHAPE_FROM_NAME[best_name]

