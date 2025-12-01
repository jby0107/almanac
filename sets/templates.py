"""Template loading and matching utilities."""

import cv2
from pathlib import Path

TEMPLATE_MOMENTS = {}

template_paths = {
    "flower": "./sets/screenshots/flower.jpg",
    "star": "./sets/screenshots/star.jpg",
    "sandglass": "./sets/screenshots/sandglass.jpg"
}


def hu_from_image(img):
    """Extract Hu moments from an image."""
    if img is None or img.size == 0:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Add edge detection step
    edges = cv2.Canny(gray, 40, 120)
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments)
    return hu


def load_templates():
    """Load template images and compute their Hu moments."""
    global TEMPLATE_MOMENTS
    TEMPLATE_MOMENTS = {}
    
    for name, path in template_paths.items():
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load template image: {path}")
            continue
        hu = hu_from_image(img)
        if hu is not None:
            TEMPLATE_MOMENTS[name] = hu
        else:
            print(f"Warning: Could not extract moments from template: {path}")


def get_template_moments():
    """Get the loaded template moments."""
    return TEMPLATE_MOMENTS


# Load templates on module import
load_templates()

