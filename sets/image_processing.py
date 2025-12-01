"""Image processing utilities for card extraction and contour detection."""

import cv2
import numpy as np


def extract_cards(frame):
    """
    Extract card regions from the game frame.
    
    Returns:
        tuple: (cards, positions) where cards is a list of cropped card images
               and positions is a list of (x1, y1, x2, y2) tuples
    """
    h, w, _ = frame.shape

    # Slight blur + edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_boxes = []
    positions = []
    good_contours = []  # filtered contours

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        area = cw * ch
        ar = cw / ch

        # filter by size & aspect ratio
        if area < 2000:
            continue
        if ar < 1.5 or ar > 3.5:
            continue

        # filter by vertical position
        if y < h * 0.25 or y > h * 0.8:
            continue

        # passed all filters
        good_contours.append(cnt)
        card_boxes.append((x, y, cw, ch))

        # crop card (with small padding)
        pad = int(0.05 * min(cw, ch))
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + cw + pad, w)
        y2 = min(y + ch + pad, h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        positions.append((x1, y1, x2, y2))

    # Sort positions first by the first element (y1), then by the first (x1)
    positions.sort(key=lambda pos: (pos[1], pos[0]))
    cards = []
    for (x1, y1, x2, y2) in positions:
        cards.append(cv2.resize(
            frame[y1+10:y2-10, x1+10:x2-10],
            (3*(x2-10 - x1-10), 3*(y2-10 - y1-10))
        ))

    return cards, positions


def find_symbol_contours(card):
    """
    Find contours of symbols within a card.
    
    Returns:
        tuple: (symbol_contours, thresh) where symbol_contours is a list of contours
               and thresh is the thresholded image
    """
    h, w, _ = card.shape
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold is usually robust to varying brightness
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 10
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    symbol_contours = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # Filter out tiny noise
        if area < 300:
            continue

        # Ignore the outer card border if it gets picked up
        if cw > 0.8 * w and ch > 0.8 * h:
            continue

        symbol_contours.append(cnt)

    # Optionally sort by x so 1/2/3 are in order
    symbol_contours = sorted(symbol_contours, key=lambda c: cv2.boundingRect(c)[0])
    return symbol_contours, thresh


def contour_to_mask(card_shape, contour, thickness=-1):
    """Create a mask from a contour."""
    mask = np.zeros(card_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=thickness)
    return mask

