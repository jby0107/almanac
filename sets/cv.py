import subprocess
from textwrap import fill
from unittest import result
import pyautogui
import time
import cv2
import numpy as np
from typing import Tuple, Literal
from logic import Solver

REGION_OFFSET_X = 5
REGION_OFFSET_Y = 100
REGION_OFFSET_W = 10
REGION_OFFSET_H = 150

Combi = Tuple[Literal[0, 1, 2], Literal[0, 1, 2], Literal[0, 1, 2], Literal[0, 1, 2]]

COLOR  = {
    "green": 0,
    "purple": 1,
    "red": 2,
}

SHAPE = {
    "flower": 0,
    "star": 1,
    "sandglass": 2,
}

NUMBER = {
    1: 0,
    2: 1,
    3: 2,
}

FILL = {
    "solid": 0,
    "striped": 1,
    "outline": 2,
}

APP_NAME = "iPhone Mirroring"

script = f'''
tell application "System Events"
    tell process "{APP_NAME}"
        if (count of windows) = 0 then
            error "No windows found"
        end if
        set winPos to position of front window
        set winSize to size of front window
        set x to item 1 of winPos as string
        set y to item 2 of winPos as string
        set w to item 1 of winSize as string
        set h to item 2 of winSize as string
        return x & "," & y & "," & w & "," & h
    end tell
end tell
'''

out = subprocess.check_output(["osascript", "-e", script]).decode().strip()
values = [v.strip() for v in out.split(",") if v.strip()]
offsets = {}
WIN_X, WIN_Y, WIN_W, WIN_H = map(int, values)
# print(WIN_X, WIN_Y, WIN_W, WIN_H)

def hu_from_image(img):
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

TEMPLATE_MOMENTS = {}
template_paths = {
    "flower": "./sets/screenshots/flower.jpg",
    "star": "./sets/screenshots/star.jpg",
    "sandglass": "./sets/screenshots/sandglass.jpg"
}

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


def click_in_window(position):
    """
    Clicks the center of the given position rectangle in the window space.
    'position' can be a tuple (x1, y1, x2, y2)
    or the special string 'center' to click the middle of the window.
    """
    if position == "center":
        x = (WIN_W // 2)
        y = (WIN_H // 2)
        screen_x = WIN_X + x
        screen_y = WIN_Y + y
    elif position == 'submit':
        x = (WIN_W // 2)
        y = (WIN_H * 0.9)
        screen_x = WIN_X + x
        screen_y = WIN_Y + y
    else:
        x1, y1, x2, y2 = position
        x = x1 + (x2 - x1) // 2 + REGION_OFFSET_X
        y = y1 + (y2 - y1) // 2 + REGION_OFFSET_Y
        screen_x = WIN_X + x
        screen_y = WIN_Y + y
    pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    pyautogui.click()
    time.sleep(0.05)

# click_in_window(WIN_W // 2, WIN_H // 2)

def extract_cards(frame):
    h, w, _ = frame.shape

    # Slight blur + edge detection
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_boxes    = []
    positions     = []
    good_contours = []   # <-- filtered contours

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        area = cw * ch
        ar   = cw / ch

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

    # --- Debug drawing ---
    # debug = frame.copy()
    # cv2.drawContours(debug, good_contours, -1, (0, 255, 0), 2)
    # # or, to draw rectangles instead:
    # # for (x, y, cw, ch) in card_boxes:
    # #     cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 255, 0), 2)

    # cv2.imshow("Filtered contours", debug)
    # cv2.waitKey(0)     # waits until any key is pressed while this window has focus
    # cv2.destroyAllWindows()

    # you probably also want to return the actual crops, so:
    # Sort positions first by the first element (x1), then by the third (x2)
    positions.sort(key=lambda pos: (pos[1], pos[0]))
    cards = []
    for (x1, y1, x2, y2) in positions:
        cards.append(cv2.resize(frame[y1+10:y2-10, x1+10:x2-10], (3*(x2-10 - x1-10), 3*(y2-10 - y1-10))))

    
    return cards, positions



def find_symbol_contours(card):
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

    # debug = gray.copy()
    # cv2.drawContours(debug, contours, -1, (0, 255, 0), 2)
    # # or, to draw rectangles instead:
    # # for (x, y, cw, ch) in card_boxes:
    # #     cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 255, 0), 2)

    # cv2.imshow("contours", debug)
    # cv2.waitKey(0)     # waits until any key is pressed while this window has focus
    # cv2.destroyAllWindows()

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
    mask = np.zeros(card_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=thickness)
    return mask

def classify_color(card, symbol_contours):
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

    # *** You will need to print avg_hue values from a few samples
    # and set thresholds correctly for your screen.
    # Rough ballpark (NOT exact; you'll tune):
    # - red: around 0 or 170–180 (wrap-around)
    # - green: around 50–80
    # - purple: around 130–160

    h = avg_hue
    # print(h)
    # Example threshold logic (placeholder – tune this):
    if h > 170 or h < 10:
        color = "purple"
    elif 20 < h < 80:
        color = "green"
    else:
        color = "red"

    return COLOR[color]

def classify_filling(card, symbol_contours):
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
        # (tune the value 200 or similar)
        ink = (inside < 210).sum()
        total = inside.size
        ratio = ink / total
        ratios.append(ratio)
        break

    if not ratios:
        return FILL["outline"]

    r = np.mean(ratios)
    # print(r)

    # Rough heuristic: tune by printing r for examples
    if r > 0.98:
        fill = "solid"
    elif r < 0.5:
        fill = "outline"
    else:
        fill = "striped"

    return FILL[fill]

def classify_shape_geometric(card, symbol_contours):
    """Classify symbol shape into flower / star / sandglass using geometric features."""
    if not symbol_contours:
        # Fallback
        return SHAPE["flower"]
    
    # Use largest contour as the symbol
    cnt = max(symbol_contours, key=cv2.contourArea)
    mask = contour_to_mask(card.shape, cnt, thickness=1)
    edge_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not edge_cnts:
        return SHAPE["flower"]
    
    # Get the largest contour from the edge detection
    edge_cnt = max(edge_cnts, key=cv2.contourArea)

    area = cv2.contourArea(edge_cnt)
    perimeter = cv2.arcLength(edge_cnt, True)

    if area <= 0 or perimeter <= 0:
        return SHAPE["flower"]

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

    # ---- Heuristic classification tuned to your three shapes ----
    print(num_defects, vertices, mean_defect_depth)
    # Sandglass: very few but deep concavities, outline otherwise smooth
    if vertices == 6 or (abs(vertices - 6) < 2 and num_defects <= 10):
        return SHAPE["sandglass"]
    # Flower: many shallow indentations (lots of defects & vertices)
    if num_defects >= 10 or (vertices >= 14 and mean_defect_depth < 10):
        return SHAPE["flower"]
    # Star: a handful of medium concavities
    return SHAPE["star"]


def classify_shape(card, symbol_contours, template_moments=TEMPLATE_MOMENTS, use_geometric=False):
    """
    Classify shape using Hu moments (default) or geometric features (fallback).
    
    Args:
        use_geometric: If True, use geometric features instead of template matching
    """
    if use_geometric:
        return classify_shape_geometric(card, symbol_contours)
    
    # We'll classify based on the largest symbol (assuming all 1/2/3 symbols are same shape)
    if not symbol_contours:
        return SHAPE["flower"]  # default/fallback
    
    if not template_moments:
        print("Warning: No template moments loaded, falling back to geometric method")
        return classify_shape_geometric(card, symbol_contours)

    # Use the largest contour (most representative symbol)
    cnt = max(symbol_contours, key=cv2.contourArea)
    
    # Calculate Hu moments from the contour directly
    moments = cv2.moments(cnt)
    if moments['m00'] == 0:
        return SHAPE["flower"]  # fallback
    
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

    return SHAPE[best_name]


img = pyautogui.screenshot(region=(
    WIN_X + REGION_OFFSET_X,
    WIN_Y + REGION_OFFSET_Y,
    WIN_W - REGION_OFFSET_W,
    WIN_H - REGION_OFFSET_H,
))
# img = cv2.imread("./sets/screenshots/screenshot.png")
frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# cv2.imwrite("./sets/screenshots/screenshot1.png", frame)
cards, positions = extract_cards(frame)
print(positions[0], positions[1], positions[2])
# print(len(cards))
# print(positions)


combinations = []
for card in cards:
    symbol_contours, thresh = find_symbol_contours(card)
    num = NUMBER[min(len(symbol_contours), 3)]
    color = classify_color(card, symbol_contours)
    filling = classify_filling(card, symbol_contours)
    shape = classify_shape(card, symbol_contours, use_geometric=True)
    combinations.append((color, shape, num, filling))

solver = Solver(combinations)
result = solver.solve()
print(result)
click_in_window("center")
for r in result:
    click_in_window(positions[r[0]])
    click_in_window(positions[r[1]])
    click_in_window(positions[r[2]])
    click_in_window('submit')
    time.sleep(0.7)