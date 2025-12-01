"""Window management and automation utilities."""

import subprocess
import time
import pyautogui
import cv2
import numpy as np

# Window region offsets
REGION_OFFSET_X = 5
REGION_OFFSET_Y = 100
REGION_OFFSET_W = 10
REGION_OFFSET_H = 150

APP_NAME = "iPhone Mirroring"

# Get window position and size using AppleScript
_script = f'''
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

_out = subprocess.check_output(["osascript", "-e", _script]).decode().strip()
_values = [v.strip() for v in _out.split(",") if v.strip()]
WIN_X, WIN_Y, WIN_W, WIN_H = map(int, _values)


def click_in_window(position):
    """
    Clicks the center of the given position rectangle in the window space.
    'position' can be a tuple (x1, y1, x2, y2)
    or the special string 'center' to click the middle of the window,
    or 'submit' to click the submit button area.
    """
    if position == "center":
        x = (WIN_W // 2)
        y = (WIN_H // 2)
        screen_x = WIN_X + x
        screen_y = WIN_Y + y
    elif position == 'submit':
        x = (WIN_W // 2)
        y = int(WIN_H * 0.9)
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


def capture_screenshot():
    """Capture a screenshot of the game window region."""
    img = pyautogui.screenshot(region=(
        WIN_X + REGION_OFFSET_X,
        WIN_Y + REGION_OFFSET_Y,
        WIN_W - REGION_OFFSET_W,
        WIN_H - REGION_OFFSET_H,
    ))
    # Convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame

