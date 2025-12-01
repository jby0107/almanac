"""Main entry point for the Set (so far only this game) game solver."""

import time
from sets.logic import Solver
from sets.window import capture_screenshot, click_in_window
from sets.image_processing import extract_cards, find_symbol_contours
from sets.classification import classify_color, classify_shape, classify_filling
from sets.constants import NUMBER_FROM_NAME


def main():
    """Main execution flow: capture, classify, solve, and click."""
    # Capture screenshot
    frame = capture_screenshot()
    
    # Extract cards from the frame
    cards, positions = extract_cards(frame)
    print(positions[0], positions[1], positions[2])
    
    # Classify each card
    combinations = []
    for card in cards:
        symbol_contours, thresh = find_symbol_contours(card)
        num = NUMBER_FROM_NAME[min(len(symbol_contours), 3)]
        color = classify_color(card, symbol_contours)
        filling = classify_filling(card, symbol_contours)
        shape = classify_shape(card, symbol_contours, use_geometric=True)
        combinations.append((color, shape, num, filling))
    
    # Solve the game
    solver = Solver(combinations)
    result = solver.solve()
    print(result)
    
    # Click the solutions
    click_in_window("center")
    for r in result:
        click_in_window(positions[r[0]])
        click_in_window(positions[r[1]])
        click_in_window(positions[r[2]])
        click_in_window('submit')
        time.sleep(0.7)


if __name__ == "__main__":
    main()
