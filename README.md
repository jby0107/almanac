# Almanac bot

Almanac bot is a project dedicated to creating bots that can play a variety of games in almanac. The ultimate goal is to have a collection of bots for all games within the "almanac" universe.

## Current Status

Currently, the project only supports the card game **Sets**.

## How it Works (for Sets)

The Sets bot operates through the following pipeline:

1.  **Screen Capture:** It takes a screenshot of the game window.
2.  **Card Extraction:** It identifies and extracts the individual cards from the screenshot.
3.  **Card Classification:** Each card is classified based on its four properties: color, shape, number, and filling.
4.  **Solving:** The bot's logic determines a valid "set" from the classified cards.
5.  **Execution:** It simulates mouse clicks to select the cards that form the set, thereby playing the game.

## Future Goals

The project aims to expand its capabilities to include bots for other games.

## Usage

To run the Sets bot, execute the following command:

```bash
python main.py
```
