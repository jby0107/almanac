"""Constants for the Set game card attributes."""

from typing import Tuple, Literal

# Type alias for a card combination
Combi = Tuple[Literal[0, 1, 2], Literal[0, 1, 2], Literal[0, 1, 2], Literal[0, 1, 2]]

# Mapping from integer codes to string names (for display/logging)
COLOR_TO_NAME = {
    0: "green",
    1: "red",
    2: "purple",
}

SHAPE_TO_NAME = {
    0: "flower",
    1: "star",
    2: "sandglass",
}

NUMBER_TO_NAME = {
    0: 1,
    1: 2,
    2: 3,
}

FILL_TO_NAME = {
    0: "solid",
    1: "striped",
    2: "outline",
}

# Mapping from string names to integer codes (for classification)
COLOR_FROM_NAME = {
    "green": 0,
    "red": 1,
    "purple": 2,
}

SHAPE_FROM_NAME = {
    "flower": 0,
    "star": 1,
    "sandglass": 2,
}

NUMBER_FROM_NAME = {
    1: 0,
    2: 1,
    3: 2,
}

FILL_FROM_NAME = {
    "solid": 0,
    "striped": 1,
    "outline": 2,
}

