"""\
Global Settings
===============

Author: Akshay Mestry, Aymaan Gillani and Joseph Giwa
Created on: Sunday, November 05 2023
Last updated on: Monday, November 06 2023

Global project settings.
"""

import os.path as p

# Relative paths used throughout the project. The ``./`` path or the
# current directory path is the base path for all the implemented code.
# Rest paths viz. the ``../docs`` and ``../data`` are mere relative
# paths for the rest of project.
PROJECT_PATHS: dict[str, str] = {
    ".": p.abspath("enhancement"),
    "../data": p.abspath("./data/"),
    "../docs": p.abspath("./docs"),
    "../logs": p.abspath("./logs/"),
    "../models": p.abspath("./models/"),
    "../results": p.abspath("./results/"),
    "../plots": p.abspath("./plots/"),
    "../bright": p.abspath("./data/bright"),
    "../dark": p.abspath("./data/dark"),
}

# Model running mode by default.
MODEL_MODE: str = "train"

# The ISO8601 standard is used for logging the timestamp of the log
# messages. The logged time is the local time of the user and not UTC.
# For UTC, the user must patch the logger object to use the ``gmtime``.
ISO8601: str = "%Y-%m-%dT%H:%M:%SZ"

# Log format for logging messages on screen or to a file.
LOG_FORMAT: str = "%(asctime)s %(levelname)8s: %(message)s"

# Size of the convolving kernel.
KERNEL_SIZE: int = 3

# Number of channels in the input image.
INPUT_CHANNELS: int = 4

# Number of channels produced by the convolution
OUTPUT_CHANNELS: int = 64
