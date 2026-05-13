"""
Small utility functions for experiments.
"""

import os
import random

import numpy as np


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def project_root():
    """
    Return the current working directory.

    The experiment scripts should usually be run from the repo root.
    """
    return os.getcwd()


def save_dataframe(df, path, index=False):
    """
    Save a pandas DataFrame after ensuring the parent directory exists.
    """
    parent = os.path.dirname(path)

    if parent:
        ensure_dir(parent)

    df.to_csv(path, index=index)


def print_section(title):
    """
    Print a simple section header.
    """
    line = "=" * len(title)
    print("\n" + line)
    print(title)
    print(line)