"""
    Download microbiome-related datasets.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import pandas as pd

__all__ = [
    "bacteria",
    "neon"
]


def normalize_categories(df: pd.DataFrame) -> None:
    """Modify the provided data frame to set negative values to 0
    and scale rows so that the sum of their values is 1.

    Args:
        df: Data frame to normalize. Modified in place.
    """
    if type(df) == pd.DataFrame:
        df[df < 0] = 0
        row_sums = df.sum(axis=1)
        df = df.div(row_sums, axis=0)
    else:
        df[df < 0] = 0
        row_sums = df.sum(axis=1)
        df = (df.T / df.sum(axis=1)).T


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Modify the provided data frame to normalize all values of the
    respective columns between 0 and 1.

    Args:
        df: Data frame to normalize.

    Returns: Normalized data frame.
    """
    return (df - df.min()) / (df.max() - df.min())
