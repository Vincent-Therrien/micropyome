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


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Modify the provided data frame to set negative values to 0
    and scale rows so that the sum of their values is 1.

    Args:
        df: Data frame to normalize.

    Returns: Row-normalized data frame.
    """
    result = df.copy()
    if type(df) == pd.DataFrame:
        result[result < 0] = 0
        row_sums = result.sum(axis=1)
        result = result.div(row_sums, axis=0)
    else:
        result[result < 0] = 0
        row_sums = result.sum(axis=1)
        result = (result.T / result.sum(axis=1)).T
    return result


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Modify the provided data frame to normalize all values of the
    respective columns between 0 and 1.

    Args:
        df: Data frame to normalize.

    Returns: Column-normalized data frame.
    """
    return (df - df.min()) / (df.max() - df.min())
