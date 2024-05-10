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


def normalize(df: pd.DataFrame) -> None:
    """Modify the provided data frame to set negative values to 0
    and scale rows so that the sum of their values is 1.

    Args:
        df: Data frame to normalize. Modified in place.
    """
    df[df < 0] = 0
    row_sums = df.sum(axis=1)
    df = df.div(row_sums, axis=0)
