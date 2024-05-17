"""
    Test module for the databases package.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import pandas as pd

from micropyome.datasets import neon, normalize


def test_normalize():
    """Test data frame normalization."""
    not_normalized = pd.DataFrame({"A": [10, 5, 0], "B": [-5, 0, 5]})
    expected = pd.DataFrame({"A": [1, 0.5, 0], "B": [0, 0.5, 1]})
    normalized = normalize(not_normalized)
    assert normalized.equals(expected)
