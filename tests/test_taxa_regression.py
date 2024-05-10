"""
    Test regression performed on taxa.

    - Author: Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

import pandas as pd

from micropyome.taxa import regression


def test_r2_score():
    """Test the r2 score metric."""
    observed = pd.DataFrame({"A": [1, 2, 3], "B": [1, 2, 3]})
    predicted = pd.DataFrame({"A": [1, 2, 3], "B": [1, 1, 1]})
    r2 = regression.r2_score_by_column(observed, predicted)
    assert r2[0] == 1.0, "Incorrect value for a prefect regression."
    expected = 1 - (5 / 2)
    assert r2[1] == expected
