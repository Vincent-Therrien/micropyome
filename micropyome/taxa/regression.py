"""
    Predict microbiome compositions with regression.

    - Authors:
        - Mohamed Achraf Bouaoune (bouaoune.mohamed_achraf@courrier.uqam.ca)
        - Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

from datetime import datetime
from typing import Callable

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from micropyome.utils import log
from micropyome.datasets import normalize


TAXONOMIC_LEVELS = [
    "fg",  # Functional group.
    "phylum",
    "class",
    "order",
    "family",
    "genus"
]


def r2_score_by_column(
        observed: np.ndarray | pd.DataFrame,
        prediction: np.ndarray | pd.DataFrame
    ) -> list[float]:
    """Calculate the coefficient of determination for each column.

    Args:
        observed: Ground truth.
        prediction: Predicted values.

    Returns: The list of R square scores given in the same order as
        the column of the arguments.
    """
    # Validate the arguments.
    if type(observed) == pd.DataFrame:
        observed = observed.to_numpy()
    if type(prediction) == pd.DataFrame:
        prediction = prediction.to_numpy()
    if observed.shape != prediction.shape:
        log.error(
            "Cannot compute the R square score of inhomogeneous data "
            + f"(shape 1: `{observed.shape}`, shape 2: `{prediction.shape}`)."
        )
        raise ValueError
    if len(observed.shape) != 2:
        log.error(
            "The R square value is to be computed on a 2D array "
            + f"(got size `{observed.shape}`)."
        )
        raise ValueError

    # Compute the score for each column.
    scores = []
    for col_index in range(observed.shape[1]):
        observed_column = observed[:, col_index]
        predicted_column = prediction[:, col_index]
        scores.append(r2_score(observed_column, predicted_column))

    return scores


def evaluate(
        model: Callable,
        x: any,
        y: any,
        ignore: list[str] | str = None,
        threshold: float = 0.0
    ) -> list[float]:
    """Evaluate the ability of a trained model to predict taxa.

    Args:
        model: An already trained model. Must support the function
            `predict(x)`.
        x: Input data.
        y: Output data.
        threshold: Values whose absolute value are inferior to this
            parameter are ignored. Can be provided only if x and y are
            `pd.DataFrame` objects.
        ignore: Name or list of names of the columns to ignore. Can
            be provided only if x and y are `pd.DataFrame` objects.

    Returns (float): The list of R square values by category.
    """
    normalize(x)
    y_pred = model.predict(x)
    y_pred = pd.DataFrame(y_pred, columns=y.columns)

    if ignore:
        if type(ignore) == str:
            ignore = [ignore]
        y = y.drop(columns=ignore)
        y_pred = y_pred.drop(columns=ignore)

    if threshold:
        for column in y.columns:
            if y[column].mean() < threshold:
                y = y.drop(columns=[column])
                y_pred = y_pred.drop(columns=[column])

    normalize(y)
    normalize(y_pred)

    return r2_score_by_column(y, y_pred)
