"""
    Predict microbiome compositions with regression.

    - Authors:
        - Mohamed Achraf Bouaoune (bouaoune.mohamed_achraf@courrier.uqam.ca)
        - Vincent Therrien (therrien.vincent.2@courrier.uqam.ca)
    - Affiliation: Département d'informatique, UQÀM
    - File creation date: May 2024
    - License: MIT
"""

from typing import Callable

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

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


def r2_score_by_row(
        observed: np.ndarray | pd.DataFrame,
        prediction: np.ndarray | pd.DataFrame
    ) -> list[float]:
    """Calculate the coefficient of determination for row.

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
        observed_row = observed[col_index, :]
        predicted_row = prediction[col_index, :]
        scores.append(r2_score(observed_row, predicted_row))

    return scores


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
    if type(y_pred) != pd.DataFrame:
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


def evaluate_models(
        models: dict[Callable],
        x_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.DataFrame,
        x_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.DataFrame,
        ignore: list[str] | str = None,
        threshold: float = 0.0
    ) -> dict:
    """Evaluate a collection of models at one taxon.

    Args:
        models: A dictionary formatted as `{"model name": Model}`. Each
            `Model` object must support the functions `fit` and
            `predict`.
        x_train: Training input data.
        y_train: Training output data.
        x_test: Test input data.
        y_test: Test output data.
        ignore: Name or list of names of the columns to ignore. Can
            be provided only if x and y are `pd.DataFrame` objects.
        threshold: Values whose absolute value are inferior to this
            parameter are ignored. Can be provided only if x and y are
            `pd.DataFrame` objects.

    Returns: R square metric of each model formatted as
        `{"model name": <r square score>}`.
    """
    log.info(f"Beginning the evaluation of {len(models)} models.")
    results = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        r = evaluate(model, x_test, y_test, ignore, threshold)
        results[model_name] = np.mean(r)
        log.trace(f"Evaluated the model `{model_name}`.")
    return results


def train_evaluate_models_multiple_taxa(
        models: dict[Callable],
        x: dict,
        y: dict,
        ignore: list[str] | str = None,
        threshold: float = 0.0,
        k_fold: int = 1
    ) -> dict:
    """Train and evaluate a collection of models at multiple taxa.

    Args:
        models: A dictionary formatted as `{"model name": Model}`. Each
            `Model` object must support the functions `fit` and
            `predict`.
        x: Input data. Must be formatted as
            `{<taxon name>: `dataFrame`}`.
        y: Output data. Must be formatted as
            `{<taxon name>: `dataFrame`}`.
        ignore: Name or list of names of the columns to ignore. Can
            be provided only if x and y are `pd.DataFrame` objects.
        threshold: Values whose absolute value are inferior to this
            parameter are ignored. Can be provided only if x and y are
            `pd.DataFrame` objects.
        k_fold: Number of k-fold splits to use.

    Returns: R square metric of each model formatted as
        `{"model name": <r square score>}`.
    """
    log.info(f"Evaluating {len(models)} models with {k_fold} splits.")
    results = {}
    for level in TAXONOMIC_LEVELS:
        log.info(f"Level: {level}")
        if not level in x:
            continue
        kf = KFold(n_splits=k_fold, shuffle=True)
        results[level] = {}
        for model_name in models:
            results[level][model_name] = []
        for i, (train_index, test_index) in enumerate(kf.split(x[level])):
            log.info(f"K-fold split: {i}")
            if type(x[level]) == pd.DataFrame:
                x_train = x[level].loc[train_index]
                x_test = x[level].loc[test_index]
            else:
                x_train = x[level][train_index]
                x_test = x[level][test_index]
            if type(y[level]) == pd.DataFrame:
                y_train = y[level].loc[train_index]
                y_test = y[level].loc[test_index]
            else:
                y_train = y[level][train_index]
                y_test = y[level][test_index]
            k_fold_result = evaluate_models(
                models, x_train, y_train, x_test, y_test,
                ignore=ignore, threshold=threshold
            )
            for model_name, r in k_fold_result.items():
                results[level][model_name].append(r)
        for model_name, rs in results[level].items():
            results[level][model_name] = np.mean(rs)
    return results


def evaluate_models_multiple_taxa(
        models: dict[Callable],
        x: dict,
        y: dict,
        ignore: list[str] | str = None,
        threshold: float = 0.0,
    ) -> dict:
    """Evaluate a collection of models at multiple taxa.

    Args:
        models: A dictionary formatted as `{"model name": Model}`. Each
            `Model` object must support the function `predict`.
        x: Input data. Must be formatted as
            `{<taxon name>: `dataFrame`}`.
        y: Output data. Must be formatted as
            `{<taxon name>: `dataFrame`}`.
        ignore: Name or list of names of the columns to ignore. Can
            be provided only if x and y are `pd.DataFrame` objects.
        threshold: Values whose absolute value are inferior to this
            parameter are ignored. Can be provided only if x and y are
            `pd.DataFrame` objects.

    Returns: R square metric of each model formatted as
        `{"model name": <r square score>}`.
    """
    log.info(f"Evaluating {len(models)} models.")
    results = {}
    for level in TAXONOMIC_LEVELS:
        log.info(f"Level: {level}")
        if not level in y:
            continue
        results[level] = {}
        for model_name, model in models.items():
            r = evaluate(model, x[level], y[level], ignore, threshold)
            results[level][model_name] = np.mean(r)
    return results
