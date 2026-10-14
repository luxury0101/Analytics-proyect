"""Métricas de regresión, clasificación y comparación contra un baseline."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def evaluar_regresion(y_true, y_pred) -> dict:
    """Calcula las métricas estándar de un modelo de regresión.

    Args:
        y_true: valores reales del target.
        y_pred: valores predichos por el modelo.

    Returns:
        Dict con claves 'rmse', 'mae', 'r2' (floats).
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluar_clasificacion(y_true, y_pred, y_proba) -> dict:
    """Calcula las métricas estándar de un modelo de clasificación binaria.

    Args:
        y_true: valores reales del target (booleano o 0/1).
        y_pred: predicciones binarias del modelo.
        y_proba: probabilidades predichas de la clase positiva.

    Returns:
        Dict con claves 'accuracy', 'f1', 'roc_auc' (floats) y
        'matriz_confusion' (lista de listas 2x2: [[tn, fp], [fn, tp]]).
    """
    # labels fija el orden [negativa, positiva] aunque falte una clase.
    matriz = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "matriz_confusion": matriz.tolist(),
    }


def comparar_con_baseline(y_train, y_eval, y_pred_modelo) -> dict:
    """Compara el accuracy del modelo contra un baseline trivial.

    El baseline predice, para TODAS las filas de y_eval, la clase
    mayoritaria observada en y_train (equivalente a
    DummyClassifier(strategy="most_frequent")).

    Args:
        y_train: target de entrenamiento, usado para determinar la
            clase mayoritaria.
        y_eval: target real del conjunto de evaluación.
        y_pred_modelo: predicciones del modelo entrenado sobre el
            mismo conjunto que y_eval.

    Returns:
        Dict con claves 'accuracy_baseline', 'accuracy_modelo', 'mejora'
        (accuracy_modelo - accuracy_baseline).
    """
    clase_mayoritaria = pd.Series(y_train).mode().iloc[0]
    y_pred_baseline = np.full(len(y_eval), clase_mayoritaria)

    accuracy_baseline = float(accuracy_score(y_eval, y_pred_baseline))
    accuracy_modelo = float(accuracy_score(y_eval, y_pred_modelo))
    return {
        "accuracy_baseline": accuracy_baseline,
        "accuracy_modelo": accuracy_modelo,
        "mejora": accuracy_modelo - accuracy_baseline,
    }
