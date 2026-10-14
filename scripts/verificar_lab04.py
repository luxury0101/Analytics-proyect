"""Verificación del Lab 04 con los CSV reales de Olist, fuera de pytest.

Uso: uv run python scripts/verificar_lab04.py --output docs/lab04_resultados.json
"""

import argparse
import json
import platform
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

from analytics import evaluate, extract, train, transform

FEATURES_REGRESION = [
    "n_items",
    "precio_total",
    "flete_total",
    "customer_state_freq",
    "dia_semana_compra",
    "mes_compra",
    "ticket_promedio_historico",
]
FEATURES_CLASIFICACION = [
    "n_items",
    "precio_total",
    "flete_total",
    "customer_state_freq",
    "tiempo_entrega_dias",
    "entregado_tarde",
    "ticket_promedio_historico",
]
UMBRAL_REVIEW_NEGATIVA = 2  # review_score <= 2 se considera negativa


def construir_dataset(data_dir: Path) -> pd.DataFrame:
    """Construye el DataFrame del Lab 03 con review_negativa por pedido.

    Args:
        data_dir: Directorio con los CSV de Olist.

    Returns:
        DataFrame con una fila por pedido, features del Lab 03 y las columnas
        review_score y review_negativa (NaN si el pedido no tiene reseña).
    """
    tablas = extract.cargar_olist(str(data_dir))
    # Un pedido puede tener varias reseñas: se conserva la más reciente.
    reseñas = (
        tablas["reviews"]
        .sort_values(["review_creation_date", "review_answer_timestamp"], kind="stable")
        .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]]
    )
    base = extract.construir_dataset_base(tablas)
    n_base = len(base)
    df = base.merge(reseñas, on="order_id", how="left")
    assert len(df) == n_base, "El join con reseñas multiplicó los pedidos."
    # customer_id es único por pedido en Olist (historial 100 % nulo); se usa
    # customer_unique_id para reconocer compras de la misma persona.
    df = (
        df.pipe(transform.agregar_features_fecha)
        .pipe(transform.agregar_encoding_estado)
        .pipe(transform.agregar_ticket_historico, columna_cliente="customer_unique_id")
    )
    df["review_negativa"] = (df["review_score"] <= UMBRAL_REVIEW_NEGATIVA).where(
        df["review_score"].notna()
    )
    return df


def generar_resultados(data_dir: Path, modelos_dir: Path) -> dict:
    """Entrena, evalúa y serializa los pipelines; retorna las evidencias.

    Args:
        data_dir: Directorio con los CSV de Olist.
        modelos_dir: Directorio donde se guardan los pipelines entrenados.

    Returns:
        Métricas de regresión (validación y test) y de clasificación, el
        baseline y la comparación con y sin class_weight="balanced".
    """
    df = construir_dataset(data_dir)

    # ── regresión: tiempo_entrega_dias ───────────────────────────────
    datos = df[df["tiempo_entrega_dias"].notna()]
    X, y = datos[FEATURES_REGRESION], datos["tiempo_entrega_dias"]
    X_resto, X_test, y_resto, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_resto, y_resto, test_size=0.25, random_state=42
    )
    pipeline_reg = train.entrenar_pipeline_regresion(X_train, y_train)
    reg_val = evaluate.evaluar_regresion(y_val, pipeline_reg.predict(X_val))
    reg_test = evaluate.evaluar_regresion(y_test, pipeline_reg.predict(X_test))
    # Referencia: predecir siempre la media de entrenamiento.
    reg_baseline_test = evaluate.evaluar_regresion(
        y_test, [float(y_train.mean())] * len(y_test)
    )
    train.guardar_pipeline(
        pipeline_reg, str(modelos_dir / "pipeline_tiempo_entrega.joblib")
    )

    # ── clasificación: review_negativa ───────────────────────────────
    datos = df[df["review_negativa"].notna()]
    Xc = datos[FEATURES_CLASIFICACION].astype(float)
    yc = datos["review_negativa"].astype(bool)
    Xc_resto, Xc_test, yc_resto, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=42, stratify=yc
    )
    Xc_train, Xc_val, yc_train, yc_val = train_test_split(
        Xc_resto, yc_resto, test_size=0.25, random_state=42, stratify=yc_resto
    )
    pipeline_clf = train.entrenar_pipeline_clasificacion(Xc_train, yc_train)
    train.guardar_pipeline(
        pipeline_clf, str(modelos_dir / "pipeline_review_negativa.joblib")
    )

    def evaluar_clf(pipeline, X_eval, y_eval) -> dict:
        pred = pipeline.predict(X_eval)
        proba = pipeline.predict_proba(X_eval)[:, 1]
        metricas = evaluate.evaluar_clasificacion(y_eval, pred, proba)
        metricas["recall_negativa"] = float(recall_score(y_eval, pred))
        return metricas

    clf_val = evaluar_clf(pipeline_clf, Xc_val, yc_val)
    clf_test = evaluar_clf(pipeline_clf, Xc_test, yc_test)
    baseline = evaluate.comparar_con_baseline(
        yc_train, yc_val, pipeline_clf.predict(Xc_val)
    )

    # 9.3: misma configuración, pero sin class_weight="balanced".
    pipeline_sin_peso = train.entrenar_pipeline_clasificacion(Xc_train, yc_train)
    pipeline_sin_peso.set_params(modelo__class_weight=None).fit(Xc_train, yc_train)
    clf_val_sin_peso = evaluar_clf(pipeline_sin_peso, Xc_val, yc_val)

    return {
        "versiones": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "definicion_review_negativa": f"review_score <= {UMBRAL_REVIEW_NEGATIVA}",
        "regresion": {
            "n_filas": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "validacion": reg_val,
            "test": reg_test,
            "baseline_media_test": reg_baseline_test,
        },
        "clasificacion": {
            "n_filas": {
                "train": len(Xc_train),
                "val": len(Xc_val),
                "test": len(Xc_test),
            },
            "prevalencia_review_negativa_train": float(yc_train.mean()),
            "baseline_validacion": baseline,
            "validacion_balanced": clf_val,
            "test_balanced": clf_test,
            "validacion_sin_class_weight": clf_val_sin_peso,
        },
    }


def main() -> None:
    """Ejecuta la verificación y escribe el JSON de resultados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/lab04_resultados.json")
    )
    args = parser.parse_args()
    resultados = generar_resultados(args.data_dir, args.models_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
