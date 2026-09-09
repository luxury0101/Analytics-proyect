"""Verificación reproducible del Lab 03 con los CSV reales, fuera de pytest.

Uso: uv run python scripts/verificar_lab03.py --output docs/lab03_resultados.json
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import pandas as pd

from analytics import eda, extract, transform


def generar_resultados(data_dir: Path) -> dict:
    """Ejecuta la integración y retorna evidencias agregadas, sin datos personales.

    Args:
        data_dir: Directorio con las cinco tablas usadas por cargar_olist.

    Returns:
        Formas, historial, encoding, perfiles, correlaciones y huellas SHA-256
        de los CSV. No modifica los archivos de entrada.
    """
    tablas = extract.cargar_olist(str(data_dir))
    base = extract.construir_dataset_base(tablas)
    original = base.copy(deep=True)
    df = (
        base.pipe(transform.agregar_features_fecha)
        .pipe(transform.agregar_encoding_estado)
        .pipe(transform.agregar_ticket_historico)
    )
    pd.testing.assert_frame_equal(base, original)
    assert len(df) == len(tablas["orders"])
    assert df["order_id"].is_unique
    assert len(df.columns) == len(base.columns) + 8

    # Proyección de las 15 columnas del PDF, sin perder compatibilidad en extract.
    legado = ["ticket_total", "total_pago", "n_cuotas"]
    base_pdf = base.drop(columns=legado)
    final_pdf = df.drop(columns=legado)
    assert base_pdf.shape[1] == 15 and final_pdf.shape[1] == 23

    historial = {}
    for columna_cliente in ["customer_id", "customer_unique_id"]:
        resultado = transform.agregar_ticket_historico(
            base, columna_cliente=columna_cliente
        )
        conteos = base.groupby(columna_cliente).size()
        nulos = resultado["ticket_promedio_historico"].isna()
        # El primer instante de cada cliente no puede usar compras simultáneas.
        primera_fecha = resultado.groupby(columna_cliente)[
            "order_purchase_timestamp"
        ].transform("min")
        es_primer_instante = resultado["order_purchase_timestamp"].eq(primera_fecha)
        historial[columna_cliente] = {
            "n_clientes": int(conteos.size),
            "n_clientes_un_pedido": int(conteos.eq(1).sum()),
            "n_clientes_recurrentes": int(conteos.gt(1).sum()),
            "pct_clientes_recurrentes": float(conteos.gt(1).mean() * 100),
            "n_filas_sin_historial": int(nulos.sum()),
            "pct_filas_sin_historial": float(nulos.mean() * 100),
            "n_filas_con_historial": int((~nulos).sum()),
            "n_filas_primer_instante": int(es_primer_instante.sum()),
            "n_filas_nulas_fuera_del_primer_instante": int(
                (nulos & ~es_primer_instante).sum()
            ),
        }

    columnas_corr = ["tiempo_entrega_dias", "n_items", "precio_total"]
    correlacion = eda.matriz_correlacion(df, columnas_corr)
    conteos_pares = {
        columna: int(df[["tiempo_entrega_dias", columna]].dropna().shape[0])
        for columna in columnas_corr[1:]
    }
    distribucion = eda.distribucion_categorica(df, "customer_state")
    # to_json normaliza escalares de pandas/numpy y nulos a JSON estándar.
    perfil = json.loads(
        pd.Series(eda.perfil_columna(df, "tiempo_entrega_dias")).to_json()
    )
    archivos = [
        "olist_orders_dataset.csv",
        "olist_customers_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
    ]
    return {
        "fuente": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        "versiones": {"python": platform.python_version(), "pandas": pd.__version__},
        "sha256_csv": {
            nombre: hashlib.sha256((data_dir / nombre).read_bytes()).hexdigest()
            for nombre in archivos
        },
        "tablas": {nombre: list(tabla.shape) for nombre, tabla in tablas.items()},
        "shape_base": list(base.shape),
        "shape_transformado": list(df.shape),
        "shape_base_enunciado": list(base_pdf.shape),
        "shape_transformado_enunciado": list(final_pdf.shape),
        "columnas_nuevas": [columna for columna in df if columna not in base],
        "historial": historial,
        "encoding_estado": {
            "n_estados": int(df["customer_state"].nunique()),
            "n_dummies_sin_drop_first": int(
                pd.get_dummies(df["customer_state"]).shape[1]
            ),
            "n_dummies_con_drop_first": int(
                pd.get_dummies(df["customer_state"], drop_first=True).shape[1]
            ),
            "n_columnas_dataframe_sin_drop_first": int(
                pd.get_dummies(df, columns=["customer_state"]).shape[1]
            ),
            "n_columnas_dataframe_con_drop_first": int(
                pd.get_dummies(df, columns=["customer_state"], drop_first=True).shape[1]
            ),
        },
        "perfil_tiempo_entrega": perfil,
        "descripcion_tiempo_entrega": json.loads(
            df["tiempo_entrega_dias"].describe().to_json()
        ),
        "correlacion": correlacion.to_dict(),
        "n_observaciones_por_correlacion": conteos_pares,
        "distribucion_estados": distribucion.to_dict(orient="index"),
    }


def main() -> None:
    """Permite regenerar el reporte desde la raíz del repositorio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/lab03_resultados.json")
    )
    args = parser.parse_args()
    try:
        resultados = generar_resultados(args.data_dir)
    except FileNotFoundError as error:
        parser.exit(
            1,
            f"Falta un CSV de Olist: {error.filename}. "
            "Descarga el dataset indicado en README.md y extráelo en data/raw/.\n",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Tabla del proyecto: {resultados['shape_base']} -> {resultados['shape_transformado']}"
    )
    print(
        f"Proyección del PDF: {resultados['shape_base_enunciado']} -> {resultados['shape_transformado_enunciado']}"
    )
    for columna, resumen in resultados["historial"].items():
        print(f"Sin historial ({columna}): {resumen['pct_filas_sin_historial']:.6f}%")
    print(f"Resultados guardados en {args.output}")


if __name__ == "__main__":
    main()
