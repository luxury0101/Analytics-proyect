"""Contratos y reportes de calidad para los datos de Olist."""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


SCHEMA_ORDERS = DataFrameSchema(
    columns={
        "order_id": Column(
            str,
            Check.str_length(32, 32),
            nullable=False,
            unique=True,
        ),
        "order_status": Column(
            str,
            Check.isin(
                [
                    "delivered",
                    "shipped",
                    "canceled",
                    "unavailable",
                    "invoiced",
                    "processing",
                    "created",
                    "approved",
                ]
            ),
        ),
        "order_purchase_timestamp": Column(
            pa.DateTime,
            nullable=False,
        ),
    },
    coerce=True,
)


SCHEMA_ITEMS = DataFrameSchema(
    columns={
        "order_id": Column(
            str,
            nullable=False,
        ),
        "order_item_id": Column(
            int,
            Check.greater_than_or_equal_to(1),
        ),
        "product_id": Column(
            str,
            nullable=False,
        ),
        "seller_id": Column(
            str,
            nullable=False,
        ),
        "price": Column(
            float,
            [
                Check.greater_than(0),
                Check.less_than_or_equal_to(7_000),
            ],
            nullable=False,
        ),
        "freight_value": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
    },
    coerce=True,
)


def validar_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    nombre: str,
) -> list[str]:
    """Valida un DataFrame y retorna los errores encontrados.

    Args:
        df: DataFrame que se validará.
        schema: Contrato Pandera usado en la validación.
        nombre: Nombre descriptivo del DataFrame.

    Returns:
        Lista vacía si pasa o una lista con los errores.
    """
    try:
        schema.validate(df)

    except pa.errors.SchemaError as error:
        return [f"{nombre}: {error}"]

    return []


def generar_reporte_calidad(
    df_entrada: pd.DataFrame,
    df_salida: pd.DataFrame,
    errores: list[str],
    nombre: str = "dataset",
) -> dict:
    """Genera un reporte de calidad con trazabilidad de filas.

    Args:
        df_entrada: DataFrame antes del filtrado.
        df_salida: DataFrame después del filtrado.
        errores: Errores detectados.
        nombre: Nombre descriptivo del dataset.

    Returns:
        Diccionario con cantidades, tasa de rechazo y errores.
    """
    filas_entrada = len(df_entrada)
    filas_salida = len(df_salida)
    filas_descartadas = filas_entrada - filas_salida

    porcentaje = filas_descartadas / filas_entrada * 100 if filas_entrada > 0 else 0.0

    tasa_rechazo = f"{porcentaje:.2f}%"

    reporte = {
        "nombre": nombre,
        "filas_entrada": filas_entrada,
        "filas_salida": filas_salida,
        "filas_descartadas": filas_descartadas,
        "tasa_rechazo": tasa_rechazo,
        "errores": errores,
    }

    print(
        f"[CALIDAD] {nombre}: {filas_entrada} entrada → "
        f"{filas_salida} salida "
        f"({filas_descartadas} descartados, {tasa_rechazo})"
    )

    return reporte
