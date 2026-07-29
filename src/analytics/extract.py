"""Módulo de ingesta de datos hacia el pipeline de analítica."""

from pathlib import Path

import pandas as pd


def cargar_csv(ruta: str) -> pd.DataFrame:
    """Carga un archivo CSV y verifica que tenga filas de datos.

    Args:
        ruta: Ruta del archivo CSV que se desea cargar.

    Returns:
        DataFrame con los datos contenidos en el archivo.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no contiene filas de datos.
    """
    dataframe = pd.read_csv(ruta)

    if dataframe.empty:
        raise ValueError("El archivo CSV no contiene filas de datos.")

    return dataframe


def cargar_olist(data_dir: str) -> dict[str, pd.DataFrame]:
    """Carga las cinco tablas principales de Olist desde un directorio.

    Args:
        data_dir: Ruta del directorio que contiene los archivos CSV de Olist.

    Returns:
        Diccionario con las tablas orders, items, customers, payments y reviews.

    Raises:
        FileNotFoundError: Si alguno de los cinco archivos no existe.
    """
    directorio = Path(data_dir)

    archivos = {
        "orders": (
            "olist_orders_dataset.csv",
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "items": ("olist_order_items_dataset.csv", None),
        "customers": ("olist_customers_dataset.csv", None),
        "payments": ("olist_order_payments_dataset.csv", None),
        "reviews": (
            "olist_order_reviews_dataset.csv",
            [
                "review_creation_date",
                "review_answer_timestamp",
            ],
        ),
    }

    tablas = {}

    for nombre, (archivo, columnas_fecha) in archivos.items():
        dataframe = pd.read_csv(
            directorio / archivo,
            parse_dates=columnas_fecha,
        )

        tablas[nombre] = dataframe

        print(f"{nombre}: {dataframe.shape[0]:,} filas × {dataframe.shape[1]} columnas")

    return tablas


def join_verificado(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    on: str | list[str],
    how: str = "left",
    nombre: str = "join",
) -> pd.DataFrame:
    """Realiza un merge y verifica que no se multipliquen las filas.

    Args:
        df_left: DataFrame izquierdo que define las filas esperadas.
        df_right: DataFrame derecho.
        on: Columna o columnas utilizadas como clave.
        how: Tipo de join.
        nombre: Nombre descriptivo del join.

    Returns:
        DataFrame resultante del merge.

    Raises:
        AssertionError: Si el resultado tiene más filas que df_left.
    """
    resultado = df_left.merge(
        df_right,
        on=on,
        how=how,
    )

    assert len(resultado) <= len(df_left), (
        f"{nombre}: el join aumentó las filas de {len(df_left)} a {len(resultado)}"
    )

    return resultado


def construir_dataset_base(
    tablas: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combina las tablas de Olist en un DataFrame por pedido.

    Args:
        tablas: Diccionario retornado por cargar_olist.

    Returns:
        DataFrame analítico con una fila por pedido.
    """
    pagos_agrupados = (
        tablas["payments"]
        .groupby("order_id", as_index=False)
        .agg(
            total_pago=("payment_value", "sum"),
            n_cuotas=("payment_installments", "max"),
        )
    )

    items_agrupados = (
        tablas["items"]
        .groupby("order_id", as_index=False)
        .agg(
            n_items=("order_item_id", "count"),
            ticket_total=("price", "sum"),
        )
    )

    dataset = join_verificado(
        tablas["orders"],
        tablas["customers"],
        on="customer_id",
        nombre="orders_customers",
    )

    dataset = join_verificado(
        dataset,
        pagos_agrupados,
        on="order_id",
        nombre="orders_payments",
    )

    dataset = join_verificado(
        dataset,
        items_agrupados,
        on="order_id",
        nombre="orders_items",
    )

    return dataset
