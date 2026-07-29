"""Tests del módulo quality.py — Lab 02."""

import pandas as pd
import pytest

from analytics.quality import (
    SCHEMA_ITEMS,
    SCHEMA_ORDERS,
    generar_reporte_calidad,
    validar_schema,
)


@pytest.fixture
def df_orders_valido():
    return pd.DataFrame(
        {
            "order_id": ["a" * 32, "b" * 32],
            "order_status": ["delivered", "shipped"],
            "order_purchase_timestamp": pd.to_datetime(["2021-01-01", "2021-02-15"]),
        }
    )


@pytest.fixture
def df_orders_status_invalido():
    return pd.DataFrame(
        {
            "order_id": ["a" * 32],
            "order_status": ["ENTREGADO"],
            "order_purchase_timestamp": pd.to_datetime(["2021-01-01"]),
        }
    )


@pytest.fixture
def df_items_precio_cero():
    return pd.DataFrame(
        {
            "order_id": ["a" * 32],
            "order_item_id": [1],
            "product_id": ["p" * 32],
            "seller_id": ["s" * 32],
            "price": [0.0],
            "freight_value": [10.0],
        }
    )


def test_schema_orders_valido_no_retorna_errores(
    df_orders_valido,
):
    errores = validar_schema(
        df_orders_valido,
        SCHEMA_ORDERS,
        "orders",
    )

    assert errores == []


def test_schema_orders_rechaza_status_invalido(
    df_orders_status_invalido,
):
    errores = validar_schema(
        df_orders_status_invalido,
        SCHEMA_ORDERS,
        "orders",
    )

    assert len(errores) > 0


def test_schema_items_rechaza_precio_cero(
    df_items_precio_cero,
):
    errores = validar_schema(
        df_items_precio_cero,
        SCHEMA_ITEMS,
        "items",
    )

    assert len(errores) > 0


def test_validar_schema_retorna_lista(
    df_orders_valido,
):
    resultado = validar_schema(
        df_orders_valido,
        SCHEMA_ORDERS,
        "orders",
    )

    assert isinstance(resultado, list)


def test_reporte_calidad_calcula_filas_correctamente():
    df_entrada = pd.DataFrame({"id": range(100)})
    df_salida = pd.DataFrame({"id": range(90)})

    reporte = generar_reporte_calidad(
        df_entrada,
        df_salida,
        errores=[],
        nombre="test",
    )

    assert reporte["filas_entrada"] == 100
    assert reporte["filas_salida"] == 90
    assert reporte["filas_descartadas"] == 10


def test_reporte_calidad_tasa_rechazo_formato():
    df_entrada = pd.DataFrame({"id": range(100)})
    df_salida = pd.DataFrame({"id": range(90)})

    reporte = generar_reporte_calidad(
        df_entrada,
        df_salida,
        errores=[],
        nombre="test",
    )

    assert reporte["tasa_rechazo"] == "10.00%"


def test_reporte_calidad_sin_descartados():
    dataframe = pd.DataFrame({"id": range(50)})

    reporte = generar_reporte_calidad(
        dataframe,
        dataframe,
        errores=[],
        nombre="test",
    )

    assert reporte["filas_descartadas"] == 0
    assert reporte["tasa_rechazo"] == "0.00%"


def test_reporte_calidad_incluye_errores():
    dataframe = pd.DataFrame({"id": range(10)})

    errores = [
        "order_status inválido en fila 3",
        "price negativo en fila 7",
    ]

    reporte = generar_reporte_calidad(
        dataframe,
        dataframe,
        errores=errores,
        nombre="test",
    )

    assert reporte["errores"] == errores
