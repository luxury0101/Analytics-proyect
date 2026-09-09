"""Regresiones de integración, fuga temporal y casos límite del Lab 03."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from analytics import eda, extract, transform


def test_fechas_fracciones_nulos_y_no_mutacion_completa():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": pd.to_datetime(
                ["2021-01-31 12:00", "2021-02-01 08:00", None]
            ),
            "order_delivered_customer_date": pd.to_datetime(
                ["2021-02-01 00:00", None, None]
            ),
            "order_estimated_delivery_date": pd.to_datetime(
                ["2021-02-02", "2021-02-05", None]
            ),
        },
        index=[8, 2, 5],
    )
    original = df.copy(deep=True)
    resultado = transform.agregar_features_fecha(df)
    assert resultado.loc[8, "tiempo_entrega_dias"] == 0.5
    assert resultado.loc[8, "retraso_entrega_dias"] == -1.0
    assert resultado.loc[8, "dia_semana_compra"] == 6
    assert resultado.loc[2, "mes_compra"] == 2
    assert pd.isna(resultado.loc[2, "tiempo_entrega_dias"])
    assert pd.isna(resultado.loc[5, "dia_semana_compra"])
    assert not resultado.loc[2, "entregado_tarde"]
    assert_frame_equal(df, original)


def test_encoding_columna_personalizada_con_nulos_no_muta():
    df = pd.DataFrame({"region": ["norte", None, "sur", "norte"]}, index=[9, 3, 7, 1])
    original = df.copy(deep=True)
    resultado = transform.agregar_encoding_estado(df, columna="region")
    assert resultado["region_freq"].tolist() == pytest.approx(
        [2 / 3, np.nan, 1 / 3, 2 / 3], nan_ok=True
    )
    assert_frame_equal(df, original)


def test_historial_simultaneos_desordenados_indices_duplicados_y_no_mutacion():
    df = pd.DataFrame(
        {
            "cliente": ["x", "y", "x", "x", "x", "y"],
            "fecha": pd.to_datetime(
                [
                    "2021-03-01",
                    "2021-01-15",
                    "2021-01-01",
                    "2021-02-01",
                    "2021-02-01",
                    "2021-04-01",
                ]
            ),
            "valor": [900.0, 50.0, 100.0, 200.0, 400.0, 80.0],
        },
        index=[8, 3, 8, 2, 1, 3],
    )
    original = df.copy(deep=True)
    resultado = transform.agregar_ticket_historico(df, "cliente", "valor", "fecha")
    assert resultado["ticket_promedio_historico"].tolist() == pytest.approx(
        [np.nan, np.nan, 100.0, 100.0, 700 / 3, 50.0], nan_ok=True
    )
    assert resultado.index.tolist() == [8, 3, 2, 1, 8, 3]
    assert_frame_equal(df, original)


def test_cambiar_precio_actual_o_futuro_no_altera_historial_previo():
    df = pd.DataFrame(
        {
            "customer_id": ["x"] * 4,
            "order_purchase_timestamp": pd.date_range("2021-01-01", periods=4),
            "precio_total": [100.0, 200.0, 300.0, 400.0],
        }
    )
    antes = transform.agregar_ticket_historico(df)
    df.loc[2:, "precio_total"] = [99999.0, 88888.0]
    despues = transform.agregar_ticket_historico(df)
    assert_series_equal(
        antes.loc[:2, "ticket_promedio_historico"],
        despues.loc[:2, "ticket_promedio_historico"],
    )


def test_historial_sin_fecha_cliente_o_precio_no_inventa_datos():
    df = pd.DataFrame(
        {
            "customer_id": ["x", "x", "x", None, "x", None],
            "order_purchase_timestamp": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    "2021-03-01",
                    "2021-01-15",
                    None,
                    "2021-04-01",
                ]
            ),
            "precio_total": [np.nan, 100.0, 200.0, 999.0, 10000.0, 888.0],
        }
    )
    resultado = transform.agregar_ticket_historico(df).sort_index()
    assert resultado["ticket_promedio_historico"].tolist() == pytest.approx(
        [np.nan, np.nan, 100.0, np.nan, np.nan, np.nan], nan_ok=True
    )


def test_transformaciones_aceptan_dataframe_vacio():
    df = pd.DataFrame(
        {
            "customer_id": pd.Series(dtype="str"),
            "customer_state": pd.Series(dtype="str"),
            "precio_total": pd.Series(dtype=float),
            "order_purchase_timestamp": pd.Series(dtype="datetime64[ns]"),
            "order_delivered_customer_date": pd.Series(dtype="datetime64[ns]"),
            "order_estimated_delivery_date": pd.Series(dtype="datetime64[ns]"),
        }
    )
    resultado = (
        df.pipe(transform.agregar_features_fecha)
        .pipe(transform.agregar_encoding_estado)
        .pipe(transform.agregar_ticket_historico)
    )
    assert resultado.shape == (0, len(df.columns) + 8)
    assert resultado["ticket_promedio_historico"].dtype == float


@pytest.mark.parametrize(
    "serie",
    [
        pd.Series([], dtype="str"),
        pd.Series([None, None], dtype="str"),
        pd.Series(pd.Categorical([None, None], categories=["a", "b"])),
    ],
)
def test_perfil_y_distribucion_sin_categorias_observadas(serie):
    df = pd.DataFrame({"categoria": serie})
    perfil = eda.perfil_columna(df, "categoria")
    assert perfil["valor_mas_frecuente"] is None
    assert perfil["freq_valor_mas_frecuente"] == 0
    assert perfil["n_unicos"] == 0
    assert perfil["pct_nulos"] == (100.0 if len(serie) else 0.0)
    distribucion = eda.distribucion_categorica(df, "categoria")
    assert distribucion.empty
    assert list(distribucion.columns) == ["conteo", "porcentaje"]


def test_perfil_mediana_desviacion_y_extremos_sin_redondear():
    df = pd.DataFrame({"valor": [1.111, 2.222, 3.333, np.nan]})
    perfil = eda.perfil_columna(df, "valor")
    assert perfil == {
        "columna": "valor",
        "tipo": "float64",
        "n_total": 4,
        "n_nulos": 1,
        "pct_nulos": 25.0,
        "n_unicos": 3,
        "media": 2.22,
        "mediana": 2.22,
        "std": 1.11,
        "min": 1.111,
        "max": 3.333,
    }


def test_perfil_numerico_nullable_sin_observaciones():
    df = pd.DataFrame({"valor": pd.Series([pd.NA, pd.NA], dtype="Float64")})
    perfil = eda.perfil_columna(df, "valor")
    assert perfil["n_nulos"] == 2
    assert all(
        pd.isna(perfil[clave]) for clave in ["media", "mediana", "std", "min", "max"]
    )


def test_eda_no_muta_ni_imprime_y_correlacion_tiene_valor_exacto(capsys):
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [3.0, 1.0, 2.0], "categoria": ["a", "a", None]}
    )
    original = df.copy(deep=True)
    eda.perfil_columna(df, "x")
    distribucion = eda.distribucion_categorica(df, "categoria")
    matriz = eda.matriz_correlacion(df, ["y", "categoria", "x"])
    assert list(matriz.columns) == ["y", "x"]
    assert matriz.loc["x", "y"] == pytest.approx(-0.5)
    assert distribucion.loc["a", "porcentaje"] == 100.0
    assert_frame_equal(df, original)
    assert capsys.readouterr().out == ""


def test_correlacion_constante_y_nula_conserva_nan_fuera_de_diagonal():
    df = pd.DataFrame({"constante": [2.0] * 3, "sin_datos": [np.nan] * 3})
    matriz = eda.matriz_correlacion(df, ["constante", "sin_datos"])
    assert matriz.loc["constante", "constante"] == 1.0
    assert matriz.loc["sin_datos", "sin_datos"] == 1.0
    assert pd.isna(matriz.loc["constante", "sin_datos"])


def test_integracion_extract_transform_eda_conserva_pedidos_y_compatibilidad():
    tablas = {
        "orders": pd.DataFrame(
            {
                "order_id": ["a", "b", "c"],
                "customer_id": ["x1", "x2", "y1"],
                "order_purchase_timestamp": pd.to_datetime(
                    ["2021-01-01", "2021-02-01", "2021-03-01"]
                ),
                "order_delivered_customer_date": pd.to_datetime(
                    ["2021-01-04", "2021-02-05", None]
                ),
                "order_estimated_delivery_date": pd.to_datetime(
                    ["2021-01-05", "2021-02-04", "2021-03-10"]
                ),
            }
        ),
        "customers": pd.DataFrame(
            {
                "customer_id": ["x1", "x2", "y1"],
                "customer_unique_id": ["x", "x", "y"],
                "customer_state": ["SP", "SP", "RJ"],
            }
        ),
        "items": pd.DataFrame(
            {
                "order_id": ["a", "a", "b"],
                "order_item_id": [1, 2, 1],
                "price": [60.0, 40.0, 200.0],
                "freight_value": [5.0, 7.0, 20.0],
            }
        ),
        "payments": pd.DataFrame(
            {
                "order_id": ["a", "a", "b"],
                "payment_value": [70.0, 42.0, 220.0],
                "payment_installments": [1, 2, 3],
            }
        ),
    }
    originales = {nombre: df.copy(deep=True) for nombre, df in tablas.items()}
    base = extract.construir_dataset_base(tablas)
    assert len(base) == 3 and base["order_id"].is_unique
    pedido_a = base.set_index("order_id").loc["a"]
    assert pedido_a["precio_total"] == pedido_a["ticket_total"] == 100.0
    assert pedido_a["flete_total"] == 12.0
    assert pedido_a["total_pago"] == 112.0 and pedido_a["n_cuotas"] == 2
    assert pedido_a["n_items"] == 2
    resultado = (
        base.pipe(transform.agregar_features_fecha)
        .pipe(transform.agregar_encoding_estado)
        .pipe(transform.agregar_ticket_historico, columna_cliente="customer_unique_id")
    )
    assert resultado.shape == (3, len(base.columns) + 8)
    assert resultado.loc[1, "ticket_promedio_historico"] == 100.0
    assert pd.isna(resultado.loc[2, "precio_total"])
    assert eda.perfil_columna(resultado, "tiempo_entrega_dias")["media"] == 3.5
    for nombre, original in originales.items():
        assert_frame_equal(tablas[nombre], original)
