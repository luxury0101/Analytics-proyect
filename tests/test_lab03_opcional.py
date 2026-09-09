"""Pruebas de las dos extensiones opcionales del Laboratorio 03."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from analytics.eda import detectar_posible_fuga
from analytics.transform import agregar_encoding_categoria_producto


def test_agrupar_raras_reduce_columnas_y_conserva_original():
    df = pd.DataFrame({"product_category_name": ["comun"] * 198 + ["rara_a", "rara_b"]})
    original = df.copy(deep=True)
    resultado = agregar_encoding_categoria_producto(df)
    columnas = resultado.filter(like="product_category_name_")
    assert columnas.shape[1] < pd.get_dummies(df["product_category_name"]).shape[1]
    assert columnas.shape[1] == 2
    assert columnas["product_category_name_otros"].sum() == 2
    assert columnas.sum(axis=1).eq(1).all()
    assert_frame_equal(df, original)


def test_umbral_uno_por_ciento_se_conserva_y_otras_raras_se_agrupan():
    df = pd.DataFrame(
        {"product_category_name": ["comun"] * 196 + ["limite"] * 2 + ["a", "b"]}
    )
    resultado = agregar_encoding_categoria_producto(df)
    columnas = resultado.filter(like="product_category_name_")
    assert columnas.shape[1] < pd.get_dummies(df["product_category_name"]).shape[1]
    assert columnas["product_category_name_limite"].sum() == 2
    assert columnas["product_category_name_otros"].sum() == 2


def test_categorias_nulas_y_no_observadas_no_crean_indicadores():
    df = pd.DataFrame(
        {
            "product_category_name": pd.Categorical(
                ["a", None, "b"], categories=["a", "b", "no_observada"]
            )
        }
    )
    resultado = agregar_encoding_categoria_producto(df)
    columnas = resultado.filter(like="product_category_name_")
    assert columnas.shape[1] == 2
    assert not columnas.iloc[1].any()


def test_feature_identica_al_objetivo_es_sospechosa():
    df = pd.DataFrame({"feature": [1, 2, 3, 4], "objetivo": [1, 2, 3, 4]})
    original = df.copy(deep=True)
    assert detectar_posible_fuga(df, "feature", "objetivo") is True
    assert_frame_equal(df, original)


def test_fuga_negativa_y_umbral_estricto():
    df = pd.DataFrame({"feature": [-1, -2, -3, -4], "objetivo": [1, 2, 3, 4]})
    assert detectar_posible_fuga(df, "feature", "objetivo") is True
    assert detectar_posible_fuga(df, "feature", "objetivo", 1.0) is False


def test_correlacion_debil_no_es_sospechosa_y_umbral_se_valida():
    df = pd.DataFrame({"feature": [1, 2, 3], "objetivo": [3, 1, 2]})
    assert detectar_posible_fuga(df, "feature", "objetivo") is False
    with pytest.raises(ValueError, match="umbral"):
        detectar_posible_fuga(df, "feature", "objetivo", 1.5)
