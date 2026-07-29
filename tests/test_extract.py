"""Tests del módulo extract.py — Lab 01."""

import pandas as pd
import pytest

from analytics.extract import cargar_csv, join_verificado


def test_cargar_csv_retorna_dataframe(tmp_path):
    """cargar_csv debe retornar un DataFrame de pandas."""
    archivo = tmp_path / "datos.csv"
    archivo.write_text(
        "ciudad,ventas\nMedellin,100\nBogota,200\n",
        encoding="utf-8",
    )

    resultado = cargar_csv(str(archivo))

    assert isinstance(resultado, pd.DataFrame)


def test_cargar_csv_carga_filas_correctamente(tmp_path):
    """El DataFrame debe tener exactamente las filas del CSV."""
    archivo = tmp_path / "datos.csv"
    archivo.write_text(
        "ciudad,ventas\nMedellin,100\nBogota,200\nCali,150\n",
        encoding="utf-8",
    )

    dataframe = cargar_csv(str(archivo))

    assert len(dataframe) == 3
    assert list(dataframe.columns) == ["ciudad", "ventas"]


def test_cargar_csv_archivo_no_existe():
    """Una ruta inexistente debe producir FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        cargar_csv("ruta/inventada/que/no/existe.csv")


def test_cargar_csv_archivo_vacio(tmp_path):
    """Un CSV sin filas de datos debe producir ValueError."""
    archivo = tmp_path / "vacio.csv"
    archivo.write_text("ciudad,ventas\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no contiene filas"):
        cargar_csv(str(archivo))


def test_join_verificado_retorna_dataframe_correcto():
    """Un join sin duplicados conserva las filas de la tabla izquierda."""
    left = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "valor": ["a", "b", "c"],
        }
    )

    right = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "extra": ["x", "y", "z"],
        }
    )

    resultado = join_verificado(left, right, on="id")

    assert isinstance(resultado, pd.DataFrame)
    assert len(resultado) == 3
    assert "extra" in resultado.columns


def test_join_verificado_falla_con_duplicados():
    """Duplicados en la clave derecha deben producir un error."""
    left = pd.DataFrame(
        {
            "id": [1, 2],
            "valor": ["a", "b"],
        }
    )

    right = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "extra": ["x", "y", "z"],
        }
    )

    with pytest.raises(AssertionError, match="filas"):
        join_verificado(
            left,
            right,
            on="id",
            nombre="test_dup",
        )


def test_join_verificado_left_join_no_pierde_filas():
    """Un left join mantiene las filas sin coincidencia."""
    left = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "valor": ["a", "b", "c"],
        }
    )

    right = pd.DataFrame(
        {
            "id": [1, 2],
            "extra": ["x", "y"],
        }
    )

    resultado = join_verificado(left, right, on="id")

    assert len(resultado) == 3
    assert pd.isna(
        resultado.loc[
            resultado["id"] == 3,
            "extra",
        ].values[0]
    )
