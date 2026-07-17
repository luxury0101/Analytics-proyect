"""Pruebas de configuración inicial del proyecto."""

from pathlib import Path


def test_dependencias_principales():
    """Verifica que las dependencias principales estén instaladas."""
    import pandas
    import pytest

    assert pandas.__version__
    assert pytest.__version__


def test_estructura_del_proyecto():
    """Verifica que exista la estructura principal del proyecto."""
    raiz = Path(__file__).resolve().parents[1]

    assert (raiz / "pyproject.toml").is_file()
    assert (raiz / "src" / "analytics").is_dir()
    assert (raiz / "data" / "raw").is_dir()


def test_paquete_importable():
    """Verifica que el paquete analytics pueda importarse."""
    import analytics

    assert analytics.__doc__
