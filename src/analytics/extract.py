"""Módulo de ingesta de datos hacia el pipeline de analítica."""

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
