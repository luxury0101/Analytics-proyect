"""Análisis exploratorio que retorna objetos reutilizables y no imprime."""

import pandas as pd


def perfil_columna(df: pd.DataFrame, columna: str) -> dict:
    """Genera un perfil estadístico sin modificar el DataFrame.

    Args:
        df: DataFrame que contiene la columna.
        columna: Nombre de la columna a perfilar.

    Returns:
        Diccionario con columna, tipo, n_total, n_nulos, pct_nulos y n_unicos.
        Si es numérica, incluye media, mediana y std muestral (ddof=1)
        redondeadas a dos decimales, min y max sin redondear. Estadísticas
        indefinidas quedan en NaN. Si no es numérica, incluye
        valor_mas_frecuente y freq_valor_mas_frecuente (None y 0 sin datos).
        Los nulos no cuentan como valores únicos ni candidatos a la moda.
        En un DataFrame vacío, pct_nulos es 0.0.
    """
    serie = df[columna]
    n_total = len(serie)
    n_nulos = int(serie.isna().sum())
    perfil = {
        "columna": columna,
        "tipo": str(serie.dtype),
        "n_total": n_total,
        "n_nulos": n_nulos,
        "pct_nulos": round(n_nulos / n_total * 100, 2) if n_total else 0.0,
        "n_unicos": int(serie.nunique()),
    }
    if pd.api.types.is_numeric_dtype(serie):
        for nombre, operacion in (
            ("media", "mean"),
            ("mediana", "median"),
            ("std", "std"),
            ("min", "min"),
            ("max", "max"),
        ):
            valor = getattr(serie, operacion)() if n_total > n_nulos else float("nan")
            if pd.isna(valor):
                valor = float("nan")
            elif nombre not in {"min", "max"}:
                valor = round(float(valor), 2)
            perfil[nombre] = valor
    else:
        conteos = serie.value_counts()
        # Un Categorical vacío puede conservar categorías con conteo cero.
        conteos = conteos[conteos > 0]
        perfil["valor_mas_frecuente"] = conteos.index[0] if len(conteos) else None
        perfil["freq_valor_mas_frecuente"] = int(conteos.iloc[0]) if len(conteos) else 0
    return perfil


def distribucion_categorica(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Retorna conteos y porcentajes de categorías en orden descendente.

    Args:
        df: DataFrame que contiene la columna.
        columna: Nombre de la columna categórica.

    Returns:
        DataFrame indexado por categoría con conteo (int) y porcentaje
        (float, dos decimales). Excluye nulos y categorías sin observaciones;
        los porcentajes se calculan sobre valores válidos y suman cerca de
        100 por redondeo. Sin valores válidos retorna un DataFrame vacío.
    """
    conteos = df[columna].value_counts()
    conteos = conteos[conteos > 0]
    resultado = conteos.astype(int).to_frame("conteo")
    total = int(conteos.sum())
    resultado["porcentaje"] = (conteos / total * 100).round(2) if total else 0.0
    return resultado


def matriz_correlacion(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Retorna la matriz de Pearson de las columnas numéricas solicitadas.

    Args:
        df: DataFrame que contiene las columnas.
        columnas: Nombres de las columnas numéricas, en el orden deseado.

    Returns:
        Matriz cuadrada, simétrica, sin modificar df. Se excluyen columnas de
        texto accidentales con numeric_only=True y se omiten nulos por pares.
        La diagonal se fija a 1.0 por el contrato del laboratorio, incluso
        para columnas constantes o sin datos; en esos casos es una convención,
        no una correlación estimable. Fuera de la diagonal, los coeficientes
        indefinidos permanecen NaN.
    """
    resultado = df[columnas].corr(method="pearson", numeric_only=True)
    for posicion in range(len(resultado)):
        resultado.iat[posicion, posicion] = 1.0
    return resultado


def detectar_posible_fuga(
    df: pd.DataFrame,
    columna_feature: str,
    columna_objetivo: str,
    umbral_correlacion: float = 0.95,
) -> bool:
    """Marca una correlación extrema como señal, no prueba, de posible fuga.

    Args:
        df: DataFrame con las dos columnas numéricas.
        columna_feature: Variable candidata a predictor.
        columna_objetivo: Variable que se quiere predecir.
        umbral_correlacion: Magnitud de Pearson a superar, entre 0 y 1.

    Returns:
        True si abs(correlación) supera estrictamente el umbral. Detecta
        relaciones tanto positivas como negativas. Una correlación indefinida
        devuelve False, lo que no garantiza ausencia de fuga.

    Raises:
        ValueError: Si el umbral no está entre 0 y 1.
        TypeError: Si alguna de las columnas no es numérica.
    """
    if not 0 <= umbral_correlacion <= 1:
        raise ValueError("El umbral de correlación debe estar entre 0 y 1.")
    for columna in (columna_feature, columna_objetivo):
        if not pd.api.types.is_numeric_dtype(df[columna]):
            raise TypeError(f"La columna {columna!r} debe ser numérica.")
    correlacion = df[columna_feature].corr(df[columna_objetivo], method="pearson")
    return bool(pd.notna(correlacion) and abs(correlacion) > umbral_correlacion)
