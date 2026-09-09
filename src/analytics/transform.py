"""Transformaciones reproducibles de Olist: fechas, encoding e historial."""

import pandas as pd


def agregar_features_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega seis features de fecha sin modificar el DataFrame recibido.

    Args:
        df: DataFrame con order_purchase_timestamp,
            order_delivered_customer_date y order_estimated_delivery_date
            de tipo datetime64.

    Returns:
        Copia con tiempo_entrega_dias y retraso_entrega_dias (float, con
        fracciones de día y NaN cuando falta una fecha), entregado_tarde
        (retraso > 0), dia_semana_compra (0=lunes, 6=domingo),
        es_fin_de_semana (sábado o domingo) y mes_compra (1-12).

    Notes:
        Una fecha ausente produce False en los indicadores booleanos; no
        equivale a confirmar una entrega puntual. Las variables de entrega
        solo se conocen después de entregar el pedido y no deben usarse como
        predictores de su propia entrega al momento de la compra.
    """
    resultado = df.copy()
    compra = resultado["order_purchase_timestamp"]
    entrega = resultado["order_delivered_customer_date"]
    estimada = resultado["order_estimated_delivery_date"]

    resultado["tiempo_entrega_dias"] = (entrega - compra).dt.total_seconds() / 86400
    resultado["retraso_entrega_dias"] = (entrega - estimada).dt.total_seconds() / 86400
    resultado["entregado_tarde"] = resultado["retraso_entrega_dias"] > 0
    resultado["dia_semana_compra"] = compra.dt.dayofweek
    resultado["es_fin_de_semana"] = resultado["dia_semana_compra"].isin([5, 6])
    resultado["mes_compra"] = compra.dt.month
    return resultado


def agregar_encoding_estado(
    df: pd.DataFrame,
    columna: str = "customer_state",
) -> pd.DataFrame:
    """Agrega frequency encoding sin modificar la entrada.

    Args:
        df: DataFrame que contiene la columna categórica.
        columna: Nombre de la columna que se codificará.

    Returns:
        Copia con f"{columna}_freq", la frecuencia relativa de cada categoría
        entre los valores no nulos del DataFrame recibido. Los nulos conservan
        NaN; una entrada vacía produce una columna float vacía.

    Notes:
        Para EDA se usa el DataFrame completo, como indica el laboratorio.
        Para entrenar un modelo hay que aprender el mapa solo en entrenamiento
        y reutilizarlo en validación/test, sin recalcularlo en esos conjuntos.
    """
    resultado = df.copy()
    frecuencias = resultado[columna].value_counts(normalize=True)
    resultado[f"{columna}_freq"] = resultado[columna].map(frecuencias).astype(float)
    return resultado


def agregar_ticket_historico(
    df: pd.DataFrame,
    columna_cliente: str = "customer_id",
    columna_valor: str = "precio_total",
    columna_fecha: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Agrega el promedio de compras estrictamente anteriores por cliente.

    Args:
        df: DataFrame con identificador, valor numérico y fecha datetime.
        columna_cliente: Identificador de agrupación. En Olist se recomienda
            customer_unique_id para reconocer compras de la misma persona;
            customer_id se conserva como valor por defecto del enunciado.
        columna_valor: Importe de cada pedido, conocido en columna_fecha.
        columna_fecha: Fecha utilizada para ordenar cronológicamente.

    Returns:
        Copia ordenada establemente por fecha, conservando las etiquetas del
        índice, con ticket_promedio_historico. El primer pedido queda en NaN.
        Los pedidos simultáneos no se usan entre sí. Se ignoran importes nulos;
        sin importes previos válidos, identificador o fecha, el resultado es NaN.

    Notes:
        Agrupar primero por cliente/instante permite excluir todo el instante
        actual del acumulado, evitando fuga incluso si hay fechas repetidas.
        Se usa suma/conteo para ponderar por pedido, no por instante.
    """
    resultado = df.sort_values(columna_fecha, kind="stable").copy()
    claves = [columna_cliente, columna_fecha]
    por_instante = resultado.groupby(claves, observed=True)[columna_valor].agg(
        suma="sum", conteo="count"
    )
    acumulados = por_instante.groupby(level=0, sort=False).cumsum()
    anteriores = acumulados.groupby(level=0, sort=False).shift(1)
    historico = anteriores["suma"] / anteriores["conteo"].replace(0, float("nan"))

    # La asignación posicional también funciona con índices duplicados.
    indice_pedidos = pd.MultiIndex.from_frame(resultado[claves])
    resultado["ticket_promedio_historico"] = historico.reindex(indice_pedidos).to_numpy(
        dtype=float, na_value=float("nan")
    )
    return resultado


def agregar_encoding_categoria_producto(
    df: pd.DataFrame,
    columna: str = "product_category_name",
) -> pd.DataFrame:
    """Agrega one-hot encoding agrupando categorías de frecuencia menor al 1%.

    Args:
        df: DataFrame que contiene la categoría del producto.
        columna: Columna categórica que se codificará.

    Returns:
        Copia con la columna original y columnas booleanas prefijadas por
        columna. Las categorías raras se agrupan en 'otros'. Los nulos no se
        agrupan como categoría y quedan con todos los indicadores en False.

    Notes:
        Las frecuencias se calculan entre valores no nulos. Una categoría con
        frecuencia exactamente del 1% se conserva. En modelado, las categorías
        y columnas deben aprenderse solo en el conjunto de entrenamiento.

    Raises:
        ValueError: Si una columna nueva coincide con una columna existente.
    """
    resultado = df.copy()
    categorias = resultado[columna].astype(object)
    frecuencias = categorias.value_counts(normalize=True)
    agrupadas = categorias.mask(categorias.map(frecuencias).lt(0.01), "otros")
    indicadores = pd.get_dummies(agrupadas, prefix=columna, dtype=bool)
    if resultado.columns.intersection(indicadores.columns).size:
        raise ValueError(
            "Ya existen columnas con los nombres del encoding de producto."
        )
    return pd.concat([resultado, indicadores], axis=1)
