# Laboratorio 03 - Transformación de datos y EDA

## Objetivo y alcance

Construir features de fecha, encoding e historial de compras reproducibles,
sin utilizar pedidos futuros, y retornar resultados de EDA reutilizables.
Se implementaron las seis funciones obligatorias, las 23 pruebas nuevas del
enunciado, las dos extensiones opcionales y pruebas adicionales de integración.

Los laboratorios anteriores conservan sus pruebas y sus columnas. Los resultados
de este documento provienen de ejecutar `scripts/verificar_lab03.py` sobre los
CSV del [dataset oficial de Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
descargados el 9 de septiembre de 2026. Las huellas SHA-256 y los resultados
numéricos quedan en [lab03_resultados.json](lab03_resultados.json).

## Implementación

| Función | Resultado y criterio |
| --- | --- |
| `agregar_features_fecha` | Seis columnas: duración y retraso en días, entrega tardía, día de semana, fin de semana y mes. |
| `agregar_encoding_estado` | Frecuencia relativa de la categoría en una columna nueva, por defecto `customer_state_freq`. |
| `agregar_ticket_historico` | Media de importes conocidos de pedidos del mismo cliente con fecha estrictamente anterior. |
| `perfil_columna` | Diccionario con conteos, nulos, valores únicos y estadísticas numéricas o moda. |
| `distribucion_categorica` | DataFrame con conteos y porcentajes ordenados de mayor a menor frecuencia. |
| `matriz_correlacion` | Matriz de Pearson para las columnas numéricas solicitadas. |
| `agregar_encoding_categoria_producto` | Extensión: one-hot encoding, agrupando categorías de frecuencia menor al 1% en `otros`. |
| `detectar_posible_fuga` | Extensión: alerta si la magnitud de Pearson supera el umbral, por defecto 0,95. |

Todas estas funciones conservan el DataFrame de entrada; las funciones de EDA no
imprimen. Los nombres de columnas e identificadores se pueden parametrizar donde
lo solicita el enunciado. Las funciones tienen anotaciones de tipo y docstrings.

### Fechas y nulos

Se utiliza `.dt.total_seconds() / 86400`, que conserva las fracciones de día. Por
ejemplo, una entrega 12 horas después de la compra equivale a 0,5 días.
Un retraso positivo indica entrega posterior a la fecha estimada; uno negativo,
entrega anticipada. Las fechas ausentes producen `NaN` en las duraciones y
`False` al comparar con cero. Ese `False` no confirma una entrega puntual:
para calcular tasas de puntualidad se deben considerar únicamente pedidos con
fechas observadas.

El perfil usa desviación estándar muestral (`ddof=1`). El frequency encoding y
la distribución excluyen nulos de su denominador. El perfil sí informa su
cantidad y porcentaje. En una entrada vacía el porcentaje de nulos es 0,0; las
estadísticas no calculables quedan en `NaN` y la moda ausente en `None` y 0.

### Historial con corrección temporal

Para un cliente y una compra en el instante `t`, se promedian solamente importes
válidos de pedidos con fecha menor que `t`. La implementación:

1. Ordena una copia por fecha con ordenación estable.
2. Agrupa por cliente e instante y calcula suma y conteo de importes válidos.
3. Acumula suma y conteo por cliente y aplica `shift(1)` dentro de cada cliente.
4. Divide la suma previa entre el conteo previo y asigna el resultado a cada pedido.

Así, varios pedidos con la misma fecha y hora reciben el mismo historial y no
se utilizan entre sí. El promedio se pondera por pedido, no por instante. El
primer instante y los historiales sin importes válidos quedan en `NaN`. También
se dejan en `NaN` filas sin identificador o sin fecha. Se conservan las etiquetas
del índice, incluso si están duplicadas.

### Integración con Lab 02

La versión previa de `construir_dataset_base()` incluía `ticket_total`,
`total_pago` y `n_cuotas`, mientras que el PDF de Lab 03 utiliza `precio_total`
y `flete_total`. Se agregó `precio_total` como alias de `ticket_total` y se
agregó el flete por pedido a partir de `freight_value`, sin retirar los datos
del laboratorio anterior ni cambiar sus pruebas.

| Tabla | Antes de transformar | Después de transformar |
| --- | --- | --- |
| Proyecto, con compatibilidad Lab 02 | 99.441 × 18 | 99.441 × 26 |
| Proyección de las columnas del PDF | 99.441 × 15 | 99.441 × 23 |

Las tres columnas adicionales son `ticket_total`, `total_pago` y `n_cuotas`.
Siempre se conservan los 99.441 pedidos y se agregan exactamente ocho features
obligatorias. La proyección del PDF se obtiene con
`df.drop(columns=["ticket_total", "total_pago", "n_cuotas"])`.
Los pedidos sin ítems conservan importes nulos; no se inventa un ticket de cero.

## Verificación

```bash
uv sync --all-groups --locked
uv run pytest tests/test_transform.py -v
uv run pytest tests/test_eda.py -v
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
uv run python scripts/verificar_lab03.py
```

| Verificación local | Resultado |
| --- | --- |
| Pruebas previas: environment, extract y quality | 18 pasan |
| `tests/test_transform.py`, casos del PDF | 12 pasan |
| `tests/test_eda.py`, casos del PDF | 11 pasan |
| Integración y casos límite adicionales | 14 pasan |
| Extensiones opcionales | 6 pasan |
| Suite completa | 61 pasan |
| Ruff, análisis y formato | Sin errores |
| Datos reales | 99.441 pedidos, 8 features nuevas, entrada sin mutaciones |

Se conservaron los valores esperados y las aserciones de los tests del PDF.
Solo se normalizó su formato y se reparó el salto de línea editorial que partía
la cadena `"2021-01-01"`. Las comparaciones con `True`/`False` se mantienen:
`pyproject.toml` excluye únicamente E712 para `tests/test_transform.py`, porque
el enunciado exige no modificar esas pruebas. Las demás reglas siguen activas.

El workflow existente ejecuta Ruff y la suite completa tanto en push como en
pull request. No necesita los CSV de Olist para funcionar. El resultado remoto
se consulta en [GitHub Actions](https://github.com/luxury0101/Analytics-proyect/actions).

## Análisis y reflexión

### 9.1 Porcentaje de filas sin ticket histórico

Con la llamada exacta del enunciado,
`agregar_ticket_historico(df)`, el **100% de las filas (99.441 de 99.441)** queda
en `NaN`. La razón es que el parámetro por defecto es `customer_id` y ese
identificador es distinto para cada pedido en Olist. Esto no demuestra que todas
las personas compraron una sola vez: para identificar compras recurrentes de la
misma persona se debe agrupar por `customer_unique_id`.

Al ejecutar `agregar_ticket_historico(df, columna_cliente="customer_unique_id")`,
se obtienen **96.412 filas sin historial, equivalentes al 96,953973%**, y 3.029
filas con historial válido. Hay 96.096 clientes únicos: 93.099 tienen un pedido
y 2.997 tienen más de uno, es decir, **3,118756% de clientes recurrentes**.

El porcentaje de filas sin historial no equivale al porcentaje de clientes de
un solo pedido: incluye la primera compra de todos los clientes, también de los
recurrentes. En esta implementación estricta hay 96.374 filas en el primer
instante de compra, incluyendo pedidos simultáneos; otras 38 filas no tienen
un importe previo válido. Estas diferencias explican por qué los nulos tampoco
coinciden exactamente con el número de clientes únicos.

### 9.2 Estados únicos frente a columnas de one-hot encoding

`df["customer_state"].nunique()` devuelve **27**. La llamada
`pd.get_dummies(df["customer_state"])` también crea **27 indicadores**, porque
`drop_first=False` es el valor por defecto. Con `drop_first=True` crea **26**:
se omite una categoría de referencia, representada por todos los indicadores
en cero. [Documentación de pandas](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).

La llamada del enunciado pasa el DataFrame completo:
`pd.get_dummies(df, columns=["customer_state"])`. Por ello el total incluye
también las columnas que no se codifican. Con nuestro DataFrame transformado de
26 columnas, el resultado tiene **26 - 1 + 27 = 52 columnas**; con
`drop_first=True`, **51**. Si se usa la proyección de 23 columnas del PDF, los
totales son 49 y 48. Hay que distinguir el número de indicadores nuevos del
número total de columnas del DataFrame resultante.

### 9.3 Correlación entre entrega, número de ítems y precio

Se obtuvo la siguiente matriz de Pearson. Para cada correlación con la duración
se utilizaron 96.476 pedidos con ambas variables observadas; no se imputaron
fechas ni se sustituyeron entregas desconocidas por cero.

| Variable | Tiempo de entrega | Número de ítems | Precio total |
| --- | ---: | ---: | ---: |
| Tiempo de entrega | 1,000000 | -0,019113 | 0,055578 |
| Número de ítems | -0,019113 | 1,000000 | 0,153063 |
| Precio total | 0,055578 | 0,153063 | 1,000000 |

La relación entre tiempo de entrega y número de ítems es prácticamente nula.
La relación con el precio total es positiva, pero también muy débil. Una posible
explicación de negocio es que algunos productos de mayor precio necesiten una
logística distinta o más preparación; sin embargo, estos coeficientes no
permiten concluir que un mayor precio cause demoras. Conviene explorar distancia,
estado de destino, vendedor y categoría del producto. La relación positiva entre
cantidad y precio tiene sentido porque más unidades pueden aumentar el valor
del pedido, aunque los precios por unidad varían mucho.

Como contexto descriptivo, la duración media fue **12,56 días**, la mediana
**10,22 días** y la desviación estándar **9,55 días**. Hay 2.965 pedidos sin fecha
de entrega observada. La correlación omite nulos por pares, de acuerdo con
[pandas.DataFrame.corr](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html).

### 9.4 Riesgo de cambiar el orden de las filas

Si se ordenan las features por fecha y otra parte del pipeline conserva el vector
objetivo en el orden original, una asignación por posición puede asociar el
objetivo de un pedido con las features de otro. Esto produce entrenamiento
incorrecto y métricas engañosas sin que necesariamente aparezca una excepción.

Se mitiga conservando `order_id` y alineando mediante esa clave, o reindexando
por el índice original cuando es único. Si hay índices duplicados, se debe
añadir antes de transformar una columna de posición y ordenar por ella al final.
La implementación conserva el índice y usa ordenación estable, pero no promete
conservar el orden original; está documentado y probado con índices duplicados.

Ejemplo de alineación explícita de un objetivo que ya existe en `base`:

```python
features = transform.agregar_ticket_historico(base)
objetivo_por_pedido = base.set_index("order_id")["objetivo"]
y_alineado = objetivo_por_pedido.reindex(features["order_id"])
# features y y_alineado corresponden al mismo orden de pedidos.
```

## Preguntas guía

- Diferencias de fechas: `.dt.total_seconds() / 86400` conserva días fraccionarios;
  `.dt.days` devuelve días enteros.
- Frecuencias relativas: `Series.value_counts(normalize=True)`.
- Historial básico: ordenar por fecha, agrupar por cliente y calcular
  `serie.shift(1).expanding().mean()` dentro de cada grupo. Cuando hay empates,
  el bloque de pedidos simultáneos debe excluirse completo, como hace este módulo.
- Detección de tipo numérico: `pd.api.types.is_numeric_dtype(serie)`.
- Conteos categóricos descendentes: `Series.value_counts()`.
- Correlación sin columnas de texto: `DataFrame.corr(numeric_only=True)`.

## Consideraciones para Lab 04

Las variables que dependen de la entrega real sirven para EDA o para construir
objetivos observados, pero aún no existen al comprar. No se deben incluir como
predictores de su propia entrega. El historial asume que el importe de un pedido
anterior ya era conocido en su fecha de compra.

El encoding del Lab 03 se calcula sobre el DataFrame recibido. Antes de entrenar,
se deben separar los datos por fecha, aprender frecuencias y categorías únicamente
con entrenamiento y reutilizar ese mapa al validar. La detección opcional de fuga
usa `abs(r) > umbral`; una correlación extrema es una alerta, no una prueba, y una
baja correlación no garantiza ausencia de fuga.

La diagonal de la matriz se fija a 1,0 para cumplir el contrato del PDF. En
variables constantes, vacías o con observaciones insuficientes es una convención:
la correlación no es estimable. Los valores indefinidos fuera de la diagonal
conservan `NaN`.
