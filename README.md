# Proyecto de Analítica de Datos - Olist

[![CI](https://github.com/luxury0101/Analytics-proyect/actions/workflows/ci.yml/badge.svg)](https://github.com/luxury0101/Analytics-proyect/actions/workflows/ci.yml)

Laboratorios 01, 02 y 03: ingesta, calidad, transformación y análisis exploratorio
con Python 3.12, pandas, Pandera, pytest y Ruff.

## Ejecutar el proyecto

Desde la raíz del repositorio, con [uv](https://docs.astral.sh/uv/) instalado:

```bash
uv sync --all-groups --locked
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

Resultado del Lab 03: **61 tests pasan**, incluidos los 41 obligatorios del PDF.
Las pruebas usan datos sintéticos y no requieren descargar Olist.

## Laboratorio 03 - Transformación y EDA

- [Implementación, decisiones y respuestas de reflexión](docs/lab03.md).
- [Resultados calculados con Olist y huellas de los archivos](docs/lab03_resultados.json).
- [Funciones de transformación](src/analytics/transform.py).
- [Funciones de EDA](src/analytics/eda.py).
- Extensiones opcionales: categorías raras y detección de posible fuga.

Para reproducir el análisis real, descargar y extraer el
[dataset oficial de Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
en `data/raw/`. Se necesitan estos cinco archivos, conservando sus nombres:

```text
olist_orders_dataset.csv
olist_customers_dataset.csv
olist_order_items_dataset.csv
olist_order_payments_dataset.csv
olist_order_reviews_dataset.csv
```

```bash
uv run python scripts/verificar_lab03.py
```

El comando regenera `docs/lab03_resultados.json`. Los CSV reales están excluidos
de Git. La tabla conserva las tres columnas de compatibilidad del Lab 02, por lo
que tiene 18 columnas antes y 26 después de transformar. El script verifica
también la proyección exacta del PDF: **99.441 × 15 -> 99.441 × 23**.

## Laboratorio 01

Configuración inicial del proyecto, entorno reproducible, pruebas automatizadas,
calidad de código y GitHub Actions.

## Análisis y reflexión

# 8.1 ¿Qué ocurrió antes de implementar cargar_csv?

Al ejecutar pytest antes de implementar la función, los tests fallaron porque
cargar_csv lanzaba NotImplementedError. Esto demuestra que pytest ejecuta las
funciones de prueba y muestra el error exacto que ocurre durante su ejecución.

# 8.2 ¿Por qué pd.read_csv lanza FileNotFoundError automáticamente?

Porque intenta abrir el archivo usando la ruta proporcionada. Si el archivo
no existe, Python genera FileNotFoundError automáticamente.

# 8.3 Diferencia entre el CI y las pruebas locales

En GitHub Actions se debe crear una máquina virtual, descargar el repositorio,
instalar Python y las dependencias antes de ejecutar las pruebas. En el computador
local, el entorno ya está instalado, por eso pytest tarda menos.

## Laboratorio 02 — Análisis y reflexión

# 9.1 Resultado de construir_dataset_base

Al ejecutar construir_dataset_base con los datos reales, el resultado tuvo
99.441 filas, correspondientes a una fila por pedido. La agregación previa de
payments e items evitó que los joins multiplicaran los pedidos. Si el resultado
tuviera más filas, probablemente se habría realizado el join con payments o
items sin agruparlos primero por order_id.

# 9.2 Validación de SCHEMA_ORDERS

Al validar la tabla orders con SCHEMA_ORDERS, la función retornó una lista con
los errores encontrados. En los datos originales se espera que el contrato se
cumpla, porque los identificadores tienen 32 caracteres, los estados pertenecen
al conjunto permitido y la fecha de compra se carga como datetime.

El número exacto de errores obtenido fue 0.

# 9.3 Uso de coerce=True

coerce=True permite que Pandera convierta una columna al tipo declarado cuando
la conversión es posible. Sin esta opción, si order_purchase_timestamp se carga
como object o string, la validación produciría un SchemaError porque el contrato
espera una columna datetime64.

# 9.4 Uso de AssertionError

Una excepción personalizada o ValueError podría expresar mejor un problema de
datos durante un join. Además, las instrucciones assert pueden eliminarse cuando
Python se ejecuta con el indicador -O. Esto haría que la validación de filas no
se ejecutara, permitiendo que un join incorrecto continuara silenciosamente.
