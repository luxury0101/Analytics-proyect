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
