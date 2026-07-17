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
