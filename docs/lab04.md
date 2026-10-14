# Laboratorio 04 - Modelado supervisado

## Objetivo y alcance

Entrenar pipelines reproducibles de regresión y clasificación sobre las features
del Lab 03 y evaluarlos con métricas apropiadas, comparando siempre contra un
baseline. Se implementaron los módulos [train.py](../src/analytics/train.py) y
[evaluate.py](../src/analytics/evaluate.py) con las siete funciones del
enunciado, y los 13 tests del PDF (`tests/test_train.py` y
`tests/test_evaluate.py`), sin modificarlos. No se realizó la extensión opcional
(sección 8).

## Implementación

| Función | Resultado y criterio |
| --- | --- |
| `entrenar_pipeline_regresion` | `Pipeline` con `imputador` (mediana), `escalador` y `modelo` (`RandomForestRegressor`), ya entrenado. |
| `entrenar_pipeline_clasificacion` | Mismos tres pasos, con `RandomForestClassifier(class_weight="balanced")`; expone `predict_proba`. |
| `guardar_pipeline` / `cargar_pipeline` | `joblib.dump` y `joblib.load`; al guardar se crea la carpeta de destino si no existe. |
| `evaluar_regresion` | `rmse` (raíz de `mean_squared_error`), `mae` y `r2`, como `float`. |
| `evaluar_clasificacion` | `accuracy`, `f1`, `roc_auc` y `matriz_confusion` `[[tn, fp], [fn, tp]]` como lista de listas. |
| `comparar_con_baseline` | Accuracy de predecir la moda de `y_train`, accuracy del modelo y su diferencia (`mejora`). |

El imputador es el primer paso para que el escalador y el modelo nunca reciban
`NaN`. Se usa la mediana aprendida en entrenamiento y reaplicada al predecir.
`matriz_confusion` fija `labels=[0, 1]`, por lo que conserva la forma 2x2 y el
orden `[negativa, positiva]` aunque en el conjunto evaluado falte una clase.
En empates, `comparar_con_baseline` toma el menor valor de la moda.

### Consideraciones del Lab 03

- `tiempo_entrega_dias` y `entregado_tarde` solo existen tras la entrega: son
  válidos como features del clasificador de reseñas (indicado por el enunciado),
  pero no habrían estado disponibles al comprar para predecir la entrega.
- `customer_state_freq` se calcula sobre todo el DataFrame. Para una evaluación
  estricta, las frecuencias deben aprenderse solo con entrenamiento.

## Verificación

```bash
uv sync --all-groups --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

Resultado: **74 tests pasan** (los 54 del PDF más las pruebas adicionales y
opcionales del Lab 03). Los tests usan datos sintéticos; no requieren Olist.

Para reproducir la verificación con los datos reales (regenera
`docs/lab04_resultados.json` y los modelos de `models/`):

```bash
uv run python scripts/verificar_lab04.py
```

## Análisis y reflexión

### Datos y protocolo

Los resultados provienen de `scripts/verificar_lab04.py` sobre los CSV de Olist
(salida completa en [lab04_resultados.json](lab04_resultados.json)):

- `review_score` se une con la reseña **más reciente** de cada pedido (un pedido
  puede tener varias); la unión conserva las 99.441 filas.
- `review_negativa` = `review_score <= 2`. El enunciado no la define; es la
  convención habitual (1 y 2 estrellas). Prevalencia en entrenamiento: 14,7 %.
- División aleatoria (`random_state=42`): 60 % entrenamiento, 20 % validación y
  20 % test. Para clasificación es estratificada. Test se evalúa una sola vez,
  al final. Regresión usa los 96.476 pedidos con fecha de entrega.
- El historial usa `customer_unique_id`. Con el `customer_id` por defecto la
  columna `ticket_promedio_historico` queda 100 % nula (en Olist es único por
  pedido) y el imputador la descarta sin avisar.
- Se guardaron `models/pipeline_tiempo_entrega.joblib` y
  `models/pipeline_review_negativa.joblib` (scikit-learn 1.9.1).

### 9.1 Mejora en accuracy frente al baseline

| Validación | Accuracy | Recall (negativa) | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| Baseline (siempre "no negativa") | 0,8531 | 0,000 | - | - |
| Modelo (`balanced`) | 0,8314 | 0,578 | 0,502 | 0,770 |

La mejora es **-0,0217**: el modelo tiene 2,2 puntos *menos* de accuracy que el
baseline. En test el modelo obtiene accuracy 0,8344, recall 0,582, F1 0,508 y
ROC-AUC 0,774. Accuracy no es la métrica adecuada para decidir el despliegue:
con 14,7 % de reseñas negativas, un baseline que nunca predice una negativa ya
alcanza 85 %, pero no detecta ninguna (recall 0). El modelo detecta 1.675 de
2.899 reseñas negativas en validación, con precisión de 0,44 (1.675 de 3.778
alertas). Por eso se debe decidir con recall, F1 y ROC-AUC, y con el costo de
cada tipo de error. El modelo aporta señal (ROC-AUC 0,77 frente a 0,5 del azar),
pero su desempeño es moderado.

### 9.2 RMSE en test frente a validación

El RMSE en test es **8,171 días**, frente a 8,138 en validación: una diferencia
de 0,033 días (0,4 %), consistente y sin señal de sobreajuste. En este
laboratorio no hubo búsqueda de hiperparámetros (se usaron los valores por
defecto) y test se midió una sola vez, por lo que no existe un proceso de
selección que pueda sobreajustarse a validación. Para dimensionar el resultado:
el baseline que predice la media tiene RMSE 9,615, de modo que el modelo mejora
el error un 15 %, con R² de 0,278 en test. Predice mejor que la media, pero la
mayor parte de la variación del tiempo de entrega no queda explicada.

### 9.3 Efecto de `class_weight="balanced"`

| Validación | `balanced` | `None` |
| --- | --- | --- |
| Recall (negativa) | 0,578 | 0,241 |
| Precisión (negativa) | 0,443 | 0,701 |
| F1 | 0,502 | 0,359 |
| Accuracy | 0,831 | 0,873 |
| ROC-AUC | 0,770 | 0,771 |

Sin pesos, el modelo se inclina hacia la clase mayoritaria: acierta más en
general (accuracy 0,873, ya por encima del baseline de 0,853) pero detecta solo
el 24 % de las reseñas negativas. Con `balanced` el recall sube a 58 % a cambio
de más falsas alarmas (precisión 0,70 -> 0,44) y menor accuracy. El ROC-AUC es
prácticamente igual: el ordenamiento de los pedidos por riesgo no cambia, solo
el punto de corte efectivo con el que se decide. Si el costo de no detectar una
reseña negativa es alto, `balanced` es preferible; si se prioriza evitar falsas
alarmas, bastaría ajustar el umbral de `predict_proba`.

### 9.4 Versión de scikit-learn y serialización

`joblib` serializa con `pickle` los objetos Python del pipeline, incluidos los
atributos internos de scikit-learn. Si el servidor tiene otra versión, esos
atributos pueden haber cambiado: `cargar_pipeline` puede fallar con errores de
atributos o módulos, o cargar con una `InconsistentVersionWarning` y producir
predicciones distintas sin avisar. Además, `pickle` ejecuta código al cargar, por
lo que solo deben cargarse archivos de origen confiable.

En `pyproject.toml` se mitiga fijando la versión exacta con la que se entrena
(por ejemplo `scikit-learn==1.9.1`, en lugar de `>=1.5`) y versionando
`uv.lock`, que el CI ya instala con `--locked`. También conviene registrar la
versión junto al archivo del modelo y reentrenar si el entorno de despliegue
cambia.
