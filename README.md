# LLM_as_a_judge

Este repositorio recoge el código, los datos intermedios y los resultados asociados al trabajo de fin de máster titulado "LLM como jueces para la evaluación automática de la corrección gramatical en gallego: una aproximación exploratoria". El objetivo principal es explorar el uso de modelos de lenguaje como jueces automáticos para evaluar correcciones gramaticales en gallego.

## Estructura del repositorio

```text
LLM_as_a_judge/
├── datasets_y_corpus/
├── ensayos_seleccion_modelo/
├── LoRA/
├── evaluacion_automatica/
├── evaluacion_manual/
├── LICENSE
└── README.md
```

## Contenido

### `datasets_y_corpus/`

Contiene los datos utilizados durante el trabajo:

* `CORTEGAL/`: corpus de partida.
* `dataset_A/`: conjunto usado para preparar y entrenar el juez.
* `dataset_B/`: conjunto reservado para evaluación.
* `gold_standard/`: anotación de referencia usada para comparar las predicciones de los modelos.

### `ensayos_seleccion_modelo/`

Incluye pruebas iniciales para seleccionar y explorar modelos antes del ajuste final. Esta parte recoge scripts y datos de prueba usados en la fase preliminar del trabajo.

### `LoRA/`

Contiene el entrenamiento mediante LoRA del modelo Selene usado como juez.

Incluye:

* `lora_selene.py`: script principal de entrenamiento.
* `splits/`: particiones de entrenamiento, validación y test.
* `checkpoint-500/` y `checkpoint-1020/`: checkpoints generados durante el entrenamiento.
* Archivos del adaptador LoRA final, tokenizer y configuración asociada.

El entrenamiento se plantea como una tarea de generación controlada: el modelo recibe un `input_corrector` y un `output_corrector`, y debe devolver únicamente la etiqueta `0` o `1`.

### `evaluacion_automatica/`

Contiene los resultados automáticos de comparación entre:

* el modelo base,
* el modelo fine-tuned con LoRA,
* y el gold standard.

El script `calculo_metricas_base_ft_gold.py` calcula métricas como accuracy, balanced accuracy, precision, recall, F1, MCC, Cohen’s kappa y matrices de confusión.

### `evaluacion_manual/`

Contiene la muestra empleada para el análisis manual de las explicaciones generadas por el modelo fine-tuned.

El script `sample_evaluacion_manual.py` genera una muestra aleatoria y añade columnas de evaluación manual, como consistencia, corrección gramatical, naturalidad y corrección factual.

## Flujo general del trabajo

1. Preparación de los datos a partir de CORTEGAL.
2. Creación de un gold standard para evaluar las salidas del corrector.
3. Entrenamiento de un juez LLM mediante LoRA.
4. Comparación automática entre modelo base, modelo fine-tuned y gold standard.
5. Análisis manual cualitativo de una muestra de explicaciones.

## Modelo

El ajuste se realiza sobre Selene mediante LoRA, con el objetivo de adaptar el modelo a una tarea concreta de evaluación gramatical en gallego sin reentrenar todos sus parámetros.

## Nota

Este repositorio está pensado como material complementario del TFM. Algunos archivos corresponden a resultados intermedios o pruebas exploratorias, por lo que la estructura refleja también el proceso real de experimentación.

## Licencia

Este repositorio se distribuye bajo licencia MIT.
