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
* `dataset_A/`: dataset usado para preparar y entrenar el juez.
* `dataset_B/`: dataset de evaluación.
* `gold_standard/`: dataset anotado de referencia usado para analizar las predicciones de los modelos.

### `ensayos_seleccion_modelo/`

Incluye las pruebas iniciales para la selección y exploración de modelos antes de coemnzar la experimentación central del trabajo; aquí se recogen los scripts y datos de prueba empleados en esa fase preliminar.

### `LoRA/`

Contiene el entrenamiento mediante Low-Rank Adaptation (LoRA) del modelo AtlaAI/Selene-1-Mini-Llama-3.1-8B usado como juez.

Incluye:

* `lora_selene.py`: script principal de fine-tuning.
* `splits/`: particiones de entrenamiento, validación y test.
* `checkpoint-500/` y `checkpoint-1020/`: checkpoints generados durante el entrenamiento.
* Archivos del adaptador LoRA final, tokenizer y configuración asociada.

### `evaluacion_automatica/`

Contiene los archivos relacionados con la comparación automática entre:

* el modelo base,
* el modelo fine-tuned con LoRA,
* y el gold standard.

El script `calculo_metricas_base_ft_gold.py` calcula métricas como precision, recall, F1, MCC o Cohen’s kappa, así como las matrices de confusión para cada modelo; se guardan aquí también los resultados obtenidos.

### `evaluacion_manual/`

Contiene la muestra aleatoria empleada para el análisis manual de las explicaciones generadas por el modelo fine-tuned.

El script `sample_evaluacion_manual.py` genera una muestra de 50 instancias y añade las columnas necesarias para la evaluación manual, como consistencia, corrección gramatical, naturalidad o corrección factual.

## Flujo general del trabajo

1. Preparación de los datos de entranmienot (dataset A) y evaluación (dataset B) a partir de CORTEGAL.
2. Creación de un gold standard (a partir del dataset B) para la evaluación automática.
3. Entrenamiento de un LLM-as-a-Judge (AtlaAI/Selene-1-Mini-Llama-3.1-8B) mediante la técnica LoRA.
4. Comparación automática entre modelo base, modelo fine-tuned y gold standard.
5. Análisis manual cualitativo de las predicciones del modelo fine-tuned, así como de una muestra de las explicaciones por dicho modelo.

## Nota

Este repositorio está pensado como material complementario del TFM. Algunos archivos corresponden a resultados intermedios o pruebas exploratorias, por lo que la estructura refleja también el proceso real de experimentación.

## Licencia

Este repositorio se distribuye bajo licencia MIT.
