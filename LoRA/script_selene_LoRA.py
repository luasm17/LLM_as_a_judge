#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# En este script hacemos el fine-tuning de Selene para adaptarlo a nuestra tarea
# de evaluación automática de correcciones gramaticales en gallego. La idea no es
# entrenar un corrector, sino un juez: el modelo recibe una frase original y una
# corrección propuesta, y aprende a decidir si esa corrección es válida o no.

# Importamos módulos generales de Python.
# os nos permite trabajar con rutas y carpetas.
# json nos permite leer y escribir archivos en formato JSONL.
# argparse nos permite pasar argumentos al script desde la terminal.
# random nos ayuda a fijar la semilla de aleatoriedad.
# inspect nos permite consultar qué argumentos acepta una clase o función concreta.
import os
import json
import argparse
import random
import inspect

# Counter nos sirve para contar cuántos ejemplos tenemos de cada clase, caso o combinación.
from collections import Counter

# Importamos PyTorch, que es la librería base sobre la que se entrena el modelo.
import torch

# Dataset, de la librería datasets, nos permite convertir nuestras listas de ejemplos
# en objetos que el Trainer de Hugging Face puede utilizar directamente.
from datasets import Dataset

# train_test_split nos permite dividir el dataset en subconjuntos de entrenamiento,
# validación y prueba.
from sklearn.model_selection import train_test_split

# Importamos las clases principales de transformers.
# AutoTokenizer carga el tokenizador asociado al modelo.
# AutoModelForCausalLM carga el modelo causal de lenguaje.
# Trainer gestiona el bucle de entrenamiento.
# TrainingArguments guarda la configuración del entrenamiento.
# DataCollatorForSeq2Seq se encarga de hacer el padding de los lotes.
# BitsAndBytesConfig permite cargar el modelo cuantizado, por ejemplo en 4 bits.
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)

# Importamos las herramientas de PEFT necesarias para hacer LoRA.
# LoRA nos permite adaptar el modelo entrenando solo un número reducido de parámetros,
# en lugar de actualizar todos los pesos del modelo base.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_jsonl(path):
    # Leemos el dataset de entrada, que esperamos que esté en formato JSONL.
    # En un archivo JSONL, cada línea contiene un ejemplo independiente en formato JSON.
    records = []

    # Abrimos el archivo usando codificación UTF-8 para poder trabajar correctamente
    # con caracteres propios del gallego, como acentos, ñ o grafías específicas.
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            # Eliminamos espacios en blanco al principio y al final de cada línea.
            line = line.strip()

            # Si encontramos una línea vacía, la ignoramos para evitar errores innecesarios.
            if not line:
                continue

            try:
                # Convertimos la línea JSON en un diccionario de Python y lo guardamos.
                obj = json.loads(line)
                records.append(obj)
            except Exception as e:
                # Si alguna línea no se puede leer como JSON, lanzamos un error indicando
                # exactamente en qué línea y archivo se produjo el problema.
                raise ValueError(f"Erro ao ler a liña {line_num} de {path}: {e}")

    # Devolvemos la lista completa de ejemplos.
    return records


def make_stratify_label(example):
    # Construimos una etiqueta auxiliar para hacer la partición estratificada.
    # No estratificamos solo por la clase 0/1, sino por la combinación entre el caso
    # del dataset y la etiqueta. Así intentamos conservar en train, validation y test
    # una distribución parecida tanto de los casos como de las clases.
    return f"{example['case']}_{example['tag']}"


def build_user_prompt(example):
    # Construimos el prompt que recibe el modelo. Aquí definimos la tarea de forma explícita:
    # evaluar la salida de un corrector gramatical en gallego.
    # El modelo debe mirar el input_corrector y el output_corrector, y devolver solo 0 o 1.
    return (
        "Tes que avaliar a saída dun corrector gramatical en galego.\n\n"
        "Criterios:\n"
        "- Devolve 0 se a corrección é válida: o output_corrector corrixe os erros do input_corrector, "
        "ou ben o input_corrector xa era correcto e o output_corrector non introduce erros novos nin sobrecorreccións.\n"
        "- Devolve 1 se a corrección é incorrecta: o output_corrector non corrixe erros presentes no input_corrector, "
        "ou ben introduce erros novos ou sobrecorreccións cando o input_corrector xa era correcto.\n\n"
        "Restricións:\n"
        "- Avalía unicamente se a corrección final é válida ou non en galego.\n"
        "- Non modifiques nin corrixas o input_corrector nin o output_corrector.\n"
        "- Responde exclusivamente con 0 ou 1.\n\n"
        f'input_corrector: "{example["input_corrector"]}"\n'
        f'output_corrector: "{example["output_corrector"]}"'
    )


def build_messages(example):
    # Convertimos cada ejemplo al formato de conversación que espera un modelo instruccional.
    # En el turno de usuario ponemos el prompt con la tarea y los textos que debe evaluar.
    # En el turno de asistente ponemos la etiqueta correcta del dataset, que es lo que el
    # modelo debe aprender a producir.
    return [
        {"role": "user", "content": build_user_prompt(example)},
        {"role": "assistant", "content": str(example["tag"])},
    ]


def tokenize_example(example, tokenizer, max_length):
    # Tokenizamos cada ejemplo para convertir el texto en ids numéricos que el modelo pueda procesar.
    # Para controlar bien el aprendizaje, construimos dos versiones del texto:
    # una con solo el prompt del usuario y otra con el diálogo completo, incluyendo la respuesta esperada.
    user_messages = [{"role": "user", "content": build_user_prompt(example)}]
    full_messages = build_messages(example)

    # Aplicamos la plantilla de chat del modelo solo al mensaje del usuario.
    # add_generation_prompt=True añade la marca que indica que a partir de ahí debe responder el asistente.
    user_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Aplicamos la plantilla de chat al diálogo completo: usuario más respuesta correcta.
    # Esta es la secuencia completa que vamos a usar como entrada del entrenamiento.
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenizamos el prompt del usuario. Esta versión nos servirá para saber cuántos tokens
    # pertenecen a las instrucciones y al par input/output.
    user_tokens = tokenizer(
        user_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )

    # Tokenizamos el diálogo completo, que incluye también la etiqueta esperada.
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )

    # Extraemos los ids de entrada y la máscara de atención.
    # input_ids contiene los tokens numéricos.
    # attention_mask indica qué posiciones son tokens reales y cuáles son padding.
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Las labels son las salidas esperadas durante el entrenamiento.
    # Partimos de una copia de input_ids porque, en modelos causales, el modelo aprende
    # a predecir el siguiente token a partir de los anteriores.
    labels = input_ids.copy()

    # Calculamos cuántos tokens corresponden al prompt del usuario.
    user_len = len(user_tokens["input_ids"])

    # Enmascaramos los tokens del prompt con -100.
    # En PyTorch y Hugging Face, el valor -100 indica que esos tokens no deben contribuir
    # al cálculo de la pérdida. Así evitamos que el modelo aprenda a copiar las instrucciones
    # y centramos el aprendizaje en la respuesta del asistente, es decir, en la etiqueta 0 o 1.
    for i in range(min(user_len, len(labels))):
        labels[i] = -100

    # Devolvemos el ejemplo tokenizado con el formato que espera el Trainer.
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def print_distribution(name, data):
    # Imprimimos la distribución del dataset para comprobar cuántos ejemplos tenemos
    # por etiqueta, por caso y por combinación de caso y etiqueta.
    # Esto nos ayuda a detectar desequilibrios antes y después de hacer los splits.
    tag_counter = Counter(x["tag"] for x in data)
    case_counter = Counter(x["case"] for x in data)
    combo_counter = Counter((x["case"], x["tag"]) for x in data)

    print(f"\n===== {name} =====")
    print(f"Total: {len(data)}")
    print(f"Tags: {dict(tag_counter)}")
    print(f"Cases: {dict(case_counter)}")
    print(f"Case x Tag: {dict(sorted(combo_counter.items()))}")


def save_jsonl(records, path):
    # Guardamos una lista de ejemplos en formato JSONL.
    # Esto nos permite conservar los splits generados por el script y reutilizarlos más tarde
    # si queremos revisar qué ejemplos quedaron en train, validation o test.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            # Eliminamos la etiqueta auxiliar de estratificación antes de guardar los ejemplos,
            # porque no forma parte del dataset original ni de la tarea que queremos modelar.
            clean_record = {k: v for k, v in record.items() if k != "stratify_label"}
            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")


def build_training_arguments(args):
    # Construimos los argumentos de entrenamiento de Hugging Face.
    # Usamos inspect para comprobar qué parámetros acepta la versión instalada de transformers.
    # Esto hace que el script sea más robusto ante diferencias entre versiones, por ejemplo
    # evaluation_strategy en unas versiones y eval_strategy en otras.
    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    # Definimos la configuración principal del entrenamiento.
    kwargs = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "seed": args.seed,
    }

    # Si tenemos GPU disponible, activamos fp16 para trabajar en media precisión.
    # Esto reduce el consumo de memoria y suele acelerar el entrenamiento.
    if torch.cuda.is_available():
        kwargs["fp16"] = True

    # Configuramos la evaluación por pasos. Mantenemos compatibilidad con distintas versiones
    # de transformers, que pueden usar evaluation_strategy o eval_strategy.
    if "evaluation_strategy" in valid_params:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in valid_params:
        kwargs["eval_strategy"] = "steps"

    # Indicamos que los checkpoints también se guarden por pasos.
    if "save_strategy" in valid_params:
        kwargs["save_strategy"] = "steps"

    # Definimos cada cuántos pasos se evalúa el modelo en el conjunto de validación.
    if "eval_steps" in valid_params:
        kwargs["eval_steps"] = args.eval_steps

    # Pedimos que al final del entrenamiento se cargue automáticamente el mejor checkpoint.
    if "load_best_model_at_end" in valid_params:
        kwargs["load_best_model_at_end"] = True

    # Usamos la pérdida de validación como criterio para elegir el mejor checkpoint.
    if "metric_for_best_model" in valid_params:
        kwargs["metric_for_best_model"] = "eval_loss"

    # Como la pérdida debe minimizarse, indicamos que un valor menor es mejor.
    if "greater_is_better" in valid_params:
        kwargs["greater_is_better"] = False

    # Filtramos los argumentos para quedarnos solo con los que acepta nuestra versión concreta
    # de TrainingArguments. Así evitamos errores por incompatibilidades entre versiones.
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return TrainingArguments(**filtered_kwargs)


def main(args):
    # Fijamos las semillas para que la partición del dataset y el entrenamiento sean más reproducibles.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\nCargando dataset...")

    # Cargamos el Dataset A desde el archivo JSONL que indiquemos al ejecutar el script.
    data = load_jsonl(args.dataset)

    # Definimos los campos que debe tener obligatoriamente cada ejemplo.
    # Estos campos reflejan la estructura que necesitamos para entrenar el juez:
    # identificadores, texto original, corrección propuesta y etiqueta 0/1.
    required_fields = {
        "case",
        "absolute_id",
        "pair_id_from_case",
        "input_corrector",
        "output_corrector",
        "tag",
    }

    # Comprobamos ejemplo por ejemplo que no falte ningún campo obligatorio.
    # Si falta algo, detenemos el proceso para no entrenar con datos incompletos.
    for i, ex in enumerate(data, start=1):
        missing = required_fields - set(ex.keys())
        if missing:
            raise ValueError(f"O exemplo {i} non ten os campos obrigatorios: {missing}")

    # Añadimos a cada ejemplo la etiqueta auxiliar de estratificación.
    # Esta etiqueta no se usa como salida del modelo, solo para dividir mejor los datos.
    for ex in data:
        ex["stratify_label"] = make_stratify_label(ex)

    # Imprimimos la distribución del dataset completo antes de hacer la división.
    print_distribution("DATASET COMPLETO", data)

    # Preparamos la lista de etiquetas de estratificación para el primer split.
    stratify_labels = [x["stratify_label"] for x in data]

    # Primero separamos el conjunto de entrenamiento del conjunto temporal.
    # Con val_size=0.2, dejamos aproximadamente un 80 por ciento para entrenamiento
    # y un 20 por ciento para dividir después entre validación y prueba.
    train_data, temp_data = train_test_split(
        data,
        test_size=args.val_size,
        stratify=stratify_labels,
        random_state=args.seed,
        shuffle=True
    )

    # Volvemos a obtener las etiquetas de estratificación, pero ahora solo para el conjunto temporal.
    temp_stratify_labels = [x["stratify_label"] for x in temp_data]

    # Dividimos el conjunto temporal en dos partes iguales: validación y test.
    # Como el conjunto temporal era el 20 por ciento, al dividirlo en dos obtenemos
    # aproximadamente 10 por ciento para validación y 10 por ciento para prueba.
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        stratify=temp_stratify_labels,
        random_state=args.seed,
        shuffle=True
    )

    # Imprimimos las distribuciones de los tres subconjuntos para revisar que el split
    # haya conservado razonablemente la proporción de casos y etiquetas.
    print_distribution("TRAIN", train_data)
    print_distribution("VALIDATION", val_data)
    print_distribution("TEST", test_data)

    print("\nCargando tokenizer...")

    # Cargamos el tokenizador correspondiente al modelo base.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Algunos modelos no tienen un token específico de padding.
    # En ese caso usamos el token de fin de secuencia como token de padding para poder
    # crear lotes con secuencias de distinta longitud.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo...")

    # Usamos device_map='auto' para que transformers coloque automáticamente el modelo
    # en los dispositivos disponibles, normalmente la GPU.
    model_kwargs = {
        "device_map": "auto",
    }

    # Si activamos --load_in_4bit, cargamos el modelo cuantizado a 4 bits.
    # Esto reduce mucho el uso de memoria. En este caso usamos cuantización nf4,
    # doble cuantización y cálculo en float16.
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["dtype"] = torch.float16
    else:
        # Si no usamos cuantización a 4 bits, cargamos igualmente el modelo en float16
        # para reducir memoria frente a float32.
        model_kwargs["dtype"] = torch.float16

    # Cargamos el modelo causal de lenguaje a partir del nombre o ruta que indiquemos.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs
    )

    # Si el modelo se ha cargado en 4 bits, lo preparamos para entrenamiento con PEFT.
    # Esta preparación ajusta algunos detalles internos para que el fine-tuning sea estable.
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configuramos LoRA. Aquí definimos cuántos parámetros adicionales vamos a entrenar
    # y en qué módulos del modelo los vamos a introducir.
    # r controla el rango de las matrices LoRA.
    # lora_alpha actúa como factor de escala.
    # lora_dropout aplica regularización durante el entrenamiento.
    # target_modules indica en qué proyecciones de atención se insertan los adaptadores.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Aplicamos LoRA al modelo base. A partir de aquí, el entrenamiento actualiza
    # principalmente los parámetros añadidos por LoRA, no todos los pesos originales del modelo.
    model = get_peft_model(model, lora_config)

    # Imprimimos cuántos parámetros son entrenables para comprobar que realmente estamos
    # haciendo fine-tuning eficiente y no entrenamiento completo del modelo.
    model.print_trainable_parameters()

    print("\nTokenizando datasets...")

    # Convertimos las listas de Python a objetos Dataset de Hugging Face.
    # Solo tokenizamos train y validation porque son los subconjuntos que se usan
    # durante el entrenamiento y la selección del checkpoint.
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Tokenizamos el conjunto de entrenamiento y eliminamos las columnas originales,
    # porque el Trainer solo necesita input_ids, attention_mask y labels.
    train_ds = train_ds.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=train_ds.column_names
    )

    # Tokenizamos el conjunto de validación del mismo modo.
    val_ds = val_ds.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=val_ds.column_names
    )

    # Creamos el data collator, que se encarga de formar lotes y aplicar padding dinámico.
    # Esto permite agrupar ejemplos de distinta longitud en un mismo batch.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Construimos los argumentos de entrenamiento.
    training_args = build_training_arguments(args)

    # Preparamos los argumentos que necesita el Trainer.
    # train_dataset se usa para ajustar el modelo.
    # eval_dataset se usa para calcular la pérdida de validación durante el entrenamiento.
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
    }

    # De nuevo usamos inspect para mantener compatibilidad entre versiones de transformers.
    # En algunas versiones el Trainer recibe tokenizer, y en otras puede recibir processing_class.
    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_valid_params = set(trainer_sig.parameters.keys())

    if "tokenizer" in trainer_valid_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_valid_params:
        trainer_kwargs["processing_class"] = tokenizer

    # Creamos el objeto Trainer, que gestionará el entrenamiento, la evaluación y los checkpoints.
    trainer = Trainer(**trainer_kwargs)

    # Definimos la carpeta donde vamos a guardar los splits generados.
    split_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # Guardamos los tres subconjuntos. Aunque este script solo usa train y validation para entrenar,
    # también guardamos test para dejarlo reservado como posible evaluación interna posterior.
    save_jsonl(train_data, os.path.join(split_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(split_dir, "validation.jsonl"))
    save_jsonl(test_data, os.path.join(split_dir, "test.jsonl"))

    print("\nComeza o adestramento...")

    # Lanzamos el entrenamiento.
    trainer.train()

    print("\nGardando modelo...")

    # Guardamos el modelo entrenado en el directorio de salida.
    # Si hemos usado LoRA, se guardan los adaptadores entrenados junto con la configuración necesaria.
    trainer.save_model(args.output_dir)

    # Guardamos también el tokenizador para poder reutilizar el modelo posteriormente
    # con la misma configuración de tokenización.
    tokenizer.save_pretrained(args.output_dir)

    print("\nListo.")
    print(f"Modelo gardado en: {args.output_dir}")
    print(f"Splits gardados en: {split_dir}")


if __name__ == "__main__":
    # Definimos los argumentos que podemos pasar al script desde la terminal.
    # Esto nos permite reutilizar el mismo código cambiando rutas, modelo o hiperparámetros
    # sin modificar directamente el archivo Python.
    parser = argparse.ArgumentParser(
        description="LoRA de Selene-1-Mini para LLM-as-a-judge de corrección gramatical en galego."
    )

    # Ruta del dataset de entrada y directorio donde se guardará el modelo entrenado.
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)

    # Modelo base sobre el que hacemos el fine-tuning. Por defecto usamos Selene-1-Mini.
    parser.add_argument(
        "--model",
        default="AtlaAI/Selene-1-Mini-Llama-3.1-8B"
    )

    # Porcentaje que separamos inicialmente para validación y test.
    # Con 0.2, el 80 por ciento queda para train y el 20 por ciento restante se divide luego
    # en validation y test.
    parser.add_argument("--val_size", type=float, default=0.2)

    # Semilla para hacer más reproducibles los splits y el entrenamiento.
    parser.add_argument("--seed", type=int, default=42)

    # Longitud máxima de las secuencias tokenizadas.
    parser.add_argument("--max_length", type=int, default=512)

    # Hiperparámetros principales del entrenamiento.
    # epochs indica el número máximo de épocas.
    # batch_size indica cuántos ejemplos procesa cada dispositivo en cada paso.
    # gradient_accumulation_steps permite acumular gradientes durante varios pasos antes
    # de actualizar los pesos, aumentando así el batch efectivo sin requerir tanta memoria.
    # learning_rate controla la magnitud de las actualizaciones del modelo.
    # weight_decay y warmup_ratio son parámetros habituales de regularización y planificación
    # de la tasa de aprendizaje.
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # Frecuencia con la que registramos logs, evaluamos en validación y guardamos checkpoints.
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)

    # Si usamos esta opción al ejecutar el script, cargamos el modelo en 4 bits.
    parser.add_argument("--load_in_4bit", action="store_true")

    # Parámetros específicos de LoRA.
    # Estos valores controlan el tamaño y comportamiento de los adaptadores entrenables.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Módulos del modelo en los que introducimos los adaptadores LoRA.
    # En este caso se aplican a las proyecciones de atención q, k, v y o.
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj"
    )

    # Parseamos los argumentos de la terminal y ejecutamos la función principal.
    args = parser.parse_args()
    main(args)
