#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importamos librerías estándar para gestionar rutas, leer archivos, interpretar argumentos y controlar la reproducibilidad.
import os
import json
import argparse
import random
import inspect
from collections import Counter

# Importamos las librerías principales de aprendizaje automático y tratamiento de datasets.
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Importamos los componentes de Transformers que utilizamos para cargar el modelo, tokenizar y entrenar.
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)

# Importamos PEFT para aplicar LoRA sin tener que ajustar todos los parámetros del modelo base.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Definimos una función auxiliar para cargar el dataset en formato JSONL.
# Leemos línea a línea para mantener el formato original de los ejemplos y detectar errores de lectura con precisión.
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except Exception as e:
                raise ValueError(f"Erro ao ler a liña {line_num} de {path}: {e}")
    return records


# Construimos una etiqueta compuesta para estratificar teniendo en cuenta tanto el caso como la clase.
# Así evitamos que la división train/validación/test pierda la distribución de los tipos de ejemplo.
def make_stratify_label(example):
    return f"{example['case']}_{example['tag']}"


# Generamos el prompt que le presentamos al modelo como usuario.
# Aquí formulamos explícitamente la tarea de evaluación y restringimos la respuesta a una etiqueta binaria.
def build_user_prompt(example):
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


# Encapsulamos el prompt y la etiqueta esperada en el formato de conversación que usa el modelo instruccional.
# De este modo, el ejemplo de entrenamiento reproduce la estructura user/assistant.
def build_messages(example):
    return [
        {"role": "user", "content": build_user_prompt(example)},
        {"role": "assistant", "content": str(example["tag"])},
    ]


# Tokenizamos cada ejemplo separando la parte del usuario de la respuesta completa.
# Lo hacemos así para poder enmascarar la pérdida sobre el prompt y entrenar solo la respuesta del assistant.
def tokenize_example(example, tokenizer, max_length):
    user_messages = [{"role": "user", "content": build_user_prompt(example)}]
    full_messages = build_messages(example)

    # Primero construimos solo el texto del usuario, añadiendo el marcador de generación para saber dónde empieza la respuesta.
    user_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Después construimos la conversación completa, incluyendo ya la etiqueta correcta como respuesta del assistant.
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenizamos el fragmento del usuario para calcular cuántos tokens debemos ignorar en la función de pérdida.
    user_tokens = tokenizer(
        user_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )

    # Tokenizamos la conversación completa, que es la secuencia real que le pasamos al modelo.
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Copiamos los input_ids como labels porque, en un modelo causal, la salida esperada es la propia secuencia desplazada internamente.
    labels = input_ids.copy()
    user_len = len(user_tokens["input_ids"])

    # Enmascaramos la parte del prompt con -100 para que el modelo no aprenda a predecir el texto del usuario, sino solo la etiqueta final.
    for i in range(min(user_len, len(labels))):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# Imprimimos la distribución de clases y casos para documentar que las particiones quedan equilibradas.
def print_distribution(name, data):
    tag_counter = Counter(x["tag"] for x in data)
    case_counter = Counter(x["case"] for x in data)
    combo_counter = Counter((x["case"], x["tag"]) for x in data)

    print(f"\n===== {name} =====")
    print(f"Total: {len(data)}")
    print(f"Tags: {dict(tag_counter)}")
    print(f"Cases: {dict(case_counter)}")
    print(f"Case x Tag: {dict(sorted(combo_counter.items()))}")


# Guardamos los splits en un JSONL limpio, eliminando la etiqueta auxiliar de estratificación.
# Esta etiqueta solo la usamos internamente y no forma parte de los datos experimentales originales.
def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            clean_record = {k: v for k, v in record.items() if k != "stratify_label"}
            f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")


# Construimos los argumentos de entrenamiento de forma compatible con distintas versiones de Transformers.
# Por eso inspeccionamos la firma de TrainingArguments y solo pasamos parámetros admitidos.
def build_training_arguments(args):
    sig = inspect.signature(TrainingArguments.__init__)
    valid_params = set(sig.parameters.keys())

    # Reunimos la configuración principal de entrenamiento en un diccionario antes de filtrarla.
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

    # Si tenemos GPU disponible, activamos fp16 para reducir memoria y acelerar el entrenamiento.
    if torch.cuda.is_available():
        kwargs["fp16"] = True

    # Adaptamos el nombre del parámetro de evaluación según la versión instalada de Transformers.
    if "evaluation_strategy" in valid_params:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in valid_params:
        kwargs["eval_strategy"] = "steps"

    if "save_strategy" in valid_params:
        kwargs["save_strategy"] = "steps"

    if "eval_steps" in valid_params:
        kwargs["eval_steps"] = args.eval_steps

    if "load_best_model_at_end" in valid_params:
        kwargs["load_best_model_at_end"] = True

    if "metric_for_best_model" in valid_params:
        kwargs["metric_for_best_model"] = "eval_loss"

    if "greater_is_better" in valid_params:
        kwargs["greater_is_better"] = False

    # Filtramos los argumentos para evitar errores cuando alguna versión de la librería no acepta un parámetro concreto.
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return TrainingArguments(**filtered_kwargs)


# Definimos el flujo principal: cargamos datos, generamos splits, preparamos modelo/tokenizer, entrenamos y guardamos resultados.
def main(args):
    # Fijamos las semillas para que la división de los datos y el entrenamiento sean lo más reproducibles posible.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\nCargando dataset...")
    # Cargamos el dataset de entrada desde la ruta indicada por línea de comandos.
    data = load_jsonl(args.dataset)

    # Declaramos los campos mínimos que debe tener cada ejemplo para que la tarea esté bien definida.
    required_fields = {
        "case",
        "absolute_id",
        "pair_id_from_case",
        "input_corrector",
        "output_corrector",
        "tag",
    }

    # Comprobamos explícitamente que todos los ejemplos tienen la estructura esperada antes de continuar.
    for i, ex in enumerate(data, start=1):
        missing = required_fields - set(ex.keys())
        if missing:
            raise ValueError(f"O exemplo {i} non ten os campos obrigatorios: {missing}")

    # Añadimos una etiqueta auxiliar case_tag para poder estratificar por tipo de caso y por etiqueta.
    for ex in data:
        ex["stratify_label"] = make_stratify_label(ex)

    print_distribution("DATASET COMPLETO", data)

    stratify_labels = [x["stratify_label"] for x in data]

    # Primero separamos el conjunto de entrenamiento de un conjunto temporal de validación+test.
    train_data, temp_data = train_test_split(
        data,
        test_size=args.val_size,
        stratify=stratify_labels,
        random_state=args.seed,
        shuffle=True
    )

    temp_stratify_labels = [x["stratify_label"] for x in temp_data]

    # Después dividimos ese conjunto temporal en dos mitades: validación y test.
    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        stratify=temp_stratify_labels,
        random_state=args.seed,
        shuffle=True
    )

    print_distribution("TRAIN", train_data)
    print_distribution("VALIDATION", val_data)
    print_distribution("TEST", test_data)

    print("\nCargando tokenizer...")
    # Cargamos el tokenizer asociado al modelo base.
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Si el tokenizer no tiene token de padding, reutilizamos el eos_token para poder formar batches con padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo...")

    # Preparamos los argumentos de carga del modelo y dejamos que Transformers reparta automáticamente el modelo entre los dispositivos disponibles.
    model_kwargs = {
        "device_map": "auto",
    }

    # Cuando usamos cuantización, preparamos el modelo para entrenamiento k-bit antes de aplicar LoRA.
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
        model_kwargs["dtype"] = torch.float16

    # Cargamos el modelo causal base sobre el que aplicaremos LoRA.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs
    )

    # Cuando usamos cuantización, preparamos el modelo para entrenamiento k-bit antes de aplicar LoRA.
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Definimos la configuración LoRA: rango, alpha, dropout y módulos donde insertamos las matrices adaptadoras.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Envolvemos el modelo base con el adaptador LoRA y dejamos entrenables solo los parámetros correspondientes.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nTokenizando datasets...")

    # Convertimos las listas de ejemplos en objetos Dataset para integrarnos con el Trainer.
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    # Tokenizamos el split de entrenamiento y retiramos las columnas textuales originales porque el Trainer ya no las necesita.
    train_ds = train_ds.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=train_ds.column_names
    )

    # Hacemos lo mismo con el split de validación, que se empleará para calcular la pérdida de evaluación.
    val_ds = val_ds.map(
        lambda x: tokenize_example(x, tokenizer, args.max_length),
        remove_columns=val_ds.column_names
    )

    # Usamos un data collator que aplica padding dinámico dentro de cada batch.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Generamos los argumentos finales de entrenamiento a partir de los parámetros recibidos por línea de comandos.
    training_args = build_training_arguments(args)

    # Preparamos los argumentos con los que inicializamos el Trainer.
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": data_collator,
    }

    # Comprobamos la firma del Trainer para usar tokenizer o processing_class según la versión de Transformers.
    trainer_sig = inspect.signature(Trainer.__init__)
    trainer_valid_params = set(trainer_sig.parameters.keys())

    if "tokenizer" in trainer_valid_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_valid_params:
        trainer_kwargs["processing_class"] = tokenizer

    # Inicializamos el Trainer, que se encargará del bucle de entrenamiento, evaluación y guardado de checkpoints.
    trainer = Trainer(**trainer_kwargs)

    # Guardamos los splits usados en este experimento junto con la salida del modelo para poder reproducir la partición exacta.
    split_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    # Exportamos las particiones finales: train, validation y test.
    save_jsonl(train_data, os.path.join(split_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(split_dir, "validation.jsonl"))
    save_jsonl(test_data, os.path.join(split_dir, "test.jsonl"))

    print("\nComeza o adestramento...")
    # Lanzamos el entrenamiento LoRA.
    trainer.train()

    print("\nGardando modelo...")
    # Guardamos el adaptador LoRA y la configuración resultante en el directorio de salida.
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\nListo.")
    print(f"Modelo gardado en: {args.output_dir}")
    print(f"Splits gardados en: {split_dir}")


# Definimos la interfaz por línea de comandos para poder lanzar el experimento de forma parametrizada.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LoRA de Selene-1-Mini para LLM-as-a-judge de corrección gramatical en galego."
    )

    # Indicamos los dos argumentos obligatorios: dataset de entrada y directorio donde guardamos el modelo.
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--model",
        default="AtlaAI/Selene-1-Mini-Llama-3.1-8B"
    )

    # Controlamos la proporción que reservamos inicialmente para validación+test.
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Fijamos la longitud máxima de contexto para evitar secuencias demasiado largas.
    parser.add_argument("--max_length", type=int, default=512)

    # Definimos los hiperparámetros principales de entrenamiento.
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # Configuramos la frecuencia de logging, evaluación y guardado de checkpoints.
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)

    # Añadimos la opción de cargar el modelo en 4-bit cuando queremos reducir memoria.
    parser.add_argument("--load_in_4bit", action="store_true")

    # Definimos los hiperparámetros propios de LoRA.
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj"
    )

    # Leemos los argumentos de línea de comandos y ejecutamos el flujo principal.
    args = parser.parse_args()
    main(args)
