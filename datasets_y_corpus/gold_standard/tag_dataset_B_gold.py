#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd


def parse_args():
    """
    Aquí definimos los argumentos del script.
    Parametrizado, pero con mis rutas  puestas por defecto.
    """
    parser = argparse.ArgumentParser(
        description="Etiquetar dataset B usando el gold standard de pairs_dataset_B.json."
    )

    parser.add_argument(
        "--input_xlsx",
        default="/home/compartido/lua/CORTEGAL_splits/dataset_B.xlsx",
        help="Ruta al Excel de entrada con las columnas Orixinal y Corrección."
    )

    parser.add_argument(
        "--gold_json",
        default="/home/compartido/lua/CORTEGAL_splits/pairs_dataset_B.json",
        help="Ruta al JSON con los pares gold."
    )

    parser.add_argument(
        "--output_dir",
        default="/home/compartido/lua/CORTEGAL_splits",
        help="Carpeta donde se guardarán los ficheros de salida."
    )

    parser.add_argument(
        "--output_name",
        default="dataset_B_tagged_gold",
        help="Nombre base de los ficheros de salida, sin extensión."
    )

    return parser.parse_args()


def cargar_excel(input_xlsx):
    """
    Aquí leemos el Excel de entrada y comprobamos que tenga exactamente
    las columnas que necesitamos para trabajar.

    Esperamos:
    - Orixinal
    - Corrección

    No renombramos todavía. Primero validamos y luego ya creamos la salida
    con los nombres nuevos.
    """
    df = pd.read_excel(input_xlsx)

    columnas_necesarias = {"Orixinal", "Corrección"}
    columnas_actuales = set(df.columns)

    if not columnas_necesarias.issubset(columnas_actuales):
        raise ValueError(
            "El Excel de entrada debe contener las columnas 'Orixinal' y 'Corrección'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    return df


def cargar_json_gold(gold_json):
    """
    Aquí cargamos el JSON gold.

    Esperamos una lista de objetos, y cada objeto debe tener al menos:
    - incorrect
    - correct_standard

    Si además existe pair_id, lo usaremos para ordenar por seguridad.
    """
    with open(gold_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El JSON gold no tiene formato de lista.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"El elemento {i} del JSON no es un objeto válido.")
        if "incorrect" not in item or "correct_standard" not in item:
            raise ValueError(
                f"El elemento {i} del JSON no contiene las claves 'incorrect' y 'correct_standard'."
            )

    if all("pair_id" in item for item in data):
        data = sorted(data, key=lambda x: x["pair_id"])

    return data


def normalizar_valor(x):
    """
    Convertimos valores vacíos o NaN a cadena vacía y el resto a string.
    No hacemos ninguna limpieza adicional. 
    Esto es importante para hacer una comparación estricta: si hay cualquier diferencia, la tag debe ser 1.
    """
    if pd.isna(x):
        return ""
    return str(x)


def etiquetar_por_orden(df_excel, gold_data):
    """
    En vez de emparejar por diccionario usando 'incorrect' como clave,
    trabajamos por posición:

    - fila 1 del Excel con elemento 1 del JSON
    - fila 2 del Excel con elemento 2 del JSON
    - etc.

    Pero, además, validamos que: Orixinal del Excel == incorrect del JSON
    Si en alguna fila no coinciden exactamente, el script se detiene.

    Después, asignamos la etiqueta:
    - 0 si Corrección == correct_standard exactamente
    - 1 si hay al menos una sola diferencia

    Además, ahora añadimos también la columna gold_standard
    en la salida, colocada antes de la tag.
    """
    if len(df_excel) != len(gold_data):
        raise ValueError(
            "El número de filas del Excel y el número de pares del JSON no coincide.\n"
            f"Filas en Excel: {len(df_excel)}\n"
            f"Pares en JSON: {len(gold_data)}"
        )

    filas_salida = []

    for i in range(len(df_excel)):
        fila_excel = df_excel.iloc[i]
        fila_gold = gold_data[i]

        input_corrector = normalizar_valor(fila_excel["Orixinal"])
        output_corrector = normalizar_valor(fila_excel["Corrección"])

        gold_incorrect = normalizar_valor(fila_gold["incorrect"])
        gold_standard = normalizar_valor(fila_gold["correct_standard"])

        if input_corrector != gold_incorrect:
            raise ValueError(
                "Desajuste entre el Excel y el JSON en la misma posición.\n"
                f"Fila Excel/JSON: {i + 1}\n"
                f"Orixinal Excel: {repr(input_corrector)}\n"
                f"incorrect JSON: {repr(gold_incorrect)}"
            )

        if output_corrector == gold_standard:
            tag = 0
        else:
            tag = 1

        filas_salida.append({
            "input_corrector": input_corrector,
            "output_corrector": output_corrector,
            "gold_standard": gold_standard,
            "tag": tag
        })

    df_out = pd.DataFrame(filas_salida)
    return df_out


def guardar_salida(df_out, output_dir, output_name):
    """
    Guardamos la salida en Excel y CSV.
    El output tendrá exactamente estas columnas:
    - input_corrector
    - output_corrector
    - gold_standard
    - tag
    """
    os.makedirs(output_dir, exist_ok=True)

    output_xlsx = os.path.join(output_dir, f"{output_name}.xlsx")
    output_csv = os.path.join(output_dir, f"{output_name}.csv")

    df_out = df_out[["input_corrector", "output_corrector", "gold_standard", "tag"]]

    df_out.to_excel(output_xlsx, index=False)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Excel guardado en: {output_xlsx}")
    print(f"CSV guardado en:   {output_csv}")


def main():
    """
    Flujo general del script:

    1. leemos argumentos
    2. cargamos el Excel
    3. cargamos el JSON gold
    4. comprobamos que ambos tienen el mismo número de elementos
    5. comparamos fila a fila
    6. asignamos tag 0 o 1 de forma estricta
    7. guardamos el resultado en Excel y CSV con el formato de salida que acabamos de explicar
    """
    args = parse_args()

    print("Leyendo Excel de entrada...")
    df_excel = cargar_excel(args.input_xlsx)

    print("Leyendo JSON gold...")
    gold_data = cargar_json_gold(args.gold_json)

    print("Etiquetando filas...")
    df_out = etiquetar_por_orden(df_excel, gold_data)

    print("Guardando salida...")
    guardar_salida(df_out, args.output_dir, args.output_name)

    print("Proceso terminado correctamente.")


if __name__ == "__main__":
    main()