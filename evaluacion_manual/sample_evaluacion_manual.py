#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Muestrea instancias aleatorias del dataset Selene FT y añade información del gold."
    )
    parser.add_argument("--base", type=str, required=True, help="Excel con tags de Selene fine-tuned")
    parser.add_argument("--gold", type=str, required=True, help="Excel con tags gold")
    parser.add_argument("--output", type=str, required=True, help="Excel de salida")
    parser.add_argument("--n", type=int, default=1, help="Número de instancias a muestrear")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument(
        "--join_cols",
        nargs="+",
        default=["input_corrector", "output_corrector"],
        help="Columnas con las que unir ambos archivos"
    )
    parser.add_argument(
        "--base_tag_col",
        type=str,
        default="tag",
        help="Nombre de la columna tag en el archivo base"
    )
    parser.add_argument(
        "--gold_tag_col",
        type=str,
        default="tag",
        help="Nombre de la columna tag en el archivo gold"
    )

    args = parser.parse_args()

    base_df = pd.read_excel(args.base)
    gold_df = pd.read_excel(args.gold)

    for col in args.join_cols:
        if col not in base_df.columns:
            raise ValueError(f"La columna '{col}' no existe en el archivo base.")
        if col not in gold_df.columns:
            raise ValueError(f"La columna '{col}' no existe en el archivo gold.")

    if args.base_tag_col not in base_df.columns:
        raise ValueError(f"La columna '{args.base_tag_col}' no existe en el archivo base.")

    if args.gold_tag_col not in gold_df.columns:
        raise ValueError(f"La columna '{args.gold_tag_col}' no existe en el archivo gold.")

    if "gold_standard" not in gold_df.columns:
        raise ValueError("La columna 'gold_standard' no existe en el archivo gold.")

    # --- GOLD ---
    gold_info = gold_df[args.join_cols + ["gold_standard", args.gold_tag_col]].copy()
    gold_info = gold_info.rename(columns={args.gold_tag_col: "tag_gold"})

    # --- MERGE ---
    merged_df = base_df.merge(gold_info, on=args.join_cols, how="left")

    if merged_df["tag_gold"].isna().sum() > 0:
        raise ValueError("Hay filas sin correspondencia en el gold.")

    # --- RENOMBRAR TAG ---
    merged_df = merged_df.rename(columns={args.base_tag_col: "tag_selene_ft"})

    # --- COLUMNA TP/TN/FP/FN (vacía) ---
    merged_df["tag_evaluacion"] = ""

    # --- COLUMNAS DE EVALUACIÓN (EN CASTELLANO) ---
    eval_cols = [
        "consistencia",
        "correccion_gramatical",
        "naturalidad",
        "correccion_factual",
    ]

    for col in eval_cols:
        merged_df[col] = ""

    # --- ORDENAR COLUMNAS ---
    cols = list(merged_df.columns)

    # gold_standard después de output_corrector
    cols.remove("gold_standard")
    idx = cols.index("output_corrector")
    cols.insert(idx + 1, "gold_standard")

    # tag_gold después de tag_selene_ft
    cols.remove("tag_gold")
    idx = cols.index("tag_selene_ft")
    cols.insert(idx + 1, "tag_gold")

    # tag_evaluacion después de tag_gold
    cols.remove("tag_evaluacion")
    idx = cols.index("tag_gold")
    cols.insert(idx + 1, "tag_evaluacion")

    # columnas de evaluación después de explanation
    for col in eval_cols:
        cols.remove(col)

    idx = cols.index("explanation")
    for i, col in enumerate(eval_cols, start=1):
        cols.insert(idx + i, col)

    merged_df = merged_df[cols]

    # --- SAMPLE ---
    if args.n > len(merged_df):
        raise ValueError(f"Se pidieron {args.n} instancias pero solo hay {len(merged_df)}.")

    sample_df = merged_df.sample(n=args.n, random_state=args.seed)

    # --- GUARDAR ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sample_df.to_excel(args.output, index=False)

    print("✔ Muestra creada correctamente")
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()