#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
)


def parse_args():
    """
    Aquí definimos los argumentos del script.

    En esta versión:
    - el análisis se hace únicamente con la columna tag
    - no se usan las explanations en ningún momento
    """
    parser = argparse.ArgumentParser(
        description="Comparar dos sistemas de etiquetado binario contra un gold usando solo la columna tag."
    )

    parser.add_argument(
        "--gold_file",
        required=True,
        help="Ruta al fichero gold con las columnas input_corrector, output_corrector y tag."
    )

    parser.add_argument(
        "--base_file",
        required=True,
        help="Ruta al fichero de predicciones del modelo base."
    )

    parser.add_argument(
        "--finetuned_file",
        required=True,
        help="Ruta al fichero de predicciones del modelo fine-tuneado."
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Carpeta donde se guardarán todos los resultados."
    )

    parser.add_argument(
        "--output_name",
        default="comparacion_modelos_vs_gold",
        help="Nombre base de los ficheros de salida, sin extensión."
    )

    return parser.parse_args()


def leer_tabla(path):
    """
    Aquí leemos automáticamente un Excel o un CSV según la extensión.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(
            f"No reconozco el formato del archivo: {path}. "
            "Usa .xlsx, .xls o .csv."
        )


def normalizar_tag_series(series, nombre_archivo):
    """
    Aquí convertimos la columna tag a enteros 0/1 y comprobamos
    que no haya valores raros.
    """
    serie = pd.to_numeric(series, errors="coerce")

    if serie.isna().any():
        filas_problematicas = serie[serie.isna()].index.tolist()
        raise ValueError(
            f"Hay tags no válidas o vacías en el archivo {nombre_archivo}. "
            f"Filas problemáticas: {filas_problematicas[:10]}"
        )

    serie = serie.astype(int)

    valores_validos = set(serie.unique().tolist())
    if not valores_validos.issubset({0, 1}):
        raise ValueError(
            f"En el archivo {nombre_archivo} hay tags distintas de 0 y 1: {valores_validos}"
        )

    return serie


def cargar_y_validar(path):
    """
    Aquí cargamos cada fichero y comprobamos que tenga las columnas mínimas. Para este análisis necesitamos:
    - input_corrector
    - output_corrector
    - tag

    No usamos explanation, así que no la exigimos.
    """
    df = leer_tabla(path)

    columnas_necesarias = {"input_corrector", "output_corrector", "tag"}
    columnas_actuales = set(df.columns)

    if not columnas_necesarias.issubset(columnas_actuales):
        raise ValueError(
            f"El archivo {path} no tiene las columnas mínimas necesarias. "
            f"Esperábamos {sorted(columnas_necesarias)} y hemos encontrado {list(df.columns)}"
        )

    df = df.copy()
    df["input_corrector"] = df["input_corrector"].fillna("").astype(str)
    df["output_corrector"] = df["output_corrector"].fillna("").astype(str)
    df["tag"] = normalizar_tag_series(df["tag"], path)

    if "gold_standard" in df.columns:
        df["gold_standard"] = df["gold_standard"].fillna("").astype(str)

    return df


def comprobar_alineacion(gold_df, pred_df, nombre_modelo):
    """
    Aquí comprobamos que el gold y las predicciones estén alineados fila a fila.
    Exigimos:
    - mismo número de filas
    - mismo input_corrector en cada posición
    - mismo output_corrector en cada posición
    """
    if len(gold_df) != len(pred_df):
        raise ValueError(
            f"El número de filas no coincide entre gold y {nombre_modelo}.\n"
            f"Gold: {len(gold_df)}\n"
            f"{nombre_modelo}: {len(pred_df)}"
        )

    mismatch_input = gold_df["input_corrector"] != pred_df["input_corrector"]
    mismatch_output = gold_df["output_corrector"] != pred_df["output_corrector"]

    if mismatch_input.any():
        idx = mismatch_input[mismatch_input].index[0]
        raise ValueError(
            f"Desalineación en input_corrector entre gold y {nombre_modelo} en la fila {idx + 1}.\n"
            f"Gold: {repr(gold_df.loc[idx, 'input_corrector'])}\n"
            f"{nombre_modelo}: {repr(pred_df.loc[idx, 'input_corrector'])}"
        )

    if mismatch_output.any():
        idx = mismatch_output[mismatch_output].index[0]
        raise ValueError(
            f"Desalineación en output_corrector entre gold y {nombre_modelo} en la fila {idx + 1}.\n"
            f"Gold: {repr(gold_df.loc[idx, 'output_corrector'])}\n"
            f"{nombre_modelo}: {repr(pred_df.loc[idx, 'output_corrector'])}"
        )


def calcular_metricas(y_true, y_pred, nombre_modelo):
    """
    Aquí calculamos las métricas principales usando solamente las tags.

    Calculamos:
    - accuracy
    - precision, recall y f1 por clase
    - macro average
    - weighted average
    - balanced accuracy
    - MCC
    - Cohen's kappa
    - matriz de confusión

    Recordatorio de etiquetas:
    - tag 0 = buena corrección
    - tag 1 = mala corrección
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    true_0_pred_0 = int(cm[0, 0])
    true_0_pred_1 = int(cm[0, 1])
    true_1_pred_0 = int(cm[1, 0])
    true_1_pred_1 = int(cm[1, 1])

    total = len(y_true)
    errores = int((y_true != y_pred).sum())
    aciertos = int((y_true == y_pred).sum())

    resumen = {
        "modelo": nombre_modelo,
        "n_ejemplos": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "aciertos": aciertos,
        "errores": errores,
        "true_0_pred_0": true_0_pred_0,
        "true_0_pred_1": true_0_pred_1,
        "true_1_pred_0": true_1_pred_0,
        "true_1_pred_1": true_1_pred_1,
    }

    metricas_por_clase = pd.DataFrame([
        {
            "modelo": nombre_modelo,
            "clase": 0,
            "interpretacion": "buena corrección",
            "precision": precision_cls[0],
            "recall": recall_cls[0],
            "f1": f1_cls[0],
            "support": int(support_cls[0]),
        },
        {
            "modelo": nombre_modelo,
            "clase": 1,
            "interpretacion": "mala corrección",
            "precision": precision_cls[1],
            "recall": recall_cls[1],
            "f1": f1_cls[1],
            "support": int(support_cls[1]),
        },
    ])

    matriz_confusion = pd.DataFrame(
        cm,
        index=["gold_0_buena", "gold_1_mala"],
        columns=["pred_0_buena", "pred_1_mala"]
    ).reset_index().rename(columns={"index": "gold_vs_pred"})

    return resumen, metricas_por_clase, matriz_confusion


def construir_comparacion_caso_a_caso(gold_df, base_df, ft_df):
    """
    Aquí construimos una tabla caso a caso para analizar
    qué ha hecho cada modelo en cada ejemplo.
    Solo dejamos la información necesaria para comparar las etiquetas.
    """
    out = pd.DataFrame()

    out["row_id"] = range(1, len(gold_df) + 1)
    out["input_corrector"] = gold_df["input_corrector"]
    out["output_corrector"] = gold_df["output_corrector"]

    if "gold_standard" in gold_df.columns:
        out["gold_standard"] = gold_df["gold_standard"]

    out["gold_tag"] = gold_df["tag"]
    out["base_tag"] = base_df["tag"]
    out["fine_tuned_tag"] = ft_df["tag"]

    out["base_ok"] = (out["base_tag"] == out["gold_tag"]).astype(int)
    out["fine_tuned_ok"] = (out["fine_tuned_tag"] == out["gold_tag"]).astype(int)

    comparacion = []

    for _, row in out.iterrows():
        base_ok = row["base_ok"]
        ft_ok = row["fine_tuned_ok"]

        if base_ok == 1 and ft_ok == 1:
            comparacion.append("empate_ambos_aciertan")
        elif base_ok == 0 and ft_ok == 0:
            comparacion.append("empate_ambos_fallan")
        elif base_ok == 1 and ft_ok == 0:
            comparacion.append("mejor_base")
        else:
            comparacion.append("mejor_fine_tuned")

    out["comparacion_por_caso"] = comparacion

    return out


def construir_resumen_comparativo(df_resumen):
    """
    Aquí construimos un pequeño resumen interpretativo para leer
    el resultado de forma rápida.

    No sustituye a las tablas detalladas, pero sí ayuda a tener
    una primera visión general de la comparación.
    """
    if len(df_resumen) != 2:
        return "No ha sido posible construir el resumen comparativo porque no hay exactamente dos modelos."

    df_ord = df_resumen.set_index("modelo")

    nombre_base = None
    nombre_ft = None

    for nombre in df_ord.index.tolist():
        if "base" in nombre.lower():
            nombre_base = nombre
        if "fine" in nombre.lower() or "tuned" in nombre.lower():
            nombre_ft = nombre

    if nombre_base is None:
        nombre_base = df_ord.index.tolist()[0]
    if nombre_ft is None:
        nombre_ft = df_ord.index.tolist()[1]

    base = df_ord.loc[nombre_base]
    ft = df_ord.loc[nombre_ft]

    texto = []
    texto.append("Comparación automática de dos modelos frente al gold")
    texto.append("")
    texto.append("Esta comparación se ha calculado únicamente con las tags.")
    texto.append("No se han usado las explanations en ningún punto del análisis.")
    texto.append("")
    texto.append(f"Número total de ejemplos evaluados: {int(base['n_ejemplos'])}")
    texto.append("")
    texto.append("Resumen global")
    texto.append(f"- Accuracy {nombre_base}: {base['accuracy']:.4f} | {nombre_ft}: {ft['accuracy']:.4f}")
    texto.append(f"- F1 macro {nombre_base}: {base['f1_macro']:.4f} | {nombre_ft}: {ft['f1_macro']:.4f}")
    texto.append(f"- Balanced accuracy {nombre_base}: {base['balanced_accuracy']:.4f} | {nombre_ft}: {ft['balanced_accuracy']:.4f}")
    texto.append(f"- MCC {nombre_base}: {base['mcc']:.4f} | {nombre_ft}: {ft['mcc']:.4f}")
    texto.append(f"- Cohen's kappa {nombre_base}: {base['cohen_kappa']:.4f} | {nombre_ft}: {ft['cohen_kappa']:.4f}")
    texto.append("")
    texto.append("Diferencias a favor del segundo modelo")
    texto.append(f"- Accuracy: {(ft['accuracy'] - base['accuracy']):.4f}")
    texto.append(f"- F1 macro: {(ft['f1_macro'] - base['f1_macro']):.4f}")
    texto.append(f"- Balanced accuracy: {(ft['balanced_accuracy'] - base['balanced_accuracy']):.4f}")

    return "\n".join(texto)


def guardar_todo(
    output_dir,
    output_name,
    df_resumen,
    df_metricas_clase,
    df_cm_base,
    df_cm_ft,
    df_caso_a_caso,
    df_errores_base,
    df_errores_ft,
    texto_resumen,
):
    """
    Aquí guardamos todos los resultados en formatos cómodos de revisar.

    Generamos:
    - un Excel con varias hojas
    - varios CSV
    - un TXT resumen
    """
    os.makedirs(output_dir, exist_ok=True)

    excel_path = os.path.join(output_dir, f"{output_name}.xlsx")
    resumen_csv = os.path.join(output_dir, f"{output_name}_resumen_modelos.csv")
    clases_csv = os.path.join(output_dir, f"{output_name}_metricas_por_clase.csv")
    casos_csv = os.path.join(output_dir, f"{output_name}_caso_a_caso.csv")
    errores_base_csv = os.path.join(output_dir, f"{output_name}_errores_base.csv")
    errores_ft_csv = os.path.join(output_dir, f"{output_name}_errores_fine_tuned.csv")
    resumen_txt = os.path.join(output_dir, f"{output_name}_resumen.txt")

    df_resumen.to_csv(resumen_csv, index=False, encoding="utf-8")
    df_metricas_clase.to_csv(clases_csv, index=False, encoding="utf-8")
    df_caso_a_caso.to_csv(casos_csv, index=False, encoding="utf-8")
    df_errores_base.to_csv(errores_base_csv, index=False, encoding="utf-8")
    df_errores_ft.to_csv(errores_ft_csv, index=False, encoding="utf-8")

    with open(resumen_txt, "w", encoding="utf-8") as f:
        f.write(texto_resumen)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_resumen.to_excel(writer, sheet_name="resumen_modelos", index=False)
        df_metricas_clase.to_excel(writer, sheet_name="metricas_por_clase", index=False)
        df_cm_base.to_excel(writer, sheet_name="matriz_conf_base", index=False)
        df_cm_ft.to_excel(writer, sheet_name="matriz_conf_ft", index=False)
        df_caso_a_caso.to_excel(writer, sheet_name="caso_a_caso", index=False)
        df_errores_base.to_excel(writer, sheet_name="errores_base", index=False)
        df_errores_ft.to_excel(writer, sheet_name="errores_fine_tuned", index=False)

    print(f"Excel comparativo guardado en: {excel_path}")
    print(f"CSV resumen guardado en:      {resumen_csv}")
    print(f"CSV por clase guardado en:    {clases_csv}")
    print(f"CSV caso a caso guardado en:  {casos_csv}")
    print(f"CSV errores base guardado en: {errores_base_csv}")
    print(f"CSV errores FT guardado en:   {errores_ft_csv}")
    print(f"TXT resumen guardado en:      {resumen_txt}")


def main():
    """
    Flujo general del script:

    1. leemos el gold
    2. leemos las predicciones del primer modelo
    3. leemos las predicciones del segundo modelo
    4. comprobamos que todo esté alineado
    5. calculamos métricas usando solo las tags
    6. construimos una tabla caso a caso
    7. separamos los errores de cada modelo
    8. guardamos todo en Excel, CSV y TXT

    El objetivo es dejar una comparación clara, reutilizable y fácil de compartir.
    """
    args = parse_args()

    print("Leyendo gold...")
    gold_df = cargar_y_validar(args.gold_file)

    print("Leyendo predicciones del modelo base...")
    base_df = cargar_y_validar(args.base_file)

    print("Leyendo predicciones del modelo fine-tuneado...")
    ft_df = cargar_y_validar(args.finetuned_file)

    print("Comprobando alineación entre gold y modelo base...")
    comprobar_alineacion(gold_df, base_df, "modelo_base")

    print("Comprobando alineación entre gold y modelo fine-tuneado...")
    comprobar_alineacion(gold_df, ft_df, "modelo_fine_tuned")

    y_true = gold_df["tag"]
    y_pred_base = base_df["tag"]
    y_pred_ft = ft_df["tag"]

    print("Calculando métricas del modelo base...")
    resumen_base, metricas_base, cm_base = calcular_metricas(
        y_true, y_pred_base, "modelo_base"
    )

    print("Calculando métricas del modelo fine-tuneado...")
    resumen_ft, metricas_ft, cm_ft = calcular_metricas(
        y_true, y_pred_ft, "modelo_fine_tuned"
    )

    df_resumen = pd.DataFrame([resumen_base, resumen_ft])
    df_metricas_clase = pd.concat([metricas_base, metricas_ft], ignore_index=True)

    print("Construyendo comparación caso a caso...")
    df_caso_a_caso = construir_comparacion_caso_a_caso(gold_df, base_df, ft_df)

    df_errores_base = df_caso_a_caso[df_caso_a_caso["base_ok"] == 0].copy()
    df_errores_ft = df_caso_a_caso[df_caso_a_caso["fine_tuned_ok"] == 0].copy()

    texto_resumen = construir_resumen_comparativo(df_resumen)

    print("Guardando resultados...")
    guardar_todo(
        output_dir=args.output_dir,
        output_name=args.output_name,
        df_resumen=df_resumen,
        df_metricas_clase=df_metricas_clase,
        df_cm_base=cm_base,
        df_cm_ft=cm_ft,
        df_caso_a_caso=df_caso_a_caso,
        df_errores_base=df_errores_base,
        df_errores_ft=df_errores_ft,
        texto_resumen=texto_resumen,
    )

    print("Proceso terminado correctamente.")


if __name__ == "__main__":
    main()