#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unimos los archivos case_1.json, case_2.json, case_3.json y case_4.json
en un único JSONL para construir el Dataset A.

Formato esperado en cada archivo de entrada:
[
  {
    "pair_id": 1,
    "input_corrector": "...",
    "output_corrector": "...",
    "tag": 0
  },
  ...
]

Formato de salida:
{
  "case": 1,
  "absolute_id": 1,
  "pair_id_from_case": 1,
  "input_corrector": "...",
  "output_corrector": "...",
  "tag": 0
}
"""

import os
import re
import json
import argparse
from typing import List, Dict, Any, Tuple


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Cargamos un archivo que puede estar en formato JSON normal
    o JSONL. Devolvemos siempre una lista de diccionarios.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # Intentamos primero como JSON completo
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"Formato JSON no soportado en {path}")
    except json.JSONDecodeError:
        pass

    # Si no era JSON normal, intentamos como JSONL
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError
                records.append(obj)
            except Exception:
                raise ValueError(
                    f"No se pudo parsear la línea {line_num} de {path} como JSONL válido."
                )

    return records


def infer_case_from_filename(path: str) -> int:
    """
    Inferimos el número de case a partir del nombre del archivo.

    Ejemplos válidos:
    - case_1.json
    - case1.json
    - case-1.json
    """
    filename = os.path.basename(path)
    match = re.search(r"case[_\- ]?([1-4])", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"No se pudo inferir el case a partir del nombre del archivo: {filename}"
        )
    return int(match.group(1))


def parse_input_args(inputs: List[str]) -> List[Tuple[str, int]]:
    """
    Permitimos dos formas:
    - /ruta/case_1.json
    - /ruta/loquesea.json:1
    """
    pairs = []

    for item in inputs:
        if ":" in item:
            path, case_str = item.rsplit(":", 1)
            try:
                case_num = int(case_str)
            except ValueError:
                raise ValueError(
                    f"No se pudo interpretar el case en '{item}'. Usa formato archivo:case"
                )
        else:
            path = item
            case_num = infer_case_from_filename(path)

        if case_num not in {1, 2, 3, 4}:
            raise ValueError(f"El case debe ser 1, 2, 3 o 4. Recibido: {case_num}")

        pairs.append((path, case_num))

    return pairs


def normalize_record(record: Dict[str, Any], case_num: int, absolute_id: int) -> Dict[str, Any]:
    """
    Reorganizamos cada ejemplo con los nombres finales que queremos.
    """
    required_fields = ["pair_id", "input_corrector", "output_corrector", "tag"]
    for field in required_fields:
        if field not in record:
            raise KeyError(
                f"Falta el campo obligatorio '{field}' en un ejemplo del case {case_num}"
            )

    normalized = {
        "case": case_num,
        "absolute_id": absolute_id,
        "pair_id_from_case": record["pair_id"],
        "input_corrector": record["input_corrector"],
        "output_corrector": record["output_corrector"],
        "tag": record["tag"],
    }

    return normalized


def save_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    """
    Guardamos la salida en formato JSONL.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Une los archivos de casos CORTEGAL en un único Dataset A en JSONL."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Lista de archivos de entrada. "
            "Pueden pasarse como /ruta/case_1.json /ruta/case_2.json ... "
            "o como /ruta/archivo.json:1 /ruta/archivo.json:2 ..."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta del archivo JSONL de salida."
    )
    parser.add_argument(
        "--sort_by_case_then_pair_id",
        action="store_true",
        help="Si se indica, ordena la salida por case y luego por pair_id_from_case."
    )

    args = parser.parse_args()

    input_pairs = parse_input_args(args.inputs)

    merged_records = []
    absolute_id = 1

    for path, case_num in input_pairs:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No existe el archivo: {path}")

        records = load_json_or_jsonl(path)

        for record in records:
            normalized = normalize_record(
                record=record,
                case_num=case_num,
                absolute_id=absolute_id
            )
            merged_records.append(normalized)
            absolute_id += 1

    if args.sort_by_case_then_pair_id:
        merged_records = sorted(
            merged_records,
            key=lambda x: (x["case"], x["pair_id_from_case"])
        )

        # Reasignamos absolute_id para que vuelva a ser consecutivo tras ordenar
        for i, record in enumerate(merged_records, start=1):
            record["absolute_id"] = i

    save_jsonl(merged_records, args.output)

    print(f"Archivos procesados: {len(input_pairs)}")
    print(f"Ejemplos totales: {len(merged_records)}")
    print(f"Salida guardada en: {args.output}")


if __name__ == "__main__":
    main()