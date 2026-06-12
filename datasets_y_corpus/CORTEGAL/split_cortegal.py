#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Dividimos CORTEGAL en 5 partes con tamaños fijos:
- 4 partes de 1700 ejemplos
- 1 parte final de 315 ejemplos

Queremos que los tipos de error (tag_id) queden repartidos entre las particiones
de la forma más equilibrada posible. Para ello:
1) Leemos los tag_id de cada ejemplo (desde la lista "corrections").
2) Calculamos objetivos por partición para cada tag_id en proporción al tamaño.
3) Asignamos ejemplos a particiones con una heurística que intenta no
   sobrepasar esos objetivos mientras respeta los tamaños exactos.

Output (5 ficheros):
- case_1.json  -> Caso 1: incorrect -> correct_standard, tag=0
- case_2.json  -> Caso 2: correct_standard -> correct_standard, tag=0
- case_3.json  -> Caso 3: correct_standard -> incorrect, tag=1
- case_4.json  -> Caso 4: incorrect -> incorrect, tag=1
- pairs.json  -> Solo pares: incorrect + correct_standard

En TODOS los outputs añadimos "pair_id" para identificar el par.
En los casos 1-4 añadimos también "Tag" (0 o 1) según el caso.
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Set


def load_cortegal_json(path: str) -> List[Dict[str, Any]]:
    """Cargamos CORTEGAL desde un JSON que es una lista de objetos."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El fichero de entrada no es una lista JSON.")
    return data


def extract_tags(example: Dict[str, Any]) -> Tuple[str, ...]:
    """
    Extraemos los tag_id asociados a un ejemplo.

    Un ejemplo puede tener múltiples correcciones (multi-etiqueta).
    Para equilibrar, consideramos que el ejemplo cuenta para TODOS sus tag_id.

    Si no hay tag_id, usamos una etiqueta de respaldo.
    """
    corr = example.get("corrections", [])
    tags: Set[str] = set()

    if isinstance(corr, list):
        for c in corr:
            if isinstance(c, dict):
                t = c.get("tag_id")
                if isinstance(t, str) and t.strip():
                    tags.add(t.strip())

    if not tags:
        tags.add("NO_TAG")

    return tuple(sorted(tags))


def validate_required_fields(example: Dict[str, Any]) -> None:
    """
    Validamos que podamos construir los pares con los campos obligatorios.
    Solo nos interesan:
    - incorrect
    - correct_standard
    """
    if "incorrect" not in example:
        raise ValueError(f"Falta el campo 'incorrect' en un ejemplo (id={example.get('id')}).")
    if "correct_standard" not in example:
        raise ValueError(f"Falta el campo 'correct_standard' en un ejemplo (id={example.get('id')}).")

    inc = example.get("incorrect")
    cor = example.get("correct_standard")

    if not isinstance(inc, str):
        raise ValueError(f"'incorrect' no es string (id={example.get('id')}). Tipo: {type(inc)}")
    if not isinstance(cor, str):
        raise ValueError(f"'correct_standard' no es string (id={example.get('id')}). Tipo: {type(cor)}")


def compute_targets_per_tag(
    tag_global_counts: Counter,
    split_sizes: List[int],
    total_examples: int
) -> Dict[str, List[int]]:
    """
    Para cada tag_id, calculamos objetivos aproximados por partición,
    proporcionalmente al tamaño de cada split.
    """
    targets: Dict[str, List[int]] = {}

    for tag, gcount in tag_global_counts.items():
        ideal = [gcount * (s / total_examples) for s in split_sizes]
        base = [int(x) for x in ideal]
        remainder = gcount - sum(base)

        decimals = sorted(
            [(i, ideal[i] - base[i]) for i in range(len(split_sizes))],
            key=lambda x: x[1],
            reverse=True
        )

        for k in range(remainder):
            idx = decimals[k % len(decimals)][0]
            base[idx] += 1

        targets[tag] = base

    return targets


def choose_best_split(
    tags: Tuple[str, ...],
    remaining_slots: List[int],
    current_tag_counts: Dict[str, List[int]],
    targets: Dict[str, List[int]]
) -> int:
    """
    Elegimos la partición más adecuada para un ejemplo respetando plazas disponibles
    y tratando de no exceder los objetivos por tag.
    """
    best_idx = None
    best_score = None

    for i in range(len(remaining_slots)):
        if remaining_slots[i] <= 0:
            continue

        overflow_increase = 0
        deficit_gain = 0

        for t in tags:
            cur = current_tag_counts[t][i]
            tgt = targets.get(t, [0] * len(remaining_slots))[i]

            before_over = max(0, cur - tgt)
            after_over = max(0, (cur + 1) - tgt)
            overflow_increase += (after_over - before_over)

            if cur < tgt:
                deficit_gain += 1

        score = (overflow_increase, -deficit_gain, -remaining_slots[i])

        if best_score is None or score < best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        raise RuntimeError("No se pudo asignar un ejemplo a ninguna partición con plazas disponibles.")

    return best_idx


def assign_examples_balanced(
    examples: List[Dict[str, Any]],
    split_sizes: List[int],
    seed: int
) -> List[List[Dict[str, Any]]]:
    """
    Asignamos ejemplos a las 5 particiones intentando equilibrar tag_id
    y respetando tamaños exactos.
    """
    total = len(examples)
    if sum(split_sizes) != total:
        raise ValueError(f"La suma de tamaños {sum(split_sizes)} no coincide con el total de ejemplos {total}.")

    for ex in examples:
        validate_required_fields(ex)

    tags_by_idx: List[Tuple[str, ...]] = []
    global_tag_counts: Counter = Counter()

    for ex in examples:
        tags = extract_tags(ex)
        tags_by_idx.append(tags)
        for t in tags:
            global_tag_counts[t] += 1

    targets = compute_targets_per_tag(global_tag_counts, split_sizes, total)

    rnd = random.Random(seed)
    indices = list(range(total))
    rnd.shuffle(indices)

    def rarity_key(idx: int) -> Tuple[int, int]:
        tags = tags_by_idx[idx]
        min_freq = min(global_tag_counts[t] for t in tags)
        return (min_freq, idx)

    indices.sort(key=rarity_key)

    splits: List[List[Dict[str, Any]]] = [[] for _ in split_sizes]
    remaining = split_sizes[:]
    current_tag_counts: Dict[str, List[int]] = defaultdict(lambda: [0] * len(split_sizes))

    for idx in indices:
        ex = examples[idx]
        tags = tags_by_idx[idx]

        best_split = choose_best_split(tags, remaining, current_tag_counts, targets)

        splits[best_split].append(ex)
        remaining[best_split] -= 1

        for t in tags:
            current_tag_counts[t][best_split] += 1

    for i, part in enumerate(splits):
        if len(part) != split_sizes[i]:
            raise RuntimeError(
                f"Error interno: la partición {i+1} tiene {len(part)} ejemplos, se esperaban {split_sizes[i]}."
            )

    return splits


def build_case_records(part: List[Dict[str, Any]], case_id: int) -> List[Dict[str, Any]]:
    """
    Construimos el formato exacto para los casos 1-4.

    IMPORTANTE:
    - Usamos EXACTAMENTE estas claves:
      pair_id, input_corrector, output_corrector, tag
    - Tag:
      * casos 1 y 2 -> Tag = 0
      * casos 3 y 4 -> Tag = 1
    """
    if case_id in (1, 2):
        tag_value = 0
    elif case_id in (3, 4):
        tag_value = 1
    else:
        raise ValueError("case_id debe ser 1, 2, 3 o 4.")

    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(part, start=1):
        inc = ex["incorrect"]
        cor = ex["correct_standard"]

        if case_id == 1:
            inp, outp = inc, cor
        elif case_id == 2:
            inp, outp = cor, cor
        elif case_id == 3:
            inp, outp = cor, inc
        else:  # case_id == 4
            inp, outp = inc, inc

        out.append({
            "pair_id": i,
            "input_corrector": inp,
            "output_corrector": outp,
            "tag": tag_value
        })

    return out


def build_pairs_only(part: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Formato de la 5ª parte: dejamos solo el par junto:
    - pair_id
    - incorrect
    - correct_standard
    """
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(part, start=1):
        out.append({
            "pair_id": i,
            "incorrect": ex["incorrect"],
            "correct_standard": ex["correct_standard"]
        })
    return out


def save_json(path: str, data: Any) -> None:
    """Guardamos JSON legible (indent=2) para poder inspeccionarlo fácilmente."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divide CORTEGAL en 5 partes (1700x4 + 315), equilibrando tag_id, y crea casos 1-4 + pares."
    )
    parser.add_argument("--input", required=True, help="Ruta al CORTEGAL.json.")
    parser.add_argument("--out_dir", required=True, help="Directorio de salida.")
    parser.add_argument(
        "--sizes",
        type=str,
        default="1700,1700,1700,1700,315",
        help="Tamaños de las 5 particiones, separados por comas. Deben sumar el total de ejemplos."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para que la partición sea reproducible.")
    args = parser.parse_args()

    split_sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    if len(split_sizes) != 5:
        raise ValueError("Necesitamos exactamente 5 tamaños en --sizes (ej: 1700,1700,1700,1700,315).")

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_cortegal_json(args.input)

    # Asignamos ejemplos equilibrando tag_id y respetando tamaños exactos
    parts = assign_examples_balanced(data, split_sizes, seed=args.seed)

    # Construimos outputs en el formato requerido
    out1 = build_case_records(parts[0], case_id=1)
    out2 = build_case_records(parts[1], case_id=2)
    out3 = build_case_records(parts[2], case_id=3)
    out4 = build_case_records(parts[3], case_id=4)
    out5 = build_pairs_only(parts[4])

    # Guardamos los 5 JSON (solo esto)
    save_json(os.path.join(args.out_dir, "case_1.json"), out1)
    save_json(os.path.join(args.out_dir, "case_2.json"), out2)
    save_json(os.path.join(args.out_dir, "case_3.json"), out3)
    save_json(os.path.join(args.out_dir, "case_4.json"), out4)
    save_json(os.path.join(args.out_dir, "pairs.json"), out5)

    print("Listo. Se han generado 5 ficheros JSON en:", args.out_dir)


if __name__ == "__main__":
    main()
