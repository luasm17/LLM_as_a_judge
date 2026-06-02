#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generamos un fichero .txt con las frases incorrectas del pairs_dataset_B, la quinta partición del CORTEGAL.

Este fichero se usa como entrada para el corrector Salamandra. A partir de estas
frases, Salamandra genera sus propias correcciones, que después se recogen en un
Excel con la frase incorrecta original y la salida producida por el corrector.
Ese Excel será el Dataset B usado en la evaluación.
"""

import argparse
import json
from pathlib import Path


def main():
    # Parametrizamos las rutas para no depender de rutas locales fijas.
    parser = argparse.ArgumentParser(
        description="Extrae las frases incorrectas de un JSON de pares y las guarda en un fichero TXT."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al JSON de entrada con los pares."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Ruta al TXT de salida con las frases incorrectas."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Leemos el JSON de pares, donde cada ejemplo contiene una frase incorrecta
    # y su corrección de referencia.
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Guardamos únicamente las frases incorrectas, una por línea, para poder
    # pasarlas después al corrector Salamandra.
    with output_path.open("w", encoding="utf-8") as out:
        for i, item in enumerate(data):
            frase = item["incorrect"]
            out.write(frase)

            # Evitamos añadir una línea vacía extra al final del fichero.
            if i < len(data) - 1:
                out.write("\n")


if __name__ == "__main__":
    main()
