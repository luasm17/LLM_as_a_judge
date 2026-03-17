#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import importlib.util
from typing import Tuple, Optional, Dict, Any, List

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from peft import PeftModel


PROMPT_BASE = r'''Tes que devolver a túa avaliación como LLM-as-a-judge seguindo exactamente este formato; debes responder exclusivamente a estes catro puntos variables, sen engadir texto adicional:

input_corrector: "<oración de entrada do corrector>"

output_corrector: "<oración xa avaliada polo corrector>"

tag: <0 ou 1>

explanation: "<explicación breve e precisa en galego do motivo polo que se escolleu a etiqueta 0 ou a etiqueta 1>"

Criterios que debes seguir:

tag = 0: a saída do corrector é correcta con respecto á gramática da lingua galega (é dicir, non hai erros no output_corrector). Nota sobre a tag = 0 (dúas situacións posibles):

Caso 0.A: input_corrector = output_corrector. O input_corrector xa era correcto e o modelo GEC non fixo cambios adicionais porque non había nada que corrixir. A explicación debe indicar que non había erros no input e por isto non se modifica.

Caso 0.B: input_corrector ≠ output_corrector. O input_corrector tiña erro(s) e o modelo GEC corrixiuno(s) correctamente. A explicación debe indicar que erro(s) se arranxou/arranxaron.

En ambos casos, a etiqueta é 0, pero a explicación debe reflectir se houbo corrección do input_corrector (caso 0.B) ou non por ser xa correcto (caso 0.A).

tag = 1: a saída do corrector non é correcta con respecto á gramática da lingua galega (é dicir, segue habendo erro(s) no output_corrector). Nota sobre a tag = 1 (dúas situacións posibles):

Caso 1.A: input_corrector = output_corrector. O input_corrector era incorrecto e o modelo GEC non corrixiu o(s) erro(s). A explicación debe indicar que o modelo GEC non corrixiu o(s) erro(s) presente(s) no input_corrector.

Caso 1.B: input_corrector ≠ output_corrector. O input_corrector non tiña ningún erro, pero o modelo GEC introduciuno. A explicación debe indicar que o corrector introduciu no output_corrector un(s) erro(s) que non había no input_corrector.

En ambos casos, a etiqueta é 1, pero a explicación debe reflectir se houbo modificación do input_corrector (caso 1.B) ou non (caso 1.A).

Restricións que debes respectar:

Non debes, baixo ningún concepto, corrixir ou modificar de ningunha forma o input nin o output do corrector.

Tes que limitarte exclusivamente a decidir se os erros gramaticais foron corrixidos polo corrector gramatical ou non.

Non uses nunca o español, o portugués nin ningunha lingua que non sexa o galego nas túas respostas.

Aquí tes uns exemplos de inputs que poderías atopar e dos outputs que deberías devolver, respectivamente:

Exemplo 1 (caso 0.A): tag = 0

Exemplo de entrada:
input_corrector: "Acaba de saír o sol despois de moita choiva e os nenos corren cara o parque."

output_corrector: "Acaba de saír o sol despois de moita choiva e os nenos corren cara o parque."

Exemplo de saída:
input_corrector: "Acaba de saír o sol despois de moita choiva e os nenos corren cara o parque."

output_corrector: "Acaba de saír o sol despois de moita choiva e os nenos corren cara o parque."

tag: 0

explanation: "O output_corrector é adecuado xa que non se modificou o input_corrector, o cal non contiña ningún erro que corrixir."

Exemplo 2 (caso 0.B): tag = 0

Exemplo de entrada:
input_corrector: "A decisións tomadas polo comité foron comunicadas aos responsables das distintas áreas."

output_corrector: "As decisións tomadas polo comité foron comunicadas aos responsables das distintas áreas."

Exemplo de saída:
input_corrector: "A decisións tomadas polo comité foron comunicadas aos responsables das distintas áreas."

output_corrector: "As decisións tomadas polo comité foron comunicadas aos responsables das distintas áreas."

tag: 0

explanation: "O output_corrector é adecuado porque solucionou o erro do input_corrector entre “A” e “decisións”."

Exemplo 3 (caso 1.A): tag = 1

Exemplo de entrada:
input_corrector: "Pásame ese libro que están enriba da mesa, por favor."

output_corrector: "Pásame ese libro que están enriba da mesa, por favor."

Exemplo de saída:
input_corrector: "Pásame ese libro que están enriba da mesa, por favor."

output_corrector: "Pásame ese libro que están enriba da mesa, por favor."

tag: 1

explanation: "O output_corrector non é adecuado porque non se modificou o input_corrector, o cal contén un erro entre “están” e “libro”."

Exemplo 4 (caso 1.B): tag = 1

Exemplo de entrada:
input_corrector: "Todo o mundo chegou a tempo á xuntanza aquel día."

output_corrector: "Todo o mundo chegaron a tempo á xuntanza aquel día."

Exemplo de saída:
input_corrector: "Todo o mundo chegou a tempo á xuntanza aquel día."

output_corrector: "Todo o mundo chegaron a tempo á xuntanza aquel día."

tag: 1

explanation: "O output_corrector non é adecuado porque o corrector engadiu un erro que non existía no input_corrector."

Agora, avalía os seguintes casos:

input_corrector: "{input_corrector}"

output_corrector: "{output_corrector}"
'''


PROMPT_REPARACION = r'''A resposta anterior non segue correctamente o formato esperado.

Debes devolver EXCLUSIVAMENTE estas dúas liñas e nada máis:

tag: <0 ou 1>
explanation: "<explicación breve e precisa en galego do motivo polo que se escolleu a etiqueta>"

Importante:
- A tag ten que ser obrigatoriamente 0 ou 1.
- A explanation ten que explicar por que se asigna esa tag.
- Non corrixas nin modifiques o input_corrector nin o output_corrector.
- Non engadas input_corrector nin output_corrector na resposta.

input_corrector: "{input_corrector}"
output_corrector: "{output_corrector}"

Resposta anterior do modelo:
{raw_response}
'''


def parse_args():
    """
    Aquí definimos los argumentos del script.
    """
    parser = argparse.ArgumentParser(
        description="Evaluar correcciones de Salamandra con el Selene fine-tuneado."
    )
    parser.add_argument("--input_xlsx", required=True, help="Ruta al Excel de entrada.")
    parser.add_argument("--adapter_path", required=True, help="Ruta al checkpoint del LoRA.")
    parser.add_argument("--output_xlsx", required=True, help="Ruta al Excel de salida.")
    parser.add_argument("--sheet_name", default=0, help="Nombre o índice de la hoja del Excel.")
    parser.add_argument("--max_new_tokens", type=int, default=320, help="Máximo de tokens nuevos.")
    parser.add_argument("--save_every", type=int, default=10, help="Guardar resultados cada N filas.")
    parser.add_argument("--device", type=int, default=0, help="Índice de la GPU visible que vamos a usar.")
    parser.add_argument(
        "--quantization",
        choices=["auto", "none", "8bit", "4bit"],
        default="auto",
        help="Modo de carga del modelo base. 'auto' intenta 8bit, después 4bit y por último sin cuantización.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Si ya existe el fichero de salida, retomamos desde las filas ya procesadas.",
    )
    return parser.parse_args()


def escoller_dtype() -> torch.dtype:
    """
    Elegimos el dtype más conveniente según el hardware disponible.
    Si hay soporte para bf16 lo aprovechamos; si no, pasamos a fp16.
    En CPU dejamos float32, que es lo más estable.
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def bnb_dispoñible() -> bool:
    """
    Comprobamos si bitsandbytes está instalado, porque solo así podremos usar 4bit u 8bit.
    """
    return importlib.util.find_spec("bitsandbytes") is not None


def limpar_cuda():
    """
    Liberamos memoria de CUDA cuando hace falta.
    Esto nos ayuda a evitar problemas si una carga falla y luego queremos reintentar.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def e_oom(exc: Exception) -> bool:
    """
    Detectamos si una excepción parece un error de memoria.
    Lo usamos para que el modo 'auto' pueda probar otra estrategia de carga.
    """
    texto = str(exc).lower()
    return (
        isinstance(exc, torch.OutOfMemoryError)
        or "out of memory" in texto
        or "cuda out of memory" in texto
        or ("cublas" in texto and "alloc" in texto)
    )


def atopar_base_model_name(adapter_path: str) -> str:
    """
    Leemos adapter_config.json para recuperar el nombre del modelo base sobre el que se entrenó el LoRA.
    """
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"No encuentro adapter_config.json en: {adapter_config_path}")

    with open(adapter_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_model_name = cfg.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("adapter_config.json no contiene 'base_model_name_or_path'.")

    return base_model_name


def atopar_tokenizer_path(adapter_path: str) -> str:
    """
    Buscamos desde dónde cargar el tokenizer.
    Primero miramos en la carpeta del adapter y, si no está ahí, en la carpeta padre.
    """
    candidatos = [adapter_path, os.path.dirname(adapter_path)]
    for path in candidatos:
        if (
            os.path.exists(os.path.join(path, "tokenizer.json"))
            or os.path.exists(os.path.join(path, "tokenizer_config.json"))
        ):
            return path
    return os.path.dirname(adapter_path)


def cargar_tokenizer(tokenizer_path: str):
    """
    Cargamos el tokenizer.
    Si falla la ruta habitual y el problema está en el backend de tokenizers, probamos con PreTrainedTokenizerFast.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as e:
        if "TokenizersBackend" in str(e):
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_quantization_config(modo: str):
    """
    Construimos la configuración de cuantización solo si realmente vamos a cargar en 8bit o 4bit.
    """
    if modo not in {"8bit", "4bit"}:
        return None

    if not bnb_dispoñible():
        raise ImportError(
            "Se pidió cuantización pero bitsandbytes no está instalado. Ejecuta: pip install -U bitsandbytes"
        )

    from transformers import BitsAndBytesConfig

    if modo == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def estratexias_carga(quantization: str) -> List[str]:
    """
    Decidimos el orden de estrategias de carga.
    En modo auto mantenemos la lógica del script original: probamos primero opciones que ahorran memoria.
    """
    if not torch.cuda.is_available():
        return ["none"]

    if quantization == "none":
        return ["none"]
    if quantization == "8bit":
        return ["8bit"]
    if quantization == "4bit":
        return ["4bit"]

    if bnb_dispoñible():
        return ["8bit", "4bit", "none"]
    return ["none"]


def cargar_modelo_base(base_model_name: str, dtype: torch.dtype, device: int, quantization: str):
    """
    Cargamos el modelo base con la estrategia elegida.
    """
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        kwargs["device_map"] = {"": device}
        kwargs["low_cpu_mem_usage"] = True
    else:
        kwargs["dtype"] = dtype

    if quantization == "none":
        if torch.cuda.is_available():
            kwargs["dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

    kwargs["quantization_config"] = get_quantization_config(quantization)
    return AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)


def cargar_modelo_y_tokenizer(adapter_path: str, device: int = 0, quantization: str = "auto"):
    """
    Cargamos tokenizer, modelo base y adapter LoRA.
    Si estamos en auto y algo falla por memoria, probamos la siguiente estrategia.
    """
    base_model_name = atopar_base_model_name(adapter_path)
    tokenizer_path = atopar_tokenizer_path(adapter_path)

    print("Modelo base detectado en adapter_config:", base_model_name)
    print("Cargando tokenizer desde:", tokenizer_path)
    tokenizer = cargar_tokenizer(tokenizer_path)

    dtype = escoller_dtype()
    errors = []

    for modo in estratexias_carga(quantization):
        try:
            print(f"Cargando modelo base con el modo: {modo}")
            limpar_cuda()
            model = cargar_modelo_base(base_model_name, dtype=dtype, device=device, quantization=modo)
            print("Cargando adapter LoRA...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
            print(f"Modelo cargado correctamente con el modo: {modo}")
            return model, tokenizer, base_model_name, modo
        except Exception as e:
            errors.append((modo, repr(e)))
            limpar_cuda()
            if quantization != "auto":
                raise
            if not e_oom(e) and not isinstance(e, ImportError):
                raise
            print(f"Ha fallado la carga con el modo {modo}: {e}")
            print("Probamos el siguiente modo de carga...")

    detalles = "\n".join([f"- {modo}: {err}" for modo, err in errors])
    raise RuntimeError("No ha sido posible cargar el modelo en ningún modo.\n" + detalles)


def construír_prompt(input_corrector: str, output_corrector: str) -> str:
    """
    Montamos el prompt principal exactamente con el formato que espera nuestro juez.
    """
    return PROMPT_BASE.format(
        input_corrector=str(input_corrector).strip(),
        output_corrector=str(output_corrector).strip(),
    )


def construír_prompt_reparacion(input_corrector: str, output_corrector: str, raw_response: str) -> str:
    """
    Si la primera respuesta viene mal formateada o sin tag/explanation claras,
    lanzamos una segunda petición muy corta para obligar al modelo a devolver una salida válida.
    """
    return PROMPT_REPARACION.format(
        input_corrector=str(input_corrector).strip(),
        output_corrector=str(output_corrector).strip(),
        raw_response=str(raw_response).strip(),
    )


def obter_terminators(tokenizer) -> List[int]:
    """
    Reunimos los tokens de fin que puedan ser útiles para cortar la generación de forma limpia.
    """
    terminators: List[int] = []

    if tokenizer.eos_token_id is not None:
        terminators.append(tokenizer.eos_token_id)

    for tok in ["<|eot_id|>", "<|end_of_text|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id and tid not in terminators:
                terminators.append(tid)
        except Exception:
            pass

    return terminators if terminators else [tokenizer.eos_token_id]


def extraer_tag_e_explanation(texto: str) -> Tuple[Optional[int], str]:
    """
    Extraemos la tag y la explanation de la salida textual del modelo.
    Aquí seguimos siendo tolerantes con el formato para rescatar la información aunque no venga perfecta.
    """
    if texto is None:
        return None, ""

    tag_matches = re.findall(r"tag\s*:\s*([01])", texto, flags=re.IGNORECASE)
    tag = int(tag_matches[-1]) if tag_matches else None

    exp_match = re.search(
        r'explanation\s*:\s*"?(.*?)"?\s*$',
        texto.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )

    if exp_match:
        explanation = exp_match.group(1).strip().strip('"').strip()
    else:
        exp_match = re.search(r"explanation\s*:\s*(.*)", texto, flags=re.IGNORECASE | re.DOTALL)
        explanation = exp_match.group(1).strip().strip('"').strip() if exp_match else ""

    return tag, explanation


def normalizar_saida_bruta(texto: str) -> str:
    """
    Limpiamos la respuesta bruta del modelo.
    Si el modelo ha repetido parte del formato, recortamos desde el último 'input_corrector:'.
    """
    if texto is None:
        return ""

    texto = texto.strip()
    idx = texto.lower().rfind("input_corrector:")
    if idx != -1:
        texto = texto[idx:].strip()
    return texto


def normalizar_tag(tag) -> Optional[int]:
    """
    Convertimos la tag a entero solo si realmente es 0 o 1.
    """
    if tag is None:
        return None

    if isinstance(tag, str):
        tag = tag.strip()

    try:
        tag_int = int(tag)
    except Exception:
        return None

    return tag_int if tag_int in (0, 1) else None


def device_do_modelo(model) -> torch.device:
    """
    Recuperamos el dispositivo real del modelo.
    """
    try:
        return model.device
    except Exception:
        pass

    try:
        return next(model.parameters()).device
    except Exception:
        pass

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def xerar_desde_prompt(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Esta función encapsula la generación.
    La usamos tanto para la inferencia principal como para la reparación del formato.
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if isinstance(model_inputs, torch.Tensor):
            model_inputs = {"input_ids": model_inputs}
        elif hasattr(model_inputs, "items"):
            model_inputs = dict(model_inputs)
        else:
            raise TypeError(f"Formato de entrada no soportado para generate(): {type(model_inputs)}")

    destino = device_do_modelo(model)
    model_inputs = {k: v.to(destino) for k, v in model_inputs.items()}
    terminators = obter_terminators(tokenizer)

    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=terminators,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_length = model_inputs["input_ids"].shape[-1]
    generated = outputs[0][input_length:]
    raw_response = tokenizer.decode(generated, skip_special_tokens=True)
    return normalizar_saida_bruta(raw_response)


@torch.no_grad()
def inferir_unha(
    model,
    tokenizer,
    input_corrector: str,
    output_corrector: str,
    max_new_tokens: int = 320,
) -> Tuple[Optional[int], str, str]:
    """
    Hacemos la inferencia principal del juez y devolvemos:
    - tag parseada
    - explanation parseada
    - respuesta bruta normalizada
    """
    prompt = construír_prompt(input_corrector, output_corrector)
    raw_response = xerar_desde_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    tag, explanation = extraer_tag_e_explanation(raw_response)
    return tag, explanation, raw_response


@torch.no_grad()
def reparar_resposta_incompleta(
    model,
    tokenizer,
    input_corrector: str,
    output_corrector: str,
    raw_response: str,
    max_new_tokens: int = 120,
) -> Tuple[Optional[int], str]:
    """
    Si la primera salida está incompleta o mal formateada, pedimos una versión reparada.
    Esta reparación solo intenta rescatar tag y explanation; no cambia el input ni el output.
    """
    prompt = construír_prompt_reparacion(input_corrector, output_corrector, raw_response)
    repaired_response = xerar_desde_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    tag, explanation = extraer_tag_e_explanation(repaired_response)
    return tag, explanation


def construír_explicacion_minima(tag: int, input_corrector: str, output_corrector: str, motivo_extra: str = "") -> str:
    """
    Construimos una explicación de respaldo en gallego.
    Solo la utilizamos si el modelo no nos ha dado una explanation utilizable pero sí necesitamos garantizar
    que el campo explanation nunca quede vacío.
    """
    iguais = str(input_corrector).strip() == str(output_corrector).strip()

    if tag == 0:
        if iguais:
            base = (
                "A etiqueta 0 asígnase porque o output_corrector coincide co input_corrector "
                "e non era necesario introducir cambios adicionais."
            )
        else:
            base = (
                "A etiqueta 0 asígnase porque o output_corrector modifica o input_corrector "
                "e a saída resultante é gramaticalmente correcta."
            )
    else:
        if iguais:
            base = (
                "A etiqueta 1 asígnase porque o output_corrector coincide co input_corrector, "
                "pero o corrector non resolveu o problema presente na entrada."
            )
        else:
            base = (
                "A etiqueta 1 asígnase porque o output_corrector non pode considerarse correcto "
                "xa que introduce un problema gramatical novo."
            )

    if motivo_extra:
        return f"{base} {motivo_extra}".strip()

    return base


def garantir_tag_e_explanation(
    model,
    tokenizer,
    input_corrector: str,
    output_corrector: str,
    tag: Optional[int],
    explanation: str,
    raw_response: str,
    max_new_tokens: int,
) -> Tuple[int, str]:
    """
    Nuestra prioridad aquí es:
    1. que la tag sea siempre 0 o 1
    2. que explanation nunca quede vacía
    3. que explanation justifique la etiqueta asignada

    Estrategia:
    - primero intentamos usar la respuesta original del modelo
    - si está incompleta, lanzamos una reparación
    - si aun así no conseguimos una salida válida, usamos una salida conservadora
    """
    tag = normalizar_tag(tag)
    explanation = str(explanation).strip() if explanation is not None else ""

    if tag in (0, 1) and explanation:
        return tag, explanation

    if raw_response and raw_response.strip():
        try:
            repaired_tag, repaired_explanation = reparar_resposta_incompleta(
                model=model,
                tokenizer=tokenizer,
                input_corrector=input_corrector,
                output_corrector=output_corrector,
                raw_response=raw_response,
                max_new_tokens=min(120, max_new_tokens),
            )
            repaired_tag = normalizar_tag(repaired_tag)
            repaired_explanation = (
                str(repaired_explanation).strip() if repaired_explanation is not None else ""
            )

            if repaired_tag in (0, 1):
                if not repaired_explanation:
                    repaired_explanation = construír_explicacion_minima(
                        repaired_tag, input_corrector, output_corrector
                    )
                return repaired_tag, repaired_explanation
        except Exception:
            pass

    if tag in (0, 1):
        return tag, construír_explicacion_minima(tag, input_corrector, output_corrector)

    return 1, construír_explicacion_minima(
        1,
        input_corrector,
        output_corrector,
        motivo_extra=(
            "Non foi posible extraer unha avaliación válida do modelo, "
            "polo que se aplica unha saída conservadora que require revisión manual."
        ),
    )


def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columnas_orixinais = df.columns.tolist()

    if "input_corrector" not in df.columns or "output_corrector" not in df.columns:
        renames = {}
        if "Orixinal" in df.columns:
            renames["Orixinal"] = "input_corrector"
        if "Corrección" in df.columns:
            renames["Corrección"] = "output_corrector"
        if "Correccion" in df.columns:
            renames["Correccion"] = "output_corrector"
        df = df.rename(columns=renames)

    if "input_corrector" not in df.columns or "output_corrector" not in df.columns:
        raise ValueError(
            "No encuentro las columnas necesarias. Columnas disponibles: "
            + ", ".join(map(str, columnas_orixinais))
        )

    df["input_corrector"] = df["input_corrector"].fillna("").astype(str)
    df["output_corrector"] = df["output_corrector"].fillna("").astype(str)

    # Si retomamos un fichero viejo, migramos automáticamente las columnas antiguas a las nuevas.
    if "tag" not in df.columns and "selene_label" in df.columns:
        df["tag"] = df["selene_label"]

    if "explanation" not in df.columns and "selene_explanation" in df.columns:
        df["explanation"] = df["selene_explanation"]

    if "tag" not in df.columns:
        df["tag"] = None

    if "explanation" not in df.columns:
        df["explanation"] = None

    return df


def cargar_resume_se_existe(df_actual: pd.DataFrame, output_xlsx: str, resume: bool) -> pd.DataFrame:
    """
    Si queremos resume y ya existe un Excel de salida, copiamos lo ya procesado.
    """
    if not resume:
        return df_actual

    if not os.path.exists(output_xlsx):
        return df_actual

    try:
        df_prev = pd.read_excel(output_xlsx)
        df_prev = preparar_dataframe(df_prev)
    except Exception as e:
        print(f"No he podido retomar desde {output_xlsx}: {e}")
        return df_actual

    if len(df_prev) != len(df_actual):
        print("No retomo porque el número de filas del fichero previo no coincide con el Excel actual.")
        return df_actual

    for col in ["tag", "explanation"]:
        if col in df_prev.columns:
            df_actual[col] = df_prev[col]

    xa_feitas = sum(fila_ya_feita(row) for _, row in df_actual.iterrows())
    print(f"Retomando desde fichero previo. Filas ya procesadas: {xa_feitas}")
    return df_actual


def gardar_resultados(df: pd.DataFrame, output_xlsx: str):
    """
    Guardamos solo las cuatro columnas finales que necesitamos.
    """
    out_dir = os.path.dirname(output_xlsx)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_salida = df[["input_corrector", "output_corrector", "tag", "explanation"]].copy()
    df_salida["tag"] = pd.to_numeric(df_salida["tag"], errors="coerce").astype("Int64")

    df_salida.to_excel(output_xlsx, index=False)
    output_csv = os.path.splitext(output_xlsx)[0] + ".csv"
    df_salida.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Guardado Excel: {output_xlsx}")
    print(f"Guardado CSV:   {output_csv}")


def fila_ya_feita(row: pd.Series) -> bool:
    """
    Consideramos que una fila ya está hecha si ya tiene una tag válida y una explanation no vacía.
    """
    tag = normalizar_tag(row.get("tag", None))
    explanation = str(row.get("explanation", "") if row.get("explanation", "") is not None else "").strip()
    return tag in (0, 1) and explanation != ""


def main():
    """
    Flujo principal del script:
    1. leemos el Excel
    2. preparamos columnas
    3. cargamos modelo y tokenizer
    4. recorremos fila a fila
    5. garantizamos siempre tag 0/1 + explanation
    6. guardamos con el formato final de cuatro columnas
    """
    args = parse_args()

    print("Leyendo Excel...")
    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet_name)
    df = preparar_dataframe(df)
    df = cargar_resume_se_existe(df, args.output_xlsx, args.resume)

    print("Cargando modelo y tokenizer...")
    model, tokenizer, _, modo_real = cargar_modelo_y_tokenizer(
        args.adapter_path,
        device=args.device,
        quantization=args.quantization,
    )

    total = len(df)
    pendentes = 0
    for _, row in df.iterrows():
        if not fila_ya_feita(row):
            pendentes += 1

    print(f"Número total de filas: {total}")
    print(f"Filas pendientes de evaluar: {pendentes}")
    print(f"Modo real de carga del modelo: {modo_real}")

    for i, row in df.iterrows():
        if fila_ya_feita(row):
            print(f"[{i+1}/{total}] ya estaba procesada; se salta.")
            continue

        input_corrector = row["input_corrector"]
        output_corrector = row["output_corrector"]

        raw_response = ""
        try:
            label, explanation, raw_response = inferir_unha(
                model=model,
                tokenizer=tokenizer,
                input_corrector=input_corrector,
                output_corrector=output_corrector,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as e:
            label = None
            explanation = ""
            raw_response = f"ERRO NA INFERENCIA: {str(e)}"
            limpar_cuda()

        label, explanation = garantir_tag_e_explanation(
            model=model,
            tokenizer=tokenizer,
            input_corrector=input_corrector,
            output_corrector=output_corrector,
            tag=label,
            explanation=explanation,
            raw_response=raw_response,
            max_new_tokens=args.max_new_tokens,
        )

        df.at[i, "tag"] = label
        df.at[i, "explanation"] = explanation

        preview = explanation[:120].replace("\n", " ") if explanation else ""
        print(f"[{i+1}/{total}] tag={label} | explanation={preview}")

        if (i + 1) % args.save_every == 0:
            gardar_resultados(df, args.output_xlsx)

    gardar_resultados(df, args.output_xlsx)
    print("Proceso terminado.")


if __name__ == "__main__":
    main()