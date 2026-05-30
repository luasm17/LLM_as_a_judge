#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
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

    La diferencia principal con la versión fine-tuneada es que ahora
    ya no pasamos una ruta a un adapter LoRA, sino el nombre o la ruta
    del modelo base que queremos cargar directamente.

    Lo dejo por defecto con Selene base, pero podemos cambiarlo
    si en algún momento queremos probar otro checkpoint base.
    """
    parser = argparse.ArgumentParser(
        description="Evaluar correcciones con Selene base, sin fine-tuning."
    )
    parser.add_argument("--input_xlsx", required=True, help="Ruta al Excel de entrada.")
    parser.add_argument(
        "--base_model_name",
        default="AtlaAI/Selene-1-Mini-Llama-3.1-8B",
        help="Nombre o ruta del modelo base que vamos a usar sin fine-tuning.",
    )
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
    Aquí elegimos el tipo de dato más conveniente según el hardware.

    Si la GPU soporta bf16, lo usamos porque suele ser una opción muy buena
    en modelos grandes. Si no, pasamos a fp16. En CPU dejamos float32
    porque es la opción más estable.
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def bnb_dispoñible() -> bool:
    """
    Esta función nos dice si bitsandbytes está instalado.

    Solo si está disponible podremos usar cuantización en 8bit o 4bit.
    Si no está, el script caerá de forma natural al modo normal.
    """
    return importlib.util.find_spec("bitsandbytes") is not None


def limpar_cuda():
    """
    Aquí liberamos memoria de CUDA cuando hace falta.

    Esto es útil si una carga falla y luego queremos intentar otra estrategia
    sin arrastrar memoria reservada de la anterior.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def e_oom(exc: Exception) -> bool:
    """
    Aquí detectamos si una excepción parece ser un error de memoria.

    Lo usamos para que el modo 'auto' pueda probar otra forma de carga
    si la anterior falla por falta de VRAM.
    """
    texto = str(exc).lower()
    return (
        isinstance(exc, torch.OutOfMemoryError)
        or "out of memory" in texto
        or "cuda out of memory" in texto
        or ("cublas" in texto and "alloc" in texto)
    )


def cargar_tokenizer(tokenizer_path: str):
    """
    Aquí cargamos el tokenizer del modelo base.

    Usamos AutoTokenizer porque así respetamos el tokenizer propio del modelo
    y su chat template. Si por algún motivo falla el backend rápido,
    intentamos una segunda vía con PreTrainedTokenizerFast.
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
    Aquí construimos la configuración de cuantización.

    Solo se usa si pedimos 8bit o 4bit. Si no, devolvemos None
    y el modelo se cargará de forma normal.
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
    Aquí decidimos en qué orden intentamos cargar el modelo.

    Si estamos en modo auto:
    - primero intentamos 8bit
    - luego 4bit
    - y por último none

    Si no hay bitsandbytes o no estamos en GPU, cargamos directamente en modo normal.
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
    Aquí cargamos el modelo base.

    Esta es la diferencia clave con tu script del fine-tuning:
    ahora no cargamos el modelo base y después un LoRA encima,
    sino que cargamos directamente el checkpoint base tal cual.
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


def cargar_modelo_y_tokenizer(base_model_name: str, device: int = 0, quantization: str = "auto"):
    """
    Esta función se encarga de cargar:
    - el tokenizer del modelo base
    - el propio modelo base

    Como aquí no hay LoRA, no hacemos ninguna carga con PEFT.
    Solo cargamos Selene base y lo dejamos en eval().
    """
    print("Modelo base:", base_model_name)
    print("Cargando tokenizer desde:", base_model_name)
    tokenizer = cargar_tokenizer(base_model_name)

    dtype = escoller_dtype()
    errors = []

    for modo in estratexias_carga(quantization):
        try:
            print(f"Cargando modelo base con el modo: {modo}")
            limpar_cuda()
            model = cargar_modelo_base(base_model_name, dtype=dtype, device=device, quantization=modo)
            model.eval()
            print(f"Modelo base cargado correctamente con el modo: {modo}")
            return model, tokenizer, modo
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
    Aquí montamos el prompt principal exactamente con el formato
    que espera nuestro juez.
    """
    return PROMPT_BASE.format(
        input_corrector=str(input_corrector).strip(),
        output_corrector=str(output_corrector).strip(),
    )


def construír_prompt_reparacion(input_corrector: str, output_corrector: str, raw_response: str) -> str:
    """
    Si la primera respuesta viene mal formateada o incompleta,
    aquí construimos un segundo prompt mucho más corto para obligar
    al modelo a devolver una tag y una explanation válidas.
    """
    return PROMPT_REPARACION.format(
        input_corrector=str(input_corrector).strip(),
        output_corrector=str(output_corrector).strip(),
        raw_response=str(raw_response).strip(),
    )


def obter_terminators(tokenizer) -> List[int]:
    """
    Aquí reunimos los tokens de finalización que puedan existir
    en este tokenizer.

    Esto ayuda a cortar la generación de forma limpia.
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
    Aquí extraemos de la respuesta del modelo los dos elementos
    que realmente nos importan:
    - la tag
    - la explanation

    Lo hacemos de forma tolerante porque a veces los modelos
    no respetan el formato al cien por cien.
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
    Aquí limpiamos la respuesta bruta del modelo.

    Si el modelo repite parte del formato o añade texto extraño,
    intentamos recortar desde el último 'input_corrector:'.
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
    Aquí convertimos la tag a entero solo si realmente es 0 o 1.
    Si no lo es, devolvemos None.
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
    Aquí recuperamos el dispositivo real donde está cargado el modelo.
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
    Esta función concentra la generación del modelo.

    Lo importante aquí es que seguimos usando apply_chat_template(),
    igual que antes, para respetar el formato conversacional
    que necesita Selene.
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
    Aquí hacemos una inferencia principal y devolvemos:
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
    Si la primera salida viene incompleta o mal formateada,
    aquí intentamos repararla con un segundo prompt más estricto.
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
    Aquí construimos una explicación mínima de respaldo.

    Solo la usamos si el modelo no devuelve una explanation útil
    pero necesitamos garantizar que el campo nunca quede vacío.
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
                "A etiqueta 1 asígnase porque o output_corrector coincide co input_corrector "
                "e o corrector non resolveu o problema presente na entrada."
            )
        else:
            base = (
                "A etiqueta 1 asígnase porque o output_corrector non pode considerarse correcto, "
                "xa que introduce un problema gramatical."
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
    Este bloque garantiza la robustez final.

    Queremos que:
    - la tag sea siempre 0 o 1
    - la explanation nunca quede vacía
    - la explanation justifique la etiqueta

    Para eso:
    1. intentamos usar la salida original
    2. si falla, intentamos una reparación
    3. si sigue fallando, aplicamos una salida conservadora
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
    """
    Aquí preparamos el DataFrame de trabajo.

    Si el Excel viene con columnas Orixinal y Corrección, las renombramos
    a los nombres internos con los que vamos a trabajar:
    - input_corrector
    - output_corrector

    Además, nos aseguramos de que existan las columnas finales:
    - tag
    - explanation
    """
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

    if "tag" not in df.columns:
        df["tag"] = None

    if "explanation" not in df.columns:
        df["explanation"] = None

    return df


def cargar_resume_se_existe(df_actual: pd.DataFrame, output_xlsx: str, resume: bool) -> pd.DataFrame:
    """
    Si el usuario pide resume y ya existe un fichero de salida,
    aquí copiamos las filas ya procesadas al DataFrame actual.
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
    Aquí guardamos los resultados finales en:
    - Excel
    - CSV

    Y dejamos solo las cuatro columnas finales que te interesan.
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
    Consideramos que una fila ya está hecha si ya tiene:
    - una tag válida
    - una explanation no vacía
    """
    tag = normalizar_tag(row.get("tag", None))
    explanation = str(row.get("explanation", "") if row.get("explanation", "") is not None else "").strip()
    return tag in (0, 1) and explanation != ""


def main():
    """
    Flujo principal del script:

    1. leemos el Excel
    2. preparamos las columnas
    3. cargamos Selene base y su tokenizer
    4. evaluamos fila a fila
    5. garantizamos siempre tag y explanation
    6. guardamos el resultado final en Excel y CSV
    """
    args = parse_args()

    print("Leyendo Excel...")
    df = pd.read_excel(args.input_xlsx, sheet_name=args.sheet_name)
    df = preparar_dataframe(df)
    df = cargar_resume_se_existe(df, args.output_xlsx, args.resume)

    print("Cargando modelo base y tokenizer...")
    model, tokenizer, modo_real = cargar_modelo_y_tokenizer(
        args.base_model_name,
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