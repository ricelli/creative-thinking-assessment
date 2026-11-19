import os
import time
import pandas as pd

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig
)

from tqdm import tqdm

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

PROJECT_ID = "project-c9a2046f-e587-4668-a90"
LOCATION = "us-central1"

PROMPT_FILE = "Avaliação de criatividade por dimensão na tarefa XPTO.txt"
DATASET_FILE = "pt_XPTO_dimensoes.csv"

QUESTION_FILES = {
    "T240Q02": "T240Q02.txt",
    "T300Q02": "T300Q02.txt",
    "T350Q02": "T350Q02.txt",
    "T570Q01": "T570Q01.txt",
}

OUTPUT_DIR = "resultados"
ERROR_LOG = "erros.log"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# VERTEX AI
# ==========================================================

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION
)

model = GenerativeModel(
    "gemini-2.5-flash"
)

# ==========================================================
# PROMPT BASE
# ==========================================================

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()

# ==========================================================
# DESCRIÇÕES DAS QUESTÕES
# ==========================================================

task_descriptions = {}

for question, filename in QUESTION_FILES.items():

    if not os.path.exists(filename):
        print(f"Arquivo não encontrado: {filename}")
        continue

    with open(filename, "r", encoding="utf-8") as f:
        task_descriptions[question] = f.read().strip()

# ==========================================================
# DATASET
# ==========================================================

df = pd.read_csv(DATASET_FILE)

print(f"\nTotal de registros: {len(df)}")

# ==========================================================
# FUNÇÃO DE AVALIAÇÃO
# ==========================================================

def avaliar_resposta(task_description, answer):

    prompt = (
        PROMPT_TEMPLATE
        .replace("{TASK_DESCRIPTION}", task_description)
        .replace("{ANSWER}", str(answer))
    )

    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            temperature=0
        )
    )

    texto = response.text.strip()

    try:

        valores = [
            float(x.strip())
            for x in texto.split(",")
        ]

        if len(valores) != 5:
            raise ValueError(
                f"Esperado 5 valores, recebido: {texto}"
            )

        return valores

    except Exception:

        raise ValueError(
            f"Não foi possível interpretar: {texto}"
        )

# ==========================================================
# LIMPA LOG
# ==========================================================

with open(ERROR_LOG, "w", encoding="utf-8") as f:
    f.write("")

# ==========================================================
# PROCESSAMENTO
# ==========================================================

for question_id, task_description in task_descriptions.items():

    print("\n" + "=" * 60)
    print(f"Processando questão {question_id}")
    print("=" * 60)

    subset = df[
        df["question"] == question_id
    ].copy()

    print(f"Total de respostas: {len(subset)}")

    resultados = []

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{question_id}_avaliado.csv"
    )

    inicio = time.time()

    for idx, row in tqdm(
        subset.iterrows(),
        total=len(subset),
        desc=question_id
    ):

        try:

            notas = avaliar_resposta(
                task_description,
                row["response"]
            )

            resultados.append({
                "question": row["question"],
                "response": row["response"],
                "originalidade": notas[0],
                "novidade": notas[1],
                "surpresa": notas[2],
                "adequacao": notas[3],
                "valor_utilidade": notas[4]
            })

        except Exception as e:

            with open(
                ERROR_LOG,
                "a",
                encoding="utf-8"
            ) as log:

                log.write(
                    f"{question_id} | linha={idx} | {e}\n"
                )

            resultados.append({
                "question": row["question"],
                "response": row["response"],
                "originalidade": None,
                "novidade": None,
                "surpresa": None,
                "adequacao": None,
                "valor_utilidade": None
            })

        # Salva a cada 50 respostas
        if len(resultados) % 50 == 0:

            pd.DataFrame(resultados).to_csv(
                output_file,
                index=False,
                encoding="utf-8-sig"
            )

    # Salva resultado final
    pd.DataFrame(resultados).to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    tempo_total = (
        time.time() - inicio
    ) / 60

    print(
        f"\nArquivo salvo: {output_file}"
    )

    print(
        f"Tempo: {tempo_total:.2f} minutos"
    )

print("\nProcessamento concluído.")
print(f"Log de erros: {ERROR_LOG}")
