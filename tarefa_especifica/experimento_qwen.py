import csv
import json
import re
import time
import urllib.request
import urllib.error
import sys
from pathlib import Path


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL = "qwen3:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Arquivo de entrada
INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "entrada.csv"

# Arquivo de saída principal
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "resultado_qwen.csv"

# Arquivo externo contendo o prompt
PROMPT_FILE = Path(
    "/home/ricelli/testes-quen/Prompt GCI cross target T370Q01.txt"
)

# ------------------------------------------------------------
# Quantidade de exemplos
#
# 10 = processa somente os 10 primeiros
# None = processa todos
# ------------------------------------------------------------

MAX_EXAMPLES = None


# ============================================================
# LEITURA DO PROMPT
# ============================================================

def carregar_prompt():
    """
    Lê o prompt do arquivo externo.

    O arquivo deve conter o marcador:

        {frase_usuario}

    Esse marcador será substituído pela resposta
    do participante (coluna 'response' do CSV).
    """

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de prompt não encontrado: {PROMPT_FILE}"
        )

    with open(
        PROMPT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        prompt = f.read()

    return prompt


# ============================================================
# CHAMADA AO OLLAMA
# ============================================================

def consultar_qwen(prompt):

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    inicio = time.time()

    with urllib.request.urlopen(
        request,
        timeout=300
    ) as response:

        resultado = json.loads(
            response.read().decode("utf-8")
        )

    tempo = time.time() - inicio

    return (
        resultado.get("response", "").strip(),
        tempo
    )


# ============================================================
# EXTRAÇÃO DA RESPOSTA
# ============================================================

def extrair_resposta(texto):
    """
    Espera que o modelo responda no formato:

        Nota (1 ou 2) | Justificativa da avaliação | Grau de confiança

    Exemplo:

        2 | Apresenta uma perspectiva inesperada e original. | 9

    Retorna:

        nota
        justificativa
        confianca
    """

    if not texto:
        return "", "", ""

    # --------------------------------------------------------
    # Remove linhas vazias
    # --------------------------------------------------------

    linhas = [
        linha.strip()
        for linha in texto.strip().splitlines()
        if linha.strip()
    ]

    if not linhas:
        return "", "", ""

    # --------------------------------------------------------
    # Usa a primeira linha da resposta
    # --------------------------------------------------------

    linha = linhas[0]

    # --------------------------------------------------------
    # Divide pelos pipes
    # --------------------------------------------------------

    partes = [
        p.strip()
        for p in linha.split("|")
    ]

    nota = ""
    justificativa = ""
    confianca = ""

    if len(partes) >= 3:

        # ----------------------------------------------------
        # NOTA
        # ----------------------------------------------------

        match_nota = re.search(
            r"\b([12])\b",
            partes[0]
        )

        if match_nota:
            nota = match_nota.group(1)

        # ----------------------------------------------------
        # JUSTIFICATIVA
        # ----------------------------------------------------

        justificativa = partes[1]

        # ----------------------------------------------------
        # CONFIANÇA
        # ----------------------------------------------------

        match_conf = re.search(
            r"\b(10|[0-9])\b\s*$",
            partes[-1]
        )

        if match_conf:
            confianca = match_conf.group(1)

    return (
        nota,
        justificativa,
        confianca
    )


# ============================================================
# CÁLCULO DAS MÉTRICAS
# ============================================================

def calcular_metricas(resultados):
    """
    Calcula:

        - Precisão
        - Revocação
        - F1

    para as classes 1 e 2.

    Também calcula a média macro.

    Classe verdadeira:
        code0_3

    Classe prevista:
        Nota
    """

    metricas = []

    # --------------------------------------------------------
    # Calcula as métricas individualmente para cada classe
    # --------------------------------------------------------

    for classe in ["1", "2"]:

        tp = 0
        fp = 0
        fn = 0
        suporte = 0

        for resultado in resultados:

            verdadeiro = str(
                resultado.get("code0_3", "")
            ).strip()

            previsto = str(
                resultado.get("Nota", "")
            ).strip()

            # ------------------------------------------------
            # Suporte = quantidade de exemplos verdadeiros
            # daquela classe
            # ------------------------------------------------

            if verdadeiro == classe:
                suporte += 1

            # ------------------------------------------------
            # Verdadeiro positivo
            # ------------------------------------------------

            if verdadeiro == classe and previsto == classe:
                tp += 1

            # ------------------------------------------------
            # Falso positivo
            # ------------------------------------------------

            elif verdadeiro != classe and previsto == classe:
                fp += 1

            # ------------------------------------------------
            # Falso negativo
            # ------------------------------------------------

            elif verdadeiro == classe and previsto != classe:
                fn += 1

        # ----------------------------------------------------
        # PRECISÃO
        #
        # TP / (TP + FP)
        # ----------------------------------------------------

        if (tp + fp) > 0:
            precisao = tp / (tp + fp)
        else:
            precisao = 0.0

        # ----------------------------------------------------
        # REVOCAÇÃO
        #
        # TP / (TP + FN)
        # ----------------------------------------------------

        if (tp + fn) > 0:
            revocacao = tp / (tp + fn)
        else:
            revocacao = 0.0

        # ----------------------------------------------------
        # F1
        #
        # 2 * precisão * revocação /
        # (precisão + revocação)
        # ----------------------------------------------------

        if (precisao + revocacao) > 0:
            f1 = (
                2 * precisao * revocacao
                / (precisao + revocacao)
            )
        else:
            f1 = 0.0

        metricas.append({
            "classe": classe,
            "precisao": precisao,
            "revocacao": revocacao,
            "f1": f1,
            "suporte": suporte
        })

    # --------------------------------------------------------
    # MÉDIA MACRO
    # --------------------------------------------------------

    macro_precisao = sum(
        m["precisao"]
        for m in metricas
    ) / len(metricas)

    macro_revocacao = sum(
        m["revocacao"]
        for m in metricas
    ) / len(metricas)

    macro_f1 = sum(
        m["f1"]
        for m in metricas
    ) / len(metricas)

    # Suporte total das classes 1 e 2
    suporte_total = sum(
        m["suporte"]
        for m in metricas
    )

    metricas.append({
        "classe": "macro",
        "precisao": macro_precisao,
        "revocacao": macro_revocacao,
        "f1": macro_f1,
        "suporte": suporte_total
    })

    return metricas


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    # ========================================================
    # VERIFICA ARQUIVO DE ENTRADA
    # ========================================================

    if not input_path.exists():

        print(
            f"ERRO: arquivo de entrada não encontrado: "
            f"{input_path}"
        )

        sys.exit(1)

    # ========================================================
    # VERIFICA ARQUIVO DE PROMPT
    # ========================================================

    if not PROMPT_FILE.exists():

        print(
            f"ERRO: arquivo de prompt não encontrado: "
            f"{PROMPT_FILE}"
        )

        sys.exit(1)

    # ========================================================
    # CARREGA PROMPT
    # ========================================================

    try:

        prompt_template = carregar_prompt()

    except Exception as e:

        print(
            f"ERRO ao carregar o prompt: {e}"
        )

        sys.exit(1)

    # ========================================================
    # INFORMAÇÕES DA EXECUÇÃO
    # ========================================================

    print("=" * 70)
    print("EXPERIMENTO QWEN3-14B")
    print("=" * 70)

    print(f"Entrada      : {input_path}")
    print(f"Saída        : {output_path}")
    print(f"Modelo       : {MODEL}")
    print(f"Prompt       : {PROMPT_FILE}")
    print("Thinking     : false")
    print("Temperature  : 0")

    print("=" * 70)

    # ========================================================
    # LEITURA DO CSV
    # ========================================================

    with open(
        input_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    print(
        f"Total de registros no CSV: {len(rows)}"
    )

    # ========================================================
    # VERIFICA SE EXISTEM REGISTROS
    # ========================================================

    if not rows:

        print(
            "ERRO: o CSV não possui registros."
        )

        sys.exit(1)

    # ========================================================
    # VERIFICA SE EXISTE A COLUNA RESPONSE
    # ========================================================

    if "response" not in rows[0]:

        print(
            "ERRO: a coluna 'response' não foi encontrada "
            "no CSV."
        )

        print(
            f"Colunas encontradas: {list(rows[0].keys())}"
        )

        sys.exit(1)

    # ========================================================
    # VERIFICA SE EXISTE A COLUNA CODE0_3
    # ========================================================

    if "code0_3" not in rows[0]:

        print(
            "ERRO: a coluna 'code0_3' não foi encontrada "
            "no CSV."
        )

        print(
            f"Colunas encontradas: {list(rows[0].keys())}"
        )

        sys.exit(1)

    # ========================================================
    # LIMITA QUANTIDADE DE EXEMPLOS
    # ========================================================

    if MAX_EXAMPLES is not None:

        rows = rows[:MAX_EXAMPLES]

    print(
        f"Registros que serão executados: "
        f"{len(rows)}"
    )

    print()

    # ========================================================
    # PROCESSAMENTO
    # ========================================================

    resultados = []

    for i, row in enumerate(
        rows,
        start=1
    ):

        # ----------------------------------------------------
        # Pega a resposta do CSV
        # ----------------------------------------------------

        response = row.get(
            "response",
            ""
        )

        # ----------------------------------------------------
        # Pega a resposta correta
        # ----------------------------------------------------

        code0_3 = row.get(
            "code0_3",
            ""
        )

        # ----------------------------------------------------
        # Insere a resposta no prompt
        # ----------------------------------------------------

        prompt = prompt_template.replace(
            "{frase_usuario}",
            response
        )

        print(
            f"[{i}/{len(rows)}] Processando..."
        )

        try:

            # ------------------------------------------------
            # Consulta o Qwen
            # ------------------------------------------------

            resposta_qwen, tempo = consultar_qwen(
                prompt
            )

            # ------------------------------------------------
            # Extrai nota, justificativa e confiança
            # ------------------------------------------------

            nota, justificativa, confianca = (
                extrair_resposta(
                    resposta_qwen
                )
            )

            # ------------------------------------------------
            # Guarda os campos desejados
            # ------------------------------------------------

            resultados.append({

                "response": response,

                "code0_3": code0_3,

                "Nota": nota,

                "Justificativa da avaliação":
                    justificativa,

                "Grau de confiança da resposta":
                    confianca
            })

            # ------------------------------------------------
            # Exibe no terminal
            # ------------------------------------------------

            print(
                f"    Resposta correta: {code0_3}"
            )

            print(
                f"    Nota            : {nota}"
            )

            print(
                f"    Justificativa   : "
                f"{justificativa}"
            )

            print(
                f"    Confiança       : "
                f"{confianca}"
            )

            print(
                f"    Tempo           : "
                f"{tempo:.2f}s"
            )

            print(
                f"    Qwen            : "
                f"{resposta_qwen}"
            )

            print()

        except Exception as e:

            # ------------------------------------------------
            # Em caso de erro, mantém as informações
            # necessárias para o CSV
            # ------------------------------------------------

            print(
                f"    ERRO: {e}"
            )

            resultados.append({

                "response": response,

                "code0_3": code0_3,

                "Nota": "",

                "Justificativa da avaliação":
                    "",

                "Grau de confiança da resposta":
                    ""
            })

    # ========================================================
    # SALVAR CSV PRINCIPAL
    # ========================================================

    campos = [

        "response",

        "code0_3",

        "Nota",

        "Justificativa da avaliação",

        "Grau de confiança da resposta"
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=campos,
            delimiter=","
        )

        writer.writeheader()

        writer.writerows(
            resultados
        )

    # ========================================================
    # CALCULAR MÉTRICAS
    # ========================================================

    metricas = calcular_metricas(
        resultados
    )

    # ========================================================
    # NOME DO CSV DE MÉTRICAS
    # ========================================================

    metricas_path = output_path.with_name(
        output_path.stem + "_metricas.csv"
    )

    # ========================================================
    # SALVAR CSV DE MÉTRICAS
    # ========================================================

    campos_metricas = [
        "classe",
        "precisao",
        "revocacao",
        "f1",
        "suporte"
    ]

    with open(
        metricas_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer_metricas = csv.DictWriter(
            f,
            fieldnames=campos_metricas,
            delimiter=","
        )

        writer_metricas.writeheader()

        for metrica in metricas:

            writer_metricas.writerow({

                "classe":
                    metrica["classe"],

                "precisao":
                    f"{metrica['precisao']:.6f}",

                "revocacao":
                    f"{metrica['revocacao']:.6f}",

                "f1":
                    f"{metrica['f1']:.6f}",

                "suporte":
                    metrica["suporte"]
            })

    # ========================================================
    # EXIBIR MÉTRICAS NO TERMINAL
    # ========================================================

    print()
    print("=" * 70)
    print("MÉTRICAS")
    print("=" * 70)

    for metrica in metricas:

        print(
            f"Classe: {metrica['classe']}"
        )

        print(
            f"    Precisão   : "
            f"{metrica['precisao']:.6f}"
        )

        print(
            f"    Revocação  : "
            f"{metrica['revocacao']:.6f}"
        )

        print(
            f"    F1         : "
            f"{metrica['f1']:.6f}"
        )

        print(
            f"    Suporte    : "
            f"{metrica['suporte']}"
        )

        print()

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print("=" * 70)
    print("RESULTADO")
    print("=" * 70)

    print(
        f"Executados : {len(resultados)}"
    )

    print(
        f"Arquivo    : {output_path}"
    )

    print(
        f"Métricas   : {metricas_path}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    main()