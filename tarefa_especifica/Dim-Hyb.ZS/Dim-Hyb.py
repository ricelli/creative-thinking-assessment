import csv
import json
import time
import urllib.request
import sys
from pathlib import Path


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL = "qwen3:14b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# ------------------------------------------------------------
# Pasta de entrada
#
# Pode ser informada na linha de comando:
#
# python avaliar_questoes.py /caminho/da/pasta
#
# Se não for informada, usa a pasta atual.
# ------------------------------------------------------------

INPUT_DIR = Path(
    sys.argv[1] if len(sys.argv) > 1 else "."
)


# ------------------------------------------------------------
# Quantidade máxima de respostas por arquivo
#
# None = processa todas
# 10   = processa somente as 10 primeiras
# ------------------------------------------------------------

MAX_EXAMPLES = None


# ============================================================
# LEITURA DO PROMPT
# ============================================================

def carregar_prompt(prompt_file):
    """
    Lê o arquivo de prompt.

    Exemplo:

        ABC.txt

    O prompt deve conter os marcadores:

        {TASK_DESCRIPTION}
        {ANSWER}

    Esses marcadores serão substituídos pelos valores
    das respectivas colunas do CSV.
    """

    if not prompt_file.exists():

        raise FileNotFoundError(
            f"Arquivo de prompt não encontrado: "
            f"{prompt_file}"
        )

    with open(
        prompt_file,
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

    data = json.dumps(
        payload,
        ensure_ascii=False
    ).encode("utf-8")

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
# SALVAR CSV DE SAÍDA
# ============================================================

def salvar_saida(
    output_path,
    resultados
):

    campos = [
        "question",
        "task_description",
        "response",
        "code0_3",
        "resposta_qwen",
        "tempo_segundos"
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
            delimiter=",",
            quoting=csv.QUOTE_ALL
        )

        writer.writeheader()

        writer.writerows(resultados)


# ============================================================
# PROCESSAR UM ARQUIVO / UMA QUESTÃO
# ============================================================

def processar_arquivo(
    prefixo,
    prompt_file,
    csv_file
):

    print()
    print("=" * 70)
    print(f"ARQUIVO: {prefixo}")
    print("=" * 70)

    print(
        f"Prompt : {prompt_file.name}"
    )

    print(
        f"CSV    : {csv_file.name}"
    )

    print(
        f"Modelo : {MODEL}"
    )

    # --------------------------------------------------------
    # CARREGA PROMPT
    # --------------------------------------------------------

    try:

        prompt_template = carregar_prompt(
            prompt_file
        )

    except Exception as e:

        print(
            f"ERRO ao carregar prompt: {e}"
        )

        return False

    # --------------------------------------------------------
    # VERIFICA MARCADORES DO PROMPT
    # --------------------------------------------------------

    marcadores_obrigatorios = [
        "{TASK_DESCRIPTION}",
        "{ANSWER}"
    ]

    marcadores_faltantes = []

    for marcador in marcadores_obrigatorios:

        if marcador not in prompt_template:

            marcadores_faltantes.append(
                marcador
            )

    if marcadores_faltantes:

        print()
        print(
            "ERRO: o prompt não contém "
            "os marcadores obrigatórios:"
        )

        for marcador in marcadores_faltantes:

            print(
                f"    {marcador}"
            )

        print()

        return False

    # --------------------------------------------------------
    # LEITURA DO CSV
    # --------------------------------------------------------

    try:

        with open(
            csv_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.DictReader(f)

            rows = list(reader)

    except Exception as e:

        print(
            f"ERRO ao ler CSV: {e}"
        )

        return False

    print()
    print(
        f"Total de registros no CSV: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # VERIFICA SE EXISTEM REGISTROS
    # --------------------------------------------------------

    if not rows:

        print(
            "ERRO: o CSV não possui registros."
        )

        return False

    # --------------------------------------------------------
    # VERIFICA COLUNAS OBRIGATÓRIAS
    # --------------------------------------------------------

    colunas_obrigatorias = [
        "question",
        "task_description",
        "response",
        "code0_3"
    ]

    colunas_encontradas = list(
        rows[0].keys()
    )

    colunas_faltantes = []

    for coluna in colunas_obrigatorias:

        if coluna not in colunas_encontradas:

            colunas_faltantes.append(
                coluna
            )

    if colunas_faltantes:

        print()
        print(
            "ERRO: o CSV não possui "
            "todas as colunas obrigatórias."
        )

        print()
        print(
            "Colunas faltantes:"
        )

        for coluna in colunas_faltantes:

            print(
                f"    - {coluna}"
            )

        print()
        print(
            "Colunas encontradas:"
        )

        for coluna in colunas_encontradas:

            print(
                f"    - {coluna}"
            )

        return False

    # --------------------------------------------------------
    # LIMITA QUANTIDADE DE EXEMPLOS
    # --------------------------------------------------------

    if MAX_EXAMPLES is not None:

        rows = rows[:MAX_EXAMPLES]

    print(
        f"Registros que serão executados: "
        f"{len(rows)}"
    )

    print()

    # --------------------------------------------------------
    # CAMINHO DO ARQUIVO DE SAÍDA
    # --------------------------------------------------------

    output_path = INPUT_DIR / (
        f"{prefixo}_resultados.csv"
    )

    # --------------------------------------------------------
    # PROCESSAMENTO
    # --------------------------------------------------------

    resultados = []

    for i, row in enumerate(
        rows,
        start=1
    ):

        # ----------------------------------------------------
        # LÊ OS CAMPOS DO CSV
        # ----------------------------------------------------

        question = row.get(
            "question",
            ""
        )

        task_description = row.get(
            "task_description",
            ""
        )

        response = row.get(
            "response",
            ""
        )

        code0_3 = row.get(
            "code0_3",
            ""
        )

        # ----------------------------------------------------
        # MONTA O PROMPT
        # ----------------------------------------------------

        prompt = prompt_template.replace(
            "{TASK_DESCRIPTION}",
            task_description
        )

        prompt = prompt.replace(
            "{ANSWER}",
            response
        )

        # ----------------------------------------------------
        # MOSTRA PROGRESSO
        # ----------------------------------------------------

        print(
            f"[{i}/{len(rows)}] "
            f"Question: {question}"
        )

        try:

            # ------------------------------------------------
            # CONSULTA O QWEN
            # ------------------------------------------------

            resposta_qwen, tempo = (
                consultar_qwen(prompt)
            )

            # ------------------------------------------------
            # GUARDA RESULTADO
            # ------------------------------------------------

            resultados.append({

                "question":
                    question,

                "task_description":
                    task_description,

                "response":
                    response,

                "code0_3":
                    code0_3,

                "resposta_qwen":
                    resposta_qwen,

                "tempo_segundos":
                    f"{tempo:.2f}"

            })

            # ------------------------------------------------
            # EXIBE RESPOSTA
            # ------------------------------------------------

            print(
                f"    Tempo: {tempo:.2f}s"
            )

            print(
                f"    Qwen: {resposta_qwen}"
            )

            print()

        except Exception as e:

            print(
                f"    ERRO: {e}"
            )

            # ------------------------------------------------
            # MESMO COM ERRO, MANTÉM O REGISTRO
            # ------------------------------------------------

            resultados.append({

                "question":
                    question,

                "task_description":
                    task_description,

                "response":
                    response,

                "code0_3":
                    code0_3,

                "resposta_qwen":
                    f"ERRO: {e}",

                "tempo_segundos":
                    ""

            })

    # ========================================================
    # SALVAR RESULTADOS
    # ========================================================

    try:

        salvar_saida(
            output_path,
            resultados
        )

    except Exception as e:

        print()
        print(
            f"ERRO ao salvar resultados: {e}"
        )

        return False

    # ========================================================
    # RESULTADO DO ARQUIVO
    # ========================================================

    print()
    print("-" * 70)

    print(
        "Resultados salvos em:"
    )

    print(
        f"    {output_path}"
    )

    print("-" * 70)

    return True


# ============================================================
# ENCONTRAR PARES TXT + CSV
# ============================================================

def encontrar_arquivos():

    """
    Procura arquivos no formato:

        ABC.txt
        ABC.csv

        DEF.txt
        DEF.csv

    O nome deve possuir exatamente três caracteres.

    Para cada TXT, procura o CSV com exatamente o
    mesmo nome-base.

    Retorna:

        (prefixo, arquivo_txt, arquivo_csv)
    """

    arquivos = []

    # --------------------------------------------------------
    # Procura todos os TXT
    # --------------------------------------------------------

    arquivos_txt = sorted(
        INPUT_DIR.glob("*.txt")
    )

    for txt_file in arquivos_txt:

        prefixo = txt_file.stem

        # ----------------------------------------------------
        # O nome deve possuir exatamente 3 letras
        # ----------------------------------------------------

        if len(prefixo) != 3:

            print(
                f"ATENÇÃO: arquivo TXT ignorado "
                f"porque não possui exatamente "
                f"3 caracteres: {txt_file.name}"
            )

            continue

        # ----------------------------------------------------
        # Verifica se são realmente letras
        # ----------------------------------------------------

        if not prefixo.isalpha():

            print(
                f"ATENÇÃO: arquivo TXT ignorado "
                f"porque o nome não possui "
                f"somente letras: {txt_file.name}"
            )

            continue

        # ----------------------------------------------------
        # Procura CSV correspondente
        # ----------------------------------------------------

        csv_file = (
            INPUT_DIR /
            f"{prefixo}.csv"
        )

        if csv_file.exists():

            arquivos.append(
                (
                    prefixo,
                    txt_file,
                    csv_file
                )
            )

        else:

            print(
                f"ATENÇÃO: encontrado "
                f"{txt_file.name}, mas não foi "
                f"encontrado {csv_file.name}."
            )

    return arquivos


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AVALIAÇÃO DE RESPOSTAS COM QWEN")
    print("QWEN3-14B / OLLAMA")
    print("=" * 70)

    # --------------------------------------------------------
    # VERIFICA PASTA
    # --------------------------------------------------------

    if not INPUT_DIR.exists():

        print()
        print(
            "ERRO: pasta de entrada não encontrada:"
        )

        print(
            f"    {INPUT_DIR}"
        )

        sys.exit(1)

    if not INPUT_DIR.is_dir():

        print()
        print(
            "ERRO: o caminho informado não é "
            "uma pasta:"
        )

        print(
            f"    {INPUT_DIR}"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # INFORMAÇÕES
    # --------------------------------------------------------

    print()
    print(
        f"Pasta de entrada : "
        f"{INPUT_DIR.resolve()}"
    )

    print(
        f"Modelo            : "
        f"{MODEL}"
    )

    print(
        f"Ollama            : "
        f"{OLLAMA_URL}"
    )

    print(
        "Temperature        : 0"
    )

    print(
        "Thinking           : false"
    )

    if MAX_EXAMPLES is None:

        print(
            "Máx. exemplos     : todos"
        )

    else:

        print(
            f"Máx. exemplos     : "
            f"{MAX_EXAMPLES}"
        )

    print("=" * 70)

    # ========================================================
    # ENCONTRAR ARQUIVOS
    # ========================================================

    arquivos = encontrar_arquivos()

    # --------------------------------------------------------
    # NENHUM PAR ENCONTRADO
    # --------------------------------------------------------

    if not arquivos:

        print()
        print(
            "ERRO: nenhum par TXT + CSV foi encontrado."
        )

        print()
        print(
            "A estrutura esperada é:"
        )

        print(
            "    ABC.txt"
        )

        print(
            "    ABC.csv"
        )

        print()

        sys.exit(1)

    # --------------------------------------------------------
    # MOSTRA ARQUIVOS ENCONTRADOS
    # --------------------------------------------------------

    print()
    print(
        f"Pares encontrados: "
        f"{len(arquivos)}"
    )

    print()

    for prefixo, txt_file, csv_file in arquivos:

        print(
            f"    {prefixo}"
            f" -> {txt_file.name}"
            f" + {csv_file.name}"
        )

    print()

    # ========================================================
    # PROCESSAR ARQUIVOS
    # ========================================================

    sucessos = 0
    erros = 0

    inicio_total = time.time()

    for indice, (
        prefixo,
        prompt_file,
        csv_file
    ) in enumerate(
        arquivos,
        start=1
    ):

        print()
        print(
            "#" * 70
        )

        print(
            f"ARQUIVO {indice}/{len(arquivos)}"
        )

        print(
            f"Prefixo: {prefixo}"
        )

        print(
            "#" * 70
        )

        try:

            sucesso = processar_arquivo(
                prefixo,
                prompt_file,
                csv_file
            )

            if sucesso:

                sucessos += 1

            else:

                erros += 1

        except KeyboardInterrupt:

            print()
            print(
                "PROCESSAMENTO INTERROMPIDO "
                "PELO USUÁRIO."
            )

            print()
            print(
                "Os arquivos já finalizados "
                "foram mantidos."
            )

            sys.exit(1)

        except Exception as e:

            erros += 1

            print()
            print(
                f"ERRO inesperado no arquivo "
                f"{prefixo}: {e}"
            )

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    tempo_total = (
        time.time() - inicio_total
    )

    print()
    print("=" * 70)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 70)

    print(
        f"Pares encontrados : "
        f"{len(arquivos)}"
    )

    print(
        f"Arquivos processados : "
        f"{sucessos}"
    )

    print(
        f"Arquivos com erro : "
        f"{erros}"
    )

    print(
        f"Tempo total : "
        f"{tempo_total:.2f} segundos"
    )

    print()

    print(
        "Arquivos gerados:"
    )

    for prefixo, _, _ in arquivos:

        output = (
            INPUT_DIR /
            f"{prefixo}_resultados.csv"
        )

        if output.exists():

            print(
                f"    {output.name}"
            )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()