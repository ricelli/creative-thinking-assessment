#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Executa o Qwen3:14b sobre a amostra chinesa do OECS/PISA 2022.

Entrada:
    /home/...

Prompts:
    /home/...

Saída:
    /home/...

Características:
- Usa somente bibliotecas da biblioteca padrão do Python.
- Utiliza o Ollama local.
- Modelo: qwen3:14b.
- Processa questão por questão.
- Mantém a resposta original e a resposta completa do Qwen.
- Extrai uma possível classe 1/2 para facilitar a análise posterior.
- Pode ser interrompido e retomado.
- A origem é identificada pelo nome do arquivo XML:
    zho-tap -> China
    zho-hkg -> Hong Kong
    zho-mac -> Macau
"""

import csv
import subprocess
import re
import time
from pathlib import Path


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_AMOSTRA = Path(
    "/home/..."
)

PASTA_PROMPTS = Path(
    "/home/..."
)

PASTA_SAIDA = Path(
    "/home/..."
)

OLLAMA = "/usr/local/bin/ollama"

MODELO = "qwen3:14b"

# Pequena pausa entre chamadas ao Ollama
ESPERA = 0.2


# ============================================================
# IDENTIFICAÇÃO DA ORIGEM
# ============================================================

def identificar_origem(xml_origem):
    """
    Identifica a origem a partir do nome do arquivo XML.

    Exemplos:
        ...zho-tap...xml -> China
        ...zho-hkg...xml -> Hong Kong
        ...zho-mac...xml -> Macau
    """

    nome = xml_origem.lower()

    if "zho-tap" in nome:
        return "China"

    if "zho-hkg" in nome:
        return "Hong Kong"

    if "zho-mac" in nome:
        return "Macau"

    return "Origem desconhecida"


# ============================================================
# CARREGAMENTO DOS PROMPTS
# ============================================================

def carregar_prompt(question):
    """
    Carrega o prompt correspondente à questão.
    """

    arquivo = PASTA_PROMPTS / f"{question}.txt"

    if not arquivo.exists():
        raise FileNotFoundError(
            f"Prompt não encontrado para {question}: {arquivo}"
        )

    return arquivo.read_text(
        encoding="utf-8"
    )


# ============================================================
# EXECUÇÃO DO QWEN
# ============================================================

def executar_qwen(prompt):
    """
    Executa o modelo Qwen através do Ollama.
    """

    try:

        resultado = subprocess.run(
            [
                OLLAMA,
                "run",
                MODELO
            ],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace"
        )

    except Exception as e:

        return "", f"ERRO_EXECUCAO: {e}"

    if resultado.returncode != 0:

        erro = resultado.stderr.strip()

        if not erro:
            erro = (
                f"código de saída {resultado.returncode}"
            )

        return "", f"ERRO_OLLAMA: {erro}"

    resposta = resultado.stdout.strip()

    return resposta, ""


# ============================================================
# EXTRAÇÃO DA CLASSIFICAÇÃO
# ============================================================

def extrair_nota(resposta):
    """
    Tenta extrair a classificação 1 ou 2 da resposta do Qwen.

    Formatos esperados:

        Nota 1 | ...
        Nota 2 | ...

    ou:

        1 | ...
        2 | ...

    A resposta original completa permanece preservada em
    resposta_qwen.
    """

    if not resposta:
        return ""

    # --------------------------------------------------------
    # Formato:
    # Nota 1 | ...
    # Nota 2 | ...
    # --------------------------------------------------------

    padrao = re.search(
        r"Nota\s*[:\(]?\s*([12])",
        resposta,
        re.IGNORECASE
    )

    if padrao:
        return padrao.group(1)

    # --------------------------------------------------------
    # Formato:
    # 1 | justificativa | confiança
    # 2 | justificativa | confiança
    # --------------------------------------------------------

    padrao = re.match(
        r"\s*([12])\s*\|",
        resposta
    )

    if padrao:
        return padrao.group(1)

    return ""


# ============================================================
# ARQUIVO DE RESULTADOS
# ============================================================

def arquivo_resultados(question):
    """
    Retorna o arquivo CSV de resultados da questão.
    """

    return PASTA_SAIDA / f"{question}_resultados.csv"


# ============================================================
# CARREGAR PROCESSADOS
# ============================================================

def carregar_processados(arquivo):
    """
    Identifica quais respostas já foram processadas.

    A chave utilizada é:

        xml_origem + student

    Isso permite interromper e posteriormente continuar
    a execução sem repetir as respostas já gravadas.
    """

    processados = set()

    if not arquivo.exists():
        return processados

    with arquivo.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            chave = (
                row.get("xml_origem", ""),
                row.get("student", "")
            )

            processados.add(chave)

    return processados


# ============================================================
# CRIAR CABEÇALHO
# ============================================================

def criar_cabecalho(arquivo):
    """
    Cria o CSV de resultados caso ele ainda não exista.
    """

    if arquivo.exists():
        return

    with arquivo.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "origem",
                "xml_origem",
                "question",
                "student",
                "code0",
                "classe_real",
                "resposta",
                "resposta_qwen",
                "classe_qwen",
                "erro"
            ]
        )

        writer.writeheader()


# ============================================================
# LEITURA DA AMOSTRA
# ============================================================

def carregar_amostra():
    """
    Carrega a amostra chinesa.
    """

    with ARQUIVO_AMOSTRA.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        # Verificação básica do cabeçalho
        campos_obrigatorios = [
            "xml_origem",
            "question",
            "student",
            "code0",
            "classe_real",
            "resposta"
        ]

        campos = reader.fieldnames or []

        faltantes = [
            campo
            for campo in campos_obrigatorios
            if campo not in campos
        ]

        if faltantes:

            raise ValueError(
                "Campos obrigatórios ausentes no CSV: "
                + ", ".join(faltantes)
                + "\n"
                + "Campos encontrados: "
                + ", ".join(campos)
            )

        return list(reader)


# ============================================================
# PROCESSAMENTO
# ============================================================

def main():

    # --------------------------------------------------------
    # Verificações iniciais
    # --------------------------------------------------------

    if not ARQUIVO_AMOSTRA.exists():

        print(
            "ERRO: arquivo de amostra não encontrado:"
        )

        print(ARQUIVO_AMOSTRA)

        return

    if not PASTA_PROMPTS.exists():

        print(
            "ERRO: pasta de prompts não encontrada:"
        )

        print(PASTA_PROMPTS)

        return

    PASTA_SAIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Carregar amostra
    # --------------------------------------------------------

    try:

        amostra = carregar_amostra()

    except Exception as e:

        print()
        print("ERRO AO LER A AMOSTRA:")
        print(e)

        return

    # --------------------------------------------------------
    # Cabeçalho
    # --------------------------------------------------------

    print("=" * 70)
    print("EXECUÇÃO QWEN - AMOSTRA CHINESA")
    print("=" * 70)

    print(
        f"Arquivo: {ARQUIVO_AMOSTRA}"
    )

    print(
        f"Modelo:  {MODELO}"
    )

    print(
        f"Total de respostas na amostra: {len(amostra)}"
    )

    print()

    # --------------------------------------------------------
    # Mostrar distribuição das origens
    # --------------------------------------------------------

    distribuicao_origem = {}

    for row in amostra:

        origem = identificar_origem(
            row["xml_origem"]
        )

        distribuicao_origem[origem] = (
            distribuicao_origem.get(origem, 0) + 1
        )

    print("Distribuição por origem:")

    for origem in sorted(distribuicao_origem):

        print(
            f"  {origem}: "
            f"{distribuicao_origem[origem]}"
        )

    print()

    # --------------------------------------------------------
    # Agrupar questões
    # --------------------------------------------------------

    questoes = sorted(
        set(
            row["question"]
            for row in amostra
        )
    )

    print(
        f"Questões encontradas: {len(questoes)}"
    )

    print()

    # --------------------------------------------------------
    # Contadores gerais
    # --------------------------------------------------------

    total_processadas = 0
    total_puladas = 0
    total_erros = 0

    # ========================================================
    # PROCESSAR QUESTÃO POR QUESTÃO
    # ========================================================

    for question in questoes:

        print("-" * 70)

        print(
            f"QUESTÃO: {question}"
        )

        print("-" * 70)

        # ----------------------------------------------------
        # Arquivo de saída
        # ----------------------------------------------------

        arquivo_saida = arquivo_resultados(
            question
        )

        criar_cabecalho(
            arquivo_saida
        )

        # ----------------------------------------------------
        # Identificar respostas já processadas
        # ----------------------------------------------------

        processados = carregar_processados(
            arquivo_saida
        )

        # ----------------------------------------------------
        # Filtrar questão
        # ----------------------------------------------------

        linhas_questao = [
            row
            for row in amostra
            if row["question"] == question
        ]

        # ----------------------------------------------------
        # Identificar pendentes
        # ----------------------------------------------------

        pendentes = []

        for row in linhas_questao:

            chave = (
                row["xml_origem"],
                row["student"]
            )

            if chave in processados:

                total_puladas += 1

            else:

                pendentes.append(row)

        print(
            f"Respostas da questão: {len(linhas_questao)}"
        )

        print(
            f"Já processadas:       "
            f"{len(linhas_questao) - len(pendentes)}"
        )

        print(
            f"Pendentes:             "
            f"{len(pendentes)}"
        )

        # ----------------------------------------------------
        # Nada pendente
        # ----------------------------------------------------

        if not pendentes:

            print(
                "Nada a processar."
            )

            continue

        # ----------------------------------------------------
        # Carregar prompt
        # ----------------------------------------------------

        try:

            prompt_base = carregar_prompt(
                question
            )

        except Exception as e:

            print(
                f"ERRO AO CARREGAR PROMPT: {e}"
            )

            total_erros += len(
                pendentes
            )

            continue

        # ----------------------------------------------------
        # Abrir CSV para acrescentar resultados
        # ----------------------------------------------------

        with arquivo_saida.open(
            "a",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "origem",
                    "xml_origem",
                    "question",
                    "student",
                    "code0",
                    "classe_real",
                    "resposta",
                    "resposta_qwen",
                    "classe_qwen",
                    "erro"
                ]
            )

            # =================================================
            # EXECUTAR RESPOSTAS
            # =================================================

            for numero, row in enumerate(
                pendentes,
                start=1
            ):

                # ------------------------------------------------
                # Identificar origem
                # ------------------------------------------------

                origem = identificar_origem(
                    row["xml_origem"]
                )

                # ------------------------------------------------
                # Resposta original
                # ------------------------------------------------

                resposta_original = row[
                    "resposta"
                ]

                # ------------------------------------------------
                # Inserir resposta no prompt
                # ------------------------------------------------

                prompt = prompt_base.replace(
                    "{frase_usuario}",
                    resposta_original
                )

                # ------------------------------------------------
                # Mostrar progresso
                # ------------------------------------------------

                print(
                    f"[{numero}/{len(pendentes)}] "
                    f"{origem} | "
                    f"{row['student']}",
                    flush=True
                )

                # ------------------------------------------------
                # Executar Qwen
                # ------------------------------------------------

                inicio = time.time()

                resposta_qwen, erro = (
                    executar_qwen(prompt)
                )

                tempo = (
                    time.time() - inicio
                )

                # ------------------------------------------------
                # Extrair classe
                # ------------------------------------------------

                classe_qwen = extrair_nota(
                    resposta_qwen
                )

                # ------------------------------------------------
                # Contabilizar erro
                # ------------------------------------------------

                if erro:

                    total_erros += 1

                # ------------------------------------------------
                # Gravar resultado
                # ------------------------------------------------

                writer.writerow({

                    "origem": origem,

                    "xml_origem":
                        row["xml_origem"],

                    "question":
                        row["question"],

                    "student":
                        row["student"],

                    "code0":
                        row["code0"],

                    "classe_real":
                        row["classe_real"],

                    "resposta":
                        resposta_original,

                    "resposta_qwen":
                        resposta_qwen,

                    "classe_qwen":
                        classe_qwen,

                    "erro":
                        erro
                })

                # ------------------------------------------------
                # IMPORTANTE:
                # grava imediatamente no disco
                # ------------------------------------------------

                f.flush()

                total_processadas += 1

                # ------------------------------------------------
                # Mostrar resultado
                # ------------------------------------------------

                if erro:

                    print(
                        f"    ERRO: {erro}"
                    )

                else:

                    resposta_resumida = (
                        resposta_qwen
                        .replace("\n", " ")
                        .strip()
                    )

                    if len(
                        resposta_resumida
                    ) > 180:

                        resposta_resumida = (
                            resposta_resumida[:180]
                            + "..."
                        )

                    print(
                        f"    Qwen: "
                        f"{resposta_resumida}"
                    )

                    print(
                        f"    Classe extraída: "
                        f"{classe_qwen or 'NÃO IDENTIFICADA'}"
                    )

                print(
                    f"    Tempo: "
                    f"{tempo:.1f}s",
                    flush=True
                )

                # ------------------------------------------------
                # Pequena pausa
                # ------------------------------------------------

                time.sleep(
                    ESPERA
                )

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print()

    print("=" * 70)
    print("PROCESSAMENTO FINALIZADO")
    print("=" * 70)

    print(
        f"Novas respostas processadas: "
        f"{total_processadas}"
    )

    print(
        f"Respostas já existentes:     "
        f"{total_puladas}"
    )

    print(
        f"Erros:                       "
        f"{total_erros}"
    )

    print()

    print(
        "Resultados em:"
    )

    print(
        PASTA_SAIDA
    )

    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()
