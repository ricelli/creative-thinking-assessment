#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cria uma amostra estratificada para o experimento Qwen em chinês.

Desenho:
- 12 questões PISA 2022 de Pensamento Criativo
- 3 origens:
    zho-tap -> China
    zho-hkg -> Hong Kong
    zho-mac -> Macau
- 60 respostas da classe 1 + 60 da classe 2
  por origem e por questão
- Total da amostra completa: 4.320 respostas

Modo teste:
- 5 respostas da classe 1 + 5 da classe 2
  por origem e por questão
- Total: 180 respostas

Somente biblioteca padrão do Python.
"""

import csv
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_XML = (
    "/home/..."
    "Arquivos OECS.../"
    "Arquivos OECS.../"
    "2022/chines"
)

ARQUIVO_SAIDA = (
    "/home.../"
    "amostra_chines_qwen.csv"
)

# Seed fixa para garantir reprodutibilidade
SEED = 20261008

# ------------------------------------------------------------
# TESTE:
# True  = 5 de cada classe por origem/questão = 180
# False = 60 de cada classe por origem/questão = 4.320
# ------------------------------------------------------------

MODO_TESTE = True

N_POR_CLASSE = 5 if MODO_TESTE else 60


# ============================================================
# QUESTÕES
# ============================================================

QUESTOES = [
    "T240Q01",
    "T240Q02",
    "T300Q01",
    "T300Q02",
    "T350Q01",
    "T350Q02",
    "T350Q03",
    "T360Q01",
    "T370Q01",
    "T570Q01",
    "T570Q02",
    "T570Q03",
]


# ============================================================
# ORIGENS
# ============================================================

ORIGENS = {
    "zho-tap": "China",
    "zho-hkg": "Hong Kong",
    "zho-mac": "Macau",
}


# ============================================================
# IDENTIFICAÇÃO DO ARQUIVO
# ============================================================

def identificar_origem(nome_arquivo):
    """
    Identifica a origem pelo código no nome do XML.
    """

    for codigo, origem in ORIGENS.items():
        if codigo in nome_arquivo:
            return origem

    return None


def identificar_questao(nome_arquivo):
    """
    Extrai TxxxQyy do nome do XML.
    """

    match = re.search(r"(T\d+Q\d+)", nome_arquivo)

    if match:
        return match.group(1)

    return None


# ============================================================
# CONVERSÃO DAS CLASSES
# ============================================================

def normalizar_classe(question, code0):
    """
    Converte o código original do OECS para a classificação
    binária utilizada no experimento.

    Questões com códigos 11-13 / 21-29:
        11, 12, 13 -> classe 1
        21-29      -> classe 2

    Questões T350Q02 e T570Q01:
        0 -> classe 1
        1 -> classe 2

    Outras questões com códigos 1/2:
        1 -> classe 1
        2 -> classe 2

    Códigos inadequados/fora da classificação:
        00, 9, 97, 99 etc. -> None
    """

    if code0 is None:
        return None

    code = str(code0).strip()

    # --------------------------------------------------------
    # Questões especiais
    # --------------------------------------------------------

    if question in {"T350Q02", "T570Q01"}:

        if code == "0":
            return 1

        if code == "1":
            return 2

        return None

    # --------------------------------------------------------
    # Questões com codificação 1/2
    # --------------------------------------------------------

    if code == "1":
        return 1

    if code == "2":
        return 2

    # --------------------------------------------------------
    # Questões com codificação 11-29
    # --------------------------------------------------------

    if code in {"11", "12", "13"}:
        return 1

    if code in {
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
    }:
        return 2

    return None


# ============================================================
# EXTRAÇÃO DO XML
# ============================================================

def extrair_respostas(caminho_xml, question, origem):

    tree = ET.parse(caminho_xml)
    root = tree.getroot()

    registros = []

    for elem in root.iter():

        if "code0" not in elem.attrib:
            continue

        code0 = elem.attrib.get("code0")

        classe = normalizar_classe(
            question,
            code0
        )

        # Ignora códigos fora das classes 1/2
        if classe not in {1, 2}:
            continue

        student = elem.attrib.get(
            "student",
            ""
        )

        # ----------------------------------------------------
        # Recupera o texto da resposta
        # ----------------------------------------------------

        resposta = "".join(
            elem.itertext()
        ).strip()

        if not resposta:
            continue

        registros.append({
            "origem": origem,
            "xml_origem": os.path.basename(caminho_xml),
            "question": question,
            "student": student,
            "code0": code0,
            "classe_real": classe,
            "resposta": resposta,
        })

    return registros


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    random.seed(SEED)

    print()
    print("=" * 80)
    print("AMOSTRAGEM DO EXPERIMENTO QWEN - CHINÊS")
    print("=" * 80)

    print()
    print("Diretório dos XMLs:")
    print(PASTA_XML)

    print()
    print(f"Seed: {SEED}")

    print(
        "Modo:",
        "TESTE (5 + 5)" if MODO_TESTE
        else "COMPLETO (60 + 60)"
    )

    print(
        f"Respostas por classe/origem/questão: "
        f"{N_POR_CLASSE}"
    )

    # ========================================================
    # LOCALIZA OS XMLs
    # ========================================================

    arquivos = []

    for nome in os.listdir(PASTA_XML):

        if not nome.endswith(".xml"):
            continue

        question = identificar_questao(nome)

        origem = identificar_origem(nome)

        if (
            question in QUESTOES
            and origem is not None
        ):
            arquivos.append(
                (
                    nome,
                    question,
                    origem
                )
            )

    print()
    print(
        f"XMLs encontrados: {len(arquivos)}"
    )

    if len(arquivos) != 36:

        print()
        print(
            "ATENÇÃO: eram esperados 36 XMLs "
            "(12 questões × 3 origens)."
        )

    # ========================================================
    # ORGANIZA RESPOSTAS
    # ========================================================

    grupos = defaultdict(
        lambda: {
            1: [],
            2: []
        }
    )

    print()
    print("-" * 80)
    print("DISTRIBUIÇÃO DAS RESPOSTAS VÁLIDAS")
    print("-" * 80)

    for nome, question, origem in sorted(arquivos):

        caminho = os.path.join(
            PASTA_XML,
            nome
        )

        try:

            registros = extrair_respostas(
                caminho,
                question,
                origem
            )

        except Exception as e:

            print()
            print(
                f"ERRO ao ler {nome}: {e}"
            )

            continue

        for registro in registros:

            grupos[
                (origem, question)
            ][
                registro["classe_real"]
            ].append(registro)

    # ========================================================
    # VERIFICAÇÃO
    # ========================================================

    faltantes = []

    for origem in ORIGENS.values():

        for question in QUESTOES:

            grupo = grupos[
                (origem, question)
            ]

            n1 = len(grupo[1])
            n2 = len(grupo[2])

            print(
                f"{origem:10s} "
                f"{question}: "
                f"classe 1={n1:3d} | "
                f"classe 2={n2:3d}"
            )

            if (
                n1 < N_POR_CLASSE
                or
                n2 < N_POR_CLASSE
            ):

                faltantes.append(
                    (
                        origem,
                        question,
                        n1,
                        n2
                    )
                )

    # ========================================================
    # INTERROMPE SE NÃO HOUVER RESPOSTAS SUFICIENTES
    # ========================================================

    if faltantes:

        print()
        print("=" * 80)
        print("ERRO: NÃO É POSSÍVEL FORMAR A AMOSTRA")
        print("=" * 80)

        for (
            origem,
            question,
            n1,
            n2
        ) in faltantes:

            print(
                f"{origem} {question}: "
                f"C1={n1}, "
                f"C2={n2}, "
                f"necessário={N_POR_CLASSE}"
            )

        raise SystemExit(1)

    # ========================================================
    # SORTEIO
    # ========================================================

    amostra = []

    for origem in ORIGENS.values():

        for question in QUESTOES:

            grupo = grupos[
                (origem, question)
            ]

            # Classe 1
            selecionados = random.sample(
                grupo[1],
                N_POR_CLASSE
            )

            amostra.extend(
                selecionados
            )

            # Classe 2
            selecionados = random.sample(
                grupo[2],
                N_POR_CLASSE
            )

            amostra.extend(
                selecionados
            )

    # ========================================================
    # EMBARALHA A AMOSTRA FINAL
    # ========================================================

    random.shuffle(amostra)

    # ========================================================
    # SALVA CSV
    # ========================================================

    campos = [
        "origem",
        "xml_origem",
        "question",
        "student",
        "code0",
        "classe_real",
        "resposta",
    ]

    with open(
        ARQUIVO_SAIDA,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:

        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        writer.writeheader()

        writer.writerows(
            amostra
        )

    # ========================================================
    # RESUMO
    # ========================================================

    print()
    print("=" * 80)
    print("AMOSTRA GERADA")
    print("=" * 80)

    print()
    print(
        f"Arquivo: {ARQUIVO_SAIDA}"
    )

    print(
        f"Total de respostas: {len(amostra)}"
    )

    # --------------------------------------------------------
    # Por origem
    # --------------------------------------------------------

    print()
    print("Por origem:")

    cont_origem = Counter(
        registro["origem"]
        for registro in amostra
    )

    for origem in ORIGENS.values():

        print(
            f"  {origem:10s}: "
            f"{cont_origem[origem]}"
        )

    # --------------------------------------------------------
    # Por classe
    # --------------------------------------------------------

    print()
    print("Por classe:")

    cont_classe = Counter(
        registro["classe_real"]
        for registro in amostra
    )

    print(
        f"  Classe 1: {cont_classe[1]}"
    )

    print(
        f"  Classe 2: {cont_classe[2]}"
    )

    # --------------------------------------------------------
    # Verificação final
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print("VERIFICAÇÃO FINAL")
    print("-" * 80)

    cont = Counter(
        (
            registro["origem"],
            registro["question"],
            registro["classe_real"]
        )
        for registro in amostra
    )

    ok = True

    for origem in ORIGENS.values():

        for question in QUESTOES:

            n1 = cont[
                (origem, question, 1)
            ]

            n2 = cont[
                (origem, question, 2)
            ]

            if (
                n1 != N_POR_CLASSE
                or
                n2 != N_POR_CLASSE
            ):

                ok = False

            print(
                f"{origem:10s} "
                f"{question}: "
                f"C1={n1:2d} "
                f"C2={n2:2d}"
            )

    print()

    if ok:

        print(
            "OK: amostra estratificada "
            "corretamente."
        )

    else:

        print(
            "ERRO: a amostra não ficou "
            "conforme o desenho esperado."
        )


if __name__ == "__main__":
    main()
