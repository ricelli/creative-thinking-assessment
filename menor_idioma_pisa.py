
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Identifica o idioma com menor quantidade de respostas
no dataset PISA 2022 - Pensamento Criativo.

Critério:
- Cada elemento <response> corresponde a uma resposta.
- O idioma é extraído do nome do arquivo XML.
- Os códigos de país/região são agrupados pelo idioma.
- A contagem é feita em todos os arquivos XML da pasta.
"""

import csv
import xml.etree.ElementTree as ET

from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_XML = Path(
    "/home/ricelli/Documentos/Doutorado/"
    "Arquivos OECS-20250827T193412Z-1-001/"
    "Arquivos OECS/2022MS-responsesXML/2022MS"
)

ARQUIVO_CSV = PASTA_XML / "quantidade_respostas_por_idioma.csv"


# ============================================================
# NOMES DOS IDIOMAS
# ============================================================

IDIOMAS = {
    "ara": "Arabic",
    "eus": "Basque",
    "est": "Estonian",
    "fao": "Faroese",
    "fin": "Finnish",
    "fra": "French",
    "glg": "Galician",
    "heb": "Hebrew",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "ind": "Indonesian",
    "isl": "Icelandic",
    "ita": "Italian",
    "kaa": "Kara-Kalpak",
    "kaz": "Kazakh",
    "kor": "Korean",
    "lav": "Latvian",
    "lit": "Lithuanian",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "mon": "Mongolian",
    "msa": "Malay",
    "nld": "Dutch",
    "pol": "Polish",
    "por": "Portuguese",
    "aze": "Azerbaijani",
    "ron": "Romanian",
    "rus": "Russian",
    "slv": "Slovenian",
    "sqi": "Albanian",
    "srp": "Serbian",
    "swe": "Swedish",
    "tha": "Thai",
    "ukr": "Ukrainian",
    "zho": "Chinese",
    "eng": "English",
    "spa": "Spanish",
    "veh": "Vehes",
}


# ============================================================
# IDENTIFICAÇÃO DO IDIOMA
# ============================================================

def identificar_idioma(arquivo):
    """
    Extrai o código do idioma do nome do arquivo.

    Exemplo:
    OECS_Responses_T240Q01_ara-jor_02162023.xml

    Retorna:
    ara
    """

    partes = arquivo.stem.split("_")

    if len(partes) < 4:
        return None

    codigo_regiao = partes[3]

    # ara-jor -> ara
    # zho-hkg -> zho
    # zho-mac -> zho

    idioma = codigo_regiao.split("-")[0].lower()

    return idioma


# ============================================================
# CONTAGEM DE RESPOSTAS
# ============================================================

def contar_respostas(arquivo):
    """
    Conta os elementos <response> presentes no XML.
    """

    try:

        total = 0

        for evento, elemento in ET.iterparse(
            arquivo,
            events=("end",)
        ):

            if elemento.tag.split("}")[-1] == "response":
                total += 1

            elemento.clear()

        return total

    except ET.ParseError as erro:

        print(f"Erro de XML: {arquivo}")
        print(erro)

        return None

    except OSError as erro:

        print(f"Erro ao abrir: {arquivo}")
        print(erro)

        return None


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    if not PASTA_XML.exists():

        print("ERRO: A pasta não existe:")
        print(PASTA_XML)

        return

    arquivos_xml = list(
        PASTA_XML.rglob("*.xml")
    )

    print("=" * 80)
    print("ANÁLISE DO DATASET PISA 2022")
    print("=" * 80)

    print(
        f"\nArquivos XML encontrados: "
        f"{len(arquivos_xml)}"
    )

    contagem_idiomas = Counter()

    arquivos_por_idioma = defaultdict(int)

    arquivos_invalidos = []

    # ========================================================
    # PROCESSAMENTO DOS ARQUIVOS
    # ========================================================

    for arquivo in arquivos_xml:

        idioma = identificar_idioma(arquivo)

        if idioma is None:

            print(
                f"Não foi possível identificar "
                f"o idioma: {arquivo.name}"
            )

            continue

        quantidade = contar_respostas(arquivo)

        if quantidade is None:

            arquivos_invalidos.append(arquivo)

            continue

        contagem_idiomas[idioma] += quantidade

        arquivos_por_idioma[idioma] += 1

    # ========================================================
    # RESULTADOS
    # ========================================================

    resultados = sorted(
        contagem_idiomas.items(),
        key=lambda item: item[1]
    )

    print("\n")
    print("=" * 80)
    print("QUANTIDADE DE RESPOSTAS POR IDIOMA")
    print("=" * 80)

    print(
        f"{'Idioma':<25}"
        f"{'Código':<12}"
        f"{'Arquivos XML':<18}"
        f"{'Respostas':>15}"
    )

    print("-" * 80)

    for idioma, quantidade in resultados:

        nome = IDIOMAS.get(
            idioma,
            f"Idioma não cadastrado ({idioma})"
        )

        numero_arquivos = arquivos_por_idioma[idioma]

        print(
            f"{nome:<25}"
            f"{idioma:<12}"
            f"{numero_arquivos:<18}"
            f"{quantidade:>15,}"
        )

    # ========================================================
    # IDENTIFICA O MENOR IDIOMA
    # ========================================================

    if contagem_idiomas:

        menor_quantidade = min(
            contagem_idiomas.values()
        )

        idiomas_menores = [
            idioma
            for idioma, quantidade
            in contagem_idiomas.items()
            if quantidade == menor_quantidade
        ]

        print("\n")
        print("=" * 80)
        print("IDIOMA(S) COM MENOR QUANTIDADE DE RESPOSTAS")
        print("=" * 80)

        for idioma in idiomas_menores:

            nome = IDIOMAS.get(
                idioma,
                f"Idioma não cadastrado ({idioma})"
            )

            print(
                f"{nome} ({idioma}): "
                f"{menor_quantidade:,} respostas"
            )

    # ========================================================
    # EXPORTAÇÃO CSV
    # ========================================================

    with open(
        ARQUIVO_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo_csv:

        escritor = csv.writer(
            arquivo_csv,
            delimiter=";"
        )

        escritor.writerow([
            "idioma",
            "codigo_idioma",
            "quantidade_arquivos",
            "quantidade_respostas"
        ])

        for idioma, quantidade in resultados:

            escritor.writerow([
                IDIOMAS.get(idioma, idioma),
                idioma,
                arquivos_por_idioma[idioma],
                quantidade
            ])

    print("\n")
    print("CSV gerado:")
    print(ARQUIVO_CSV)

    print(
        f"\nArquivos XML inválidos: "
        f"{len(arquivos_invalidos)}"
    )

    print("\nProcessamento concluído!")


if __name__ == "__main__":
    main()
