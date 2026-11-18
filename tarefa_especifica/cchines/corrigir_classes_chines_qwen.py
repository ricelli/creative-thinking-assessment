#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corrige a extração das classes 1 e 2 nas respostas já geradas pelo Qwen.

- Usa somente a biblioteca padrão do Python.
- Não executa novamente o Qwen.
- Preserva a coluna resposta_qwen.
- Atualiza somente a coluna classe_qwen.
- Cria uma cópia de segurança .bak antes da primeira alteração.
- Mostra os resultados por questão e no total.
"""

import csv
import re
import shutil
from pathlib import Path
from collections import Counter


PASTA = Path(
    "/home/.../"
    "resultados_chines_qwen"
)

PADRAO_ARQUIVOS = "T*Q*_resultados.csv"


def limpar_texto(texto):
    """Remove sequências ANSI e caracteres de controle do terminal."""
    if not texto:
        return ""

    # Remove sequências ANSI, como códigos de cursor e limpeza de linha.
    texto = re.sub(
        r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])",
        "",
        texto,
    )

    # Remove caracteres de controle, preservando quebras de linha,
    # tabulações e retornos de carro.
    texto = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        texto,
    )

    # Trata backspaces eventualmente registrados na saída.
    while "\x08" in texto:
        novo_texto = re.sub(r".?\x08", "", texto)
        if novo_texto == texto:
            texto = texto.replace("\x08", "")
            break
        texto = novo_texto

    return texto


def extrair_nota(resposta):
    """
    Extrai a classificação final 1 ou 2.

    Exemplos reconhecidos:
        1 | Justificativa | 8
        2 | Justificativa | 9
        Nota (2) | Justificativa | Grau de confiança (9)
        Nota: 1 | Justificativa ...
        Nota 2 | Justificativa ...
    """
    texto = limpar_texto(resposta)

    if not texto.strip():
        return ""

    linhas = [
        linha.strip()
        for linha in texto.splitlines()
        if linha.strip()
    ]

    # 1. Procura primeiro o formato principal nas linhas finais.
    for linha in reversed(linhas):
        match = re.match(r"^([12])\s*\|", linha)
        if match:
            return match.group(1)

    # 2. Procura a nota explícita nas linhas finais.
    for linha in reversed(linhas):
        match = re.match(
            r"^Nota\s*(?:"
            r"\(\s*([12])\s*\)"
            r"|:\s*([12])\b"
            r"|\s+([12])\b"
            r")",
            linha,
            flags=re.IGNORECASE,
        )

        if match:
            for grupo in match.groups():
                if grupo:
                    return grupo

    return ""


def processar_arquivo(caminho):
    """Atualiza classe_qwen e retorna estatísticas do arquivo."""

    backup = caminho.with_suffix(caminho.suffix + ".bak")

    # Cria backup somente se ainda não existir.
    if not backup.exists():
        shutil.copy2(caminho, backup)

    # Lê todas as linhas e preserva as colunas existentes.
    with caminho.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as arquivo:
        leitor = csv.DictReader(arquivo)

        if not leitor.fieldnames:
            print(f"ERRO: arquivo vazio: {caminho.name}")
            return None

        campos = list(leitor.fieldnames)
        linhas = list(leitor)

    colunas_obrigatorias = {"resposta_qwen", "classe_qwen"}
    ausentes = colunas_obrigatorias - set(campos)

    if ausentes:
        print(
            f"ERRO: {caminho.name}: "
            f"colunas ausentes: {', '.join(sorted(ausentes))}"
        )
        return None

    estatisticas = Counter()

    for linha in linhas:
        resposta = linha.get("resposta_qwen", "")
        classe = extrair_nota(resposta)

        # Substitui a classe extraída anteriormente.
        linha["classe_qwen"] = classe

        if classe in ("1", "2"):
            estatisticas["validas"] += 1
        else:
            estatisticas["invalidas"] += 1

    estatisticas["total"] = len(linhas)

    # Regrava o arquivo com as mesmas colunas.
    with caminho.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as arquivo:
        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos,
            extrasaction="ignore",
        )
        escritor.writeheader()
        escritor.writerows(linhas)

    return estatisticas


def main():
    if not PASTA.exists():
        print(f"ERRO: pasta não encontrada:\n{PASTA}")
        return

    arquivos = sorted(PASTA.glob(PADRAO_ARQUIVOS))

    if not arquivos:
        print(f"Nenhum arquivo encontrado em:\n{PASTA}")
        return

    print(f"Pasta de resultados: {PASTA}")
    print(f"Arquivos encontrados: {len(arquivos)}")
    print("=" * 76)

    total_geral = Counter()
    processados = 0

    for caminho in arquivos:
        estatisticas = processar_arquivo(caminho)

        if estatisticas is None:
            continue

        processados += 1
        total_geral.update(estatisticas)

        print(
            f"{caminho.name}: "
            f"total={estatisticas['total']} | "
            f"válidas={estatisticas['validas']} | "
            f"inválidas={estatisticas['invalidas']}"
        )

    print("=" * 76)
    print(f"Arquivos processados: {processados}")
    print(
        f"TOTAL: {total_geral['total']} | "
        f"Válidas={total_geral['validas']} | "
        f"Inválidas={total_geral['invalidas']}"
    )

    if total_geral["total"] > 0:
        percentual = (
            100 * total_geral["validas"] / total_geral["total"]
        )
        print(
            "Percentual de classificações extraídas: "
            f"{percentual:.2f}%"
        )

    print("\nCópias de segurança: arquivos com extensão .csv.bak")
    print("Concluído.")


if __name__ == "__main__":
    main()
