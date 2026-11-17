
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calcula precisão, revocação e F1 para o experimento
chinês do Qwen3:14b no OECS/PISA 2022.

Saídas:
1. Métricas por questão e classe.
2. Métricas macro por questão.
3. Métricas acumuladas por tarefa (GCI, GDI e EII).
4. Resumo da quantidade de respostas válidas e inválidas.

Utiliza apenas a biblioteca padrão do Python.
"""

import csv
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_RESULTADOS = Path(
    "/home/ricelli/Documentos/Doutorado/"
    "resultados_chines_qwen"
)

PASTA_METRICAS = PASTA_RESULTADOS / "metricas"

CLASSES = ["1", "2"]

TAREFAS = {
    "GCI": [
        "T240Q01",
        "T300Q01",
        "T350Q01",
        "T360Q01",
        "T370Q01",
        "T570Q02",
    ],
    "GDI": [
        "T240Q02",
        "T300Q02",
        "T350Q02",
        "T570Q01",
    ],
    "EII": [
        "T350Q03",
        "T570Q03",
    ],
}

QUESTOES = {
    q
    for questoes in TAREFAS.values()
    for q in questoes
}


# ============================================================
# LEITURA DOS RESULTADOS
# ============================================================

def ler_csv(caminho):
    with caminho.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:
        return list(csv.DictReader(arquivo))


def normalizar_classe(valor):
    """
    Aceita apenas classes 1 e 2.

    Não converte automaticamente outros valores em classe.
    """

    if valor is None:
        return ""

    valor = str(valor).strip()

    if valor in CLASSES:
        return valor

    return ""


def carregar_dados():
    """
    Lê os CSVs por questão e valida os campos necessários.
    """

    dados = {}
    problemas = []

    for question in sorted(QUESTOES):
        arquivo = (
            PASTA_RESULTADOS
            / f"{question}_resultados.csv"
        )

        if not arquivo.exists():
            problemas.append(
                f"Arquivo ausente: {arquivo.name}"
            )
            continue

        linhas = ler_csv(arquivo)

        obrigatorios = {
            "classe_real",
            "classe_qwen",
        }

        if not linhas:
            problemas.append(
                f"Arquivo sem registros: {arquivo.name}"
            )
            dados[question] = []
            continue

        campos = set(linhas[0].keys())
        faltantes = obrigatorios - campos

        if faltantes:
            problemas.append(
                f"{arquivo.name}: campos ausentes: "
                + ", ".join(sorted(faltantes))
            )
            continue

        dados[question] = linhas

    return dados, problemas


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(linhas):
    """
    Calcula métricas para as classes 1 e 2.

    Uma predição inválida é excluída das métricas principais.
    Ela também é contabilizada separadamente.

    A matriz de confusão é construída usando somente registros
    com classe real válida (1/2). Uma predição inválida em um
    registro com classe real válida conta como predição errada
    para a revocação da classe verdadeira.
    """

    matriz = {
        real: {pred: 0 for pred in CLASSES}
        for real in CLASSES
    }

    suporte_real = Counter()
    suporte_predito = Counter()

    total = len(linhas)
    validas = 0
    invalidas = 0
    reais_invalidas = 0

    for linha in linhas:
        real = normalizar_classe(
            linha.get("classe_real", "")
        )

        pred = normalizar_classe(
            linha.get("classe_qwen", "")
        )

        # Classe real ausente ou fora de 1/2:
        # não pode integrar a avaliação binária.
        if real not in CLASSES:
            reais_invalidas += 1
            continue

        suporte_real[real] += 1

        # Predição inválida: erro para a classe verdadeira.
        if pred not in CLASSES:
            invalidas += 1
            continue

        validas += 1
        matriz[real][pred] += 1
        suporte_predito[pred] += 1

    metricas = {}

    for classe in CLASSES:
        tp = matriz[classe][classe]

        fp = sum(
            matriz[outra][classe]
            for outra in CLASSES
            if outra != classe
        )

        fn = (
            suporte_real[classe]
            - tp
        )

        precisao = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        revocacao = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precisao * revocacao
            / (precisao + revocacao)
            if (precisao + revocacao) > 0
            else 0.0
        )

        metricas[classe] = {
            "precisao": precisao,
            "revocacao": revocacao,
            "f1": f1,
            "suporte": suporte_real[classe],
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    # Macro: média aritmética entre as classes 1 e 2.
    macro = {
        "precisao": sum(
            metricas[c]["precisao"]
            for c in CLASSES
        ) / len(CLASSES),

        "revocacao": sum(
            metricas[c]["revocacao"]
            for c in CLASSES
        ) / len(CLASSES),

        "f1": sum(
            metricas[c]["f1"]
            for c in CLASSES
        ) / len(CLASSES),
    }

    return {
        "total": total,
        "validas": validas,
        "invalidas": invalidas,
        "reais_invalidas": reais_invalidas,
        "suporte_real": suporte_real,
        "matriz": matriz,
        "classes": metricas,
        "macro": macro,
    }


# ============================================================
# REGISTROS DE SAÍDA
# ============================================================

def registro_metrica(nivel, nome, classe, resultado):
    """
    Gera uma linha padronizada para o CSV de métricas.
    """

    if classe in CLASSES:
        m = resultado["classes"][classe]

        return {
            "nivel": nivel,
            "nome": nome,
            "classe": classe,
            "total": resultado["total"],
            "validas": resultado["validas"],
            "invalidas": resultado["invalidas"],
            "reais_invalidas": resultado["reais_invalidas"],
            "suporte": m["suporte"],
            "precisao": m["precisao"],
            "revocacao": m["revocacao"],
            "f1": m["f1"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        }

    return {
        "nivel": nivel,
        "nome": nome,
        "classe": "macro",
        "total": resultado["total"],
        "validas": resultado["validas"],
        "invalidas": resultado["invalidas"],
        "reais_invalidas": resultado["reais_invalidas"],
        "suporte": (
            resultado["suporte_real"]["1"]
            + resultado["suporte_real"]["2"]
        ),
        "precisao": resultado["macro"]["precisao"],
        "revocacao": resultado["macro"]["revocacao"],
        "f1": resultado["macro"]["f1"],
        "tp": "",
        "fp": "",
        "fn": "",
    }


def salvar_csv(caminho, linhas):
    campos = [
        "nivel",
        "nome",
        "classe",
        "total",
        "validas",
        "invalidas",
        "reais_invalidas",
        "suporte",
        "precisao",
        "revocacao",
        "f1",
        "tp",
        "fp",
        "fn",
    ]

    with caminho.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:
        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )
        writer.writeheader()
        writer.writerows(linhas)


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    if not PASTA_RESULTADOS.exists():
        print(
            "ERRO: pasta de resultados não encontrada:"
        )
        print(PASTA_RESULTADOS)
        return

    PASTA_METRICAS.mkdir(
        parents=True,
        exist_ok=True
    )

    dados, problemas = carregar_dados()

    if problemas:
        print("AVISOS NA LEITURA DOS ARQUIVOS:")
        for problema in problemas:
            print(f" - {problema}")
        print()

    linhas_questoes = []
    linhas_tarefas = []
    linhas_validacao = []

    print("=" * 78)
    print("MÉTRICAS DO EXPERIMENTO CHINÊS — QWEN3:14B")
    print("=" * 78)

    # --------------------------------------------------------
    # Métricas por questão
    # --------------------------------------------------------

    dados_tarefa = {
        tarefa: []
        for tarefa in TAREFAS
    }

    for tarefa, questoes in TAREFAS.items():
        for question in questoes:
            if question in dados:
                dados_tarefa[tarefa].extend(
                    dados[question]
                )

    for tarefa, questoes in TAREFAS.items():
        for question in questoes:
            if question not in dados:
                continue

            linhas = dados[question]
            resultado = calcular_metricas(linhas)

            print()
            print(
                f"{question} ({tarefa})"
            )
            print(
                f"Total={resultado['total']} | "
                f"Predições válidas={resultado['validas']} | "
                f"Inválidas={resultado['invalidas']} | "
                f"Classes reais inválidas="
                f"{resultado['reais_invalidas']}"
            )

            for classe in CLASSES:
                m = resultado["classes"][classe]

                print(
                    f"  Classe {classe}: "
                    f"P={m['precisao']:.4f}, "
                    f"R={m['revocacao']:.4f}, "
                    f"F1={m['f1']:.4f}, "
                    f"suporte={m['suporte']}"
                )

                linhas_questoes.append(
                    registro_metrica(
                        "questao",
                        question,
                        classe,
                        resultado
                    )
                )

            macro = resultado["macro"]

            print(
                f"  Macro: "
                f"P={macro['precisao']:.4f}, "
                f"R={macro['revocacao']:.4f}, "
                f"F1={macro['f1']:.4f}"
            )

            linhas_questoes.append(
                registro_metrica(
                    "questao",
                    question,
                    "macro",
                    resultado
                )
            )

            linhas_validacao.append({
                "nivel": "questao",
                "nome": question,
                "total": resultado["total"],
                "predicoes_validas": resultado["validas"],
                "predicoes_invalidas": resultado["invalidas"],
                "classes_reais_invalidas": resultado["reais_invalidas"],
            })

    # --------------------------------------------------------
    # Métricas acumuladas por tarefa
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("MÉTRICAS ACUMULADAS POR TAREFA")
    print("=" * 78)

    for tarefa, linhas in dados_tarefa.items():
        if not linhas:
            print(
                f"\n{tarefa}: sem dados disponíveis."
            )
            continue

        resultado = calcular_metricas(linhas)

        print()
        print(
            f"{tarefa}: questões={len(TAREFAS[tarefa])}"
        )

        print(
            f"  Total={resultado['total']} | "
            f"Predições válidas={resultado['validas']} | "
            f"Inválidas={resultado['invalidas']} | "
            f"Classes reais inválidas="
            f"{resultado['reais_invalidas']}"
        )

        for classe in CLASSES:
            m = resultado["classes"][classe]

            print(
                f"  Classe {classe}: "
                f"P={m['precisao']:.4f}, "
                f"R={m['revocacao']:.4f}, "
                f"F1={m['f1']:.4f}, "
                f"suporte={m['suporte']}"
            )

            linhas_tarefas.append(
                registro_metrica(
                    "tarefa",
                    tarefa,
                    classe,
                    resultado
                )
            )

        macro = resultado["macro"]

        print(
            f"  Macro: "
            f"P={macro['precisao']:.4f}, "
            f"R={macro['revocacao']:.4f}, "
            f"F1={macro['f1']:.4f}"
        )

        linhas_tarefas.append(
            registro_metrica(
                "tarefa",
                tarefa,
                "macro",
                resultado
            )
        )

        linhas_validacao.append({
            "nivel": "tarefa",
            "nome": tarefa,
            "total": resultado["total"],
            "predicoes_validas": resultado["validas"],
            "predicoes_invalidas": resultado["invalidas"],
            "classes_reais_invalidas": resultado["reais_invalidas"],
        })

    # --------------------------------------------------------
    # Salvar CSVs
    # --------------------------------------------------------

    arquivo_questoes = (
        PASTA_METRICAS
        / "metricas_por_questao.csv"
    )

    arquivo_tarefas = (
        PASTA_METRICAS
        / "metricas_acumuladas_por_tarefa.csv"
    )

    arquivo_validacao = (
        PASTA_METRICAS
        / "resumo_validacao.csv"
    )

    salvar_csv(
        arquivo_questoes,
        linhas_questoes
    )

    salvar_csv(
        arquivo_tarefas,
        linhas_tarefas
    )

    with arquivo_validacao.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as arquivo:
        campos = [
            "nivel",
            "nome",
            "total",
            "predicoes_validas",
            "predicoes_invalidas",
            "classes_reais_invalidas",
        ]

        writer = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )
        writer.writeheader()
        writer.writerows(linhas_validacao)

    print()
    print("=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)
    print(arquivo_questoes)
    print(arquivo_tarefas)
    print(arquivo_validacao)
    print("=" * 78)


if __name__ == "__main__":
    main()