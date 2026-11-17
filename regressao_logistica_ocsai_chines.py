
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regressão logística sobre os escores do OCSAI para classificar
respostas de pensamento criativo nas classes 1 e 2.

Experimentos:
1. Somente o escore de originalidade.
2. Escore de originalidade + confiança do OCSAI.

Avaliação:
- Validação cruzada estratificada de até 5 partições.
- Métricas por questão e por tarefa (GCI, GDI e EII).
- Precisão, revocação, F1 por classe e macro-F1.
- Acurácia.
- Predições fora da amostra.

Usa somente a biblioteca padrão do Python.
"""

import csv
import math
import random
from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA_ENTRADA = Path(
    "/home/ricelli/Documentos/Doutorado/"
    "resultados_chines_ocsai"
)

PASTA_SAIDA = PASTA_ENTRADA / "regressao_logistica"

NUM_PARTICOES = 5
SEMENTE = 42
REGULARIZACAO = 0.01
MAX_ITERACOES = 10000
TOLERANCIA = 1e-8
TAXA_APRENDIZAGEM = 0.1

QUESTOES_POR_TAREFA = {
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

TAREFA_POR_QUESTAO = {
    questao: tarefa
    for tarefa, questoes in QUESTOES_POR_TAREFA.items()
    for questao in questoes
}

EXPERIMENTOS = {
    "originalidade": [
        "ocsai_originality",
    ],
    "originalidade_confianca": [
        "ocsai_originality",
        "ocsai_confidence",
    ],
}

CLASSES = [1, 2]


# ============================================================
# LEITURA E VALIDAÇÃO DOS DADOS
# ============================================================

def converter_float(valor):
    """Converte valor para float ou retorna None."""
    if valor is None:
        return None

    valor = str(valor).strip()

    if not valor:
        return None

    try:
        numero = float(valor)
    except (ValueError, TypeError):
        return None

    if not math.isfinite(numero):
        return None

    return numero


def carregar_dados():
    registros = []
    excluidos = []

    if not PASTA_ENTRADA.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {PASTA_ENTRADA}"
        )

    arquivos = sorted(
        PASTA_ENTRADA.glob("*_ocsai_resultados.csv")
    )

    if not arquivos:
        raise FileNotFoundError(
            "Nenhum arquivo *_ocsai_resultados.csv foi encontrado."
        )

    for arquivo in arquivos:
        questao = arquivo.name.replace(
            "_ocsai_resultados.csv", ""
        )

        if questao not in TAREFA_POR_QUESTAO:
            continue

        with open(
            arquivo,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            leitor = csv.DictReader(f)

            colunas = leitor.fieldnames or []

            obrigatorias = {
                "classe_real",
                "ocsai_originality",
                "ocsai_confidence",
                "resposta",
            }

            faltantes = obrigatorias - set(colunas)

            if faltantes:
                raise ValueError(
                    f"Colunas ausentes em {arquivo.name}: "
                    f"{sorted(faltantes)}"
                )

            for numero_linha, linha in enumerate(leitor, start=2):
                identificador = (
                    f"{arquivo.name}:linha {numero_linha}"
                )

                motivo = None

                if str(linha.get("erro_api", "")).strip():
                    motivo = "erro_api"

                classe_texto = str(
                    linha.get("classe_real", "")
                ).strip()

                try:
                    classe = int(classe_texto)
                except (ValueError, TypeError):
                    classe = None

                if motivo is None and classe not in CLASSES:
                    motivo = "classe_real_invalida"

                originalidade = converter_float(
                    linha.get("ocsai_originality")
                )

                confianca = converter_float(
                    linha.get("ocsai_confidence")
                )

                # A originalidade é necessária para ambos os
                # experimentos.
                if motivo is None and originalidade is None:
                    motivo = "originalidade_ausente_ou_invalida"

                if motivo is not None:
                    excluidos.append({
                        "arquivo": arquivo.name,
                        "identificador": identificador,
                        "question": questao,
                        "motivo_exclusao": motivo,
                        "classe_real": classe_texto,
                        "ocsai_originality": linha.get(
                            "ocsai_originality", ""
                        ),
                        "ocsai_confidence": linha.get(
                            "ocsai_confidence", ""
                        ),
                    })
                    continue

                registro = dict(linha)
                registro["_arquivo"] = arquivo.name
                registro["_questao"] = questao
                registro["_tarefa"] = TAREFA_POR_QUESTAO[questao]
                registro["_classe"] = classe
                registro["_originalidade"] = originalidade
                registro["_confianca"] = confianca
                registro["_identificador"] = identificador

                registros.append(registro)

    return registros, excluidos


# ============================================================
# REGRESSÃO LOGÍSTICA
# ============================================================

def sigmoid(z):
    """Função sigmoide numericamente estável."""
    if z >= 0:
        e = math.exp(-min(z, 700))
        return 1.0 / (1.0 + e)

    e = math.exp(max(z, -700))
    return e / (1.0 + e)


def calcular_media_desvio(matriz):
    """Calcula média e desvio-padrão do conjunto de treinamento."""
    n = len(matriz)
    p = len(matriz[0])

    medias = []
    desvios = []

    for j in range(p):
        valores = [linha[j] for linha in matriz]
        media = sum(valores) / n

        variancia = sum(
            (valor - media) ** 2 for valor in valores
        ) / n

        desvio = math.sqrt(variancia)

        if desvio < 1e-12:
            desvio = 1.0

        medias.append(media)
        desvios.append(desvio)

    return medias, desvios


def padronizar(matriz, medias, desvios):
    return [
        [
            (valor - medias[j]) / desvios[j]
            for j, valor in enumerate(linha)
        ]
        for linha in matriz
    ]


def treinar_regressao_logistica(matriz, rotulos_binarios):
    """
    Treina regressão logística binária com gradiente descendente.

    Os rótulos de treinamento precisam estar codificados como:
        classe real 1 -> 0
        classe real 2 -> 1
    """
    if not matriz:
        raise ValueError("Conjunto de treinamento vazio.")

    medias, desvios = calcular_media_desvio(matriz)
    X = padronizar(matriz, medias, desvios)

    n = len(X)
    p = len(X[0])

    # beta[0] é o intercepto.
    beta = [0.0] * (p + 1)

    for _ in range(MAX_ITERACOES):
        gradiente = [0.0] * (p + 1)

        for linha, y in zip(X, rotulos_binarios):
            z = beta[0]

            for j in range(p):
                z += beta[j + 1] * linha[j]

            probabilidade = sigmoid(z)
            erro = probabilidade - y

            gradiente[0] += erro

            for j in range(p):
                gradiente[j + 1] += erro * linha[j]

        gradiente[0] /= n

        for j in range(p):
            gradiente[j + 1] = (
                gradiente[j + 1] / n
                + REGULARIZACAO * beta[j + 1]
            )

        norma = max(abs(v) for v in gradiente)

        if norma < TOLERANCIA:
            break

        for j in range(p + 1):
            beta[j] -= (
                TAXA_APRENDIZAGEM * gradiente[j]
            )

    return {
        "beta": beta,
        "medias": medias,
        "desvios": desvios,
    }


def prever_probabilidade(modelo, linha):
    """Estima a probabilidade de a resposta pertencer à classe 2."""
    beta = modelo["beta"]
    medias = modelo["medias"]
    desvios = modelo["desvios"]

    z = beta[0]

    for j, valor in enumerate(linha):
        valor_padronizado = (
            (valor - medias[j]) / desvios[j]
        )
        z += beta[j + 1] * valor_padronizado

    return sigmoid(z)


def prever_classe(modelo, linha):
    """
    Usa 0,5 como limiar sobre a probabilidade estimada.

    Retorna a classe original 1 ou 2 e a probabilidade da classe 2.
    """
    probabilidade = prever_probabilidade(modelo, linha)

    classe = 2 if probabilidade >= 0.5 else 1

    return classe, probabilidade


# ============================================================
# VALIDAÇÃO CRUZADA ESTRATIFICADA
# ============================================================

def criar_particoes(rotulos_originais, numero_particoes, semente):
    """
    Divide os índices em partições estratificadas.

    IMPORTANTE:
    Esta função recebe as classes originais 1 e 2,
    e não os rótulos binários 0 e 1.
    """
    gerador = random.Random(semente)

    indices_por_classe = {
        classe: [
            i for i, rotulo in enumerate(rotulos_originais)
            if rotulo == classe
        ]
        for classe in CLASSES
    }

    menor_classe = min(
        len(indices_por_classe[classe])
        for classe in CLASSES
    )

    if menor_classe < 2:
        raise ValueError(
            "São necessárias pelo menos duas observações "
            "de cada classe para validação cruzada."
        )

    k = min(numero_particoes, menor_classe)

    particoes = [[] for _ in range(k)]

    for classe in CLASSES:
        indices = indices_por_classe[classe][:]
        gerador.shuffle(indices)

        for posicao, indice in enumerate(indices):
            particoes[posicao % k].append(indice)

    return particoes


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(rotulos_reais, rotulos_preditos):
    resultados = {}

    for classe in CLASSES:
        vp = sum(
            1
            for real, pred in zip(
                rotulos_reais, rotulos_preditos
            )
            if real == classe and pred == classe
        )

        fp = sum(
            1
            for real, pred in zip(
                rotulos_reais, rotulos_preditos
            )
            if real != classe and pred == classe
        )

        fn = sum(
            1
            for real, pred in zip(
                rotulos_reais, rotulos_preditos
            )
            if real == classe and pred != classe
        )

        suporte = sum(
            1 for real in rotulos_reais if real == classe
        )

        precisao = (
            vp / (vp + fp)
            if (vp + fp) > 0 else 0.0
        )

        revocacao = (
            vp / (vp + fn)
            if (vp + fn) > 0 else 0.0
        )

        f1 = (
            2 * precisao * revocacao / (precisao + revocacao)
            if (precisao + revocacao) > 0 else 0.0
        )

        resultados[str(classe)] = {
            "precisao": precisao,
            "revocacao": revocacao,
            "f1": f1,
            "suporte": suporte,
        }

    resultados["MACRO"] = {
        "precisao": sum(
            resultados[str(c)]["precisao"] for c in CLASSES
        ) / len(CLASSES),

        "revocacao": sum(
            resultados[str(c)]["revocacao"] for c in CLASSES
        ) / len(CLASSES),

        "f1": sum(
            resultados[str(c)]["f1"] for c in CLASSES
        ) / len(CLASSES),

        "suporte": len(rotulos_reais),
    }

    acertos = sum(
        real == pred
        for real, pred in zip(
            rotulos_reais, rotulos_preditos
        )
    )

    resultados["MACRO"]["acuracia"] = (
        acertos / len(rotulos_reais)
        if rotulos_reais else 0.0
    )

    return resultados


# ============================================================
# EXECUÇÃO DE UM EXPERIMENTO
# ============================================================

def executar_experimento(
    dados,
    nome_grupo,
    tipo_grupo,
    nome_experimento,
    atributos,
    semente
):
    """
    Executa a validação cruzada para uma questão ou tarefa.

    Os rótulos permanecem como 1 e 2 para a estratificação
    e para o cálculo das métricas. Somente o treinamento
    converte 1/2 em 0/1.
    """
    dados = sorted(
        dados,
        key=lambda r: r["_identificador"]
    )

    matriz = []
    rotulos_originais = []
    registros_validos = []

    for registro in dados:
        valores = []

        for atributo in atributos:
            if atributo == "ocsai_originality":
                valor = registro["_originalidade"]
            elif atributo == "ocsai_confidence":
                valor = registro["_confianca"]
            else:
                raise ValueError(
                    f"Atributo não reconhecido: {atributo}"
                )

            if valor is None:
                continue

            valores.append(valor)

        # Se algum atributo necessário estiver ausente,
        # exclui-se a resposta apenas deste experimento.
        if len(valores) != len(atributos):
            continue

        matriz.append(valores)
        rotulos_originais.append(registro["_classe"])
        registros_validos.append(registro)

    contagem = Counter(rotulos_originais)

    if any(contagem[c] == 0 for c in CLASSES):
        print(
            f"AVISO: {tipo_grupo}={nome_grupo}, "
            f"experimento={nome_experimento} ignorado: "
            f"classe 1={contagem[1]}, classe 2={contagem[2]}."
        )
        return [], []

    particoes = criar_particoes(
        rotulos_originais,
        NUM_PARTICOES,
        semente
    )

    predicoes = [None] * len(registros_validos)
    probabilidades = [None] * len(registros_validos)

    for numero_fold, indices_teste in enumerate(
        particoes, start=1
    ):
        conjunto_teste = set(indices_teste)

        indices_treino = [
            i for i in range(len(matriz))
            if i not in conjunto_teste
        ]

        X_treino = [matriz[i] for i in indices_treino]

        # CORREÇÃO PRINCIPAL:
        # converter as classes reais 1/2 para 0/1 SOMENTE
        # no conjunto usado para treinar a regressão logística.
        y_treino = [
            1 if rotulos_originais[i] == 2 else 0
            for i in indices_treino
        ]

        modelo = treinar_regressao_logistica(
            X_treino,
            y_treino
        )

        for indice in indices_teste:
            classe, probabilidade = prever_classe(
                modelo,
                matriz[indice]
            )

            predicoes[indice] = classe
            probabilidades[indice] = probabilidade

    metricas = calcular_metricas(
        rotulos_originais,
        predicoes
    )

    linhas_metricas = []

    for classe_nome in ["1", "2", "MACRO"]:
        linha = metricas[classe_nome]

        linhas_metricas.append({
            "nivel": tipo_grupo,
            "grupo": nome_grupo,
            "experimento": nome_experimento,
            "variaveis": "+".join(atributos),
            "classe": classe_nome,
            "precisao": f"{linha['precisao']:.6f}",
            "revocacao": f"{linha['revocacao']:.6f}",
            "f1": f"{linha['f1']:.6f}",
            "suporte": linha["suporte"],
            "acuracia": (
                f"{metricas['MACRO']['acuracia']:.6f}"
                if classe_nome == "MACRO" else ""
            ),
            "n": len(registros_validos),
            "particoes": len(particoes),
        })

    linhas_predicoes = []

    for i, registro in enumerate(registros_validos):
        linhas_predicoes.append({
            "nivel": tipo_grupo,
            "grupo": nome_grupo,
            "experimento": nome_experimento,
            "variaveis": "+".join(atributos),
            "arquivo_origem": registro["_arquivo"],
            "origem": registro.get("origem", ""),
            "xml_origem": registro.get("xml_origem", ""),
            "question": registro["_questao"],
            "student": registro.get("student", ""),
            "classe_real": rotulos_originais[i],
            "classe_predita": predicoes[i],
            "probabilidade_classe_2": (
                f"{probabilidades[i]:.8f}"
            ),
            "ocsai_originality": registro.get(
                "ocsai_originality", ""
            ),
            "ocsai_confidence": registro.get(
                "ocsai_confidence", ""
            ),
            "resposta": registro.get("resposta", ""),
        })

    print(
        f"{tipo_grupo}={nome_grupo} | "
        f"experimento={nome_experimento} | "
        f"n={len(registros_validos)} | "
        f"classe 1={contagem[1]} | "
        f"classe 2={contagem[2]} | "
        f"F1 macro={metricas['MACRO']['f1']:.4f}"
    )

    return linhas_metricas, linhas_predicoes


# ============================================================
# SALVAMENTO DOS CSVs
# ============================================================

def salvar_csv(caminho, linhas):
    if not linhas:
        print(f"AVISO: nenhum registro para salvar em {caminho}")
        return

    with open(
        caminho,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=list(linhas[0].keys()),
            extrasaction="ignore"
        )
        escritor.writeheader()
        escritor.writerows(linhas)


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("REGRESSÃO LOGÍSTICA SOBRE OS ESCORES DO OCSAI")
    print("=" * 72)
    print(f"Entrada: {PASTA_ENTRADA}")
    print(f"Saída:   {PASTA_SAIDA}")
    print()

    registros, excluidos = carregar_dados()

    print(f"Registros válidos na leitura: {len(registros)}")
    print(f"Registros excluídos na leitura: {len(excluidos)}")
    print()

    if not registros:
        raise RuntimeError(
            "Não há registros válidos para executar os experimentos."
        )

    metricas_totais = []
    predicoes_totais = []
    contador_grupo = 0

    # --------------------------------------------------------
    # 1. Experimentos por questão
    # --------------------------------------------------------

    for tarefa, questoes in QUESTOES_POR_TAREFA.items():
        for questao in questoes:
            dados_questao = [
                r for r in registros
                if r["_questao"] == questao
            ]

            if not dados_questao:
                print(f"AVISO: nenhum registro para {questao}")
                continue

            for nome_experimento, atributos in EXPERIMENTOS.items():
                contador_grupo += 1

                metricas, predicoes = executar_experimento(
                    dados=dados_questao,
                    nome_grupo=questao,
                    tipo_grupo="QUESTAO",
                    nome_experimento=nome_experimento,
                    atributos=atributos,
                    semente=SEMENTE + contador_grupo
                )

                metricas_totais.extend(metricas)
                predicoes_totais.extend(predicoes)

    # --------------------------------------------------------
    # 2. Experimentos agregados por tarefa
    # --------------------------------------------------------

    for tarefa, questoes in QUESTOES_POR_TAREFA.items():
        dados_tarefa = [
            r for r in registros
            if r["_questao"] in questoes
        ]

        if not dados_tarefa:
            print(f"AVISO: nenhum registro para tarefa {tarefa}")
            continue

        for nome_experimento, atributos in EXPERIMENTOS.items():
            contador_grupo += 1

            metricas, predicoes = executar_experimento(
                dados=dados_tarefa,
                nome_grupo=tarefa,
                tipo_grupo="TAREFA",
                nome_experimento=nome_experimento,
                atributos=atributos,
                semente=SEMENTE + contador_grupo
            )

            metricas_totais.extend(metricas)
            predicoes_totais.extend(predicoes)

    # --------------------------------------------------------
    # 3. Salvar resultados
    # --------------------------------------------------------

    arquivo_metricas = (
        PASTA_SAIDA / "metricas_regressao_logistica.csv"
    )

    arquivo_predicoes = (
        PASTA_SAIDA / "predicoes_fora_amostra.csv"
    )

    arquivo_excluidos = (
        PASTA_SAIDA / "registros_excluidos.csv"
    )

    salvar_csv(arquivo_metricas, metricas_totais)
    salvar_csv(arquivo_predicoes, predicoes_totais)
    salvar_csv(arquivo_excluidos, excluidos)

    print()
    print("=" * 72)
    print("EXECUÇÃO CONCLUÍDA")
    print("=" * 72)
    print(f"Métricas:  {arquivo_metricas}")
    print(f"Predições: {arquivo_predicoes}")
    print(f"Excluídos: {arquivo_excluidos}")


if __name__ == "__main__":
    main()