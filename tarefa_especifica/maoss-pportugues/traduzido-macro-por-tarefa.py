
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimento MAoSS - PISA/OECS 2022
=================================

Etapas:
1. Lê as amostras já traduzidas em:
   resultados_experimentos_qwen_oecs_traduzido/<combinacao>/amostras/

2. Associa cada resposta ao enunciado correspondente em inglês.

3. Gera os CSVs de entrada para o MAoSS e um arquivo de mapeamento.

4. Após a execução do MAoSS, lê os escores prediction e executa
   regressão logística com validação cruzada estratificada de 5 folds.

5. Calcula precisão, revocação (recall) e F1 para as classes 1 e 2,
   macro-F1 por questão, tarefa e conjunto agregado.

Modos:
    preparar  - prepara os CSVs para submissão ao MAoSS
    avaliar   - avalia os escores retornados pelo MAoSS
    status    - verifica quais arquivos já foram preparados/avaliados

Observação:
A submissão ao MAoSS é feita pela interface do serviço.
"""

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        f1_score,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    sys.exit(
        "ERRO: scikit-learn não está instalado neste ambiente.\n"
        "Instale-o no ambiente virtual ativo com:\n"
        "python -m pip install scikit-learn"
    )


# ---------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------

BASE = Path.home() / "Documentos" / "Doutorado"

INPUT_ROOT = (
    BASE / "resultados_experimentos_qwen_oecs_traduzido"
)

OUTPUT_ROOT = BASE / "resultados_experimentos_maoss_traduzido"

PROMPTS_FILE = (
    BASE
    / "Arquivos ..."
)

COMBINACOES = [
    "america_latina",
    "america_sul",
    "espanha",
    "peninsula_iberica",
    "todos",
]

QUESTOES_GCI = [
    "T240Q01",
    "T300Q01",
    "T350Q01",
    "T360Q01",
    "T370Q01",
    "T570Q02",
]

QUESTOES_GDI = [
    "T240Q02",
    "T300Q02",
    "T350Q02",
    "T570Q01",
]

QUESTOES_EII = [
    "T350Q03",
    "T570Q03",
]

QUESTOES = QUESTOES_GCI + QUESTOES_GDI + QUESTOES_EII

MAPA_TAREFAS = {
    **{q: "GCI" for q in QUESTOES_GCI},
    **{q: "GDI" for q in QUESTOES_GDI},
    **{q: "EII" for q in QUESTOES_EII},
}

RANDOM_STATE = 20261015
N_SPLITS = 5


# ---------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------


def ler_csv(caminho):
    """Lê os CSVs do experimento usando vírgula como delimitador."""
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    with caminho.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter=",",
            quotechar='"',
            doublequote=True,
            skipinitialspace=False,
        )

        if not reader.fieldnames:
            raise ValueError(
                f"CSV sem cabeçalho: {caminho}"
            )

        registros = list(reader)

    return registros



def escrever_csv(caminho, colunas, registros):
    """Grava CSV com cabeçalho e codificação UTF-8."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with caminho.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=colunas,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(registros)


def texto(valor):
    if valor is None:
        return ""
    return str(valor).strip()


def numero(valor):
    """Converte um valor numérico, aceitando vírgula decimal."""
    valor = texto(valor)

    if not valor:
        return None

    valor = valor.replace(",", ".")

    try:
        resultado = float(valor)
    except ValueError:
        return None

    if not math.isfinite(resultado):
        return None

    return resultado


# ---------------------------------------------------------------------
# ENUNCIADOS
# ---------------------------------------------------------------------

def carregar_enunciados():
    """
    Espera um CSV com as colunas:
        question,item

    question: código da questão, por exemplo T570Q01
    item: enunciado/estímulo em inglês.
    """
    registros = ler_csv(PROMPTS_FILE)
    enunciados = {}

    for r in registros:
        questao = texto(r.get("question"))
        item = texto(r.get("item"))

        if questao and item:
            enunciados[questao] = item

    faltantes = sorted(set(QUESTOES) - set(enunciados))

    if faltantes:
        raise ValueError(
            "Não foram encontrados enunciados para estas questões no "
            f"arquivo {PROMPTS_FILE}:\n"
            + ", ".join(faltantes)
            + "\nConfira os nomes das colunas question e item."
        )

    return enunciados


# ---------------------------------------------------------------------
# IDENTIFICADORES E AMOSTRAS
# ---------------------------------------------------------------------

def criar_id(combinacao, questao, indice, student):
    """
    Identificador estável por linha.
    O índice diferencia registros mesmo se student estiver repetido.
    """
    texto_id = (
        f"{combinacao}|{questao}|{indice}|{texto(student)}"
    )
    hash_curto = hashlib.sha1(
        texto_id.encode("utf-8")
    ).hexdigest()[:10]

    return f"{combinacao}_{questao}_{indice:05d}_{hash_curto}"


def carregar_amostras_combinacao(combinacao, enunciados):
    pasta = INPUT_ROOT / combinacao / "amostras"

    if not pasta.is_dir():
        raise FileNotFoundError(
            f"Pasta de amostras não encontrada: {pasta}"
        )

    registros_completos = []
    problemas = []

    for questao in QUESTOES:
        arquivo = pasta / f"{questao}_amostra.csv"

        if not arquivo.exists():
            problemas.append(f"Arquivo ausente: {arquivo}")
            continue

        linhas = ler_csv(arquivo)

        if not linhas:
            problemas.append(f"Arquivo sem registros: {arquivo}")
            continue

        for indice, linha in enumerate(linhas, start=1):
            questao_lida = texto(linha.get("questao"))

            if questao_lida and questao_lida != questao:
                problemas.append(
                    f"{arquivo}: questão inesperada "
                    f"{questao_lida!r} na linha {indice}"
                )
                continue

            resposta = texto(linha.get("resposta"))
            classe_texto = texto(linha.get("classe_real"))

            if not resposta:
                problemas.append(
                    f"{arquivo}: resposta vazia na linha {indice}"
                )
                continue

            try:
                classe = int(classe_texto)
            except (TypeError, ValueError):
                problemas.append(
                    f"{arquivo}: classe_real inválida na linha {indice}: "
                    f"{classe_texto!r}"
                )
                continue

            if classe not in (1, 2):
                problemas.append(
                    f"{arquivo}: classe_real deve ser 1 ou 2; "
                    f"linha {indice}: {classe}"
                )
                continue

            student = texto(linha.get("student"))
            registro_id = criar_id(
                combinacao, questao, indice, student
            )

            registros_completos.append({
                "id": registro_id,
                "combinacao": combinacao,
                "questao": questao,
                "tarefa": MAPA_TAREFAS[questao],
                "student": student,
                "pais": texto(linha.get("pais")),
                "pais_nome": texto(linha.get("pais_nome")),
                "idioma": texto(linha.get("idioma")),
                "arquivo_xml": texto(linha.get("arquivo_xml")),
                "code0": texto(linha.get("code0")),
                "classe_real": classe,
                "item": enunciados[questao],
                "response": resposta,
            })

    if problemas:
        mensagem = "\n".join(problemas[:30])
        if len(problemas) > 30:
            mensagem += (
                f"\n... e mais {len(problemas) - 30} problema(s)."
            )
        raise ValueError(
            f"Problemas nos arquivos de {combinacao}:\n{mensagem}"
        )

    return registros_completos


# ---------------------------------------------------------------------
# PREPARAÇÃO DA ENTRADA MAoSS
# ---------------------------------------------------------------------

def preparar_combinacao(combinacao, enunciados):
    registros = carregar_amostras_combinacao(
        combinacao, enunciados
    )

    pasta_saida = OUTPUT_ROOT / combinacao
    pasta_saida.mkdir(parents=True, exist_ok=True)

    # Este arquivo contém os campos necessários ao experimento.
    # O campo id permite associar o score à resposta original.
    entrada_maoss = [
        {
            "id": r["id"],
            "item": r["item"],
            "response": r["response"],
        }
        for r in registros
    ]

    # Mapeamento separado com classes reais e metadados.
    mapeamento = [
        {
            "id": r["id"],
            "combinacao": r["combinacao"],
            "questao": r["questao"],
            "tarefa": r["tarefa"],
            "student": r["student"],
            "pais": r["pais"],
            "pais_nome": r["pais_nome"],
            "idioma": r["idioma"],
            "arquivo_xml": r["arquivo_xml"],
            "code0": r["code0"],
            "classe_real": r["classe_real"],
        }
        for r in registros
    ]

    escrever_csv(
        pasta_saida / "maoss_entrada_cap.csv",
        ["id", "item", "response"],
        entrada_maoss,
    )

    escrever_csv(
        pasta_saida / "maoss_mapeamento.csv",
        [
            "id", "combinacao", "questao", "tarefa",
            "student", "pais", "pais_nome", "idioma",
            "arquivo_xml", "code0", "classe_real",
        ],
        mapeamento,
    )

    resumo = []

    for questao in QUESTOES:
        subset = [r for r in registros if r["questao"] == questao]

        if not subset:
            continue

        n1 = sum(r["classe_real"] == 1 for r in subset)
        n2 = sum(r["classe_real"] == 2 for r in subset)

        resumo.append({
            "combinacao": combinacao,
            "questao": questao,
            "tarefa": MAPA_TAREFAS[questao],
            "n_total": len(subset),
            "n_classe_1": n1,
            "n_classe_2": n2,
        })

    escrever_csv(
        pasta_saida / "resumo_entrada.csv",
        [
            "combinacao", "questao", "tarefa", "n_total",
            "n_classe_1", "n_classe_2",
        ],
        resumo,
    )

    print(
        f"[OK] {combinacao}: {len(registros)} respostas preparadas"
    )
    print(f"     Entrada: {pasta_saida / 'maoss_entrada_cap.csv'}")
    print(f"     Mapeamento: {pasta_saida / 'maoss_mapeamento.csv'}")

    return len(registros)


def preparar_todas():
    enunciados = carregar_enunciados()
    total = 0

    for combinacao in COMBINACOES:
        total += preparar_combinacao(combinacao, enunciados)

    print("\n" + "=" * 72)
    print("PREPARAÇÃO CONCLUÍDA")
    print("=" * 72)
    print(f"Total de respostas preparadas: {total}")
    print(f"Pasta de saída: {OUTPUT_ROOT}")
    print(
        "\nAgora envie cada arquivo maoss_entrada_cap.csv ao MAoSS. "
        "Salve os resultados de cada combinação na pasta correspondente "
        "com o nome maoss_resultados_cap.csv."
    )


# ---------------------------------------------------------------------
# LEITURA DOS ESCORES MAoSS
# ---------------------------------------------------------------------

def carregar_resultados_maoss(combinacao):
    pasta = OUTPUT_ROOT / combinacao
    arquivo_scores = pasta / "maoss_resultados_cap.csv"
    arquivo_mapa = pasta / "maoss_mapeamento.csv"

    if not arquivo_scores.exists():
        raise FileNotFoundError(
            f"Resultado MAoSS não encontrado: {arquivo_scores}\n"
            "Envie a entrada ao MAoSS e salve o CSV de resultados "
            "com esse nome antes de executar o modo avaliar."
        )

    if not arquivo_mapa.exists():
        raise FileNotFoundError(
            f"Mapeamento não encontrado: {arquivo_mapa}\n"
            "Execute primeiro o modo preparar."
        )

    scores = ler_csv(arquivo_scores)
    mapa = ler_csv(arquivo_mapa)

    if not scores:
        raise ValueError(f"CSV de resultados vazio: {arquivo_scores}")

    if "prediction" not in scores[0]:
        raise ValueError(
            f"O arquivo {arquivo_scores} precisa conter a coluna "
            "'prediction'. Colunas encontradas: "
            + ", ".join(scores[0].keys())
        )

    mapa_por_id = {texto(r.get("id")): r for r in mapa}
    linhas_com_score = []

    # Caso o resultado tenha ID, a associação é feita pelo ID.
    tem_id = "id" in scores[0]

    if tem_id:
        ids_scores = [texto(r.get("id")) for r in scores]

        if any(not x for x in ids_scores):
            raise ValueError(
                f"Há IDs vazios em {arquivo_scores}. "
                "Confira se o MAoSS preservou a coluna id."
            )

        if len(ids_scores) != len(set(ids_scores)):
            raise ValueError(
                f"Há IDs duplicados em {arquivo_scores}."
            )

        ids_mapa = set(mapa_por_id)
        ids_resultados = set(ids_scores)

        ausentes = ids_mapa - ids_resultados
        extras = ids_resultados - ids_mapa

        if ausentes or extras:
            raise ValueError(
                f"Os IDs dos resultados não correspondem ao mapeamento "
                f"de {combinacao}. "
                f"IDs sem resultado: {len(ausentes)}; "
                f"IDs desconhecidos: {len(extras)}. "
                "Não é seguro associar os escores automaticamente."
            )

        for score_linha in scores:
            registro_id = texto(score_linha.get("id"))
            original = mapa_por_id[registro_id]
            prediction = numero(score_linha.get("prediction"))

            if prediction is None:
                raise ValueError(
                    f"prediction inválida para id={registro_id}: "
                    f"{score_linha.get('prediction')!r}"
                )

            linhas_com_score.append({
                **original,
                "prediction": prediction,
            })

    else:
        # Fallback: somente usa a ordem se a quantidade de linhas
        # for idêntica. O MAoSS deve preservar a ordem original.
        if len(scores) != len(mapa):
            raise ValueError(
                f"O resultado de {combinacao} não contém coluna id e "
                f"tem {len(scores)} linhas, mas o mapeamento tem "
                f"{len(mapa)}. Não é seguro associar os registros."
            )

        print(
            f"[AVISO] {combinacao}: o resultado não contém coluna id. "
            "A associação será feita pela ordem das linhas. "
            "Confirme que o MAoSS não alterou essa ordem."
        )

        for original, score_linha in zip(mapa, scores):
            prediction = numero(score_linha.get("prediction"))

            if prediction is None:
                raise ValueError(
                    f"prediction inválida para id={original['id']}: "
                    f"{score_linha.get('prediction')!r}"
                )

            linhas_com_score.append({
                **original,
                "prediction": prediction,
            })

    return linhas_com_score


# ---------------------------------------------------------------------
# REGRESSÃO LOGÍSTICA E MÉTRICAS
# ---------------------------------------------------------------------

def avaliar_grupo(registros, nome_grupo, nivel):
    """
    Treina uma regressão logística sobre o score prediction do MAoSS.
    As classes-alvo são classe_real 1 e 2.
    """
    if not registros:
        return None, []

    y = [int(r["classe_real"]) for r in registros]
    X = [[float(r["prediction"])] for r in registros]

    contagem = {
        classe: y.count(classe)
        for classe in (1, 2)
    }

    if min(contagem.values()) < N_SPLITS:
        print(
            f"[AVISO] Grupo {nome_grupo}: não há pelo menos "
            f"{N_SPLITS} exemplos de cada classe. "
            "Grupo não avaliado."
        )
        return None, []

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    modelo = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=RANDOM_STATE,
        ),
    )

    predicoes = cross_val_predict(
        modelo,
        X,
        y,
        cv=cv,
        method="predict",
    )

    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y,
        predicoes,
        labels=[1, 2],
        zero_division=0,
    )

    metricas = {
        "nivel": nivel,
        "grupo": nome_grupo,
        "n": len(registros),
        "n_classe_1": contagem[1],
        "n_classe_2": contagem[2],
        "accuracy": accuracy_score(y, predicoes),
        "precision_classe_1": precisao[0],
        "recall_classe_1": recall[0],
        "f1_classe_1": f1[0],
        "suporte_classe_1": suporte[0],
        "precision_classe_2": precisao[1],
        "recall_classe_2": recall[1],
        "f1_classe_2": f1[1],
        "suporte_classe_2": suporte[1],
        "macro_precision": float(precisao.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": f1_score(
            y, predicoes, labels=[1, 2],
            average="macro", zero_division=0,
        ),
    }

    registros_preditos = []

    for registro, previsto in zip(registros, predicoes):
        registros_preditos.append({
            "id": registro["id"],
            "combinacao": registro["combinacao"],
            "questao": registro["questao"],
            "tarefa": registro["tarefa"],
            "student": registro["student"],
            "classe_real": int(registro["classe_real"]),
            "prediction_maoss": registro["prediction"],
            "classe_predita_cv": int(previsto),
            "grupo_avaliacao": nome_grupo,
            "nivel_avaliacao": nivel,
        })

    return metricas, registros_preditos


def avaliar_combinacao(combinacao):
    registros = carregar_resultados_maoss(combinacao)

    pasta_saida = OUTPUT_ROOT / combinacao
    metricas_todas = []
    predicoes_todas = []

    # Por questão
    for questao in QUESTOES:
        subset = [r for r in registros if r["questao"] == questao]

        if not subset:
            continue

        metricas, predicoes = avaliar_grupo(
            subset, questao, "questao"
        )

        if metricas:
            metricas_todas.append(metricas)
            predicoes_todas.extend(predicoes)

    # Por tarefa: GCI, GDI, EII
    for tarefa in ("GCI", "GDI", "EII"):
        subset = [r for r in registros if r["tarefa"] == tarefa]

        metricas, predicoes = avaliar_grupo(
            subset, tarefa, "tarefa"
        )

        if metricas:
            metricas_todas.append(metricas)
            predicoes_todas.extend(predicoes)

    # Agregado: todas as 12 questões
    metricas, predicoes = avaliar_grupo(
        registros, "PISA_completo", "agregado"
    )

    if metricas:
        metricas_todas.append(metricas)
        predicoes_todas.extend(predicoes)

    if not metricas_todas:
        raise ValueError(
            f"Nenhum grupo de {combinacao} pôde ser avaliado. "
            "Confira as classes e a quantidade de registros."
        )

    colunas_metricas = [
        "nivel", "grupo", "n", "n_classe_1", "n_classe_2",
        "accuracy",
        "precision_classe_1", "recall_classe_1", "f1_classe_1",
        "suporte_classe_1",
        "precision_classe_2", "recall_classe_2", "f1_classe_2",
        "suporte_classe_2",
        "macro_precision", "macro_recall", "macro_f1",
    ]

    escrever_csv(
        pasta_saida / "metricas_regressao_logistica.csv",
        colunas_metricas,
        metricas_todas,
    )

    escrever_csv(
        pasta_saida / "predicoes_validacao_cruzada.csv",
        [
            "id", "combinacao", "questao", "tarefa", "student",
            "classe_real", "prediction_maoss", "classe_predita_cv",
            "grupo_avaliacao", "nivel_avaliacao",
        ],
        predicoes_todas,
    )

    print("\n" + "-" * 72)
    print(f"RESULTADOS: {combinacao}")
    print("-" * 72)

    # Na tela, exibir somente as três métricas macro agrupadas por tarefa.
    # Os resultados por questão e o agregado completo continuam salvos no CSV.
    print("\nMÉTRICAS MACRO POR TAREFA")
    print(f"{'Tarefa':<8} | {'Precisão macro':>14} | {'Revocação macro':>15} | {'F-medida macro':>14}")
    print("-" * 64)

    for m in metricas_todas:
        if m["nivel"] == "tarefa":
            print(
                f"{m['grupo']:<8} | "
                f"{m['macro_precision']:>14.4f} | "
                f"{m['macro_recall']:>15.4f} | "
                f"{m['macro_f1']:>14.4f}"
            )

    print(f"\nMétricas: {pasta_saida / 'metricas_regressao_logistica.csv'}")
    print(f"Predições: {pasta_saida / 'predicoes_validacao_cruzada.csv'}")

    return metricas_todas


def avaliar_todas():
    todas_metricas = []

    for combinacao in COMBINACOES:
        metricas = avaliar_combinacao(combinacao)

        for m in metricas:
            todas_metricas.append({
                "combinacao": combinacao,
                **m,
            })

    colunas = [
        "combinacao", "nivel", "grupo", "n",
        "n_classe_1", "n_classe_2", "accuracy",
        "precision_classe_1", "recall_classe_1", "f1_classe_1",
        "suporte_classe_1",
        "precision_classe_2", "recall_classe_2", "f1_classe_2",
        "suporte_classe_2",
        "macro_precision", "macro_recall", "macro_f1",
    ]

    escrever_csv(
        OUTPUT_ROOT / "metricas_todas_combinacoes.csv",
        colunas,
        todas_metricas,
    )

    print("\n" + "=" * 72)
    print("AVALIAÇÃO DE TODAS AS COMBINAÇÕES CONCLUÍDA")
    print("=" * 72)
    print(
        "Arquivo consolidado:",
        OUTPUT_ROOT / "metricas_todas_combinacoes.csv",
    )


# ---------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------

def mostrar_status():
    print("\nSTATUS DOS ARQUIVOS MAoSS")
    print("=" * 72)

    for combinacao in COMBINACOES:
        pasta = OUTPUT_ROOT / combinacao
        entrada = pasta / "maoss_entrada_cap.csv"
        mapa = pasta / "maoss_mapeamento.csv"
        resultado = pasta / "maoss_resultados_cap.csv"
        metricas = pasta / "metricas_regressao_logistica.csv"

        print(f"\n{combinacao}")
        print(f"  Entrada preparada: {'SIM' if entrada.exists() else 'NÃO'}")
        print(f"  Mapeamento:        {'SIM' if mapa.exists() else 'NÃO'}")
        print(f"  Resultado MAoSS:   {'SIM' if resultado.exists() else 'NÃO'}")
        print(f"  Métricas:          {'SIM' if metricas.exists() else 'NÃO'}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experimento MAoSS para amostras PISA/OECS traduzidas."
    )

    parser.add_argument(
        "modo",
        choices=["preparar", "avaliar", "status"],
        help="Etapa do experimento.",
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.modo == "preparar":
        preparar_todas()

    elif args.modo == "avaliar":
        avaliar_todas()

    elif args.modo == "status":
        mostrar_status()


if __name__ == "__main__":
    main()
