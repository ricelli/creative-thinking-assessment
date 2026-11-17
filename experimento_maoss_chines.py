#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimento MAoSS/CAP — respostas chinesas do PISA 2022.

Entrada:
  ~/Documentos/Doutorado/amostra_chines_qwen.csv

Estímulos em inglês:
  Arquivo enunciados_pisa_ingles.csv do experimento MAoSS em inglês.

Etapas:
  preparar -> gera CSV para envio manual à CAP
  avaliar  -> associa os escores e executa regressão logística

Requer scikit-learn apenas para a etapa avaliar.
"""

from pathlib import Path
from collections import Counter, defaultdict, deque
import csv
import math
import sys
import argparse

BASE = Path.home() / "Documentos" / "Doutorado"
INPUT = BASE / "amostra_chines_qwen.csv"

ENGLISH_PROMPTS = (
    BASE
    / "Arquivos OECS-20250827T193412Z-1-001"
    / "Arquivos OECS"
    / "2022MS-responsesXML"
    / "2022MS"
    / "ingles"
    / "amostra_qwen"
    / "maoss_experimento"
    / "enunciados_pisa_ingles.csv"
)

OUT = BASE / "maoss_experimento_chines"
UPLOAD = OUT / "maoss_entrada_cap.csv"
SCORED = OUT / "maoss_resultados_cap.csv"
MERGED = OUT / "maoss_resultados_com_classes.csv"
METRICS = OUT / "maoss_metricas.csv"
PREDICTIONS = OUT / "maoss_predicoes_oof.csv"

QUESTION_TO_TASK = {
    "T240Q01": "GCI",
    "T300Q01": "GCI",
    "T350Q01": "GCI",
    "T360Q01": "GCI",
    "T370Q01": "GCI",
    "T570Q02": "GCI",
    "T240Q02": "GDI",
    "T300Q02": "GDI",
    "T350Q02": "GDI",
    "T570Q01": "GDI",
    "T350Q03": "EII",
    "T570Q03": "EII",
}


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        return list(csv.DictReader(f, dialect=dialect))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    return (value or "").strip()


def load_prompts():
    if not ENGLISH_PROMPTS.exists():
        raise FileNotFoundError(
            "Não encontrei os enunciados do experimento em inglês:\n"
            f"{ENGLISH_PROMPTS}\n\n"
            "Copie para esse local o arquivo enunciados_pisa_ingles.csv "
            "usado no experimento MAoSS em inglês."
        )

    rows = read_csv(ENGLISH_PROMPTS)
    if not rows or not {"question", "item"}.issubset(rows[0]):
        raise ValueError(
            "O arquivo de enunciados precisa ter as colunas question,item."
        )

    prompts = {
        clean(r.get("question")): clean(r.get("item"))
        for r in rows
    }

    missing = [
        q for q in QUESTION_TO_TASK
        if not prompts.get(q)
    ]
    if missing:
        raise ValueError(
            "Faltam estímulos em inglês para: " + ", ".join(missing)
        )

    return prompts


def load_sample():
    if not INPUT.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT}")

    rows = read_csv(INPUT)
    if not rows:
        raise ValueError("O arquivo de amostra está vazio.")

    required = {"question", "resposta", "classe_real"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            "Colunas ausentes no CSV: " + ", ".join(sorted(missing))
        )

    records = []
    for i, row in enumerate(rows, start=1):
        question = clean(row.get("question"))
        response = clean(row.get("resposta"))

        if question not in QUESTION_TO_TASK or not response:
            continue

        try:
            label = int(float(clean(row.get("classe_real"))))
        except (ValueError, TypeError):
            continue

        if label not in (1, 2):
            continue

        # O identificador permite associar o escore à resposta original.
        rid = f"CHN_{i:06d}"

        records.append({
            "id": rid,
            "question": question,
            "task": QUESTION_TO_TASK[question],
            "classe_real": label,
            "item": "",
            "response": response,
            "origem": clean(row.get("origem")),
            "xml_origem": clean(row.get("xml_origem")),
            "student": clean(row.get("student")),
        })

    if not records:
        raise ValueError(
            "Nenhuma resposta válida com classe_real 1 ou 2 foi encontrada."
        )

    return records


def prepare():
    prompts = load_prompts()
    records = load_sample()

    for r in records:
        r["item"] = prompts[r["question"]]

    fields = [
        "id", "question", "task", "classe_real",
        "item", "response", "origem", "xml_origem", "student"
    ]
    write_csv(UPLOAD, records, fields)

    print("\nPreparação concluída.")
    print(f"Entrada original: {INPUT}")
    print(f"CSV para enviar ao MAoSS: {UPLOAD}")
    print(f"Respostas preparadas: {len(records)}")
    print("\nRespostas por questão:")
    for q, n in sorted(Counter(r["question"] for r in records).items()):
        print(f"  {q}: {n}")

    print("\nPróximos passos:")
    print("1. Abra https://cap.ist.psu.edu/maoss")
    print("2. Envie maoss_entrada_cap.csv.")
    print("3. Baixe o CSV pontuado pelo MAoSS.")
    print(f"4. Salve-o como: {SCORED}")
    print("5. Execute: python3 experimento_maoss_chines.py avaliar")
    print(
        "\nImportante: mantenha a coluna id no arquivo retornado, "
        "se a plataforma permitir preservar colunas adicionais."
    )


def associate_scores(original, scored):
    """
    Associa prediction às respostas originais.
    Prioridade: id; alternativa: correspondência por item + response,
    respeitando a ordem das ocorrências duplicadas.
    """
    if not scored:
        raise ValueError("O arquivo de escores está vazio.")

    if "prediction" not in scored[0]:
        raise ValueError(
            "Não encontrei a coluna prediction no arquivo retornado pelo MAoSS. "
            f"Colunas: {list(scored[0])}"
        )

    original_by_id = {r["id"]: r for r in original}
    merged = []

    # Primeiro tenta usar os identificadores preservados pelo CAP.
    if "id" in scored[0]:
        for row in scored:
            rid = clean(row.get("id"))
            if rid not in original_by_id:
                continue
            try:
                score = float(clean(row.get("prediction")).replace(",", "."))
            except (ValueError, TypeError):
                continue
            if math.isfinite(score):
                merged.append({
                    **original_by_id[rid],
                    "maoss_prediction": score,
                })

        if merged:
            return merged

    # Se o CAP não preservou id, tenta item + response.
    # Filas preservam respostas repetidas sem reutilizar um mesmo registro.
    queues = defaultdict(deque)
    for r in original:
        queues[(r["item"], r["response"])].append(r)

    for row in scored:
        key = (
            clean(row.get("item")),
            clean(row.get("response")),
        )
        if not queues[key]:
            continue
        try:
            score = float(clean(row.get("prediction")).replace(",", "."))
        except (ValueError, TypeError):
            continue
        if math.isfinite(score):
            base = queues[key].popleft()
            merged.append({**base, "maoss_prediction": score})

    if not merged and len(scored) == len(original):
        print(
            "[AVISO] Não foi possível associar por id ou texto. "
            "Verifique manualmente a ordem e as colunas do CSV retornado."
        )

    return merged


def evaluate():
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            f1_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "A avaliação requer scikit-learn no Python usado para executar "
            "o script. Instale-o nesse ambiente antes de continuar."
        ) from exc

    original = load_sample()
    prompts = load_prompts()
    for r in original:
        r["item"] = prompts[r["question"]]

    if not SCORED.exists():
        raise FileNotFoundError(
            "Arquivo pontuado não encontrado:\n"
            f"{SCORED}\n"
            "Envie o CSV ao MAoSS, baixe o resultado e salve nesse local."
        )

    scored = read_csv(SCORED)
    merged = associate_scores(original, scored)

    if not merged:
        raise ValueError(
            "Não foi possível associar os escores às respostas originais. "
            "Confira o CSV baixado da CAP."
        )

    write_csv(
        MERGED, merged,
        [
            "id", "question", "task", "classe_real", "response",
            "maoss_prediction", "origem", "xml_origem", "student"
        ],
    )

    print(f"\nRespostas válidas na amostra: {len(original)}")
    print(f"Respostas associadas a escores: {len(merged)}")
    print(f"Respostas sem escore: {len(original) - len(merged)}")

    groups = []
    for q in sorted(QUESTION_TO_TASK):
        g = [r for r in merged if r["question"] == q]
        if g:
            groups.append((f"QUESTAO_{q}", g))

    for task in ("GCI", "GDI", "EII"):
        g = [r for r in merged if r["task"] == task]
        if g:
            groups.append((f"TAREFA_{task}", g))

    groups.append(("AGREGADO_12_QUESTOES", merged))

    metrics = []
    oof_rows = []

    for name, group in groups:
        y = [int(r["classe_real"]) for r in group]
        x = [[float(r["maoss_prediction"])] for r in group]
        counts = Counter(y)

        if len(counts) < 2 or min(counts.values()) < 5:
            print(
                f"[PULADO] {name}: são necessários pelo menos "
                "5 exemplos de cada classe para CV estratificada de 5 folds."
            )
            continue

        cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=20261014
        )
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=20261014),
        )
        pred = cross_val_predict(model, x, y, cv=cv, method="predict")

        p, r, f, _ = precision_recall_fscore_support(
            y, pred, labels=[1, 2], zero_division=0
        )

        row = {
            "grupo": name,
            "n": len(y),
            "n_classe_1": counts[1],
            "n_classe_2": counts[2],
            "acuracia": round(accuracy_score(y, pred), 6),
            "precisao_classe_1": round(p[0], 6),
            "revocacao_classe_1": round(r[0], 6),
            "f1_classe_1": round(f[0], 6),
            "precisao_classe_2": round(p[1], 6),
            "revocacao_classe_2": round(r[1], 6),
            "f1_classe_2": round(f[1], 6),
            "macro_f1": round(
                f1_score(y, pred, labels=[1, 2], average="macro", zero_division=0),
                6,
            ),
        }
        metrics.append(row)

        for rec, actual, prediction in zip(group, y, pred):
            oof_rows.append({
                "id": rec["id"],
                "question": rec["question"],
                "task": rec["task"],
                "classe_real": actual,
                "maoss_prediction": rec["maoss_prediction"],
                "classe_predita_oof": int(prediction),
                "grupo_avaliacao": name,
            })

        print(
            f"{name}: n={len(y)}, macro-F1={row['macro_f1']:.4f}, "
            f"F1 classe 1={row['f1_classe_1']:.4f}, "
            f"F1 classe 2={row['f1_classe_2']:.4f}"
        )

    if not metrics:
        raise ValueError(
            "Nenhum grupo pôde ser avaliado. Confira as quantidades por classe."
        )

    write_csv(METRICS, metrics, list(metrics[0]))
    write_csv(
        PREDICTIONS, oof_rows,
        [
            "id", "question", "task", "classe_real",
            "maoss_prediction", "classe_predita_oof", "grupo_avaliacao"
        ],
    )

    print("\nAvaliação concluída.")
    print(f"Resultados associados: {MERGED}")
    print(f"Métricas: {METRICS}")
    print(f"Predições out-of-fold: {PREDICTIONS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["preparar", "avaliar"])
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.modo == "preparar":
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        sys.exit(1)