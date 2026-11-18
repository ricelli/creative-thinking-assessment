#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepara dados PT/BR/Macau para MAoSS e avalia as pontuações CAP com
regressão logística e validação cruzada estratificada de 5 folds.

Uso:
  python3 experimento_maoss_pt_br_macao.py preparar
  # Envie maoss_entrada_cap.csv ao MAoSS/CAP e baixe o CSV com prediction.
  python3 experimento_maoss_pt_br_macao.py avaliar --resultados caminho/resultado_maoss.csv
  # Ou, depois de obter o resultado:
  python3 experimento_maoss_pt_br_macao.py tudo --resultados caminho/resultado_maoss.csv

Entradas:
  ~/pt_eii.csv
  ~/pt_gci.csv
  ~/t_gdi.csv

Usa somente source=test, code0_3 nas classes 1 e 2, e response_US
(já traduzida). Os enunciados em inglês são reutilizados de
maoss_experimento/enunciados_pisa_ingles.csv.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path.home() / "Documentos" / "Doutorado"
DATASET_DIR = BASE / "dataset-pt" / "dataset"
OUT_DIR = DATASET_DIR / "maoss_experimento"

INPUT_FILES = {
    "EII": DATASET_DIR / "pt_eii.csv",
    "GCI": DATASET_DIR / "pt_gci.csv",
    "GDI": DATASET_DIR / "pt_gdi.csv",
}
PROMPTS_FILE = (
    BASE
    / "Arquivos ..."

)
QUESTOES_GCI = ["T240Q01", "T300Q01", "T350Q01", "T360Q01", "T370Q01", "T570Q02"]
QUESTOES_GDI = ["T240Q02", "T300Q02", "T350Q02", "T570Q01"]
QUESTOES_EII = ["T350Q03", "T570Q03"]
TASK_BY_QUESTION = {
    **{q: "GCI" for q in QUESTOES_GCI},
    **{q: "GDI" for q in QUESTOES_GDI},
    **{q: "EII" for q in QUESTOES_EII},
}
RANDOM_STATE = 20261015
N_SPLITS = 5


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Não encontrei o arquivo: {path}")
    # Os arquivos podem ter BOM e separador ;. Detecta ; ou ,.
    last_error = None
    for sep in (";", ",", "\t"):
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=str, keep_default_na=False)
            if len(df.columns) > 1:
                df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
                return df
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Não consegui ler {path}: {last_error}")


def normalize_question(value: object) -> str:
    return str(value).strip().upper()


def preparar() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = read_csv_flexible(PROMPTS_FILE)
    if not {"question", "item"}.issubset(prompts.columns):
        raise ValueError(
            f"O arquivo de enunciados precisa ter colunas 'question' e 'item'. "
            f"Encontrei: {list(prompts.columns)}"
        )
    prompts["question_key"] = prompts["question"].map(normalize_question)
    prompts = prompts.drop_duplicates("question_key", keep="first")
    prompt_map = dict(zip(prompts["question_key"], prompts["item"]))

    frames = []
    for task, path in INPUT_FILES.items():
        df = read_csv_flexible(path)
        required = {"question", "student", "code0_3", "response_US", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: faltam colunas {sorted(missing)}; encontrei {list(df.columns)}")
        df["question"] = df["question"].map(normalize_question)
        df["source_norm"] = df["source"].astype(str).str.strip().str.casefold()
        df = df[df["source_norm"] == "test"].copy()
        df["classe_real"] = pd.to_numeric(df["code0_3"], errors="coerce")
        df = df[df["classe_real"].isin([1, 2])].copy()
        df["response_US"] = df["response_US"].astype(str).str.strip()
        df = df[df["response_US"].ne("")].copy()
        df["task"] = task
        df["item"] = df["question"].map(prompt_map)
        missing_prompts = sorted(df.loc[df["item"].isna(), "question"].unique())
        if missing_prompts:
            raise ValueError(
                f"Não há enunciado em inglês para {missing_prompts} no arquivo {PROMPTS_FILE}"
            )
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError("Nenhum arquivo de entrada foi processado.")
    data = pd.concat(frames, ignore_index=True)
    data.insert(0, "id", [f"PTBRM-{i:07d}" for i in range(1, len(data) + 1)])
    # Mapeamento completo para ligar as respostas pontuadas às linhas originais.
    mapping_cols = [
        "id", "question", "task", "student", "classe_real", "response",
        "response_US", "source", "source_file",
    ]
    data["response"] = data["response"] if "response" in data.columns else ""
    for col in mapping_cols:
        if col not in data.columns:
            data[col] = ""
    data["classe_real"] = data["classe_real"].astype(int)

    # Formato de entrada do CAP/MAoSS utilizado nos experimentos anteriores.
    # O MAoSS/CAP espera o esquema mínimo com apenas "item" e "response".
    # CSV padrão separado por vírgulas, UTF-8 sem BOM. Não incluir metadados
    # (id, tarefa, classe etc.) no arquivo enviado ao sistema.
    cap = pd.DataFrame({
        "id": data["id"].astype(str),
        "item": data["item"].astype(str),
        "response": data["response_US"].astype(str),
    })
    cap_path = OUT_DIR / "maoss_entrada_cap.csv"
    # Reproduz o formato do arquivo da America Latina que funcionou:
    # UTF-8 com BOM, separador vírgula, colunas id,item,response e
    # terminadores de linha padrão do pandas (sem "\\n" literal).
    cap.to_csv(
        cap_path,
        index=False,
        encoding="utf-8-sig",
        sep=",",
        quoting=csv.QUOTE_MINIMAL,
    )

    data[mapping_cols + ["item"]].to_csv(
        OUT_DIR / "maoss_mapeamento.csv", index=False, encoding="utf-8-sig", sep=";"
    )
    # Também guarda resumo para verificar a amostra antes do upload.
    summary = (
        data.groupby(["task", "question", "classe_real"], dropna=False)
        .size().rename("n").reset_index()
    )
    summary.to_csv(OUT_DIR / "maoss_resumo_amostra.csv", index=False, encoding="utf-8-sig", sep=";")
    print(f"Preparação concluída. Registros incluídos: {len(data)}")
    print(f"Arquivo para enviar ao MAoSS/CAP: {cap_path}")
    print(f"Mapeamento: {OUT_DIR / 'maoss_mapeamento.csv'}")
    print(f"Resumo da amostra: {OUT_DIR / 'maoss_resumo_amostra.csv'}")
    print("\nContagem por tarefa e classe:")
    print(pd.crosstab(data["task"], data["classe_real"]).to_string())


def metric_row(y_true, y_pred, scope_type, scope, class_label=None):
    if class_label is None:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1, 2], average="macro", zero_division=0
        )
        return {
            "nivel": scope_type, "grupo": scope, "classe": "macro",
            "precisao": p, "revocacao": r, "f1": f,
            "n": len(y_true),
        }
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[class_label], average=None, zero_division=0
    )
    return {
        "nivel": scope_type, "grupo": scope, "classe": int(class_label),
        "precisao": float(p[0]), "revocacao": float(r[0]), "f1": float(f[0]),
        "n": len(y_true), "suporte_real_classe": int(support[0]),
    }


def score_subset(df: pd.DataFrame, group_name: str, group_value: str):
    """Retorna predições OOF e métricas. Se não há amostra suficiente, registra motivo."""
    y = df["classe_real"].astype(int).to_numpy()
    X = df[["prediction"]].astype(float).to_numpy()
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() < N_SPLITS:
        return None, [{
            "nivel": group_name, "grupo": group_value, "status": "ignorado",
            "motivo": f"Necessário ter as duas classes e pelo menos {N_SPLITS} exemplos por classe",
            "n": len(df), "n_classe_1": int((y == 1).sum()), "n_classe_2": int((y == 2).sum()),
        }]
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
    ))
    try:
        pred = cross_val_predict(model, X, y, cv=cv, method="predict")
    except Exception as exc:
        return None, [{
            "nivel": group_name, "grupo": group_value, "status": "erro",
            "motivo": str(exc), "n": len(df),
        }]
    out = df.copy()
    out["predicao_logistica_oof"] = pred.astype(int)
    out["acertou"] = (out["classe_real"].astype(int) == out["predicao_logistica_oof"]).astype(int)
    rows = [metric_row(y, pred, group_name, group_value)]
    rows.extend(metric_row(y, pred, group_name, group_value, c) for c in (1, 2))
    for row in rows:
        row["status"] = "ok"
        row["acuracia"] = accuracy_score(y, pred)
    return out, rows


def avaliar(resultados_path: Path) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = read_csv_flexible(resultados_path)
    # MAoSS/CAP deve preservar o id e devolver a pontuação numérica prediction.
    if not any(str(c).strip().casefold() == "prediction" for c in results.columns):
        raise ValueError(
            "O CSV retornado pelo MAoSS precisa conter a coluna 'prediction'. "
            f"Colunas encontradas: {list(results.columns)}"
        )
    # Aceita retorno com id (se a plataforma o preservou) ou sem id.
    # Sem id, associa prediction à ordem original das linhas, que deve ser mantida
    # pelo MAoSS. Não descartamos linhas antes da associação para evitar deslocamento.
    results.columns = [str(c).replace("\\ufeff", "").strip() for c in results.columns]
    pred_col = next((c for c in results.columns if c.casefold() == "prediction"), None)
    if pred_col is None:
        raise ValueError(f"Não encontrei a coluna prediction. Colunas: {list(results.columns)}")
    results[pred_col] = pd.to_numeric(results[pred_col], errors="coerce")
    mapping_path = OUT_DIR / "maoss_mapeamento.csv"
    mapping = read_csv_flexible(mapping_path)
    mapping["id"] = mapping["id"].astype(str).str.strip()
    mapping["classe_real"] = pd.to_numeric(mapping["classe_real"], errors="coerce")
    if "id" in results.columns:
        results["id"] = results["id"].astype(str).str.strip()
        valid_results = results.dropna(subset=[pred_col])
        merged = mapping.merge(
            valid_results[["id", pred_col]].rename(columns={pred_col: "prediction"}),
            on="id", how="inner", validate="one_to_one"
        )
    else:
        if len(results) != len(mapping):
            raise ValueError(
                f"O resultado do MAoSS tem {len(results)} linhas, mas o mapeamento tem {len(mapping)}. "
                "Como o MAoSS não devolveu a coluna id, é necessário que todas as linhas sejam devolvidas "
                "na mesma ordem; verifique o CSV de saída."
            )
        if results[pred_col].isna().any():
            raise ValueError(
                "Há prediction vazia/não numérica no resultado do MAoSS. Corrija o arquivo antes de avaliar."
            )
        merged = mapping.copy()
        merged["prediction"] = results[pred_col].to_numpy()
    merged = merged[merged["classe_real"].isin([1, 2])].copy()
    merged["classe_real"] = merged["classe_real"].astype(int)
    if merged.empty:
        raise ValueError("Nenhum registro foi associado. Verifique se o MAoSS preservou os IDs.")

    merged.to_csv(OUT_DIR / "maoss_resultados_com_classes.csv", index=False, encoding="utf-8-sig", sep=";")
    metric_rows, prediction_frames, skipped_rows = [], [], []

    # Resultado geral, por tarefa e por questão. O geral também é útil para o relatório.
    scopes = [("geral", "todos", merged)]
    scopes.extend(("tarefa", task, sub) for task, sub in merged.groupby("task"))
    scopes.extend(("questao", question, sub) for question, sub in merged.groupby("question"))
    for level, name, sub in scopes:
        pred_df, rows = score_subset(sub, level, str(name))
        metric_rows.extend(rows)
        if pred_df is not None:
            prediction_frames.append(pred_df)
        else:
            skipped_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics.to_csv(OUT_DIR / "maoss_metricas.csv", index=False, encoding="utf-8-sig", sep=";")
        metrics[metrics["nivel"].eq("tarefa")].to_csv(
            OUT_DIR / "maoss_metricas_por_tarefa.csv", index=False, encoding="utf-8-sig", sep=";"
        )
        metrics[metrics["nivel"].eq("questao")].to_csv(
            OUT_DIR / "maoss_metricas_por_questao.csv", index=False, encoding="utf-8-sig", sep=";"
        )
        metrics[metrics["classe"].astype(str).isin(["1", "2"])].to_csv(
            OUT_DIR / "maoss_metricas_por_classe.csv", index=False, encoding="utf-8-sig", sep=";"
        )
    if prediction_frames:
        # Predições OOF geradas em análises separadas (geral/tarefa/questão).
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            OUT_DIR / "maoss_predicoes_oof.csv", index=False, encoding="utf-8-sig", sep=";"
        )
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            OUT_DIR / "maoss_grupos_nao_avaliados.csv", index=False, encoding="utf-8-sig", sep=";"
        )

    print(f"Avaliação concluída. Registros com pontuação MAoSS associada: {len(merged)}")
    print(f"Resultados salvos em: {OUT_DIR}")
    task_metrics = metrics[(metrics["nivel"] == "tarefa") & (metrics["classe"].astype(str) == "macro")]
    if not task_metrics.empty:
        print("\nPrecisão, revocação e F1 macro por tarefa:")
        print(task_metrics[["grupo", "precisao", "revocacao", "f1", "n"]].to_string(index=False))
    if skipped_rows:
        print(f"\nAviso: {len(skipped_rows)} grupo(s) não foram avaliados; veja maoss_grupos_nao_avaliados.csv.")


def main():
    parser = argparse.ArgumentParser(description="Preparação MAoSS e avaliação por regressão logística.")
    parser.add_argument("acao", choices=["preparar", "avaliar", "tudo"])
    parser.add_argument(
        "--resultados", type=Path,
        help="CSV de saída do MAoSS/CAP contendo id e prediction. Necessário em avaliar/tudo."
    )
    args = parser.parse_args()
    if args.acao in ("preparar", "tudo"):
        preparar()
    if args.acao in ("avaliar", "tudo"):
        if args.resultados is None:
            parser.error("Informe --resultados com o CSV pontuado pelo MAoSS.")
        avaliar(args.resultados)


if __name__ == "__main__":
    main()
