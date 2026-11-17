#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preparação e avaliação MAoSS/CAP para amostras OECS em português/espanhol.

Modos:
  preparar  traduz respostas para inglês e gera CSVs para upload manual no CAP.
  avaliar   lê os CSVs baixados do CAP e calcula métricas com regressão logística.
  tudo      prepara e, se já houver arquivos pontuados, avalia.

Dependências:
  python -m pip install deep-translator scikit-learn

A tradução usa GoogleTranslator (serviço Google Tradutor acessado pela biblioteca
 deep-translator; não é a API oficial Google Cloud). O uso depende de internet e
 pode estar sujeito a limites/bloqueios do serviço.
"""

import argparse
import csv
import hashlib
import math
import random
import re
import sys
import time
from pathlib import Path

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

QUESTION_TO_TASK = {
    "T240Q01": "GCI", "T300Q01": "GCI", "T350Q01": "GCI",
    "T360Q01": "GCI", "T370Q01": "GCI", "T570Q02": "GCI",
    "T240Q02": "GDI", "T300Q02": "GDI", "T350Q02": "GDI",
    "T570Q01": "GDI",
    "T350Q03": "EII", "T570Q03": "EII",
}
COMBINACOES = [
    "america_latina",
    "america_sul",
    "espanha",
    "peninsula_iberica",
    "todos",
]
BASE_AMOSTRAS = Path.home() / "Documentos/Doutorado/resultados_experimentos_qwen_oecs"
BASE_ENUNCIADOS = (
    Path.home() / "Documentos/Doutorado/Arquivos OECS-20250827T193412Z-1-001/"
    "Arquivos OECS/2022MS-responsesXML/2022MS/ingles/amostra_qwen/maoss_experimento"
)
ARQUIVO_ENUNCIADOS = BASE_ENUNCIADOS / "enunciados_pisa_ingles.csv"
PASTA_SAIDA = Path.home() / "Documentos/Doutorado/maoss_google_portugues_espanhol"
SEED_CV = 20261015


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def clean(value):
    return "" if value is None else str(value).strip()


def stable_id(combo, row, question, index):
    # Identificador curto e determinístico; não depende só do student, que pode repetir.
    source = "|".join([
        combo, question, clean(row.get("arquivo_xml")), clean(row.get("student")),
        clean(row.get("pais")), str(index), clean(row.get("resposta"))
    ])
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:20]


def load_prompts():
    if not ARQUIVO_ENUNCIADOS.exists():
        raise FileNotFoundError(
            f"Arquivo de enunciados em inglês não encontrado:\n{ARQUIVO_ENUNCIADOS}\n"
            "Confira o caminho no início do script. O CSV deve ter colunas 'question' e 'item'."
        )
    rows = read_csv(ARQUIVO_ENUNCIADOS)
    if not rows or not {"question", "item"}.issubset(rows[0].keys()):
        raise ValueError(f"O arquivo {ARQUIVO_ENUNCIADOS} precisa ter colunas question,item.")
    prompts = {clean(r.get("question")): clean(r.get("item")) for r in rows}
    missing = [q for q in QUESTION_TO_TASK if not prompts.get(q)]
    if missing:
        raise ValueError("Faltam enunciados em inglês para: " + ", ".join(missing))
    return prompts


def load_combo_samples(combo):
    folder = BASE_AMOSTRAS / combo / "amostras"
    if not folder.exists():
        raise FileNotFoundError(f"Pasta de amostras não encontrada: {folder}")
    files = sorted(folder.glob("T*_amostra.csv"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo T*_amostra.csv em {folder}")
    records = []
    for path in files:
        question_from_name = path.name.split("_")[0]
        if question_from_name not in QUESTION_TO_TASK:
            continue
        for i, row in enumerate(read_csv(path), start=1):
            question = clean(row.get("questao")) or question_from_name
            if question not in QUESTION_TO_TASK:
                continue
            response = clean(row.get("resposta"))
            try:
                label = int(float(clean(row.get("classe_real"))))
            except (TypeError, ValueError):
                continue
            if not response or label not in (1, 2):
                continue
            rec = {
                "id": stable_id(combo, row, question, i),
                "combinacao": combo,
                "question": question,
                "task": QUESTION_TO_TASK[question],
                "classe_real": label,
                "response_original": response,
                "response_english": "",
                "pais": clean(row.get("pais")),
                "pais_nome": clean(row.get("pais_nome")),
                "idioma": clean(row.get("idioma")),
                "student": clean(row.get("student")),
                "arquivo_xml": clean(row.get("arquivo_xml")),
                "source_file": path.name,
            }
            records.append(rec)
    if not records:
        raise ValueError(f"Nenhuma resposta válida (classe_real 1/2) em {folder}")
    # IDs garantem unicidade e mantêm as respostas da amostra de cada combinação.
    return sorted(records, key=lambda r: (r["question"], r["id"]))


def translate_records(records, cache_path, delay=0.35):
    if GoogleTranslator is None:
        raise RuntimeError(
            "Biblioteca deep-translator não instalada. Execute:\n"
            "python -m pip install deep-translator"
        )
    cache = {}
    if cache_path.exists():
        for row in read_csv(cache_path):
            original = clean(row.get("response_original"))
            translated = clean(row.get("response_english"))
            if original and translated:
                cache[original] = translated
    translator = GoogleTranslator(source="auto", target="en")
    total = len(records)
    for i, rec in enumerate(records, start=1):
        original = rec["response_original"]
        if original in cache:
            rec["response_english"] = cache[original]
            continue
        # Tentativas limitadas: não marca tradução falha como tradução válida.
        last_error = None
        for attempt in range(3):
            try:
                translated = translator.translate(original)
                if not translated or not clean(translated):
                    raise ValueError("O Google Tradutor retornou texto vazio.")
                rec["response_english"] = clean(translated)
                cache[original] = rec["response_english"]
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        if not rec["response_english"]:
            print(f"\n[ERRO TRADUÇÃO] {i}/{total}, {rec['question']}: {last_error}")
        elif i % 10 == 0 or i == total:
            print(f"Traduções: {i}/{total}")
        # Salva checkpoint após cada item, inclusive as traduções reaproveitadas.
        write_csv(cache_path, [
            {"response_original": k, "response_english": v}
            for k, v in cache.items()
        ], ["response_original", "response_english"])
        time.sleep(delay)
    return records


def prepare_combo(combo, prompts):
    out_dir = PASTA_SAIDA / combo
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "respostas_traduzidas.csv"
    cache_path = out_dir / "cache_traducoes.csv"
    upload_path = out_dir / "maoss_entrada_cap.csv"

    records = load_combo_samples(combo)
    print(f"\n[{combo}] Respostas válidas: {len(records)}")
    records = translate_records(records, cache_path)
    write_csv(full_path, records, [
        "id", "combinacao", "question", "task", "classe_real",
        "response_original", "response_english", "pais", "pais_nome", "idioma",
        "student", "arquivo_xml", "source_file"
    ])

    upload_rows = []
    failed = 0
    for rec in records:
        if not rec["response_english"]:
            failed += 1
            continue
        upload_rows.append({
            "id": rec["id"],
            "question": rec["question"],
            "task": rec["task"],
            "classe_real": rec["classe_real"],
            "item": prompts[rec["question"]],
            "response": rec["response_english"],
            "source_file": rec["source_file"],
        })
    write_csv(upload_path, upload_rows, [
        "id", "question", "task", "classe_real", "item", "response", "source_file"
    ])
    print(f"[{combo}] Traduções válidas: {len(upload_rows)}; falhas: {failed}")
    print(f"[{combo}] Arquivo para upload CAP: {upload_path}")


def load_scored(combo):
    out_dir = PASTA_SAIDA / combo
    scored_path = out_dir / "maoss_resultados_cap.csv"
    full_path = out_dir / "respostas_traduzidas.csv"
    if not scored_path.exists():
        raise FileNotFoundError(
            f"Não encontrei {scored_path}. Envie maoss_entrada_cap.csv ao CAP/MAoSS, "
            "baixe o resultado pontuado e salve-o com esse nome."
        )
    base = read_csv(full_path)
    base_by_id = {clean(r.get("id")): r for r in base}
    scored = read_csv(scored_path)
    if not scored or "prediction" not in scored[0]:
        raise ValueError(f"O arquivo pontuado {scored_path} precisa conter a coluna 'prediction'.")
    merged, seen = [], set()
    for row in scored:
        rid = clean(row.get("id"))
        if not rid or rid not in base_by_id or rid in seen:
            continue
        try:
            score = float(clean(row.get("prediction")).replace(",", "."))
        except (ValueError, TypeError):
            continue
        if not math.isfinite(score):
            continue
        merged.append({**base_by_id[rid], "maoss_prediction": score})
        seen.add(rid)
    if not merged:
        raise ValueError(
            "Não consegui associar escores via coluna id. Verifique se o CAP preservou a coluna id "
            "e se prediction contém números. Não associei automaticamente por texto para evitar erros."
        )
    write_csv(out_dir / "respostas_com_escores.csv", merged, [
        "id", "combinacao", "question", "task", "classe_real", "response_original",
        "response_english", "maoss_prediction", "pais", "pais_nome", "idioma",
        "student", "arquivo_xml", "source_file"
    ])
    return out_dir, merged


def evaluate_combo(combo):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Instale scikit-learn: python -m pip install scikit-learn") from exc

    out_dir, records = load_scored(combo)
    groups = []
    for q in sorted(QUESTION_TO_TASK):
        group = [r for r in records if r["question"] == q]
        if group:
            groups.append((f"QUESTAO_{q}", group))
    for task in ("GCI", "GDI", "EII"):
        group = [r for r in records if r["task"] == task]
        if group:
            groups.append((f"TAREFA_{task}", group))
    groups.append(("AGREGADO_12_QUESTOES", records))

    metrics, oof = [], []
    for group_name, group in groups:
        y = [int(r["classe_real"]) for r in group]
        x = [[float(r["maoss_prediction"])] for r in group]
        n1, n2 = y.count(1), y.count(2)
        if min(n1, n2) < 5:
            print(f"[{combo}] [PULADO] {group_name}: classe 1={n1}, classe 2={n2}; exige >=5 por classe.")
            continue
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_CV)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED_CV)
        )
        pred = cross_val_predict(model, x, y, cv=cv, method="predict")
        p, r, f, support = precision_recall_fscore_support(y, pred, labels=[1, 2], zero_division=0)
        metrics.append({
            "combinacao": combo, "grupo": group_name, "n": len(y),
            "n_classe_1": n1, "n_classe_2": n2,
            "acuracia": round(accuracy_score(y, pred), 6),
            "precisao_classe_1": round(p[0], 6), "revocacao_classe_1": round(r[0], 6),
            "f1_classe_1": round(f[0], 6), "precisao_classe_2": round(p[1], 6),
            "revocacao_classe_2": round(r[1], 6), "f1_classe_2": round(f[1], 6),
            "macro_precision": round(float(p.mean()), 6), "macro_recall": round(float(r.mean()), 6),
            "macro_f1": round(f1_score(y, pred, labels=[1, 2], average="macro", zero_division=0), 6),
        })
        for rec, prediction in zip(group, pred):
            oof.append({
                "id": rec["id"], "combinacao": combo, "question": rec["question"],
                "task": rec["task"], "classe_real": rec["classe_real"],
                "maoss_prediction": rec["maoss_prediction"],
                "classe_predita_oof": int(prediction), "grupo_avaliacao": group_name,
            })
        print(f"[{combo}] {group_name}: n={len(y)}, macro-F1={metrics[-1]['macro_f1']:.4f}")

    if metrics:
        write_csv(out_dir / "maoss_metricas.csv", metrics, list(metrics[0].keys()))
        write_csv(out_dir / "maoss_predicoes_oof.csv", oof, list(oof[0].keys()))
        print(f"[{combo}] Métricas: {out_dir / 'maoss_metricas.csv'}")
    else:
        print(f"[{combo}] Nenhum grupo avaliável.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Experimento MAoSS com tradução Google para português/espanhol.")
    parser.add_argument("modo", choices=["preparar", "avaliar", "tudo"])
    parser.add_argument("--combinacao", choices=COMBINACOES, help="Por padrão, processa as seis combinações.")
    args = parser.parse_args()
    combos = [args.combinacao] if args.combinacao else COMBINACOES
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    if args.modo in ("preparar", "tudo"):
        prompts = load_prompts()
        for combo in combos:
            try:
                prepare_combo(combo, prompts)
            except Exception as exc:
                print(f"\n[ERRO PREPARAÇÃO {combo}] {exc}", file=sys.stderr)
                if args.combinacao:
                    raise

    if args.modo in ("avaliar", "tudo"):
        all_metrics = []
        for combo in combos:
            try:
                all_metrics.extend(evaluate_combo(combo))
            except Exception as exc:
                print(f"\n[AVISO AVALIAÇÃO {combo}] {exc}", file=sys.stderr)
                if args.combinacao:
                    raise
        if all_metrics:
            fields = list(all_metrics[0].keys())
            write_csv(PASTA_SAIDA / "metricas_todas_combinacoes.csv", all_metrics, fields)
            print(f"\nMétricas consolidadas: {PASTA_SAIDA / 'metricas_todas_combinacoes.csv'}")

if __name__ == "__main__":
    main()
