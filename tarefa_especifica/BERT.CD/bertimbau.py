# ============================================================
# BERTimbau - Classificação em múltiplos conjuntos
# ============================================================
#
# Esperado:
#
# TREINO:
#   Cross-target-TREINO.csv
#
# TESTE:
#   Cross-target-teste.csv
#
# Colunas:
#   Texto  -> response
#   Classe -> code0_3
#
# O código:
#   - encontra automaticamente os pares treino/teste
#   - treina um BERTimbau para cada questão
#   - calcula accuracy, precision, recall e f1
#   - salva predições e métricas
#
# IMPORTANTE:
#   Labels 1 e 2 são convertidos internamente:
#
#       1 -> 0
#       2 -> 1
#
#   MAS as saídas finais são salvas novamente como:
#
#       1 e 2
#
# ============================================================

# ============================================================
# BERTimbau - Classificação em múltiplos conjuntos
# ============================================================

import os
import re
import glob
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_DADOS = "./dados"
PASTA_RESULTADOS = "./resultados"

COLUNA_TEXTO = "response"
COLUNA_LABEL = "code0_3"

MODELO = "neuralmind/bert-base-portuguese-cased"

MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# ============================================================
# GPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE:", device)
print("=" * 70)

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODELO)

# ============================================================
# TOKENIZAÇÃO
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples[COLUNA_TEXTO],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# ============================================================
# MÉTRICAS PARA O TRAINER
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    # macro
    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1": f1_macro
    }

# ============================================================
# LOCALIZAR ARQUIVOS
# ============================================================

train_files = glob.glob(
    os.path.join(PASTA_DADOS, "*treino*.csv")
)

test_files = glob.glob(
    os.path.join(PASTA_DADOS, "*teste*.csv")
)

print(f"Treinos encontrados: {len(train_files)}")
print(f"Testes encontrados : {len(test_files)}")

# ============================================================
# ORGANIZAR PARES
# ============================================================

pares = {}

# ---------- treino ----------
for file in train_files:

    nome = os.path.basename(file)

    match = re.search(r"(T\d+Q\d+)", nome)

    if match:

        questao = match.group(1)

        if questao not in pares:
            pares[questao] = {}

        pares[questao]["train"] = file

# ---------- teste ----------
for file in test_files:

    nome = os.path.basename(file)

    match = re.search(r"(T\d+Q\d+)", nome)

    if match:

        questao = match.group(1)

        if questao not in pares:
            pares[questao] = {}

        pares[questao]["test"] = file

# ============================================================
# MÉTRICAS FINAIS
# ============================================================

metricas_finais = []

# ============================================================
# LOOP PRINCIPAL
# ============================================================

for questao, arquivos in pares.items():

    print("\n" + "=" * 70)
    print(f"PROCESSANDO QUESTÃO: {questao}")
    print("=" * 70)

    if "train" not in arquivos or "test" not in arquivos:

        print("Par treino/teste incompleto. Pulando.")
        continue

    train_path = arquivos["train"]
    test_path = arquivos["test"]

    print("TREINO:", train_path)
    print("TESTE :", test_path)

    # ========================================================
    # LEITURA
    # ========================================================

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # remover NaN
    train_df = train_df.dropna(
        subset=[COLUNA_TEXTO, COLUNA_LABEL]
    )

    test_df = test_df.dropna(
        subset=[COLUNA_TEXTO, COLUNA_LABEL]
    )

    # ========================================================
    # CONVERTER LABELS PARA INT
    # ========================================================

    train_df[COLUNA_LABEL] = (
        train_df[COLUNA_LABEL].astype(int)
    )

    test_df[COLUNA_LABEL] = (
        test_df[COLUNA_LABEL].astype(int)
    )

    # ========================================================
    # MAPEAMENTO DE LABELS
    # ========================================================

    classes_originais = sorted(
        list(
            set(train_df[COLUNA_LABEL].unique()).union(
                set(test_df[COLUNA_LABEL].unique())
            )
        )
    )

    print("\nClasses originais:", classes_originais)

    label2id = {
        label: idx
        for idx, label in enumerate(classes_originais)
    }

    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    print("Mapeamento:", label2id)

    # aplicar mapeamento
    train_df[COLUNA_LABEL] = (
        train_df[COLUNA_LABEL].map(label2id)
    )

    test_df[COLUNA_LABEL] = (
        test_df[COLUNA_LABEL].map(label2id)
    )

    num_labels = len(classes_originais)

    print("Número de classes:", num_labels)

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = Dataset.from_pandas(
        train_df[[COLUNA_TEXTO, COLUNA_LABEL]]
    )

    test_dataset = Dataset.from_pandas(
        test_df[[COLUNA_TEXTO, COLUNA_LABEL]]
    )

    # renomear labels
    train_dataset = train_dataset.rename_column(
        COLUNA_LABEL,
        "labels"
    )

    test_dataset = test_dataset.rename_column(
        COLUNA_LABEL,
        "labels"
    )

    # tokenização
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )

    # formato torch
    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    # ========================================================
    # MODELO
    # ========================================================

    model = AutoModelForSequenceClassification.from_pretrained(
        MODELO,
        num_labels=num_labels
    )

    # ========================================================
    # ARGUMENTOS DE TREINO
    # ========================================================

    output_dir = os.path.join(
        PASTA_RESULTADOS,
        questao
    )

    training_args = TrainingArguments(
        output_dir=output_dir,

        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=LEARNING_RATE,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        num_train_epochs=EPOCHS,

        weight_decay=0.01,

        logging_steps=10,

        load_best_model_at_end=True,

        metric_for_best_model="f1",
        greater_is_better=True,

        fp16=torch.cuda.is_available(),

        report_to="none"
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # ========================================================
    # TREINAMENTO
    # ========================================================

    trainer.train()

    # ========================================================
    # PREDIÇÕES
    # ========================================================

    resultados = trainer.predict(test_dataset)

    preds = np.argmax(
        resultados.predictions,
        axis=-1
    )

    labels = resultados.label_ids

    # ========================================================
    # MÉTRICAS
    # ========================================================

    acc = accuracy_score(labels, preds)

    # métricas por classe
    precision_classes, recall_classes, f1_classes, support_classes = (
        precision_recall_fscore_support(
            labels,
            preds,
            labels=list(range(num_labels)),
            average=None,
            zero_division=0
        )
    )

    # métricas macro
    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0
        )
    )

    # métricas weighted
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="weighted",
            zero_division=0
        )
    )

    print("\nRESULTADOS")
    print("-" * 40)

    print(f"Accuracy : {acc:.4f}")

    print("\n--- MÉTRICAS MACRO ---")

    print(f"Precision Macro : {precision_macro:.4f}")
    print(f"Recall Macro    : {recall_macro:.4f}")
    print(f"F1 Macro        : {f1_macro:.4f}")

    print("\n--- MÉTRICAS WEIGHTED ---")

    print(f"Precision Weighted : {precision_weighted:.4f}")
    print(f"Recall Weighted    : {recall_weighted:.4f}")
    print(f"F1 Weighted        : {f1_weighted:.4f}")

    print("\n--- MÉTRICAS POR CLASSE ---")

    for i in range(num_labels):

        classe_original = id2label[i]

        print(f"\nClasse {classe_original}")

        print(f"Precision : {precision_classes[i]:.4f}")
        print(f"Recall    : {recall_classes[i]:.4f}")
        print(f"F1-score  : {f1_classes[i]:.4f}")
        print(f"Support   : {support_classes[i]}")

    # ========================================================
    # RELATÓRIO
    # ========================================================

    print("\nCLASSIFICATION REPORT")
    print("-" * 40)

    report = classification_report(
        labels,
        preds,
        labels=list(range(num_labels)),
        target_names=[
            str(id2label[i])
            for i in range(num_labels)
        ],
        zero_division=0
    )

    print(report)

    # ========================================================
    # CONVERTER PREVISÕES PARA LABELS ORIGINAIS
    # ========================================================

    preds_originais = [
        id2label[p]
        for p in preds
    ]

    labels_originais = [
        id2label[l]
        for l in labels
    ]

    # ========================================================
    # SALVAR PREDIÇÕES
    # ========================================================

    output_pred = test_df.copy()

    output_pred["classe_real"] = labels_originais
    output_pred["predicao"] = preds_originais

    pred_path = os.path.join(
        PASTA_RESULTADOS,
        f"{questao}_predicoes.csv"
    )

    output_pred.to_csv(
        pred_path,
        index=False
    )

    print("\nPredições salvas em:")
    print(pred_path)

    # ========================================================
    # SALVAR MÉTRICAS
    # ========================================================

    linha_metricas = {
        "questao": questao,

        "accuracy": acc,

        # macro
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,

        # weighted
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted
    }

    # métricas por classe
    for i in range(num_labels):

        classe_original = id2label[i]

        linha_metricas[
            f"precision_classe_{classe_original}"
        ] = precision_classes[i]

        linha_metricas[
            f"recall_classe_{classe_original}"
        ] = recall_classes[i]

        linha_metricas[
            f"f1_classe_{classe_original}"
        ] = f1_classes[i]

        linha_metricas[
            f"support_classe_{classe_original}"
        ] = support_classes[i]

    metricas_finais.append(linha_metricas)

# ============================================================
# SALVAR MÉTRICAS GERAIS
# ============================================================

metricas_df = pd.DataFrame(metricas_finais)

metricas_path = os.path.join(
    PASTA_RESULTADOS,
    "metricas_finais.csv"
)

metricas_df.to_csv(
    metricas_path,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PROCESSAMENTO FINALIZADO")
print("=" * 70)

print("\nMÉTRICAS FINAIS")
print(metricas_df)

print("\nArquivo salvo:")
print(metricas_path)
