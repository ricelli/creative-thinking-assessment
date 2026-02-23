import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold

# =============================
# CONFIGURAÇÕES
# =============================

PASTA_TREINO = "treino"
PASTA_TESTE = "teste"
PASTA_RESULTADOS = "resultados"

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

metricas_gerais_lista = []

# =============================
# LOOP PRINCIPAL
# =============================

for arquivo in os.listdir(PASTA_TREINO):

    caminho_treino = os.path.join(PASTA_TREINO, arquivo)
    caminho_teste = os.path.join(PASTA_TESTE, arquivo)

    if not os.path.exists(caminho_teste):
        print(f"Arquivo de teste não encontrado para {arquivo}")
        continue

    print(f"\nProcessando: {arquivo}")

    df_train = pd.read_csv(caminho_treino)
    df_test = pd.read_csv(caminho_teste)

    y_train = df_train["label"]
    y_test = df_test["label"]

    X_train = df_train.drop(columns=["label"])
    X_test = df_test.drop(columns=["label"])

    # Mantém apenas colunas numéricas
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # =============================
    # PIPELINE
    # =============================

    modelo = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # =============================
    # VALIDAÇÃO CRUZADA
    # =============================

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    cv_results = cross_validate(
        modelo,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    # =============================
    # TREINO FINAL
    # =============================

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    # =============================
    # MÉTRICAS GERAIS
    # =============================

    acc = accuracy_score(y_test, y_pred)

    prec_macro = precision_score(y_test, y_pred, average="macro")
    rec_macro = recall_score(y_test, y_pred, average="macro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    prec_weighted = precision_score(y_test, y_pred, average="weighted")
    rec_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # =============================
    # MÉTRICAS POR CLASSE
    # =============================

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=sorted(y_test.unique())
    )

    # =============================
    # MATRIZ DE CONFUSÃO
    # =============================

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm)

    cm_df.to_csv(
        os.path.join(PASTA_RESULTADOS, f"matriz_confusao_{arquivo}.csv"),
        index=False
    )

    # =============================
    # SALVAR MÉTRICAS
    # =============================

    metricas_dict = {
        "arquivo": arquivo,

        # CV
        "cv_accuracy_mean": np.mean(cv_results["test_accuracy"]),
        "cv_accuracy_std": np.std(cv_results["test_accuracy"]),
        "cv_precision_macro_mean": np.mean(cv_results["test_precision_macro"]),
        "cv_recall_macro_mean": np.mean(cv_results["test_recall_macro"]),
        "cv_f1_macro_mean": np.mean(cv_results["test_f1_macro"]),

        # Teste geral
        "test_accuracy": acc,
        "test_precision_macro": prec_macro,
        "test_recall_macro": rec_macro,
        "test_f1_macro": f1_macro,
        "test_precision_weighted": prec_weighted,
        "test_recall_weighted": rec_weighted,
        "test_f1_weighted": f1_weighted
    }

    # Adiciona métricas por classe dinamicamente
    classes = sorted(y_test.unique())
    for i, classe in enumerate(classes):
        metricas_dict[f"precision_classe_{classe}"] = precision[i]
        metricas_dict[f"recall_classe_{classe}"] = recall[i]
        metricas_dict[f"f1_classe_{classe}"] = f1[i]
        metricas_dict[f"support_classe_{classe}"] = support[i]

    metricas_df = pd.DataFrame([metricas_dict])

    metricas_df.to_csv(
        os.path.join(PASTA_RESULTADOS, f"metricas_{arquivo}.csv"),
        index=False
    )

    metricas_gerais_lista.append(metricas_dict)

# =============================
# CSV CONSOLIDADO
# =============================

df_final = pd.DataFrame(metricas_gerais_lista)
df_final.to_csv(
    os.path.join(PASTA_RESULTADOS, "metricas_consolidadas.csv"),
    index=False
)

print("\nProcessamento finalizado!")