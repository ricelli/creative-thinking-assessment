import pandas as pd
import glob
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

PASTA = "dados"
SAIDA = "relatorio_final.csv"

COLUNAS_X = ["originality","confidence"]
COLUNA_CLASSE = "classe"

resultados = []

arquivos = glob.glob(os.path.join(PASTA, "*.csv"))

for arquivo in arquivos:

    try:
        df = pd.read_csv(arquivo)
    except:
        print("Erro lendo:", arquivo)
        continue

    nome_questao = os.path.basename(arquivo)

    # garantir colunas
    colunas_necessarias = set(COLUNAS_X + [COLUNA_CLASSE,"source"])

    if not colunas_necessarias.issubset(df.columns):
        print("Ignorado (colunas faltando):", arquivo)
        continue

    # limpar dados
    for col in COLUNAS_X:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[COLUNAS_X] = df[COLUNAS_X].fillna(0)

    df[COLUNA_CLASSE] = pd.to_numeric(df[COLUNA_CLASSE], errors="coerce")

    df = df[df[COLUNA_CLASSE].isin([1,2])]

    train = df[df["source"] == "train"]
    test  = df[df["source"] == "test"]

    if len(train) == 0 or len(test) == 0:
        print("Sem train/test:", arquivo)
        continue

    X_train = train[COLUNAS_X]
    y_train = train[COLUNA_CLASSE]

    X_test = test[COLUNAS_X]
    y_test = test[COLUNA_CLASSE]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    resultados.append({

        "arquivo": nome_questao,

        "train_size": len(train),
        "test_size": len(test),

        "accuracy": accuracy_score(y_test, y_pred),

        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),

        "precision_classe_1": report["1"]["precision"] if "1" in report else None,
        "recall_classe_1": report["1"]["recall"] if "1" in report else None,
        "f1_classe_1": report["1"]["f1-score"] if "1" in report else None,

        "precision_classe_2": report["2"]["precision"] if "2" in report else None,
        "recall_classe_2": report["2"]["recall"] if "2" in report else None,
        "f1_classe_2": report["2"]["f1-score"] if "2" in report else None,

        "coef_originality": model.coef_[0][0],
        "coef_confidence": model.coef_[0][1],
        "intercept": model.intercept_[0]

    })

saida = pd.DataFrame(resultados)
saida.to_csv(SAIDA, index=False)

print()
print("Arquivos analisados:", len(saida))
print("Relatório:", SAIDA)