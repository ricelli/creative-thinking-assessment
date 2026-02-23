import os
import pandas as pd

PASTAS = ["treino", "teste"]
COLUNAS_NUMERICAS = ["originality", "elaboration"]

def verificar_pasta(pasta):
    print(f"\n========== Verificando pasta: {pasta} ==========")

    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".csv"):
            caminho = os.path.join(pasta, arquivo)
            print(f"\n--- Arquivo: {arquivo} ---")

            df = pd.read_csv(caminho)

            for coluna in COLUNAS_NUMERICAS:
                if coluna not in df.columns:
                    print(f"\nColuna {coluna} NÃO encontrada.")
                    continue

                # tenta converter
                convertido = pd.to_numeric(df[coluna], errors="coerce")

                # pega qualquer linha que tenha NaN (antes ou depois)
                mask_problem = convertido.isna()

                if mask_problem.any():
                    print(f"\nProblemas encontrados na coluna: {coluna}")

                    for idx, linha in df[mask_problem].iterrows():
                        print(f"\nLinha número: {idx}")
                        print(linha.to_string())
                        print("-" * 60)
                else:
                    print(f"\nColuna {coluna}: OK (sem NaN)")

for pasta in PASTAS:
    verificar_pasta(pasta)