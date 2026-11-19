import os
import re
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PASTA = "/home/ricelli/testes-quen/Creat-Hyb"

ARQUIVO_DADOS = os.path.join(
    PASTA,
    "dados_regressao_logistica.csv"
)

ARQUIVO_PREDICOES = os.path.join(
    PASTA,
    "predicoes_regressao_logistica.csv"
)

ARQUIVO_METRICAS = os.path.join(
    PASTA,
    "metricas_regressao_logistica.csv"
)

ARQUIVO_COEFICIENTES = os.path.join(
    PASTA,
    "coeficientes_regressao_logistica.csv"
)


# ============================================================
# FUNÇÃO: NORMALIZAR RESPONSE
# ============================================================

def normalizar_response(texto):

    if pd.isna(texto):
        return ""

    return str(texto).strip()


# ============================================================
# FUNÇÃO: EXTRAIR OS 8 ESCORES
# ============================================================

def extrair_escores(texto):

    if pd.isna(texto):
        return None

    texto = str(texto)

    escores = {}

    padrao = re.compile(
        r"\|\s*Ideia\s*(\d+)\s*\|\s*(-?\d+(?:[.,]\d+)?)\s*\|",
        re.IGNORECASE
    )

    encontrados = padrao.findall(texto)

    for ideia, escore in encontrados:

        ideia = int(ideia)

        if 1 <= ideia <= 8:

            escore = escore.replace(",", ".")

            escores[ideia] = float(escore)

    if not all(
        i in escores
        for i in range(1, 9)
    ):
        return None

    return [
        escores[1],
        escores[2],
        escores[3],
        escores[4],
        escores[5],
        escores[6],
        escores[7],
        escores[8]
    ]


# ============================================================
# FUNÇÃO: LER CSV
# ============================================================

def carregar_csv(caminho):

    return pd.read_csv(
        caminho,
        dtype=str,
        keep_default_na=False
    )


# ============================================================
# FUNÇÃO: ENCONTRAR GDI / GCI / EII
# ============================================================

def encontrar_arquivo_original(codigo_questao):

    candidatos = [

        os.path.join(
            PASTA,
            f"GDI-{codigo_questao}.csv"
        ),

        os.path.join(
            PASTA,
            f"GCI-{codigo_questao}.csv"
        ),

        os.path.join(
            PASTA,
            f"EII-{codigo_questao}.csv"
        )

    ]

    encontrados = [

        caminho

        for caminho in candidatos

        if os.path.exists(caminho)

    ]

    if len(encontrados) == 0:
        return None

    if len(encontrados) > 1:
        return encontrados

    return encontrados[0]


# ============================================================
# LOCALIZAR RESULTADOS
# ============================================================

arquivos_resultados = sorted(

    glob.glob(
        os.path.join(
            PASTA,
            "*_resultados.csv"
        )
    )

)


print()
print("=" * 80)
print("REGRESSÃO LOGÍSTICA - AVALIAÇÃO DAS IDEIAS")
print("=" * 80)
print()

print(
    f"Arquivos de resultados encontrados: "
    f"{len(arquivos_resultados)}"
)

print()


# ============================================================
# PROCESSAR ARQUIVOS
# ============================================================

todos_dados = []

total_arquivos = 0
total_respostas = 0
total_correspondencias = 0
total_sem_correspondencia = 0
total_erros_escores = 0
total_multiplos_originais = 0


for arquivo_resultados in arquivos_resultados:

    nome_resultados = os.path.basename(
        arquivo_resultados
    )

    codigo_questao = nome_resultados.replace(
        "_resultados.csv",
        ""
    )

    print("=" * 80)
    print(f"QUESTÃO: {codigo_questao}")
    print("=" * 80)

    print(
        f"Resultados: {nome_resultados}"
    )

    # --------------------------------------------------------
    # Procurar GDI / GCI / EII
    # --------------------------------------------------------

    arquivo_original = encontrar_arquivo_original(
        codigo_questao
    )

    if arquivo_original is None:

        print(
            "AVISO: nenhum GDI/GCI/EII encontrado."
        )

        print()

        continue

    if isinstance(
        arquivo_original,
        list
    ):

        total_multiplos_originais += 1

        print(
            "ATENÇÃO: encontrados múltiplos "
            "arquivos originais:"
        )

        for caminho in arquivo_original:

            print(
                f"  - {os.path.basename(caminho)}"
            )

        print(
            "Questão ignorada."
        )

        print()

        continue

    print(
        f"Original:   "
        f"{os.path.basename(arquivo_original)}"
    )

    print()

    # --------------------------------------------------------
    # Ler arquivos
    # --------------------------------------------------------

    try:

        df_resultados = carregar_csv(
            arquivo_resultados
        )

        df_original = carregar_csv(
            arquivo_original
        )

    except Exception as erro:

        print(
            f"ERRO ao ler arquivos: {erro}"
        )

        print()

        continue

    total_arquivos += 1

    # --------------------------------------------------------
    # Verificar colunas
    # --------------------------------------------------------

    necessarias_resultados = {
        "response",
        "resposta_qwen"
    }

    necessarias_original = {
        "response",
        "code0_3"
    }

    faltantes_resultados = (
        necessarias_resultados
        - set(df_resultados.columns)
    )

    faltantes_original = (
        necessarias_original
        - set(df_original.columns)
    )

    if faltantes_resultados:

        print(
            "ERRO: colunas ausentes no arquivo "
            f"{nome_resultados}:"
        )

        print(
            faltantes_resultados
        )

        continue

    if faltantes_original:

        print(
            "ERRO: colunas ausentes no arquivo "
            f"{os.path.basename(arquivo_original)}:"
        )

        print(
            faltantes_original
        )

        continue

    # --------------------------------------------------------
    # Normalizar response
    # --------------------------------------------------------

    df_resultados[
        "_response_match"
    ] = (

        df_resultados[
            "response"
        ]

        .apply(
            normalizar_response
        )

    )

    df_original[
        "_response_match"
    ] = (

        df_original[
            "response"
        ]

        .apply(
            normalizar_response
        )

    )

    total_respostas += len(
        df_resultados
    )

    # --------------------------------------------------------
    # Mapa response -> code0_3
    # --------------------------------------------------------

    mapa_code = {}

    for _, linha in df_original.iterrows():

        response = linha[
            "_response_match"
        ]

        code = str(
            linha["code0_3"]
        ).strip()

        if response == "":
            continue

        if code not in {"1", "2"}:
            continue

        if response not in mapa_code:

            mapa_code[
                response
            ] = int(code)

    # --------------------------------------------------------
    # Processar respostas
    # --------------------------------------------------------

    correspondencias = 0
    sem_correspondencia = 0
    erros_escores = 0

    for _, linha in df_resultados.iterrows():

        response = linha[
            "_response_match"
        ]

        # ----------------------------------------------------
        # Correspondência
        # ----------------------------------------------------

        if response not in mapa_code:

            total_sem_correspondencia += 1
            sem_correspondencia += 1

            continue

        code0_3 = mapa_code[
            response
        ]

        # ----------------------------------------------------
        # Escores
        # ----------------------------------------------------

        escores = extrair_escores(
            linha["resposta_qwen"]
        )

        if escores is None:

            total_erros_escores += 1
            erros_escores += 1

            print(
                "AVISO: não foi possível extrair "
                f"os 8 escores - índice "
                f"{linha.get('indice', '')}"
            )

            continue

        # ----------------------------------------------------
        # Registro
        # ----------------------------------------------------

        registro = {

            "questao":
                codigo_questao,

            "indice":
                linha.get(
                    "indice",
                    ""
                ),

            "response":
                linha["response"],

            "ideia_1":
                escores[0],

            "ideia_2":
                escores[1],

            "ideia_3":
                escores[2],

            "ideia_4":
                escores[3],

            "ideia_5":
                escores[4],

            "ideia_6":
                escores[5],

            "ideia_7":
                escores[6],

            "ideia_8":
                escores[7],

            "code0_3":
                code0_3
        }

        todos_dados.append(
            registro
        )

        total_correspondencias += 1
        correspondencias += 1

    print(
        f"Respostas:             "
        f"{len(df_resultados)}"
    )

    print(
        f"Correspondências:      "
        f"{correspondencias}"
    )

    print(
        f"Sem correspondência:   "
        f"{sem_correspondencia}"
    )

    print(
        f"Erro nos escores:      "
        f"{erros_escores}"
    )

    print()


# ============================================================
# VERIFICAR DADOS
# ============================================================

print("=" * 80)
print("RESUMO")
print("=" * 80)

print(
    f"Arquivos processados:       "
    f"{total_arquivos}"
)

print(
    f"Respostas encontradas:      "
    f"{total_respostas}"
)

print(
    f"Correspondências:           "
    f"{total_correspondencias}"
)

print(
    f"Sem correspondência:        "
    f"{total_sem_correspondencia}"
)

print(
    f"Erros nos escores:          "
    f"{total_erros_escores}"
)

print(
    f"Múltiplos GDI/GCI/EII:      "
    f"{total_multiplos_originais}"
)

print()


if len(todos_dados) == 0:

    print(
        "ERRO: nenhum dado válido."
    )

    exit()


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    todos_dados
)


# ============================================================
# SALVAR DADOS
# ============================================================

df.to_csv(

    ARQUIVO_DADOS,

    index=False,

    encoding="utf-8-sig"

)

print(
    f"Dados salvos em:\n{ARQUIVO_DADOS}"
)

print()


# ============================================================
# VARIÁVEIS
# ============================================================

features = [

    "ideia_1",
    "ideia_2",
    "ideia_3",
    "ideia_4",
    "ideia_5",
    "ideia_6",
    "ideia_7",
    "ideia_8"

]


X = df[
    features
]

y = df[
    "code0_3"
]


# ============================================================
# TREINO / TESTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print("=" * 80)
print("TREINO / TESTE")
print("=" * 80)

print(
    f"Treinamento: {len(X_train)}"
)

print(
    f"Teste:       {len(X_test)}"
)

print()


# ============================================================
# MODELO
# ============================================================

modelo = LogisticRegression(

    max_iter=1000,

    random_state=42

)


modelo.fit(

    X_train,

    y_train

)


# ============================================================
# PREDIÇÕES
# ============================================================

y_pred = modelo.predict(
    X_test
)

y_prob = modelo.predict_proba(
    X_test
)


# ============================================================
# DATAFRAME DE TESTE
# ============================================================

df_test = df.loc[
    X_test.index
].copy()


df_test[
    "classe_prevista"
] = y_pred


classes_modelo = modelo.classes_


for i, classe in enumerate(
    classes_modelo
):

    df_test[
        f"probabilidade_classe_{classe}"
    ] = y_prob[:, i]


df_test[
    "acertou"
] = (

    df_test[
        "code0_3"
    ]

    ==

    df_test[
        "classe_prevista"
    ]

)


# ============================================================
# SALVAR PREDIÇÕES
# ============================================================

df_test.to_csv(

    ARQUIVO_PREDICOES,

    index=False,

    encoding="utf-8-sig"

)


# ============================================================
# MÉTRICAS POR QUESTÃO
# ============================================================

metricas = []


questoes = sorted(
    df_test[
        "questao"
    ].unique()
)


print("=" * 80)
print("MÉTRICAS POR QUESTÃO")
print("=" * 80)
print()


for questao in questoes:

    df_q = df_test[
        df_test["questao"] == questao
    ]

    y_real_q = df_q[
        "code0_3"
    ]

    y_pred_q = df_q[
        "classe_prevista"
    ]

    n_q = len(df_q)

    # --------------------------------------------------------
    # Quantidade por classe
    # --------------------------------------------------------

    n_classe_1 = (
        y_real_q == 1
    ).sum()

    n_classe_2 = (
        y_real_q == 2
    ).sum()

    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    precision_q = precision_score(

        y_real_q,
        y_pred_q,

        labels=[1, 2],

        average=None,

        zero_division=0

    )

    recall_q = recall_score(

        y_real_q,
        y_pred_q,

        labels=[1, 2],

        average=None,

        zero_division=0

    )

    f1_q = f1_score(

        y_real_q,
        y_pred_q,

        labels=[1, 2],

        average=None,

        zero_division=0

    )

    accuracy_q = accuracy_score(

        y_real_q,
        y_pred_q

    )

    precision_macro_q = precision_score(

        y_real_q,
        y_pred_q,

        labels=[1, 2],

        average="macro",

        zero_division=0

    )

    recall_macro_q = recall_score(

        y_real_q,
        y_pred_q,

        labels=[1, 2],

        average="macro",

        zero_division=0

    )

    f1_macro_q = f1_score(

        y_real_q,
        y_pred_q,

        labels=[1, 2],

        average="macro",

        zero_division=0

    )

    # --------------------------------------------------------
    # Exibir
    # --------------------------------------------------------

    print(
        f"{questao}: "
        f"n={n_q} | "
        f"classe 1={n_classe_1} | "
        f"classe 2={n_classe_2}"
    )

    print(
        f"  Classe 1: "
        f"P={precision_q[0]:.4f} "
        f"R={recall_q[0]:.4f} "
        f"F1={f1_q[0]:.4f}"
    )

    print(
        f"  Classe 2: "
        f"P={precision_q[1]:.4f} "
        f"R={recall_q[1]:.4f} "
        f"F1={f1_q[1]:.4f}"
    )

    print(
        f"  Macro:    "
        f"P={precision_macro_q:.4f} "
        f"R={recall_macro_q:.4f} "
        f"F1={f1_macro_q:.4f}"
    )

    print(
        f"  Acurácia: "
        f"{accuracy_q:.4f}"
    )

    print()

    # --------------------------------------------------------
    # Adicionar registros
    # --------------------------------------------------------

    metricas.append({

        "questao":
            questao,

        "classe":
            1,

        "n_teste":
            n_q,

        "n_classe_1":
            n_classe_1,

        "n_classe_2":
            n_classe_2,

        "precisao":
            precision_q[0],

        "revocacao":
            recall_q[0],

        "f1":
            f1_q[0],

        "acuracia":
            accuracy_q

    })


    metricas.append({

        "questao":
            questao,

        "classe":
            2,

        "n_teste":
            n_q,

        "n_classe_1":
            n_classe_1,

        "n_classe_2":
            n_classe_2,

        "precisao":
            precision_q[1],

        "revocacao":
            recall_q[1],

        "f1":
            f1_q[1],

        "acuracia":
            accuracy_q

    })


    metricas.append({

        "questao":
            questao,

        "classe":
            "macro",

        "n_teste":
            n_q,

        "n_classe_1":
            n_classe_1,

        "n_classe_2":
            n_classe_2,

        "precisao":
            precision_macro_q,

        "revocacao":
            recall_macro_q,

        "f1":
            f1_macro_q,

        "acuracia":
            accuracy_q

    })


# ============================================================
# MÉTRICAS GERAIS
# ============================================================

print("=" * 80)
print("MÉTRICAS GERAIS")
print("=" * 80)
print()


precision_geral = precision_score(

    y_test,
    y_pred,

    labels=[1, 2],

    average=None,

    zero_division=0

)

recall_geral = recall_score(

    y_test,
    y_pred,

    labels=[1, 2],

    average=None,

    zero_division=0

)

f1_geral = f1_score(

    y_test,
    y_pred,

    labels=[1, 2],

    average=None,

    zero_division=0

)

accuracy_geral = accuracy_score(

    y_test,
    y_pred

)


precision_macro_geral = precision_score(

    y_test,
    y_pred,

    average="macro",

    zero_division=0

)

recall_macro_geral = recall_score(

    y_test,
    y_pred,

    average="macro",

    zero_division=0

)

f1_macro_geral = f1_score(

    y_test,
    y_pred,

    average="macro",

    zero_division=0

)


print(
    f"Classe 1: "
    f"P={precision_geral[0]:.4f} "
    f"R={recall_geral[0]:.4f} "
    f"F1={f1_geral[0]:.4f}"
)

print(
    f"Classe 2: "
    f"P={precision_geral[1]:.4f} "
    f"R={recall_geral[1]:.4f} "
    f"F1={f1_geral[1]:.4f}"
)

print(
    f"Macro: "
    f"P={precision_macro_geral:.4f} "
    f"R={recall_macro_geral:.4f} "
    f"F1={f1_macro_geral:.4f}"
)

print(
    f"Acurácia: "
    f"{accuracy_geral:.4f}"
)

print()


# ============================================================
# ADICIONAR MÉTRICAS GERAIS AO CSV
# ============================================================

n_geral = len(
    y_test
)

n_geral_classe_1 = (
    y_test == 1
).sum()

n_geral_classe_2 = (
    y_test == 2
).sum()


metricas.append({

    "questao":
        "GERAL",

    "classe":
        1,

    "n_teste":
        n_geral,

    "n_classe_1":
        n_geral_classe_1,

    "n_classe_2":
        n_geral_classe_2,

    "precisao":
        precision_geral[0],

    "revocacao":
        recall_geral[0],

    "f1":
        f1_geral[0],

    "acuracia":
        accuracy_geral

})


metricas.append({

    "questao":
        "GERAL",

    "classe":
        2,

    "n_teste":
        n_geral,

    "n_classe_1":
        n_geral_classe_1,

    "n_classe_2":
        n_geral_classe_2,

    "precisao":
        precision_geral[1],

    "revocacao":
        recall_geral[1],

    "f1":
        f1_geral[1],

    "acuracia":
        accuracy_geral

})


metricas.append({

    "questao":
        "GERAL",

    "classe":
        "macro",

    "n_teste":
        n_geral,

    "n_classe_1":
        n_geral_classe_1,

    "n_classe_2":
        n_geral_classe_2,

    "precisao":
        precision_macro_geral,

    "revocacao":
        recall_macro_geral,

    "f1":
        f1_macro_geral,

    "acuracia":
        accuracy_geral

})


# ============================================================
# SALVAR MÉTRICAS
# ============================================================

df_metricas = pd.DataFrame(
    metricas
)

df_metricas.to_csv(

    ARQUIVO_METRICAS,

    index=False,

    encoding="utf-8-sig"

)


# ============================================================
# COEFICIENTES
# ============================================================

coeficientes = pd.DataFrame({

    "variavel":
        features,

    "coeficiente":
        modelo.coef_[0]

})

coeficientes.to_csv(

    ARQUIVO_COEFICIENTES,

    index=False,

    encoding="utf-8-sig"

)


# ============================================================
# MATRIZ DE CONFUSÃO GERAL
# ============================================================

matriz = confusion_matrix(

    y_test,
    y_pred,

    labels=[1, 2]

)


print("=" * 80)
print("MATRIZ DE CONFUSÃO GERAL")
print("=" * 80)

print()

print(
    "                 PREVISTO"
)

print(
    "              1          2"
)

print(
    f"REAL 1    "
    f"{matriz[0, 0]:8d}"
    f" {matriz[0, 1]:10d}"
)

print(
    f"REAL 2    "
    f"{matriz[1, 0]:8d}"
    f" {matriz[1, 1]:10d}"
)

print()


# ============================================================
# FINAL
# ============================================================

print("=" * 80)
print("ARQUIVOS GERADOS")
print("=" * 80)

print()

print(
    f"Dados utilizados:\n"
    f"{ARQUIVO_DADOS}"
)

print()

print(
    f"Predições:\n"
    f"{ARQUIVO_PREDICOES}"
)

print()

print(
    f"Métricas por questão e geral:\n"
    f"{ARQUIVO_METRICAS}"
)

print()

print(
    f"Coeficientes:\n"
    f"{ARQUIVO_COEFICIENTES}"
)

print()

print("=" * 80)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 80)