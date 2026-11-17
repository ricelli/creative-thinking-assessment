import time
import pandas as pd
import translators as ts
from tqdm import tqdm

# Caminhos dos arquivos
FICHEIRO_ENTRADA = "/home/ricelli/Documentos/Doutorado/resultados_experimentos_qwen_oecs/america_latina/amostras/T240Q01_amostra.csv"
FICHEIRO_SAIDA = "/home/ricelli/Documentos/Doutorado/resultados_experimentos_qwen_oecs/america_latina/amostras/T240Q01_amostra_traduzida.csv"

def traduzir_turno_com_retry(texto, max_retries=5):
    """
    Traduz um turno individual garantindo a retentativa em caso de instabilidade de rede.
    """
    if not texto.strip():
        return texto

    for tentativa in range(1, max_retries + 1):
        try:
            # Traduz via Google Translate sem limite de cota
            traduzido = ts.translate_text(texto, translator='google', from_language='pt', to_language='en')
            if traduzido:
                return traduzido
        except Exception:
            time.sleep(1.5 * tentativa)

    return texto

def traduzir_resposta(texto):
    if pd.isna(texto) or not str(texto).strip():
        return texto

    turnos = str(texto).split("X|X")
    turnos_traduzidos = []

    for turno in turnos:
        if turno.strip():
            traduzido = traduzir_turno_com_retry(turno)
            turnos_traduzidos.append(traduzido)
        else:
            turnos_traduzidos.append(turno)

    # Pausa curta para evitar limitação de requisições por IP
    time.sleep(0.2)
    return "X|X".join(turnos_traduzidos)

def executar():
    print(f"Lendo o arquivo em: {FICHEIRO_ENTRADA}...")
    try:
        df = pd.read_csv(FICHEIRO_ENTRADA, encoding="utf-8-sig")
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {FICHEIRO_ENTRADA}")
        return

    # Filtra as linhas cujo idioma seja português ('por' ou 'pt')
    mascara_pt = df['idioma'].astype(str).str.lower().isin(['por', 'pt'])
    indices_pt = df[mascara_pt].index

    print(f"Total de linhas no arquivo: {len(df)}")
    print(f"Linhas em Português a traduzir: {len(indices_pt)}\n")

    for idx in tqdm(indices_pt, desc="Traduzindo (PT -> EN)", unit="linha"):
        df.loc[idx, 'resposta'] = traduzir_resposta(df.loc[idx, 'resposta'])

    # Salva o arquivo CSV atualizado
    df.to_csv(FICHEIRO_SAIDA, index=False, encoding="utf-8-sig")
    
    print(f"\n[SUCESSO] Processamento concluído!")
    print(f"Resultado salvo em: {FICHEIRO_SAIDA}\n")

    # Exibe amostra das 3 primeiras linhas traduzidas para conferência direta no terminal
    print("--- CONFERÊNCIA DAS LINHAS TRADUZIDAS (BRASIL) ---")
    mostra_pt = df[df['idioma'] == 'por']
    for r in mostra_pt['resposta'].head(3):
        print("->", r)

if __name__ == "__main__":
    executar()