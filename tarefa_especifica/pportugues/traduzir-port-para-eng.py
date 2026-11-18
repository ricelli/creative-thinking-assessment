import os
import time
import logging
import pandas as pd
import translators as ts
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError

BASE_DIR = "/home/..."
OUTPUT_DIR = "/home/..."
LOG_FILE = os.path.join(OUTPUT_DIR, "erros_traducao.log")

# 2 workers mantém a velocidade boa sem gerar bloqueios de IP
MAX_WORKERS = 2
TIMEOUT_SEGUNDOS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def traduzir_com_timeout_e_status(texto, max_retries=3):
    if not str(texto).strip():
        return texto

    for tentativa in range(1, max_retries + 1):
        inicio = time.time()
        
        with ThreadPoolExecutor(max_workers=1) as single_executor:
            future = single_executor.submit(
                ts.translate_text, 
                str(texto), 
                translator='bing',  # Alternado para BING para evitar o erro 429 do Google
                from_language='pt', 
                to_language='en'
            )
            
            try:
                traduzido = future.result(timeout=TIMEOUT_SEGUNDOS)
                tempo_decorrido = time.time() - inicio
                
                if traduzido:
                    time.sleep(0.4)  # Pausa de segurança entre requisições
                    return traduzido

            except TimeoutError:
                tqdm.write(f"⌛ [TIMEOUT] Reenviando via Bing (Tentativa {tentativa}/{max_retries})...")
            except Exception as e:
                tqdm.write(f"❌ [ERRO NET] Tentativa {tentativa}/{max_retries}: {e}")

        # Se falhou por limite, aguarda 3 a 6 segundos para liberar a conexão
        time.sleep(3.0 * tentativa)

    tqdm.write(f"⚠️ [IGNORADO] Não foi possível traduzir o trecho: '{texto[:30]}...'")
    return texto

def traduzir_resposta_paralela(texto):
    if pd.isna(texto) or not str(texto).strip():
        return texto

    turnos = str(texto).split("X|X")
    turnos_traduzidos = []

    for turno in turnos:
        if turno.strip():
            traduzido = traduzir_com_timeout_e_status(turno)
            turnos_traduzidos.append(traduzido)
        else:
            turnos_traduzidos.append(turno)

    return "X|X".join(turnos_traduzidos)

def processar_csv(tarefa):
    caminho_origem, caminho_destino = tarefa
    
    # Se já foi processado e salvo previamente, pula o arquivo
    if os.path.exists(caminho_destino):
        return caminho_destino

    try:
        df = pd.read_csv(caminho_origem, encoding='utf-8-sig')

        if 'idioma' in df.columns and 'resposta' in df.columns:
            mascara_pt = df['idioma'].astype(str).str.lower().isin(['por', 'pt'])
            indices_pt = df[mascara_pt].index

            for idx in indices_pt:
                df.loc[idx, 'resposta'] = traduzir_resposta_paralela(df.loc[idx, 'resposta'])

        os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)
        df.to_csv(caminho_destino, index=False, encoding='utf-8-sig')
        return caminho_destino

    except Exception as e:
        logging.error(f"Erro ao processar {caminho_origem}: {e}")
        return None

def mapear_arquivos():
    lista_tarefas = []
    for root, dirs, files in os.walk(BASE_DIR):
        if os.path.basename(root).lower() == "amostras":
            for file in files:
                if file.endswith(".csv"):
                    caminho_origem = os.path.join(root, file)
                    rel_path = os.path.relpath(caminho_origem, BASE_DIR)
                    caminho_destino = os.path.join(OUTPUT_DIR, rel_path)
                    lista_tarefas.append((caminho_origem, caminho_destino))
    return lista_tarefas

def executar():
    tarefas = mapear_arquivos()
    total = len(tarefas)

    print(f"Total de {total} arquivos CSV mapeados.")
    print(f"Rodando tradução via Bing com MAX_WORKERS = {MAX_WORKERS}...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(processar_csv, t) for t in tarefas]
        for _ in tqdm(as_completed(futures), total=total, desc="Progresso Geral", unit="arquivo"):
            pass

    print(f"\n[CONCLUÍDO] Todos os arquivos foram processados!")
    print(f"Diretório final: {OUTPUT_DIR}")

if __name__ == "__main__":
    executar()
