import os
import pandas as pd
from tqdm import tqdm
import argostranslate.package
import argostranslate.translate

# Caminhos dos diretórios
BASE_DIR = "/home/..."
OUTPUT_DIR = "/home/..."

def carregar_modelo_traducao():
    """
    Garante o download e carregamento do pacote de tradução PT -> EN offline.
    """
    print("Verificando/Instalando pacote de tradução offline (PT -> EN)...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    
    package_to_install = next(
        filter(lambda x: x.from_code == "pt" and x.to_code == "en", available_packages),
        None
    )
    
    if package_to_install:
        argostranslate.package.install_from_path(package_to_install.download())
        print("Pacote de idioma instalado com sucesso!")
    else:
        print("Pacote já instalado ou pronto para uso.")

def traduzir_resposta_offline(texto):
    """
    Traduz a linha mantendo os delimitadores X|X.
    """
    if pd.isna(texto) or not str(texto).strip():
        return texto

    turnos = str(texto).split("X|X")
    turnos_traduzidos = []

    for turno in turnos:
        if turno.strip():
            try:
                # Tradução feita 100% em memória local
                traduzido = argostranslate.translate.translate(turno, "pt", "en")
                turnos_traduzidos.append(traduzido)
            except Exception:
                turnos_traduzidos.append(turno)
        else:
            turnos_traduzidos.append(turno)

    return "X|X".join(turnos_traduzidos)

def processar_csv(caminho_origem, caminho_destino):
    # Se já foi traduzido e salvo, pula o arquivo
    if os.path.exists(caminho_destino):
        return

    try:
        df = pd.read_csv(caminho_origem, encoding='utf-8-sig')

        if 'idioma' in df.columns and 'resposta' in df.columns:
            mascara_pt = df['idioma'].astype(str).str.lower().isin(['por', 'pt'])
            indices_pt = df[mascara_pt].index

            for idx in indices_pt:
                df.loc[idx, 'resposta'] = traduzir_resposta_offline(df.loc[idx, 'resposta'])

        os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)
        df.to_csv(caminho_destino, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f"Erro ao processar {caminho_origem}: {e}")

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
    carregar_modelo_traducao()
    
    tarefas = mapear_arquivos()
    total = len(tarefas)

    print(f"\nIniciando tradução offline de {total} arquivos CSV...")

    for origem, destino in tqdm(tarefas, desc="Progresso da Tradução", unit="arquivo"):
        processar_csv(origem, destino)

    print(f"\n[CONCLUÍDO COM SUCESSO] Todos os arquivos foram traduzidos offline!")
    print(f"Diretório de saída: {OUTPUT_DIR}")

if __name__ == "__main__":
    executar()
