#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline OCSAI 2 para a amostra chinesa do PISA 2022.

Entrada:
    /home/ricelli/Documentos/Doutorado/amostra_chines_qwen.csv

Saída:
    /home/ricelli/Documentos/Doutorado/resultados_chines_ocsai/

Características:
- Usa somente a biblioteca padrão do Python.
- Envia somente a resposta original, nunca resposta_qwen.
- Preserva as colunas da amostra.
- Salva resultados por questão.
- Permite retomar processamento interrompido.
- Registra erros da API.
"""

import argparse
import csv
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ARQUIVO_ENTRADA = Path(
    "/home/ricelli/Documentos/Doutorado/"
    "amostra_chines_qwen.csv"
)

PASTA_SAIDA = Path(
    "/home/ricelli/Documentos/Doutorado/"
    "resultados_chines_ocsai"
)

API_URL = "https://openscoring.du.edu/llm"
MODELO = "ocsai2"
TASK = "CUSTOM"

# Código de idioma a validar na primeira chamada.
IDIOMA = "zho"

BATCH_SIZE = 25
SLEEP_ENTRE_REQUISICOES = 1.0
TIMEOUT = 300


QUESTOES = [
    "T240Q01",
    "T300Q01",
    "T350Q01",
    "T360Q01",
    "T370Q01",
    "T570Q02",
    "T240Q02",
    "T300Q02",
    "T350Q02",
    "T570Q01",
    "T350Q03",
    "T570Q03",
]

DIMENSOES = {
    "T240Q01": "GCI",
    "T300Q01": "GCI",
    "T350Q01": "GCI",
    "T360Q01": "GCI",
    "T370Q01": "GCI",
    "T570Q02": "GCI",
    "T240Q02": "GDI",
    "T300Q02": "GDI",
    "T350Q02": "GDI",
    "T570Q01": "GDI",
    "T350Q03": "EII",
    "T570Q03": "EII",
}


# ============================================================
# ESTÍMULOS
# Mantidos conforme o script original fornecido.
# ============================================================

ESTIMULOS = {
    "T350Q03": (
        "Escreva uma continuação para a história abaixo, com base "
        "na história e na imagem da capa do livro que possui uma "
        "árvore frondosa e três pessoas com guarda-chuvas abertos "
        "caminhando ao lado. É possível ver a sombra das árvores "
        "e também das pessoas no solo, e o céu amarelado com poucas "
        "nuvens, dando a entender que estão caminhando no fim do dia.\n\n"
        "Esse é o início da história: "
        '"Dois homens e uma mulher que não se conheciam encontraram '
        "abrigo sob a mesma árvore. Quando começaram a conversar, "
        "descobriram que todos tinham um olho castanho, um olho azul, "
        "não conheciam seus pais e podiam ler a mente de outras pessoas. "
        "Eles concluíram que tinham que ser parentes de alguma forma "
        'e decidiram fazer uma jornada para descobrir de onde todos eles vieram."'
    ),
    "T350Q01": (
        "Escrever uma ideia original de enredo para um livro através "
        "do estímulo: uma capa de livro que contém um astronauta "
        "olhando para o céu em tons de laranja, rosa e azul, e uma "
        "nave espacial pousada ao lado. O solo é árido, porém apresenta "
        "um espaço verde atrás, é cercado de montanhas ou estruturas "
        "pedregosas e no céu é possível visualizar um astro se pondo."
    ),
    "T240Q01": (
        "Gerar diálogos para uma história em quadrinhos com quatro "
        "quadros e seis caixinhas de diálogo entre o Sol e a Terra "
        "com o separador X|X."
    ),
    "T570Q02": (
        "Escrever uma continuação de um diálogo para o seguinte item: "
        '"Seu amigo sugere uma ideia de história de filme em que Rob '
        "(o robô) poderia ter uma aparência e sentimentos humanos. "
        "No filme, Rob está disfarçando sua verdadeira identidade e "
        "tentando viver uma vida humana. Você decide desenvolver essa "
        "ideia mais a fundo e começa a trabalhar no roteiro do filme. "
        "Em uma cena-chave, Leo descobre que Rob é um robô. Então, Rob "
        'revela que há algo que Leo não sabe. Escreva uma continuação '
        'original para o diálogo entre Rob e Leo."\n\n'
        "Leo: Você mentiu para mim esse tempo todo! Isso precisa parar "
        "agora ou eu vou contar a todo mundo quem você realmente é.\n"
        "Rob: Espere! Tem algo que você não sabe...\n"
        "Rob:Leo:"
    ),
    "T300Q01": (
        "Dar um título adequado para uma imagem que contém o planeta "
        "Terra dentro do bulbo de uma lâmpada redonda."
    ),
    "T570Q01": (
        "Escreva 2 ideias diferentes de histórias de filmes sobre "
        "um humano chamado Leo, que interage com um robô inteligente "
        "chamado Rob. Você não precisa escrever a história completa "
        "do filme, apenas explicar sobre o que será o filme. As duas "
        "histórias devem ser tão diferentes uma da outra quanto "
        "possível separadas por X|X."
    ),
    "T570Q03": (
        "Escreva um final alternativo para o filme com base na ideia "
        "de que Rob conseguiu convencer Leo a não revelar sua "
        "identidade robótica a ninguém.\n\n"
        "Rob consegue convencer Leo a não revelar sua identidade "
        "robótica a ninguém. O robô e o humano se tornam melhores "
        "amigos. Leo ajuda Rob a se comportar ainda mais como um "
        "humano. Para preservar seu segredo, os dois amigos decidem "
        "começar uma nova vida em uma nova cidade onde ninguém os conhece."
    ),
    "T350Q02": (
        "Escrever duas histórias diferentes separadas por X|X para "
        "uma capa de livro que contém um balão de ar quente de estilo "
        "antigo, com uma grande esfera vermelha como o balão principal. "
        "Abaixo do balão, há uma estrutura com asas semelhantes a "
        "penas ou velas, junto com uma pequena cesta onde duas pessoas "
        "parecem estar posicionadas. O design remete a invenções aéreas "
        "do século XIX, com um estilo ilustrativo e detalhado. O fundo é azul claro."
    ),
    "T300Q02": (
        "Geração de títulos para uma ilustração tão diferentes entre "
        "si quanto possível com o separador X|X. A ilustração possui "
        "um livro aberto sem nada escrito e páginas amarelas em um "
        "campo com céu claro e com nuvens, grama no solo, uma árvore "
        "e uma cadeira de madeira vazia."
    ),
    "T240Q02": (
        "Gerar diálogos para uma história em quadrinhos com quatro "
        "quadros e seis caixinhas de diálogo entre o Sol e a Terra "
        "com o separador X|X."
    ),
    "T370Q01": (
        'Escrever uma ideia original de enredo para um livro através '
        'do estímulo: uma capa de livro vermelho com letras maiúsculas '
        'brancas escrito "2983".'
    ),
    "T360Q01": (
        'Escrever uma ideia original de enredo para um livro através '
        'do estímulo: uma capa de livro vermelho com letras maiúsculas '
        'brancas escrito "andar para trás".'
    ),
}


# ============================================================
# CSV E IDENTIFICAÇÃO
# ============================================================

def carregar_csv(caminho):
    with caminho.open("r", encoding="utf-8-sig", newline="") as f:
        leitor = csv.DictReader(f)
        if not leitor.fieldnames:
            raise RuntimeError(f"CSV sem cabeçalho: {caminho}")
        return list(leitor), list(leitor.fieldnames)


def obter_questao(linha):
    valor = (linha.get("question") or "").strip()
    return valor


def obter_resposta(linha):
    # Deliberadamente não usa resposta_qwen.
    return linha.get("resposta", "") or ""


def chave_linha(linha):
    # A combinação mantém a identidade de cada resposta da amostra.
    return (
        linha.get("xml_origem", ""),
        linha.get("response_index", ""),
        linha.get("student", ""),
        linha.get("question", ""),
    )


def caminho_saida(questao):
    return PASTA_SAIDA / f"{questao}_ocsai_resultados.csv"


def carregar_resultados_existentes(caminho):
    resultados = {}

    if not caminho.exists():
        return resultados

    linhas, _ = carregar_csv(caminho)

    for linha in linhas:
        chave = chave_linha(linha)
        originalidade = (linha.get("ocsai_originality") or "").strip()
        erro = (linha.get("erro_api") or "").strip()

        # Erros e resultados vazios podem ser tentados novamente.
        if originalidade and not erro:
            resultados[chave] = linha

    return resultados


def salvar_resultados(caminho, linhas, campos):
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    with caminho.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(
            f,
            fieldnames=campos,
            extrasaction="ignore",
        )
        escritor.writeheader()
        escritor.writerows(linhas)


# ============================================================
# API OCSAI
# ============================================================

def construir_input_csv(respostas):
    buffer = io.StringIO(newline="")
    escritor = csv.writer(buffer, lineterminator="\n")

    for resposta in respostas:
        escritor.writerow([resposta])

    return buffer.getvalue()


def chamar_ocsai(respostas, estimulo):
    parametros = {
        "input": construir_input_csv(respostas),
        "model": MODELO,
        "task": TASK,
        "language": IDIOMA,
        "prompt": estimulo,
        "prompt_in_input": "false",
        "elab_method": "none",
        "logprob_scoring": "true",
    }

    corpo = urllib.parse.urlencode(parametros).encode("utf-8")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    api_key = os.environ.get("OCSAI_API_KEY")
    if api_key:
        headers["X-API-KEY"] = api_key

    requisicao = urllib.request.Request(
        API_URL,
        data=corpo,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=TIMEOUT) as resposta:
            conteudo = resposta.read().decode("utf-8")
            return json.loads(conteudo)

    except urllib.error.HTTPError as erro:
        corpo_erro = erro.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {erro.code}: {corpo_erro}") from erro

    except urllib.error.URLError as erro:
        raise RuntimeError(f"Erro de conexão: {erro}") from erro


def extrair_scores(resultado):
    if not isinstance(resultado, dict):
        raise RuntimeError("Resposta da API não é um objeto JSON.")

    scores = resultado.get("scores")
    if not isinstance(scores, list):
        raise RuntimeError(
            "A resposta da API não contém uma lista na chave 'scores'. "
            f"Resposta recebida: {str(resultado)[:1000]}"
        )

    return scores


# ============================================================
# PROCESSAMENTO
# ============================================================

def processar_questao(questao, amostra, campos_originais):
    if questao not in ESTIMULOS:
        print(f"[ERRO] Estímulo não definido: {questao}")
        return

    linhas_questao = [
        dict(linha) for linha in amostra
        if obter_questao(linha) == questao
    ]

    if not linhas_questao:
        print(f"[AVISO] Nenhuma resposta encontrada para {questao}.")
        return

    arquivo_saida = caminho_saida(questao)
    resultados_anteriores = carregar_resultados_existentes(arquivo_saida)

    campos = list(campos_originais)
    campos_novos = [
        "ocsai_modelo",
        "ocsai_task",
        "ocsai_language",
        "ocsai_originality",
        "ocsai_confidence",
        "ocsai_flags",
        "erro_api",
    ]

    for campo in campos_novos:
        if campo not in campos:
            campos.append(campo)

    # Recupera resultados anteriores para permitir retomada.
    for linha in linhas_questao:
        chave = chave_linha(linha)
        if chave in resultados_anteriores:
            linha.update(resultados_anteriores[chave])
        else:
            linha["ocsai_modelo"] = MODELO
            linha["ocsai_task"] = TASK
            linha["ocsai_language"] = IDIOMA
            linha["ocsai_originality"] = ""
            linha["ocsai_confidence"] = ""
            linha["ocsai_flags"] = ""
            linha["erro_api"] = ""

    pendentes = [
        linha for linha in linhas_questao
        if not (
            (linha.get("ocsai_originality") or "").strip()
            and not (linha.get("erro_api") or "").strip()
        )
    ]

    print("\n" + "=" * 72)
    print(f"Questão: {questao} | Tarefa: {DIMENSOES[questao]}")
    print(f"Respostas na amostra: {len(linhas_questao)}")
    print(f"Resultados já válidos: {len(linhas_questao) - len(pendentes)}")
    print(f"Pendentes: {len(pendentes)}")
    print(f"Idioma enviado à API: {IDIOMA}")
    print(f"Saída: {arquivo_saida}")
    print("=" * 72)

    if not pendentes:
        print("[OK] Questão já processada.")
        return

    for inicio in range(0, len(pendentes), BATCH_SIZE):
        lote = pendentes[inicio:inicio + BATCH_SIZE]
        numero_lote = inicio // BATCH_SIZE + 1
        total_lotes = (len(pendentes) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nLote {numero_lote}/{total_lotes}: {len(lote)} respostas")

        # Somente as respostas originais são enviadas à API.
        respostas = [obter_resposta(linha) for linha in lote]

        try:
            resultado = chamar_ocsai(
                respostas=respostas,
                estimulo=ESTIMULOS[questao],
            )
            scores = extrair_scores(resultado)

            if len(scores) != len(lote):
                raise RuntimeError(
                    f"API retornou {len(scores)} scores para "
                    f"{len(lote)} respostas."
                )

            for linha, score in zip(lote, scores):
                linha["ocsai_originality"] = score.get("originality", "")
                linha["ocsai_confidence"] = score.get("confidence", "")

                flags = score.get("flags", "")
                if isinstance(flags, (dict, list)):
                    flags = json.dumps(flags, ensure_ascii=False)

                linha["ocsai_flags"] = flags
                linha["erro_api"] = ""

            print(f"[OK] {len(scores)} resultados recebidos.")

        except Exception as erro:
            mensagem = f"{type(erro).__name__}: {erro}"
            print(f"[ERRO] {mensagem}")

            for linha in lote:
                linha["ocsai_originality"] = ""
                linha["ocsai_confidence"] = ""
                linha["ocsai_flags"] = ""
                linha["erro_api"] = mensagem

        # Salva todos os resultados após cada lote.
        # Assim, uma interrupção não apaga o que já foi processado.
        salvar_resultados(arquivo_saida, linhas_questao, campos)

        if inicio + BATCH_SIZE < len(pendentes):
            time.sleep(SLEEP_ENTRE_REQUISICOES)

    def tem_valor(valor):
        if valor is None:
            return False
        return str(valor).strip() != ""

    validos = sum(
        1 for linha in linhas_questao
        if tem_valor(linha.get("ocsai_originality"))
        and not tem_valor(linha.get("erro_api"))
    )
    erros = len(linhas_questao) - validos

    print(f"\nConcluída {questao}: válidas={validos}, sem resultado válido={erros}")
    print(f"Arquivo: {arquivo_saida}")


# ============================================================
# LISTAGEM E EXECUÇÃO
# ============================================================

def listar():
    print(f"Entrada: {ARQUIVO_ENTRADA}")
    print(f"Saída:   {PASTA_SAIDA}")
    print(f"Modelo:  {MODELO}")
    print(f"Tarefa:  {TASK}")
    print(f"Idioma:  {IDIOMA}")
    print(f"Questões com estímulos: {len(ESTIMULOS)}")
    for questao in QUESTOES:
        print(f"  {questao} ({DIMENSOES[questao]})")


def main():
    parser = argparse.ArgumentParser(
        description="OCSAI 2 para a amostra chinesa do PISA 2022."
    )
    parser.add_argument("--listar", action="store_true")
    parser.add_argument("--questao", choices=QUESTOES)
    parser.add_argument("--todas", action="store_true")
    args = parser.parse_args()

    if args.listar:
        listar()
        return

    if not args.questao and not args.todas:
        parser.print_help()
        return

    if not ARQUIVO_ENTRADA.exists():
        raise SystemExit(
            "Arquivo de entrada não encontrado:\n"
            f"{ARQUIVO_ENTRADA}"
        )

    amostra, campos = carregar_csv(ARQUIVO_ENTRADA)

    if "question" not in campos:
        raise SystemExit(
            "A coluna 'question' não existe no CSV de entrada. "
            f"Colunas encontradas: {campos}"
        )

    if "resposta" not in campos:
        raise SystemExit(
            "A coluna 'resposta' não existe no CSV. "
            "O script exige a resposta original e não usa resposta_qwen. "
            f"Colunas encontradas: {campos}"
        )

    if args.questao:
        processar_questao(args.questao, amostra, campos)
    else:
        for questao in QUESTOES:
            processar_questao(questao, amostra, campos)


if __name__ == "__main__":
    main()
