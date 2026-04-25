# Bibliotecas de processamento
import pandas as pd
import numpy as np
import requests

# Link da base de dados de escalonamento de tareafas
url = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/CCNFP10g1a.txt"
df_txt = requests.get( url )

# Obter requisição
req = requests.get( url )
texto = texto = req.text

# Carregar base de Dados
def carregar_base_dados( url : str ) -> pd.DataFrame:
    #Obter o texto
    response = requests.get(url)
    texto = response.text
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]

    # As primeiras 10 linhas são as demandas dos nós
    demandas = [float(x) for x in linhas[:10]]

    # Dicionário para capturar os blocos de dados
    blocos = {
        "Aij": [],
        "Bij": [],
        "Cij": []
    }

    # Mapeamento de palavras-chave para nossas chaves do dicionário
    chave_atual = None

    for linha in linhas[10:]:
        if "1st Variable" in linha:
            chave_atual = "Aij"
            continue
        elif "2nd Variable" in linha:
            chave_atual = "Bij"
            continue
        elif "Fixed Cost" in linha:
            chave_atual = "Cij"
            continue

        if chave_atual:
            blocos[chave_atual].append(float(linha))

    # Criar o DataFrame
    df = pd.DataFrame(blocos)

    # Criando os índices de Origem (i) e Destino (j)
    # Como são 100 valores, assumimos uma matriz 10x10
    n_nos = 10
    df['origem'] = [i // n_nos for i in range(len(df))]
    df['destino'] = [i % n_nos for i in range(len(df))]

    # Reorganizar colunas para melhor leitura
    df = df[['origem', 'destino', 'Aij', 'Bij', 'Cij']]

    return df

# Converter tipo de atributos da base
def tratar_base( df : pd.DataFrame ) -> pd.DataFrame:
  for coluna in df.columns:
    if df[coluna].dtype == float:
      df[coluna] = df[coluna].astype(int)

  return df

# Execução
url = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/CCNFP10g1a.txt"
df = carregar_base_dados(url)
df_tratado = tratar_base( df )

def calcular_fitness(solucao_fluxo, df_tratado):
    """
    Calcula o custo total (Fitness) de uma solução.
    
    Parâmetros:
    - solucao_fluxo: Um array/lista com a quantidade de fluxo x_ij alocada 
      para cada linha do DataFrame df_tratado.
    - df_tratado: O DataFrame com as colunas Aij, Bij e Cij.
    """
    custo_total = 0
    
    # Iteramos sobre cada arco (linha do dataframe) e a solução proposta
    for i, fluxo in enumerate(solucao_fluxo):
        if fluxo > 0:
            # Recupera os parâmetros do custo para este arco
            custo_fixo = df_tratado.iloc[i]['Cij']
            A = df_tratado.iloc[i]['Aij']
            B = df_tratado.iloc[i]['Bij']
            
            # Cálculo do custo côncavo: Fixo + Variável (A*x + B*x^2)
            # Nota: Em problemas de custo côncavo, B costuma ser negativo ou 
            # a função usa uma raiz, dependendo da instância específica.
            custo_arco = custo_fixo + (A * fluxo) + (B * (fluxo**2))
            custo_total += custo_arco
            
    return custo_total

# Escolher uma linha
linha = 15
problema_escolhido = df.iloc[linha]
vetor_problema = np.array(problema_escolhido)
vetor_problema

# Como calcular fitness
calcular_fitness( problema_escolhido, df_tratado  )
