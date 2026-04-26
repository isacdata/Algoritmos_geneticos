import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests

def carregar_base_dados(url: str) -> pd.DataFrame:
    """Baixa o texto, extrai os blocos de variáveis e cria o DataFrame."""
    response = requests.get(url)
    texto = response.text
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]

    # Dicionário para capturar os blocos de dados
    blocos = {"Aij": [], "Bij": [], "Cij": []}
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

    df = pd.DataFrame(blocos)

    # Criando os índices de Origem (i) e Destino (j) - Matriz 10x10
    n_nos = 10
    df['origem'] = [i // n_nos for i in range(len(df))]
    df['destino'] = [i % n_nos for i in range(len(df))]
    
    # Reorganiza colunas
    df = df[['origem', 'destino', 'Aij', 'Bij', 'Cij']]
    return df

def tratar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Converte tipos float irreais para int para simplificar."""
    for coluna in df.columns:
        if df[coluna].dtype == float:
            df[coluna] = df[coluna].astype(int)
    return df