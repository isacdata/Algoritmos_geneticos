from utils import preprocessing as p
from utils import genetic as g

import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/CCNFP10g1a.txt"
df_bruto = p.carregar_base_dados(url)
df_tratado = p.tratar_base(df_bruto)
    
# Salva o CSV conforme solicitado pelo pré-processamento original
arquivo_csv = "data/base_dados_tratada.csv"
df_tratado.to_csv(arquivo_csv, index=False)
print(f"Base de dados salva com sucesso em '{arquivo_csv}'.")
    
# Carrega do CSV para garantir a integração das pontas
df_csv = pd.read_csv(arquivo_csv)
    
print("\n2. Iniciando Algoritmo Genético...")
# Ajuste max_flow de acordo com a regra de negócios do seu fluxo
solucao, custo, hist_best, hist_avg, tempo = g.genetic_algorithm_ccnfp(
        df_tratado=df_csv, 
        pop_size=50, 
        generations=150, 
        mut_rate=0.05, 
        max_flow=10 
    )
    
print("-" * 80)
print(f"Custo Final (Menor é Melhor): {custo}")
print(f"Tempo de execução: {tempo:.4f} segundos")
print(f"Distribuição do Fluxo (Arco a Arco):\n{solucao}")
print("-" * 80)

# Plotagem
plt.figure(figsize=(10, 5))
# Inverti as cores para psicologia visual: custo caindo fica bem em vermelho/verde
plt.plot(hist_best, label='Best Fitness (Menor Custo)', color='green', linewidth=2)
plt.plot(hist_avg, label='Average Fitness', color='orange', linestyle='--')
plt.title('Convergência do AG - Problema de Fluxo de Custo Côncavo')
plt.xlabel('Gerações')
plt.ylabel('Custo Total')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("images/convergencia_ag.png", dpi=300)  # Salva a figura em alta resolução
plt.show()