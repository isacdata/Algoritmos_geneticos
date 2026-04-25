import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import ast

# --- 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
def parse_array(string_data):
    """Converte a string '[1 2 3]' ou '[1, 2, 3]' em um array numpy."""
    # Remove colchetes e limpa espaços extras
    clean_str = string_data.replace('[', '').replace(']', '').strip()
    # Lida com múltiplos espaços entre números no CSV
    return np.fromstring(clean_str, sep=' ')

def load_instance(df, row_index=0):
    weights = parse_array(df.iloc[row_index]['Weights'])
    prices = parse_array(df.iloc[row_index]['Prices'])
    capacity = df.iloc[row_index]['Capacity']
    return weights, prices, capacity

# --- 2. FUNÇÕES DO ALGORITMO GENÉTICO ---

def calculate_fitness(population, weights, prices, capacity):
    """
    Calcula o valor total dos itens. 
    Se ultrapassar a capacidade, o fitness é 0 (penalização).
    """
    # Produto escalar: Soma(gene * preço) e Soma(gene * peso)
    total_weights = np.dot(population, weights)
    total_prices = np.dot(population, prices)
    
    # Indivíduos que excedem a capacidade recebem fitness 0
    fitness = np.where(total_weights <= capacity, total_prices, 0)
    return fitness

def tournament_selection(population, fitness, k=3):
    """Seleciona o melhor entre k indivíduos aleatórios."""
    idx = np.random.randint(0, len(population), size=k)
    best_idx = idx[np.argmax(fitness[idx])]
    return population[best_idx].copy()

def crossover(parent1, parent2, rate=0.8):
    """Crossover de ponto único."""
    if np.random.rand() < rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual, rate=0.01):
    """Mutação por inversão de bit (bit-flip)."""
    mask = np.random.rand(len(individual)) < rate
    individual[mask] = 1 - individual[mask]
    return individual

# --- 3. LOOP EVOLUTIVO PRINCIPAL ---

def genetic_algorithm(weights, prices, capacity, pop_size=50, generations=100, mut_rate=0.05):
    n_items = len(weights)
    # População inicial aleatória (binária)
    population = np.random.randint(2, size=(pop_size, n_items))
    
    best_fitness_history = []
    avg_fitness_history = []
    
    start_time = time.time()
    
    for gen in range(generations):
        fitness = calculate_fitness(population, weights, prices, capacity)
        
        # Estatísticas da geração
        best_fitness_history.append(np.max(fitness))
        avg_fitness_history.append(np.mean(fitness))
        
        new_population = []
        
        # Elitismo: Mantém o melhor indivíduo
        best_ind = population[np.argmax(fitness)].copy()
        new_population.append(best_ind)
        
        # Preenche o restante da nova população
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)
            
            c1, c2 = crossover(p1, p2)
            
            new_population.append(mutate(c1, mut_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(c2, mut_rate))
                
        population = np.array(new_population)
        
    end_time = time.time()
    
    return best_ind, np.max(fitness), best_fitness_history, avg_fitness_history, (end_time - start_time)

# --- 4. EXECUÇÃO E VISUALIZAÇÃO ---

# Carregar o arquivo enviado na imagem
try:
    df = pd.read_csv('knapsack_5_items.csv')
    # Selecionando a primeira instância para teste
    w, p, cap = load_instance(df, row_index=0)
    
    print(f"Iniciando GA para instância com {len(w)} itens. Capacidade: {cap}")
    
    # Rodar o algoritmo
    best_sol, best_val, hist_best, hist_avg, duration = genetic_algorithm(w, p, cap, pop_size=30, generations=50)
    
    print("-" * 30)
    print(f"Resultado Final: {best_val}")
    print(f"Melhor Cromossomo: {best_sol}")
    print(f"Tempo de execução: {duration:.4f} segundos")
    print("-" * 30)
    
    # Plotagem dos resultados (Etapa 3 do Trabalho)
    plt.figure(figsize=(10, 5))
    plt.plot(hist_best, label='Best Fitness', color='blue', linewidth=2)
    plt.plot(hist_avg, label='Average Fitness', color='orange', linestyle='--')
    plt.title('Convergência do Algoritmo Genético - Problema da Mochila')
    plt.xlabel('Gerações')
    plt.ylabel('Valor Total (Fitness)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

except FileNotFoundError:
    print("Erro: Certifique-se de que o arquivo 'knapsack_5_items.csv' está na mesma pasta.")