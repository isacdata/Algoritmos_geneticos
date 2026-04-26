import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests

def calcular_fitness_individual(solucao_fluxo, df_tratado):
    """Calcula o custo total (Fitness) de uma única solução (Custo Côncavo)."""
    custo_total = 0
    # Usando vetores do numpy para acelerar o cálculo
    A = df_tratado['Aij'].values
    B = df_tratado['Bij'].values
    C = df_tratado['Cij'].values
    
    # Máscara onde o fluxo é maior que zero
    ativos = solucao_fluxo > 0
    
    # Cálculo vetorizado: Fixo + A*x + B*(x^2) apenas onde fluxo > 0
    custos = C[ativos] + (A[ativos] * solucao_fluxo[ativos]) + (B[ativos] * (solucao_fluxo[ativos]**2))
    custo_total = np.sum(custos)
    
    return custo_total

def calculate_population_fitness(population, df_tratado):
    """Calcula o fitness para toda a população."""
    return np.array([calcular_fitness_individual(ind, df_tratado) for ind in population])

def tournament_selection_min(population, fitness, k=3):
    """Seleciona o indivíduo com MENOR custo entre k aleatórios."""
    idx = np.random.randint(0, len(population), size=k)
    best_idx = idx[np.argmin(fitness[idx])] # np.argmin porque queremos minimizar o custo
    return population[best_idx].copy()

def crossover_mean(parent1, parent2, rate=0.8):
    """Crossover por Média: os filhos são a média aritmética dos pais."""
    if np.random.rand() < rate:
        # Calcula a média exata entre os dois pais
        child = (parent1 + parent2) / 2.0
        return child.copy(), child.copy()
        
    return parent1.copy(), parent2.copy()

def mutate_gaussian(individual, rate=0.05, desvio_padrao=1.0):
    """Mutação Gaussiana: adiciona ruído de uma distribuição normal."""
    # Convertendo para float64 caso o array venha como inteiro
    individual = individual.astype(np.float64) 
    
    mask = np.random.rand(len(individual)) < rate
    num_mutations = np.sum(mask)
    
    if num_mutations > 0:
        # Sorteia ruídos de uma distribuição normal (média 0, desvio padrão definido)
        ruido = np.random.normal(loc=0.0, scale=desvio_padrao, size=num_mutations)
        individual[mask] += ruido
        
        # Trava de segurança: impede que o fluxo fique negativo
        individual[individual < 0] = 0.0 
        
    return individual

def genetic_algorithm_ccnfp(df_tratado, pop_size=50, generations=100, mut_rate=0.05, max_flow=20):
    n_items = len(df_tratado)
    
    # MUDANÇA 1: População inicial agora usa floats contínuos (uniforme entre 0 e max_flow)
    population = np.random.uniform(0, max_flow, size=(pop_size, n_items))
    
    best_fitness_history = []
    avg_fitness_history = []
    
    start_time = time.time()
    
    for gen in range(generations):
        fitness = calculate_population_fitness(population, df_tratado)
        
        melhor_custo_atual = np.min(fitness)
        best_fitness_history.append(melhor_custo_atual)
        avg_fitness_history.append(np.mean(fitness))
        
        new_population = []
        
        best_ind = population[np.argmin(fitness)].copy()
        new_population.append(best_ind)
        
        while len(new_population) < pop_size:
            p1 = tournament_selection_min(population, fitness)
            p2 = tournament_selection_min(population, fitness)
            
            c1, c2 = crossover_mean(p1, p2)
            
            new_population.append(mutate_gaussian(c1, mut_rate, desvio_padrao=2.0))
            if len(new_population) < pop_size:
                new_population.append(mutate_gaussian(c2, mut_rate, desvio_padrao=2.0))
                
        population = np.array(new_population)
        
    end_time = time.time()
    
    final_fitness = calculate_population_fitness(population, df_tratado)
    best_ind_final = population[np.argmin(final_fitness)].copy()
    
    return best_ind_final, np.min(final_fitness), best_fitness_history, avg_fitness_history, (end_time - start_time)