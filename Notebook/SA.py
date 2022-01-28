import numpy as np
import math
import random
from FS import evaluate

def generate_candidate(solution, distance = 1):
    
    random_indices = random.sample(range(len(solution)), distance)
    
    for i in random_indices:
        solution[i] = 1 - solution[i]
    
    return solution

def binary_simulated_annealing(features_len, n_iterations, temperature): 
    
    best_solution = np.random.choice([0, 1], features_len)
    best_score = evaluate(best_solution)
    
    current_solution, current_score = best_solution, best_score
    print("initial_solution: {}, initial_score: {}".format(current_solution, 1-current_score))
    
    for i in range(n_iterations):
        
        candidate_solution = generate_candidate(current_solution.copy())
        candidate_score = evaluate(candidate_solution)
        
        if candidate_score < best_score:
            
            best_solution, best_score = candidate_solution, candidate_score
            print("{}: {} ~~~> {:.3f}".format(i, best_solution, 1-best_score))
        
        delta = candidate_score - current_score
        current_t = temperature / float(i+1)
        prob = math.exp(-delta / current_t)
        
        if delta < 0 or np.random.rand() < prob:
            
            current_solution, current_score = candidate_solution, candidate_score
    
    return {"best_solution": best_solution,
           "best_score": 1-best_score}