#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:02:51 2020

@author: gregmeyer
"""

import numpy as np
from numpy.random import choice
import pandas as pd
import random
import json

from numpy.random import choice
choice([0, 0.9, 0.99], 1, p=[0.2,0.7,0.1])


param_grid = {
    'architecture': ['efficientnet'],
    'epochs': [200],
    'batch_size': [4 ,8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'learning_rate_type': ['constant', 'cosine_decay'], # implement percentage weighting
    'learning_rate': [0.1, 0.01, 0.001, 0.0001, 0.00001],
    'patience': [7],
    'weight_initialisation': ['noisy-student'],
    'optimiser': ['sgd', 'adam'],
    'momentum': [0, 0.9, 0.99],
    'nesterov': ['True', 'False'],
    'label_smoothing': [0, 0.01, 0.05],
    'dropout': [0, 0.2, 0.5],
    'target_size': [224],
    'class_weights': ['True', 'False'],
    'warmup_epochs': [3,5,7]
}


def random_search(param_grid, max_evals=200):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = list(param_grid.keys())+ ['TestAcc'],
                           index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in range(max_evals):
        
        # Choose random hyperparameters
# =============================================================================
#         hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
# =============================================================================
        hyperparameters = {}
        for k, v in param_grid.items():
            if k=='batch_size':
                hyperparameters[k] = choice(v, 1, p=[0.06]+[0.11]*8+[0.06])[0]
            elif k=='learning_rate':
                hyperparameters[k] = choice(v, 1, p=[0.08]+[0.23]*4)[0]
            elif k=='nesterov':
                hyperparameters[k] = choice(v, 1, p=[0.7, 0.3])[0]
            elif k=='class_weights':
                hyperparameters[k] = choice(v, 1, p=[0.95, 0.05])[0]
            
            else:
                hyperparameters[k] = choice(v, 1)[0]
                
            if type(hyperparameters[k])==np.int64:
                hyperparameters[k]=int(hyperparameters[k])
        
        # Fill in dataframe
        results.loc[i, :] = list(hyperparameters.values()) + ['-']
        
        # Drop duplicate parameter combos (unlikely to occur anyway
        results.drop_duplicates(inplace=True)
        
        json_params = json.dumps(hyperparameters, indent=4)

        with open('../configs/config' + str(i) + '.json', 'w', encoding='utf-8') as file:
            file.write(json_params)
        
    return results 

def sort(results):
    """Sort combinations of hyperparameters by the highest test accuracies."""
    results.sort_values('TestAcc', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    
    return results


random_search(param_grid, max_evals = 3)