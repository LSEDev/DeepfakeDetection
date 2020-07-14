#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:02:51 2020

@author: gregmeyer
"""

import numpy as np
import pandas as pd
import random
import json

param_grid = {
    'architecture': ['xception', 'mobilenet', 'efficientnet', 'densenet'],
    'epochs': list(range(20, 160, 20)),
    'batch_size': [4 ,8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    'learning_rate_type': ['constant', 'cosine_decay', 'increasing'],
    'learning_rate': list(np.logspace(np.log10(0.00005), np.log10(0.005), base = 10, num = 1000)),
    'patience': list(range(2, 10, 2)),
    'weight_initialisation': ['imagenet', 'noisy-student', 'xavier'],
    'optimiser': ['sgd', 'adam'],
    'label_smoothing': list(np.linspace(0.005, 0.04, 7)),
    'momentum': [0.1, 0.3, 0.5],
    'dropout': list(np.linspace(0, 0.7, num=10)),
    'target_size': [224,256],
    'class_weights': [True, False]
}


def random_search(param_grid, max_evals=200):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = list(param_grid.keys())+ ['TestAcc'],
                           index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in range(max_evals):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        
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


print(random_search(param_grid, max_evals = 2))