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
    'epochs': [30],
    'learning_rate_type': ['constant', 'cosine_decay'], 
    'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
    'weight_decay':[0, 1e-1, 1e-2, 1e-4],
    'patience': [7],
    'weight_initialisation': ['noisy-student'],
    'optimiser': ['sgd', 'adam'],
    'momentum': [0, 0.9, 0.99],
    'nesterov': ['True', 'False'],
    'label_smoothing': [0, 0.01, 0.05],
    'dropout': [0, 0.2, 0.5],
    'target_size': [224],
    'class_weights': ['True'],
    'warmup_epochs': [0, 2, 4]
}


def random_search(param_grid, max_evals=200):
    """Random search for hyperparameter optimization"""
    
    # Dataframe for results
    results = pd.DataFrame(columns = list(param_grid.keys())+ ['TestAcc'],
                           index = list(range(max_evals)))
    
    # Keep searching until reach max evaluations
    for i in range(max_evals):
        
        # Choose random hyperparameters
        hyperparameters = {}
        for k, v in param_grid.items():
            if k=='batch_size':
                hyperparameters[k] = choice(v, 1, p=[0.15, 0.3, 0.25, 0.25, 0.05])[0]
            elif k=='learning_rate':
                hyperparameters[k] = choice(v, 1, p=[0.05, 0.1, 0.45, 0.4])[0]
            elif k=='nesterov':
                hyperparameters[k] = choice(v, 1, p=[0.9, 0.1])[0]
            elif k=='optimiser':
                hyperparameters[k] = choice(v, 1, p=[0.8, 0.2])[0]
            
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
    
    results.to_csv('../configs/config_df.csv')
        
    return results 

def sort(results):
    """Sort combinations of hyperparameters by the highest test accuracies."""
    results.sort_values('TestAcc', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    
    return results


random_search(param_grid, max_evals = 100)