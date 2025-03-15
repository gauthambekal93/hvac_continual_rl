# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:22:30 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:31:44 2024

@author: gauthambekal93
"""

import os

import numpy as np
#import pandas as pd
import random

import torch
import torch.nn as nn

#import torch.optim as optim
#import torch.distributions as dist

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

import time 
#import re
from env_model import ReaTZon_Model, TDryBul_Model, Reward_Model

from memory_module import Environment_Memory_Train
mse_loss = nn.MSELoss()

import pickle
import json

with open('all_paths.json', 'r') as openfile:  json_data = json.load(openfile)

exp_path = json_data['experiment_path']


def initialize_env_model(env, env_model_attributes):     
     
     realT_zon_model = ReaTZon_Model(env, env_model_attributes)
    
     dry_bulb_model = TDryBul_Model(env, env_model_attributes)
     
     reward_model = Reward_Model(env, env_model_attributes)
     
     env_memory_train = Environment_Memory_Train(env_model_attributes["env_train_buffer_size"] ,  env_model_attributes["train_test_ratio"])
     
     return realT_zon_model, dry_bulb_model, reward_model, env_memory_train #, env_memory_test
 

def load_env_model(realT_zon_model, reward_model, env_memory_train):
    
    model_files = [f for f in os.listdir(exp_path + '/Models/' ) if ('realT_zon_model' in f) and f.endswith('.pkl')]
    
    if model_files:
        
        with open(exp_path + '/Models/'+model_files[0], "rb") as f:
               realT_zon_model = pickle.load(f)
              
    model_files = [f for f in os.listdir(exp_path + '/Models/' ) if ('reward_model.pkl' in f) and f.endswith('.pkl')]
     
    
    if model_files:
         with open(exp_path + '/Models/'+model_files[0], "rb") as f:
                reward_model = pickle.load(f)
           
            
    if os.path.exists(exp_path + '/Models/'+"env_train_data.pkl"):
        
        with open(exp_path + '/Models/'+"env_train_data.pkl", "rb") as f:
               env_memory_train = pickle.load(f)
    

    return realT_zon_model, reward_model, env_memory_train




def calculate_train_loss( hypernet, y_pred, y, task_index):
    
    beta, regularizer = 0.01, 0.0   
    
    weights, bias = hypernet.generate_weights_bias( task_index)
    
    for previous_task_index in range(0, task_index): 
        
        weights_old, bias_old = hypernet.generate_weights_bias( previous_task_index)
        
        for layer_no in range(len( weights )):
                
            regularizer = regularizer  + mse_loss( weights[layer_no] , weights_old[layer_no] ) + mse_loss( bias[layer_no], bias_old[layer_no] )
    
    mse = mse_loss(y_pred, y ) 
    
    loss = mse  + beta * regularizer    
          
    return loss
    
       
   
def validate_model(hypernet, target_model, validation_X, validation_y, task_index, no_of_models):
    
    with torch.no_grad():
        weights, bias = hypernet.generate_weights_bias(task_index , no_of_models)
    
        target_model.update_params( weights , bias, no_of_models )
    
        predictions = target_model.predict_target(validation_X) 
    
    final_predictions = torch.mean(predictions, dim = 0)
    
    validation_loss = mse_loss( final_predictions, validation_y ) 
    
    #uncertanity  = torch.mean( torch.std ( predictions , dim = 0) )
    
    return  validation_loss.item(), final_predictions
    
  


def train_neuralnet(model, env_memory, exp_path, env_model_attributes ):
    
    train_X, train_y, validation_X, validation_y = model.get_dataset(env_memory)
    
    task_index = env_model_attributes["task_index"]
    
    batch_size = env_model_attributes["batch_size"]
    
    no_of_models = env_model_attributes["no_of_models"]
    
    
    if  ( np.random.rand() >= 0.9 )  and (len(train_X) > batch_size):
        no_of_updates = env_model_attributes["no_of_updates"]
    else:
        no_of_updates = 0
            
    indices  = torch.randperm(len(train_X) ) 
        
    start, end = 0, batch_size
    
    
    for step in range(no_of_updates):
        
        index = indices [start : end] 
        
        batch_X , batch_y = train_X[index], train_y[index]
    
        weights, bias = model.hypernet.generate_weights_bias(task_index, no_of_models )
        
        model.target_model.update_params(weights, bias, no_of_models )
        
        predictions = model.target_model.predict_target(batch_X)  
            
        predictions = torch.mean(predictions, dim =0)
        
        loss = calculate_train_loss( model.hypernet , predictions, batch_y, task_index )
        
        model.hypernet_optimizer.zero_grad()
        
        torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), max_norm=1.0)
        
        loss.backward()   
        
        model.hypernet_optimizer.step()        
        
        start =  end
        
        end = end + batch_size         
    
           
    validation_loss, validation_predictions = validate_model( model.hypernet, model.target_model, validation_X, validation_y, task_index , no_of_models)
             
    print("Model: " ,model.name, "Hypernet Validation Loss: ", validation_loss)
    
    print("------------------------------------------------------------------------------------")        
                 
'''             
import torch

torch.set_printoptions(sci_mode=False)  # Disable scientific notation
'''