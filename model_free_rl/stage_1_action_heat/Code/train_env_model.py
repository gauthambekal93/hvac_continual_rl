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

import torch.optim as optim
#import torch.distributions as dist

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

import time 
import re
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
     
     return realT_zon_model, dry_bulb_model, reward_model
 

def load_env_model(realT_zon_model, dry_bulb_model, reward_model, env_model_attributes):
    
    temp1, temp2, temp3 = False, False, False
    
    task_index = env_model_attributes["task_index"]
    
    model_files = [f for f in os.listdir(exp_path + '/Models/' ) if ('hypernet_{0}'.format( realT_zon_model.name) in f) and f.endswith('.pkl')]
    
    
    if model_files:
        
        checkpoint = torch.load(exp_path + '/Models/'+model_files[0])
        
        realT_zon_model.hypernet.load_state_dict(checkpoint['model_state_dict'])
        
        realT_zon_model.hypernet_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        
        if task_index in checkpoint["task_index_loss"].keys() :
            
            temp1 = True
            
            realT_zon_model.initial_training = True
            
            realT_zon_model.current_loss = checkpoint["task_index_loss"][task_index]
            
        
    model_files = [f for f in os.listdir(exp_path + '/Models/' ) if ('hypernet_{0}'.format( dry_bulb_model.name) in f) and f.endswith('.pkl')]
    
    if model_files:
         
         checkpoint = torch.load(exp_path + '/Models/'+model_files[0])
         
         dry_bulb_model.hypernet.load_state_dict(checkpoint['model_state_dict'])
         
         dry_bulb_model.hypernet_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         
         if task_index in checkpoint["task_index_loss"].keys() :
             
             temp2 = True
             
             dry_bulb_model.initial_training = True
         
             dry_bulb_model.current_loss = checkpoint["task_index_loss"][task_index]
             
         
    model_files = [f for f in os.listdir(exp_path + '/Models/' ) if ('hypernet_{0}'.format(reward_model.name) in f) and f.endswith('.pkl')]
     
    if model_files:
          
          checkpoint = torch.load(exp_path + '/Models/'+model_files[0])
          
          reward_model.hypernet.load_state_dict(checkpoint['model_state_dict'])
          
          reward_model.hypernet_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          
          if task_index in checkpoint["task_index_loss"].keys() :
             
             temp3 = True
              
             reward_model.initial_training = True   
          
             reward_model.current_loss = checkpoint["task_index_loss"][task_index]
             
          
    if os.path.exists(exp_path + '/Models/'+"env_train_data.pkl"):
        
        with open(exp_path + '/Models/'+"env_train_data.pkl", "rb") as f:
               env_memory_train = pickle.load(f)
    
    
    
    
    if temp1 and temp2 and temp3:
          completed_initial_env_train = True
    else:
          completed_initial_env_train = False
          
    return realT_zon_model, dry_bulb_model, reward_model, env_memory_train, completed_initial_env_train



def calculate_train_loss( epoch, y_pred, y, task_index, hypernet, hypernet_old):
    
    beta, regularizer = 0.01, 0.0   
    
    for previous_task_index in range(0, task_index): 
        
        weights, bias = hypernet.generate_weights_bias( previous_task_index)
        
        weights_old, bias_old = hypernet_old.generate_weights_bias( previous_task_index)
        
        for layer_no in range(len( weights )):
                
            regularizer = regularizer  + mse_loss( hypernet.W_mus[layer_no] , hypernet_old.W_mus[layer_no] ) + mse_loss( hypernet.b_mus[layer_no], hypernet_old.b_mus[layer_no] )
    
    mse = mse_loss(y_pred, y ) 
    
    loss = mse  + beta * regularizer    
    
    if epoch %10 == 0:
        print("Epoch ",epoch, "MSE ", mse.item(), "Regularizer ", beta * regularizer,"Train Loss ", loss.item()   )
        print("------------------------------------------------------------------------------------")
          
    return loss, hypernet
    
       
   
def validate_model(validation_X, validation_y, hypernet, target_model, task_index, sample_size = 1000):
    
    with torch.no_grad():
        weights, bias = hypernet.generate_weights_bias(task_index , sample_size)
    
        target_model.update_params( weights , bias, sample_size )
    
        predictions = target_model.predict_target(validation_X) 
    
    final_predictions = torch.mean(predictions, dim = 0)
    
    validation_loss = mse_loss( torch.mean(predictions, dim = 0), validation_y ) 
    
    uncertanity  = torch.mean( torch.std ( predictions , dim = 0) )

    print("Hypernet Validation Loss: ", validation_loss.item(), "Uncertanity: ", uncertanity.item() )
    
    print("------------------------------------------------------------------------------------")
    
    return  validation_loss.item(), final_predictions
    
  


def train_hypernet(model, env_memory, exp_path, env_model_attributes ):
    
    task_index, epochs, batch_size = env_model_attributes["task_index"], env_model_attributes["epochs"] , env_model_attributes["batch_size"]
    
    #hypernet_old =  env_model_attributes["hypernet_old"]  #I dont think a second model is needed
    
    train_X, train_y, validation_X, validation_y = model.get_dataset(env_memory)
    
    sample_size = 100
    
    #loss_threshold = env_model_attributes["{0}_loss_thresh".format(model.name) ]
    
    """IF initial training is completed initially, we only train with a 
       small set of random examples which is 5 time the batch size. """
    
    if model.initial_training is True :
        
        indices  = torch.randperm(len(train_X) ) [ : int( len(train_X) *0.1 ) ] 
        
        train_X, train_y = train_X[indices], train_y[indices]
    
        epochs = 1
             
    for epoch in range(epochs):
                
        indices  = torch.randperm(len(train_X) )
        
        for batch_no in range( 0, int(len(train_X) ), batch_size ):
            
            index = indices [batch_no : batch_no + batch_size]
            
            batch_X , batch_y = train_X[index], train_y[index]
            
            weights, bias = model.hypernet.generate_weights_bias(task_index, sample_size )   #weights and bias generated initially is nearly 20 times larger than other code
           
            model.target_model.update_params(weights, bias, sample_size)
            
            predictions = model.target_model.predict_target(batch_X)   #even before any gradient update this produced hugre predictions avg (825)
                
            predictions = torch.mean(predictions, dim =0)
        
            loss, hypernet = calculate_train_loss(epoch, predictions, batch_y, task_index, model.hypernet, model.hypernet_old )
            
            model.hypernet_optimizer.zero_grad()
            
            torch.nn.utils.clip_grad_norm_(model.hypernet.parameters(), max_norm=1.0)
            
            loss.backward()   
            
            model.hypernet_optimizer.step()        
           
       
        if epoch % 10 ==0 :
            
             validation_loss, validation_predictions = validate_model(validation_X, validation_y, model.hypernet, model.target_model, task_index )
             
             model_files = [f for f in os.listdir(exp_path + '/Models/' ) if ('hypernet_{0}'.format(model.name) in f) and f.endswith('.pkl')]
             
             if validation_loss < model.current_loss:
                     
                 if model_files:
                     
                     checkpoint = torch.load(exp_path + '/Models/'+model_files[0])
                     
                     checkpoint["model_name"] = model.name
                     checkpoint["model_state_dict"] = model.hypernet.state_dict()
                     checkpoint["optimizer_state_dict"] = model.hypernet_optimizer.state_dict()
                     checkpoint["task_index_loss"][task_index] = validation_loss
                         
                 else:

                    checkpoint = { "model_name":model.name,
                                   'model_state_dict': model.hypernet.state_dict(),  
                                   'optimizer_state_dict': model.hypernet_optimizer.state_dict() , 
                                    "task_index_loss": { task_index: validation_loss } }
                    
                         
                 torch.save(checkpoint, exp_path + '/Models/'+'hypernet_'+model.name+'.pkl')
                 
                 model.initial_training = True
                 
                 model.current_loss = validation_loss 
   
             else:
                 
                if model_files:
                    
                     checkpoint = torch.load(exp_path + '/Models/'+model_files[0])
                     
                     model.hypernet.load_state_dict(checkpoint['model_state_dict'])
                     
                     model.hypernet_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                     
                