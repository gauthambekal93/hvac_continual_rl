# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:18:33 2024

@author: gauthambekal93
"""
import os

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
import torch.distributions as dist

from collections import deque  
import time

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

#num_tasks = 3

#we will store the complete data instead of just whats just required.

        


class Target():
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        self.weight1_shape =   (input_dim, hidden_dim)
        
        self.bias1_shape =     (1, hidden_dim) 
        
        self.weight2_shape =   (hidden_dim, hidden_dim)
        
        self.bias2_shape =     (1, hidden_dim) 
        
        self.weight3_shape =   (hidden_dim, output_dim )
        
        self.bias3_shape =     (1, output_dim) 
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
    def predict_target(self, X):
        
        logits = self.leaky_relu (  torch.matmul ( X, self.weight1)  + self.bias1 )  
        
        logits = self.leaky_relu (  torch.matmul ( logits, self.weight2 ) + self.bias2 )
        
        logits = self.leaky_relu  ( torch.matmul ( logits, self.weight3 ) + self.bias3 )
        
        return logits
   
    def update_params(self, weights, bias , sample_size):
        
        self.weight1 = weights[0].reshape( (sample_size,) + self.weight1_shape )
    
        self.bias1 = bias[0].reshape((sample_size,) + self.bias1_shape )
        
        self.weight2 = weights[1].reshape( (sample_size,) + self.weight2_shape )
        
        self.bias2 = bias[1].reshape((sample_size,) + self.bias2_shape )
        
        self.weight3 = weights[2].reshape( (sample_size,) + self.weight3_shape )
    
        self.bias3 = bias[2].reshape((sample_size,) + self.bias3_shape )
        
        


class Hypernet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers  ):
        super().__init__()
        
        self.common1 = nn.Linear(input_dim, hidden_dim)
        
        self.common2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.weight1 = nn.Linear(hidden_dim, w1_dim * 2)
        
        self.bias1 = nn.Linear(hidden_dim, b1_dim * 2)
        
        self.weight2 = nn.Linear(hidden_dim, w2_dim * 2)
        
        self.bias2 = nn.Linear(hidden_dim, b2_dim * 2)
        
        self.weight3 = nn.Linear(hidden_dim, w3_dim * 2)
        
        self.bias3 = nn.Linear(hidden_dim, b3_dim * 2)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        self.normal_dist = dist.Normal(0, 1)

        self.W_mus, self.W_stds, self.b_mus, self.b_stds = [], [], [] , []
        
        self.num_tasks = num_tasks
        
        self.num_layers = num_layers
        
    def task_conditioned(self, X, layer_no):
       
       logits = self.leaky_relu ( self.common1(X) )
       
       logits = self.leaky_relu ( self.common2(logits) )
       
       if layer_no ==0 :
           
           w_logits = self.leaky_relu (self.weight1 (logits)) 
           
           b_logits = self.leaky_relu (self.bias1 (logits))
           
       if layer_no == 1 :
           
           w_logits = self.leaky_relu (self.weight2 (logits)) 
           
           b_logits = self.leaky_relu (self.bias2 (logits))
           
       if layer_no == 2 :
           
           w_logits = self.leaky_relu (self.weight3 (logits)) 
           
           b_logits = self.leaky_relu (self.bias3 (logits))
        
       
       W_mu, W_std = torch.chunk(w_logits, 2, dim=1)
   
       W_std = torch.exp(W_std)
       
       b_mu, b_std = torch.chunk(b_logits, 2, dim=1)
   
       b_std = torch.exp(b_std)
       
       return W_mu, W_std, b_mu, b_std
   
    
   
    def generate_distribution(self, W_mu, W_std, b_mu, b_std , sample_size):
       
       samples = self.normal_dist.sample(( sample_size, W_mu.shape[1] ))  #sample from unit normal distribution
       
       w_z =  ( W_mu ) + (samples * W_std)
       
       #w_z = torch.mean(w_z, dim =0 ).reshape(1, -1)
       
       samples = self.normal_dist.sample(( sample_size, b_mu.shape[1] ))  #sample from unit normal distribution
       
       b_z = ( b_mu ) + (samples * b_std)
       
       #b_z = torch.mean(b_z, dim =0 ).reshape(1, -1)
       
       return w_z, b_z 
   
    
    
    def generate_weights_bias(self, task_index, sample_size = 1):
    
        
        task_id =  torch.nn.functional.one_hot( torch.tensor(task_index) , num_classes = self.num_tasks) * 0.1  
        
        weights, bias = [], []
        
        self.W_mus, self.W_stds, self.b_mus, self.b_stds = [], [], [] , []
        
        for i in range(0, self.num_layers):
            
            layer_id = torch.nn.functional.one_hot( torch.tensor(i) , num_classes = self.num_layers) * 0.1  
            
            X = torch.cat( [ task_id, layer_id ] ).to(dtype=torch.float32)
            
            if X.dim() ==1: 
                X = X.reshape(1,-1)
        
            W_mu, W_std, b_mu, b_std = self.task_conditioned( X, i )        
            
            self.W_mus.append(W_mu)
            
            self.W_stds.append(W_std)
            
            self.b_mus.append(b_mu)
            
            self.b_stds.append(b_std)
            
            W, b = self.generate_distribution(W_mu, W_std, b_mu, b_std, sample_size )
            
            weights.append(W )
            
            bias.append(b)
            
        return weights, bias




class ReaTZon_Model:
    
    def __init__(self, env, env_model_attributes):
        
        self.name = "ReaTZon_Model"
        
        self.initial_training = False
        
        self.current_loss = 1.0
        
        t_input_dim =  env_model_attributes["real_zone_input"]
        
        t_hidden_dim = env_model_attributes["hidden_layers"]
        
        t_output_dim = env_model_attributes["real_zone_output"]
        
        num_layers = env_model_attributes["num_layers"]
        
        num_tasks =  env_model_attributes["num_tasks"]
        
        self.target_model = Target( t_input_dim, t_hidden_dim, t_output_dim )
        
        h_input_dim = num_tasks + num_layers   #we need to definetask_id and layer_id as input to the hypernet model
        
        w1_dim = self.target_model.weight1_shape[0] * self.target_model.weight1_shape[1] 
        b1_dim = self.target_model.bias1_shape[1]
        
        w2_dim = self.target_model.weight2_shape[0] * self.target_model.weight2_shape[1] 
        b2_dim = self.target_model.bias2_shape[1] 
        
        w3_dim = self.target_model.weight3_shape[0] * self.target_model.weight3_shape[1] 
        b3_dim = self.target_model.bias3_shape[1] 
        

        h_hidden_dim = max(w1_dim, w2_dim, w3_dim)
        
        self.hypernet = Hypernet(  h_input_dim, h_hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers ) 
         
        self.hypernet_optimizer = optim.Adam( self.hypernet.parameters(), lr =  env_model_attributes["lr"] )  #was 0.001

        self.hypernet_old = None
        
        input_states = ['reaTZon_y',"TDryBul_pred_0", 'TDryBul_pred_900', 'TDryBul_pred_1800', 'TDryBul_pred_2700', 'TDryBul_pred_3600', 'TDryBul_pred_4500','TDryBul_pred_5400','TDryBul_pred_6300','TDryBul_pred_7200']
        
        self.input_state_index = [ i for i, obs in enumerate( env.observations) if obs in input_states ]
        
        self.output_state_index = [ i for i, obs in enumerate( env.observations) if obs in ["reaTZon_y"] ]
        
        
        
    def get_dataset(self, env_memory):
        
            states = torch.cat( list(env_memory.states_train), dim =0 )[:, self.input_state_index  ] 
            
            actions = torch.cat( list(env_memory.actions_train), dim =0 ) 
            
            train_X =  torch.cat([ states, actions ] , dim = 1 )
            
            train_y = torch.cat( list(env_memory.next_states_train), dim =0 ) [:, self.output_state_index  ] 
            
            
            states = torch.cat( list(env_memory.states_validation), dim =0 ) [:, self.input_state_index  ] 
            
            actions = torch.cat( list(env_memory.actions_validation), dim =0 ) 
            
            validation_X =  torch.cat([ states, actions ] , dim = 1 )
            
            validation_y = torch.cat( list(env_memory.next_states_validation), dim =0 )[:, self.output_state_index  ] 
            
            
            return train_X, train_y, validation_X, validation_y
        
        
            
        

class TDryBul_Model:
    
    def __init__(self, env, env_model_attributes):
        
        self.name = "TDryBul_Model"
        
        self.initial_training = False
        
        self.current_loss = 1.0
        
        t_input_dim =  env_model_attributes["dry_bulb_input"]
        
        t_hidden_dim = env_model_attributes["hidden_layers"]
        
        t_output_dim = env_model_attributes["dry_bulb_output"]
        
        num_layers = env_model_attributes["num_layers"]
        
        num_tasks =  env_model_attributes["num_tasks"]
        
        self.target_model = Target( t_input_dim, t_hidden_dim, t_output_dim )
        
        h_input_dim = num_tasks + num_layers   #we need to definetask_id and layer_id as input to the hypernet model
        
        w1_dim = self.target_model.weight1_shape[0] * self.target_model.weight1_shape[1] 
        b1_dim = self.target_model.bias1_shape[1]
        
        w2_dim = self.target_model.weight2_shape[0] * self.target_model.weight2_shape[1] 
        b2_dim = self.target_model.bias2_shape[1] 
        
        w3_dim = self.target_model.weight3_shape[0] * self.target_model.weight3_shape[1] 
        b3_dim = self.target_model.bias3_shape[1] 
        

        h_hidden_dim = max(w1_dim, w2_dim, w3_dim)
        
        self.hypernet = Hypernet(  h_input_dim, h_hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers ) 
         
        self.hypernet_optimizer = optim.Adam( self.hypernet.parameters(), lr =  env_model_attributes["lr"] )  #was 0.001

        self.hypernet_old = None
        
        input_states =  ["TDryBul_pred_0", 'TDryBul_pred_900', 'TDryBul_pred_1800', 'TDryBul_pred_2700', 'TDryBul_pred_3600', 'TDryBul_pred_4500','TDryBul_pred_5400','TDryBul_pred_6300','TDryBul_pred_7200']
        
        self.input_state_index = [ i for i, obs in enumerate( env.observations) if obs in input_states ]
        
        self.output_state_index = [ i for i, obs in enumerate( env.observations) if obs in ['TDryBul_pred_7200'] ]
        
    def get_dataset(self, env_memory):
            
            train_X = torch.cat( list(env_memory.states_train), dim =0 ) [:, self.input_state_index  ] 
            
            train_y = torch.cat( list(env_memory.next_states_train), dim =0 )[:, self.output_state_index  ] 
            
            validation_X = torch.cat( list(env_memory.states_validation), dim =0 ) [:, self.input_state_index  ] 
            
            validation_y = torch.cat( list(env_memory.next_states_validation), dim =0 )[:, self.output_state_index  ] 
            
            return train_X, train_y, validation_X, validation_y
        

        
    
class Reward_Model:
    def __init__(self, env, env_model_attributes):
        
        self.name = "Reward_Model"
        
        self.initial_training = False
        
        self.current_loss = 1.0
        
        t_input_dim = env_model_attributes["reward_model_input"]
        
        t_hidden_dim = env_model_attributes["hidden_layers"]
        
        t_output_dim = env_model_attributes["reward_model_output"]
        
        num_layers = env_model_attributes["num_layers"]
        
        num_tasks =  env_model_attributes["num_tasks"]
        
        self.target_model = Target( t_input_dim, t_hidden_dim, t_output_dim )
        
        h_input_dim = num_tasks + num_layers   #we need to definetask_id and layer_id as input to the hypernet model
        
        w1_dim = self.target_model.weight1_shape[0] * self.target_model.weight1_shape[1] 
        b1_dim = self.target_model.bias1_shape[1]
        
        w2_dim = self.target_model.weight2_shape[0] * self.target_model.weight2_shape[1] 
        b2_dim = self.target_model.bias2_shape[1] 
        
        w3_dim = self.target_model.weight3_shape[0] * self.target_model.weight3_shape[1] 
        b3_dim = self.target_model.bias3_shape[1] 
        

        h_hidden_dim = max(w1_dim, w2_dim, w3_dim)
        
        self.hypernet = Hypernet(  h_input_dim, h_hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers ) 
         
        self.hypernet_optimizer = optim.Adam( self.hypernet.parameters(), lr = env_model_attributes["lr"] )         
        
        self.hypernet_old = None
        
        self.min_reward, self.max_reward = None, None
        
        input_states = ['reaTZon_y','reaTSetCoo_y','reaTSetHea_y'] 
        
        self.input_state_index = [ i for i, obs in enumerate( env.observations) if obs in input_states ]
        
        '''
        input_states = ['reaTZon_y',"LowerSetp[1]", "UpperSetp[1]" ]
        
        self.input_state_index = []
        
        for i, obs in enumerate( env.observations ) :
            
            if [ input_state for input_state in input_states if input_state in obs ]:
                self.input_state_index .append(i)
        '''

    def scale(self, y):
        
        y = ( 2 * (y - self.min_reward ) / ( self.max_reward  - self.min_reward) ) - 1
        
        return y 
    
    def inverse_scale(self, y ):
        
        y = ( (y + 1) * ( self.max_reward  - self.min_reward) ) / 2  + self.min_reward
        
        return y
    
    
    def get_dataset(self, env_memory):
        
            states = torch.cat( list(env_memory.states_train), dim =0 ) [:, self.input_state_index  ] 
            
            actions = torch.cat( list(env_memory.actions_train), dim =0 ) 
            

            train_X =  torch.cat([ states, actions ] , dim = 1 )
            
            train_y = torch.cat( list(env_memory.rewards_train), dim =0 )
        
            
            
            states = torch.cat( list(env_memory.states_validation), dim =0 )[:, self.input_state_index  ] 
            
            actions = torch.cat( list(env_memory.actions_validation), dim =0 ) 
            
            
            validation_X =  torch.cat([ states, actions ] , dim = 1 )
            
            validation_y = torch.cat( list(env_memory.rewards_validation), dim =0 )
            
       
            if self.min_reward is None:
                self.min_reward = torch.min(train_y)
            
            if self.max_reward is None:
                self.max_reward = torch.max(train_y)
            
            
            return train_X, train_y, validation_X, validation_y
        
        























           