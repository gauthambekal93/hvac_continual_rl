# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:18:33 2024

@author: gauthambekal93
"""

import torch
import torch.nn as nn

import torch.optim as optim
import torch.distributions as dist




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
        
        logits =  torch.matmul ( logits, self.weight3 ) + self.bias3 
        
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
        
        self.weight1 = nn.Linear(hidden_dim, w1_dim )  #was w1_dim * 2
        
        self.bias1 = nn.Linear(hidden_dim, b1_dim )
        
        self.weight2 = nn.Linear(hidden_dim, w2_dim )
        
        self.bias2 = nn.Linear(hidden_dim, b2_dim )
        
        self.weight3 = nn.Linear(hidden_dim, w3_dim )
        
        self.bias3 = nn.Linear(hidden_dim, b3_dim )
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        self.normal_dist = dist.Normal(0, 1)

        self.W_mus, self.W_stds, self.b_mus, self.b_stds = [], [], [] , []
        
        self.num_tasks = num_tasks
        
        self.num_layers = num_layers
        
    def task_conditioned(self, X, layer_no):
       
       logits = self.leaky_relu ( self.common1(X) )
       
       logits = self.leaky_relu ( self.common2(logits) )
       
       if layer_no ==0 :
           
           w_logits = self.weight1 (logits) 
           
           b_logits = self.bias1 (logits)
           
           return w_logits, b_logits
           
       if layer_no == 1 :
           
           w_logits = self.weight2 (logits) 
           
           b_logits = self.bias2 (logits)
           
           return w_logits, b_logits
           
       if layer_no == 2 :
           
           w_logits = self.weight3 (logits) 
           
           b_logits = self.bias3 (logits)
           
           return w_logits, b_logits
        

    def generate_weights_bias(self, task_index, device, no_of_models  =  1):

        task_id =  torch.nn.functional.one_hot( torch.tensor(task_index) , num_classes = self.num_tasks).float() 
        
        # Add small Gaussian noise,  Scale noise (0.05 controls intensity)
        task_id = torch.stack( [  task_id + torch.randn_like(task_id) * 0.05  for _ in range(no_of_models) ] , dim  = 0)
        
        weights, bias = [], []
                
        for i in range(0, self.num_layers):
            
            layer_id = torch.nn.functional.one_hot( torch.tensor(i) , num_classes = self.num_layers) .repeat(no_of_models, 1)
            
            X = torch.cat( [ task_id, layer_id ] , dim = 1).to(device).to(dtype=torch.float32)
            
            if X.dim() ==1: 
                X = X.reshape(1,-1)
            
            W, b = self.task_conditioned( X, i )  
            
            weights.append(W )
            
            bias.append(b)
            
        return weights, bias


        


class ReaTZon_Model:
    
    def __init__(self, env, learnt_env_attributes, device):
        
        self.name = "ReaTZon_Model"
        
        self.initial_training = False
        
        no_of_models = 5
        
        self.current_loss = no_of_models*[1.0]
        
        t_input_dim =  learnt_env_attributes["real_zone_input"]
        
        t_hidden_dim = learnt_env_attributes["hidden_size"]
        
        t_output_dim = learnt_env_attributes["real_zone_output"]
        
        num_layers = learnt_env_attributes["num_layers"]
        
        num_tasks =  learnt_env_attributes["num_tasks"]
        
        self.target_model = Target( t_input_dim, t_hidden_dim, t_output_dim )
        
        h_input_dim = num_tasks + num_layers   #we need to definetask_id and layer_id as input to the hypernet model
        
        w1_dim = self.target_model.weight1_shape[0] * self.target_model.weight1_shape[1] 
        b1_dim = self.target_model.bias1_shape[1]
        
        w2_dim = self.target_model.weight2_shape[0] * self.target_model.weight2_shape[1] 
        b2_dim = self.target_model.bias2_shape[1] 
        
        w3_dim = self.target_model.weight3_shape[0] * self.target_model.weight3_shape[1] 
        b3_dim = self.target_model.bias3_shape[1] 
        

        h_hidden_dim = max(w1_dim, w2_dim, w3_dim)
        
        
        self.hypernet = Hypernet(  h_input_dim, h_hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers ).to(device) 
        
        self.hypernet_optimizer =  optim.Adam( self.hypernet.parameters(), lr =  learnt_env_attributes["lr"] ) 
        
        
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
    
    def __init__(self, env, learnt_env_attributes, device):
        
        self.name = "TDryBul_Model"
        
        self.initial_training = False
        
        no_of_models = 5
        
        self.current_loss = no_of_models*[1.0]
        
        t_input_dim =  learnt_env_attributes["dry_bulb_input"]
        
        t_hidden_dim = learnt_env_attributes["hidden_size"]
        
        t_output_dim = learnt_env_attributes["dry_bulb_output"]
        
        num_layers = learnt_env_attributes["num_layers"]
        
        num_tasks =  learnt_env_attributes["num_tasks"]
        
        self.target_model = Target( t_input_dim, t_hidden_dim, t_output_dim )
        
        h_input_dim = num_tasks + num_layers   #we need to definetask_id and layer_id as input to the hypernet model
        
        w1_dim = self.target_model.weight1_shape[0] * self.target_model.weight1_shape[1] 
        b1_dim = self.target_model.bias1_shape[1]
        
        w2_dim = self.target_model.weight2_shape[0] * self.target_model.weight2_shape[1] 
        b2_dim = self.target_model.bias2_shape[1] 
        
        w3_dim = self.target_model.weight3_shape[0] * self.target_model.weight3_shape[1] 
        b3_dim = self.target_model.bias3_shape[1]
        

        h_hidden_dim = max(w1_dim, w2_dim, w3_dim)
        
        self.hypernet = Hypernet(  h_input_dim, h_hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers ).to(device) 
        
        self.hypernet_optimizer = optim.Adam( self.hypernet.parameters(), lr =  learnt_env_attributes["lr"] )

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
    def __init__(self, env, learnt_env_attributes, device):
        
        self.name = "Reward_Model"
        
        self.initial_training = False
        
        no_of_models = 5
        
        self.current_loss = no_of_models*[1.0]
        
        t_input_dim = learnt_env_attributes["reward_model_input"]
        
        t_hidden_dim = learnt_env_attributes["hidden_size"]
        
        t_output_dim = learnt_env_attributes["reward_model_output"]
        
        
        num_layers = learnt_env_attributes["num_layers"]
        
        num_tasks =  learnt_env_attributes["num_tasks"]
        
        self.target_model = Target( t_input_dim, t_hidden_dim, t_output_dim )
        
        h_input_dim = num_tasks + num_layers   #we need to definetask_id and layer_id as input to the hypernet model
        
        w1_dim = self.target_model.weight1_shape[0] * self.target_model.weight1_shape[1] 
        b1_dim = self.target_model.bias1_shape[1]
        
        w2_dim = self.target_model.weight2_shape[0] * self.target_model.weight2_shape[1] 
        b2_dim = self.target_model.bias2_shape[1] 
        
        w3_dim = self.target_model.weight3_shape[0] * self.target_model.weight3_shape[1] 
        b3_dim = self.target_model.bias3_shape[1] 
        

        h_hidden_dim = max(w1_dim, w2_dim, w3_dim)
        
        self.hypernet =  Hypernet(  h_input_dim, h_hidden_dim, w1_dim, b1_dim, w2_dim, b2_dim, w3_dim, b3_dim, num_tasks, num_layers ).to(device)
        
        self.hypernet_optimizer = optim.Adam(  self.hypernet.parameters(), lr =  learnt_env_attributes["lr"] ) 
         
        
        self.min_reward, self.max_reward = None, None
        
        input_states = ['reaTZon_y','reaTSetCoo_y','reaTSetHea_y'] 
        
        self.input_state_index = [ i for i, obs in enumerate( env.observations) if obs in input_states ]


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
        
        























           