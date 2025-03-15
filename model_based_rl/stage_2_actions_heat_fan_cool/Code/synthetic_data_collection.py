# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:11:59 2025

@author: gauthambekal93
"""
import os

import numpy as np
#import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
import time 
#import torch.distributions as dist

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  



def generate_test_predictions(validation_X, hypernet, target_model, task_index, sample_size):
    
    with torch.no_grad():
        
        weights, bias = hypernet.generate_weights_bias(task_index , sample_size)
    
        target_model.update_params( weights , bias, sample_size )
    
        predictions = target_model.predict_target(validation_X) 
    
    final_predictions = torch.mean(predictions, dim = 0)
    
    uncertanities  = torch.abs( torch.std ( predictions , dim = 0)  ) 
    
    return  final_predictions, uncertanities




def create_synthetic_data( actor, realT_zon_model, dry_bulb_model, reward_model, agent_actual_memory, agent_synthetic_memory, env_attributes, env_model_attributes, env ):

      
      task_index = env_model_attributes["task_index"]
      
      sample_size = env_model_attributes["no_of_synthetic_samples"]
      
      no_of_models = env_model_attributes["no_of_models"]
      
      state_samples, _, _, _, next_state_samples, done_samples =  agent_actual_memory.sample_memory( sample_size = sample_size)
          
      
      
      actions, _, _ = actor.select_action( state_samples[ :,   env_attributes["state_mask"] ]  )  #9 actions to be sampled by the agent per, state
      
      actions = actions.detach().clone()
      
      #temp =  np.where(discrete_actions == 0, -1 , 1)  #was 0 instead of -1 
      
      #actions = torch.cat( [ actions, torch.tensor(temp) , torch.tensor(temp) ], axis = 1  )
      
      
      
      """ Predict the Zone operative temperature"""
      test_X =  torch.cat([ state_samples[:, realT_zon_model.input_state_index ], actions ] , dim = 1 )
      
      predicted_realT_zon, _ = generate_test_predictions(test_X, realT_zon_model.hypernet, realT_zon_model.target_model, task_index, no_of_models )

      
      
      """ Obtain the Dry bulb temperature"""

      predicted_dry_bulb = next_state_samples[:, dry_bulb_model.input_state_index ][:, -1:]
      
      """ Create the next sate predictions"""
      time_diff = agent_actual_memory.next_states[0][0,0] - agent_actual_memory.states[0][0,0]
      
      time_data = state_samples[:, [0]] + time_diff
      
      next_state_predictions = torch.empty(( len(state_samples),  0 ))
      
      dry_bulb_indices = []
      
      for observation in env.observations:
          
          if "time" in observation:
              next_state_predictions  = torch.cat ( [ next_state_predictions, time_data ] , dim = 1)

              
          if "reaTZon" in observation:
              next_state_predictions = torch.cat ( [ next_state_predictions, predicted_realT_zon ] , dim = 1)


          if ("TDryBul" in observation ) and len( dry_bulb_indices ) == 0:
              
              dry_bulb_indices = [i for i , obs in enumerate( env.observations) if "TDryBul" in obs]
              
              dry_bulb = state_samples[:, dry_bulb_indices]
              
              dry_bulb = torch.cat([dry_bulb, predicted_dry_bulb] , dim =1)
              
              dry_bulb =  dry_bulb[: , 1:]
              
              next_state_predictions = torch.cat ( [ next_state_predictions, dry_bulb ] , dim = 1)
              
              
          if "reaTSetCoo" in observation:
              tmp = torch.empty((next_state_predictions.shape[0], 1))
              
              next_state_predictions = torch.cat ( [ next_state_predictions, tmp ] , dim = 1)
          
            
          if "reaTSetHea_y" in observation:
              tmp = torch.empty((next_state_predictions.shape[0], 1))
              
              next_state_predictions = torch.cat ( [ next_state_predictions, tmp ] , dim = 1)
              

      """ Predict the Reward """
      test_X = torch.cat([ state_samples[:, reward_model.input_state_index], actions ] , dim = 1 )
      
      predicted_rewards, uncertanities = generate_test_predictions(test_X, reward_model.hypernet, reward_model.target_model, task_index, no_of_models )


      """Save the Synthetic Data """
      for state_sample, action, reward, next_state, done, uncertanity in zip( state_samples, actions, predicted_rewards, next_state_predictions, done_samples, uncertanities) :
          agent_synthetic_memory.remember (state_sample, action, reward, next_state, done , uncertanity)
      
      




        