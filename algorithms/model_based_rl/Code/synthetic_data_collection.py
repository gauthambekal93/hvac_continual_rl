# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:11:59 2025

@author: gauthambekal93
"""

import numpy as np
import torch


def generate_test_predictions(validation_X, hypernet, target_model, task_index, sample_size, device):
    
    with torch.no_grad():
        
        weights, bias = hypernet.generate_weights_bias(task_index , device, sample_size)
    
        target_model.update_params( weights , bias, sample_size )
    
        predictions = target_model.predict_target(validation_X) 
    
    final_predictions = torch.mean(predictions, dim = 0)
    
    uncertanities  = torch.abs( torch.std ( predictions , dim = 0)  ) 
    
    return  final_predictions, uncertanities




def create_synthetic_data( actor, realT_zon_model, dry_bulb_model, reward_model, agent_actual_memory, agent_synthetic_memory, real_env_attributes, agent_attributes, learnt_env_attributes, env, device, exp_name ):

      
      task_index = learnt_env_attributes["task_index"]
      
      sample_size = learnt_env_attributes["no_of_synthetic_samples"]
      
      no_of_models = learnt_env_attributes["no_of_models"]
      
      state_samples, _, _, _, next_state_samples, done_samples =  agent_actual_memory.sample_memory( sample_size = sample_size)
          
      
      
      actions, discrete_actions, _ = actor.select_action( state_samples[ :,   agent_attributes["state_mask"] ]  )
     
      actions = actions.detach().clone()
      
      if ('task_1_stage_1' == exp_name) or ('task_1_stage_3' == exp_name):
          
          temp =  np.where(discrete_actions == 0, -1 , 1)  #was 0 instead of -1 
          
          actions = torch.cat( [ actions, torch.tensor(temp).to(device) , torch.tensor(temp).to(device) ], axis = 1  )
      
      if ('task_1_stage_1' != exp_name ) and ('task_2_stage_2' != exp_name) and ('task_1_stage_3' != exp_name):
    
          raise Exception("Experiment name has been incorrectly given")
      
      """ Predict the Zone operative temperature"""
      test_X =  torch.cat([ state_samples[:, realT_zon_model.input_state_index ], actions ] , dim = 1 )
      
      predicted_realT_zon, _ = generate_test_predictions(test_X, realT_zon_model.hypernet, realT_zon_model.target_model, task_index, no_of_models, device )

      
      
      """ Obtain the Dry bulb temperature"""

      predicted_dry_bulb = next_state_samples[:, dry_bulb_model.input_state_index ][:, -1:]
      
      """ Create the next sate predictions"""
      time_diff = agent_actual_memory.next_states[0][0,0] - agent_actual_memory.states[0][0,0]
      
      time_data = state_samples[:, [0]] + time_diff
      
      next_state_predictions = torch.empty(( len(state_samples),  0 )).to(device)
      
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
              tmp = torch.empty((next_state_predictions.shape[0], 1)).to(device)
              
              next_state_predictions = torch.cat ( [ next_state_predictions, tmp ] , dim = 1)
          
            
          if "reaTSetHea_y" in observation:
              tmp = torch.empty((next_state_predictions.shape[0], 1)).to(device)
              
              next_state_predictions = torch.cat ( [ next_state_predictions, tmp ] , dim = 1)
              

      """ Predict the Reward """
      test_X = torch.cat([ state_samples[:, reward_model.input_state_index], actions ] , dim = 1 )
      
      predicted_rewards, uncertanities = generate_test_predictions(test_X, reward_model.hypernet, reward_model.target_model, task_index, no_of_models , device)


      """Save the Synthetic Data """
      for state_sample, action, reward, next_state, done, uncertanity in zip( state_samples, actions, predicted_rewards, next_state_predictions, done_samples, uncertanities) :
          agent_synthetic_memory.remember (state_sample, action, reward, next_state, done , uncertanity)
      
      




        