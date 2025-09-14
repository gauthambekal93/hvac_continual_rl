# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 15:31:10 2025

@author: gauthambekal93
"""

import os
import sys
sys.path.append("C:/Users/gauthambekal93/Research/hvac_continual_rl/common/codes")
import torch
import numpy as np
import pandas as pd
import csv
import random
import time 

import json
from simulation_environments import bestest_hydronic_heat_pump

from save_results import save_models, save_train_results, save_test_results

from train_agent import initialize_agent, load_models , collect_from_actual_env, compute_target, train_critic, train_actor, update_target_critic, get_train_data


def get_configurations(data_config_path, model_config_path):

    with open(data_config_path, 'r') as f:
       
        data_params = json.load(f)
     
    with open(model_config_path, 'r') as f:
       
        model_params = json.load(f)
        
    return data_params, model_params      
    

def set_seed(seed):
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
if __name__ == '__main__':
    
    project_root = os.path.abspath( os.path.join(os.getcwd(), "..","..", ".."))
     
    data_config_path = os.path.join(project_root, "configuration_files","data", "task_1","stage_1_action_heat.json") 
    
    model_config_path = os.path.join(project_root, "configuration_files","models", "task_1", "model_free_sac", "2.json") 
    
    data_params, model_params = get_configurations(data_config_path, model_config_path)

    set_seed(model_params["seed"])
    
    env,  env_attributes  = bestest_hydronic_heat_pump( data_params, model_params)
    
    
    n_training_episodes = int( env_attributes['n_training_episodes'] )

    max_t = int(env_attributes['max_t'])

    gamma = float(env_attributes['gamma'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    metrics_path = model_params["metrics_path"]
    
    print("Device ", device)
    
    """Initialize and load the RL model and memory """
    actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_actual_memory = initialize_agent(env_attributes)

    #actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_actual_memory, last_loaded_episode = load_models(actor, actor_optimizer, critic_1, critic_optimizer_1, critic_2, critic_optimizer_2, critic_target_1, critic_target_2, agent_actual_memory, env_attributes)

    last_loaded_episode = 0
       
    if last_loaded_episode !=0:  
        
        temp = pd.read_csv(metrics_path)
        temp = temp.loc[temp['episode']<=last_loaded_episode ]
        temp.to_csv(metrics_path, index = False)
        
        filtered_df = temp.loc[temp["Type"] == "Train", ["episode", "extrinsic_reward"]]
        plot_scores_train_extrinsic = filtered_df.set_index("episode")["extrinsic_reward"].to_dict()
        
        if len(temp)>0:
           train_time = temp.loc[ temp["Type"] == "Train"]['time_steps'].iloc[-1]
        else:
             train_time = 0
    else:        
        
        with open(metrics_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_1_loss','critic_2_loss','q_predictions','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','reward'])
            file.close()
        
        plot_scores_train_extrinsic = {}
        
        train_time = 0
    
    
    plot_scores_train , train_time = {}, 0

    for i_episode in range(last_loaded_episode + 1, n_training_episodes+1): 
            
            start_time = time.time()
        
            done, time_step = 0, 1
            
            state, _ = env.reset(options = i_episode)
            
            episode_rewards, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss, q_predictions = [], [], [], [], []
            
            while done==0:
                
                if time_step % 100 == 0:  
                    print("Episode: ", i_episode, time_step, "No of memory elements: ",len(agent_actual_memory.states) )
                
                action, discrete_action, _ = actor.select_action(state [ env_attributes["state_mask"] ] )
                
                next_state, reward, done = collect_from_actual_env( env, discrete_action )  
                
                agent_actual_memory.remember( state, action, discrete_action, reward, next_state, done)
                
                time_step +=1
                
                plot_scores_train[i_episode] = plot_scores_train.get(i_episode, 0) + reward
                
                episode_rewards.append(reward) 
           
                if True:        
    
                    for _ in range(env_attributes["no_of_updates"]):
                        
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_done = get_train_data( agent_actual_memory , env_attributes )
                     
                        q_val_target = compute_target( actor, critic_target_1, critic_target_2, batch_rewards, batch_next_states, batch_done, gamma )
                        
                        l1, l2 = train_critic( critic_1, critic_2, critic_optimizer_1, critic_optimizer_2, batch_states, batch_actions , q_val_target )
                        
                        a1 = train_actor(actor, critic_1, critic_2, actor_optimizer, batch_states) 
                        
                        update_target_critic(critic_target_1, critic_target_2, critic_1, critic_2)
                        
                        episode_critic_1_loss.append(l1)
                        
                        episode_critic_2_loss.append(l2)
                        
                        episode_actor_loss.append(a1)
                        
                        q_predictions.append(q_val_target.mean().item())
                        
                state = next_state.copy()                
        
            
            train_time = train_time + ( time.time() - start_time)            
            
            save_train_results(i_episode, train_time, episode_rewards, plot_scores_train, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss, q_predictions, data_params, model_params, env_attributes , env )
            
            save_test_results(i_episode, env, env_attributes, data_params, model_params, actor) 
             
            save_models(i_episode, model_params, actor,actor_optimizer,critic_1, critic_optimizer_1 , critic_2 , critic_optimizer_2, agent_actual_memory) 
    

    
    
    
    
    