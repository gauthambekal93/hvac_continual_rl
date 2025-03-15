# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 18:38:10 2024

@author: gauthambekal93
"""
import sys
sys.path.insert(0, r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V31/Code_Results_Models_V9/Code')

import os
os.chdir(r'C:/Users/gauthambekal93/Research/model_based_rl/bestest_hydronic_heat_pump/dynamic_electricity_pricing/Experiment_V31/Code_Results_Models_V9/Code')

import numpy as np
import torch
import pandas as pd
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

 
#import torch.optim as optim
#import torch.nn as nn
import json
import time
import warnings
import csv

# Filter out the specific UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.step_period to get variables from other wrappers is deprecated*")
warnings.filterwarnings("ignore", category=UserWarning, message="WARN: env.get_kpis to get variables from other wrappers is deprecated*")

from simulation_environments import bestest_hydronic_heat_pump

from save_results import save_models, save_train_results, save_test_results

from train_agent import initialize_agent, load_models , update_hyperparameters, collect_from_actual_env, compute_target, train_critic, train_actor, update_target_critic, get_train_data

from train_env_model import initialize_env_model, load_env_model, train_hypernet

from synthetic_data_collection import  create_synthetic_data

env, env_attributes, env_model_attributes  = bestest_hydronic_heat_pump()

n_training_episodes = int( env_attributes['n_training_episodes'] )

max_t = int(env_attributes['max_t'])

gamma = float(env_attributes['gamma'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device ", device)


with open('all_paths.json', 'r') as openfile:  json_data = json.load(openfile)

exp_path = json_data['experiment_path']
metrics_path = json_data['metrics_path']
rl_data_path = json_data['rl_data_path']




train_agent_start_episode = 2
rho = 0.99
alpha = 0.15


last_loaded_episode = 0

"""Initialize and load the RL model and memory """
actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_actual_memory = initialize_agent(env_attributes)

actor, actor_optimizer, critic_1 , critic_optimizer_1, critic_2 , critic_optimizer_2, critic_target_1, critic_target_2, agent_actual_memory, last_loaded_episode = load_models(actor, actor_optimizer, critic_1, critic_optimizer_1, critic_2, critic_optimizer_2, critic_target_1, critic_target_2, agent_actual_memory, env_attributes)


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
        writer.writerow(['Type','episode', 'time_steps', 'Length', 'Date', 'actor_loss', 'critic_1_loss','critic_2_loss','q_predictions','cost_tot', 'emis_tot','ener_tot','idis_tot','pdih_tot','pele_tot','pgas_tot','tdis_tot','extrinsic_reward'])
        file.close()
    
    plot_scores_train_extrinsic = {}
    
    train_time = 0
     
    
env_type = env_attributes["type"]
        


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
            
            plot_scores_train_extrinsic[i_episode] = plot_scores_train_extrinsic.get(i_episode, 0) + reward
            
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
        
            
        save_train_results(i_episode, metrics_path, env , exp_path, env_attributes["points"], train_time, episode_rewards, plot_scores_train_extrinsic, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss, q_predictions, env_type)
        
        save_test_results(i_episode, metrics_path, env, env_attributes["state_mask"], exp_path, env_attributes["points"], actor, env_type) 
        
        save_models(i_episode, exp_path, actor,actor_optimizer,critic_1, critic_optimizer_1 , critic_2 , critic_optimizer_2, agent_actual_memory, env_type)    
        





