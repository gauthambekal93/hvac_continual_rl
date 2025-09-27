# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:48:10 2024

@author: gauthambekal93
"""

import os

import torch
import csv
#import os
import datetime
import time
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle 

from updated_plot_2 import test_agent, get_plots
#from simulation_environments import max_episode_length, start_time_tests,episode_length_test, warmup_period_test


year = 2024

test_jan17_time, test_apr19_time, test_nov15_time, test_dec08_time = 0, 0, 0, 0



def save_models(i_episode, actor, actor_optimizer, critic_1, critic_optimizer_1 , critic_2 , critic_optimizer_2, agent_actual_memory, agent_attributes, learnt_env_attributes = None, realT_zon_model = None, reward_model = None):
    
    rl_model_path = agent_attributes["save_model_path"]
    
    print("save policy model....")
    
    checkpoint = { 'model_state_dict': actor.state_dict(),  'optimizer_state_dict': actor_optimizer.state_dict() }
    
    torch.save(checkpoint, rl_model_path+'actor_model_'+str(i_episode)+'_.pkl')
    

    checkpoint = { 'model_state_dict': critic_1.state_dict(),  'optimizer_state_dict': critic_optimizer_1.state_dict() }
    
    torch.save(checkpoint, rl_model_path+'critic_model_1_'+str(i_episode)+'_.pkl')       
    
    
    checkpoint = { 'model_state_dict': critic_2.state_dict(),  'optimizer_state_dict': critic_optimizer_2.state_dict() }
    
    torch.save(checkpoint,  rl_model_path + 'critic_model_2_'+str(i_episode)+'_.pkl')    
    
   
    with open(rl_model_path + "agent_actual_data.pkl" , "wb") as f:
      pickle.dump(agent_actual_memory, f)
    
    if learnt_env_attributes is None:
        return 
    
    env_model_path = learnt_env_attributes["save_model_path"]
    
    checkpoint = { 'model_state_dict': realT_zon_model.hypernet.state_dict(),  'optimizer_state_dict': realT_zon_model.hypernet_optimizer.state_dict() }
      
    torch.save(checkpoint,  env_model_path + 'realT_zon_model_'+str(i_episode)+'_.pkl')    
      
    checkpoint = { 'model_state_dict': reward_model.hypernet.state_dict(),  'optimizer_state_dict': reward_model.hypernet_optimizer.state_dict() }
      
    torch.save(checkpoint,  env_model_path + 'reward_model_'+str(i_episode)+'_.pkl')
     

def plot_and_save_specific(plt, day_of_year, specific_result_path, i_episode, save_to_file, data_type ):
    
    plt.tight_layout()
    
    if save_to_file:
        #plt.savefig(exp_path + '/'+ data_type+'/'  + 'train_episode_' + str(i_episode) +'_'+str(int(day_of_year))+ '.png',  bbox_inches='tight')
        plt.savefig(specific_result_path  + data_type +'_'+ str(i_episode) +'_'+str(int(day_of_year))+ '.png',  bbox_inches='tight')
    plt.show()
    


def plot_and_save_overall(plt, env, env_type, consolidated_result_path, metrics_path, i_episode, plot_scores, max_episode_length, day_of_year, episode_actor_loss , episode_critic_1_loss, episode_critic_2_loss , q_predictions, time_taken, train_date, type_of_data):
    
    with open(metrics_path, 'a', newline='') as file:
            
        writer = csv.writer(file)   
        
        if i_episode % 1 == 0:
             
             if type_of_data=="train":
                 
                 print("=========Train=========")
                 
                 print("Actor Loss ", np.mean(episode_actor_loss) , "Critic 1 Loss ", np.mean(episode_critic_1_loss),"Critic 2 Loss ", np.mean(episode_critic_2_loss))
             
                 print("KPIs \n ",  env.get_kpis() )
             
    
             kpis = env.get_kpis()
             
             if type_of_data == "train":
                 tmp = [type_of_data, i_episode, time_taken, max_episode_length/24/3600,  train_date.strftime("%B %d, %Y") , np.mean(episode_actor_loss), np.mean(episode_critic_1_loss), np.mean(episode_critic_2_loss) , np.mean(q_predictions) ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores.values())[-1] ]
             else:
                 tmp = [type_of_data, i_episode, time_taken, max_episode_length/24/3600,  train_date.strftime("%B %d, %Y") , episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss , q_predictions ] +[kpis['cost_tot'] , kpis['emis_tot'], kpis['ener_tot'], kpis['idis_tot'], kpis['pdih_tot'],kpis['pele_tot'],kpis['pgas_tot'],kpis['tdis_tot'] ] + [list(plot_scores.values())[-1] ]
             
             writer.writerow(tmp )
             
             
             plt.title( type_of_data+'_'+env_type)
             plt.xlabel('Episodes')
             plt.ylabel(type_of_data+ ' rewards')
             plt.plot( list(plot_scores.keys()), list(plot_scores.values()) )
             plt.tight_layout()
             plt.savefig(consolidated_result_path+ '/'+type_of_data +'_'+ env_type + '.png')
             plt.close() 

        file.close()
     
        
def save_train_results(i_episode, train_time, episode_rewards, plot_scores_train, episode_actor_loss, episode_critic_1_loss, episode_critic_2_loss, q_predictions, real_env_attributes, agent_attributes , env ):    
    
    metrics_path = agent_attributes["metrics_path"]
    specific_result_path = agent_attributes["individual_train_results"]
    consolidated_result_path = agent_attributes["consolidated_results"]
    points = real_env_attributes["points"]
    max_episode_length = real_env_attributes["max_episode_length"]
    env_type = real_env_attributes["type"]
    save_to_file = agent_attributes["save_to_file"]
    
         
    plt, day_of_year = get_plots(env, episode_rewards, points = points, log_dir=os.getcwd(), save_to_file=save_to_file, testcase = env_type)
    
    train_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
    
    plot_and_save_specific(plt, day_of_year, specific_result_path, i_episode, save_to_file, data_type = "Train")
    
    plot_and_save_overall(plt, env, env_type, consolidated_result_path, metrics_path, i_episode, plot_scores_train, max_episode_length, day_of_year, episode_actor_loss , episode_critic_1_loss, episode_critic_2_loss , q_predictions, train_time, train_date, 'train')
    
  
    
    print("Completed training ...")


def save_test_results(i_episode, env, real_env_attributes, agent_attributes , actor):
    
    metrics_path = agent_attributes["metrics_path"]
    specific_result_path = agent_attributes["individual_test_results"] 
    consolidated_result_path = agent_attributes["consolidated_results"]
    
    max_episode_length = real_env_attributes["max_episode_length"]
    
    env_type = real_env_attributes["type"]
    
    save_to_file =  agent_attributes["save_to_file"]
    
    start_time_tests = real_env_attributes["start_time_tests"]
    
    df = pd.read_csv(metrics_path)
    
    df["Date"] = pd.to_datetime(df['Date'], format='mixed')
    
    
    filtered_df = df.loc[ df['Date'] == '2024-01-16', ["episode", "reward"]]
    
    plot_scores_test_jan17 = filtered_df.set_index("episode")["reward"].to_dict()
    
    temp = df.loc[ df['Date'] == '2024-01-16']['time_steps']
    
    if len(temp) ==0:
        test_jan17_time = 0
    else:
        test_jan17_time = temp.iloc[-1]
    
    start = time.time()
    
    observations, actions, rewards_test, kpis, plt, day_of_year, results = test_agent(env, real_env_attributes, agent_attributes, actor, start_time_tests[0] )
    
    test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
    
    plot_scores_test_jan17[i_episode] = sum(rewards_test)
    
    test_jan17_time = test_jan17_time + (time.time() - start)
    
    plot_and_save_specific(plt, day_of_year, specific_result_path, i_episode, save_to_file, data_type = "Test")       
    
    plot_and_save_overall(plt, env, env_type, consolidated_result_path, metrics_path, i_episode, plot_scores_test_jan17, max_episode_length, day_of_year, "NA" , "NA", "NA" , "NA", test_jan17_time, test_date, 'test_jan17')
  
    
  
    filtered_df = df.loc[ df['Date'] == '2024-04-17', ["episode", "reward"]]
    
    plot_scores_test_apr19 = filtered_df.set_index("episode")["reward"].to_dict()
    
    temp = df.loc[ df['Date'] == '2024-04-17']['time_steps']
    
    if len(temp) ==0:
        test_apr19_time = 0
    else:
        test_apr19_time = temp .iloc[-1]
        
    start = time.time()
    
    observations, actions, rewards_test, kpis, plt, day_of_year, results = test_agent(env, real_env_attributes, agent_attributes, actor , start_time_tests[1] )
    
    test_date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=int(day_of_year) - 1)
    
    plot_scores_test_apr19[i_episode] = sum(rewards_test)
    
    test_apr19_time = test_apr19_time + (time.time() - start)
    
    plot_and_save_specific(plt, day_of_year, specific_result_path, i_episode, save_to_file, data_type = "Test")
    
    plot_and_save_overall(plt, env, env_type, consolidated_result_path, metrics_path, i_episode, plot_scores_test_apr19, max_episode_length, day_of_year, "NA" , "NA", "NA" , "NA", test_apr19_time, test_date, 'test_apr19')

    
  