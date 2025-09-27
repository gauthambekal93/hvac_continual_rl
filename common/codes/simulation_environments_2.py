# -*- coding: utf-8 -*-
"""
Created on Sun May 12 18:50:32 2024

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:36:01 2024

@author: gauthambekal93
"""

import random

from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper, DiscretizedObservationWrapper, NormalizedObservationWrapper
from boptestGymEnv import BoptestGymEnvRewardClipping, BoptestGymEnvRewardWeightDiscomfort

from collections import OrderedDict
import torch



def bestest_hydronic_heat_pump(data_params, model_params):
    

    
    url = data_params["urls"]["base_url"] #'http://127.0.0.1:5000' 
    #operational_data = data_params["urls"]["operational_data"] #'http://127.0.0.1:5000/initialize'
    
    
    warmup_period =data_params["time_params"]["warmup_period"] #24*3600
    episode_length_test =data_params["time_params"]["episode_length_test"] # 14*24*3600 #was 7*24*3600
    
    start_time_train = data_params["time_params"]["start_time_train"]  # 5173160
    #Jan 17. April 19, Nov 15, Dec 08 all for 2023
    start_time_tests    = data_params["time_params"]["start_time_tests"] #[(23-7)*24*3600, (115-7)*24*3600] #(325-7)*24*3600, (348-7)*24*3600]  # (Jan 16 to Jan 30) and (April 18 to May 02) 
    
    
    max_episode_length = data_params["time_params"]["max_episode_length"] #14*24*3600 #was 7*24*3600
    random_start_time = data_params["time_params"]["random_start_time"] #True # was False 
    step_period = data_params["time_params"]["step_period"]  #900
    #render_episodes =data_params["time_params"]["render_episodes"]# False # False
    
    predictive_period =  data_params["time_params"]["predictive_period"] #2*3600   #2700 #was 24*3600 #None # 2*3600  
    regressive_period = data_params["time_params"]["regressive_period"] #None #None #2700 #was 6*3600 #None
    
      
    actions = data_params["actions"] #['oveHeaPumY_u'] #had oveFan_u

    points =  data_params["points"] #["reaTZon_y","reaTSetHea_y","reaTSetCoo_y","oveHeaPumY_u", 'oveFan_u','ovePum_u', "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y"]
    


    observations          = OrderedDict(
                            [('time', data_params["observations"]["time"] ),
                             ('reaTZon_y', data_params["observations"]["reaTZon_y"] ),
                             ('TDryBul',  data_params["observations"]["TDryBul"] ),
                             ('reaTSetCoo_y', data_params["observations"]["reaTSetCoo_y"] ),
                             ('reaTSetHea_y', data_params["observations"]["reaTSetHea_y"] ),
             
                             ])       
    '''
    observations          = OrderedDict(
                            [('time',(0,604800)),
                             ('reaTZon_y',(280.,310.)),
                             ('TDryBul',(265,303)),
                             ('reaTSetCoo_y', (296, 303) ),
                             ('reaTSetHea_y', (288, 295) ),
             
                             ])      
    
    '''
    
    excluding_periods = []
    for start_time_test in start_time_tests:
         excluding_periods.append((start_time_test, start_time_test+episode_length_test))
         
    
    excluding_periods.append( data_params["excluding_periods"] )
    
    n_bins_act = data_params["n_bins_act"]  
    
    n_training_episodes = model_params["rl_agent"]["n_training_episodes"] 
    
    
    def find_train_start_time():
        
           if regressive_period is not None:
               bgn_year_margin = regressive_period
           else:
               bgn_year_margin = 0
               
           while True:
               start_time = random.randint(0 + bgn_year_margin, 3.1536e+7-bgn_year_margin)  
               
               end_time = start_time + max_episode_length
               
               tmp = [ x for x in  excluding_periods if (start_time not in range(x[0], x[1]) ) and (end_time not in range(x[0], x[1]) ) ]
                    
               if len(tmp) == len(excluding_periods) :
                   
                   return start_time
               
    train_periods= [ find_train_start_time() for i in range(n_training_episodes) ]
    
    
    env = BoptestGymEnv(      
                url                  = url,
                actions              = actions,
                observations         = observations,
                predictive_period    = predictive_period,            
                regressive_period    = regressive_period,             
                #scenario              = {'electricity_price':'dynamic'#highly_dynamic
                #                         },
                random_start_time    = random_start_time,   #was True
                excluding_periods    = excluding_periods,    #was commented
                max_episode_length   = max_episode_length, #was  7*24*3600, 
                warmup_period        = warmup_period,
                step_period          = step_period,
                start_time         = start_time_train,
                train_periods = train_periods
                #render_episodes      =  render_episodes
                )  
    
        
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act)  #action space same as original code
        
        
    #obs_dim = data_params["obs_dim"] #13
    
    excluded_states = data_params["excluded_states"] #["reaTSetHea_y","reaTSetCoo_y"] #
    
    state_masks = []
    
    for observation in env.observations:
        if [ excluded_state for excluded_state in excluded_states if excluded_state in observation ]:
            state_masks.append(False)
        else:
            state_masks.append ( True )
        
    #env_attributes should have sample and batch size seperately. sample size can change based on if synthetic data is present or not. batch size will be constant.
    real_env_attributes = {
                         "state_space": len([ state_mask for state_mask in state_masks if state_mask ]) , #str( obs_dim  - len(excluded_states) ) ,  
                         "actions": actions,
                         "action_space": env.action_space.n,
                         "action_bins": n_bins_act + 1,
                         "points":points,
                         "type": data_params["env_type"],
                         "max_episode_length": max_episode_length,
                         "start_time_tests": start_time_tests,
                         "warmup_period" : warmup_period ,
                         "episode_length_test" : episode_length_test,
                         "warmup_period_test":data_params["time_params"]["warmup_period_test"],
                         "random_start_time" : random_start_time,
                         "start_time_train": start_time_train, 
                         "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        } 
    
    agent_attributes = {
                        "hidden_size": model_params["rl_agent"]["hidden_size"] , 
                        "n_training_episodes": n_training_episodes,
                        "max_t": max_episode_length // step_period, 
                        "gamma": model_params["rl_agent"]["gamma"], 
                        "actor_lr": model_params["rl_agent"]["actor_lr"],
                        "critic_lr": model_params["rl_agent"]["critic_lr"], 
                        "rho": model_params["rl_agent"]["rho"],
                        "alpha": model_params["rl_agent"]["alpha"],
                        "no_of_action_types":len(actions),
                        "real_buffer_size": model_params["rl_agent"]["real_buffer_size"], 
                        "synthetic_buffer_size": model_params["rl_agent"]["synthetic_buffer_size"], 
                        "multiplicative_factor": model_params["rl_agent"]["multiplicative_factor"], 
                        "batch_size": model_params["rl_agent"]["batch_size"], 
                        "no_of_updates": model_params["rl_agent"]["no_of_updates"], 
                        "state_mask": state_masks,
                        "action_mask": [True if d == "True" else False for d in data_params["action_mask"]],
                        "metrics_path": model_params["rl_agent"]["metrics_path"],
                        "exp_path":  model_params["rl_agent"]["exp_path"],
                        "save_model_path":  model_params["rl_agent"]["save_model_path"],
                        "individual_train_results": model_params["rl_agent"]["individual_train_results"],
                        "individual_test_results":  model_params["rl_agent"]["individual_test_results"],
                        "consolidated_results":  model_params["rl_agent"]["consolidated_results"],
                        "save_to_file": True if model_params["rl_agent"]["save_to_file"]=="True" else False,
                        "plot": True if model_params["rl_agent"]["plot"]=="True" else False,
                        "exp_name": model_params["exp_name"] ,
                        "load_model_path": model_params["rl_agent"]["load_model_path"],
                        "resume_from_episode": model_params["rl_agent"]["resume_from_episode"]         

        }
    
    
    if model_params["rl_agent"]["algorithm_type"]=="model_free_sac":
        return env, real_env_attributes,  agent_attributes
    
    
    learnt_env_attributes = {
                        "num_tasks": model_params["learnt_world_model"]["num_tasks"],
                        "real_zone_input": model_params["learnt_world_model"]["real_zone_input"], 
                        "dry_bulb_input": model_params["learnt_world_model"]["dry_bulb_input"], 
                        "reward_model_input": model_params["learnt_world_model"]["reward_model_input"], 
                        "real_zone_output": model_params["learnt_world_model"]["real_zone_output"],
                        "dry_bulb_output": model_params["learnt_world_model"]["dry_bulb_output"] , 
                        "reward_model_output": model_params["learnt_world_model"]["reward_model_output"],
                        "lr": model_params["learnt_world_model"]["lr"],  
                        "task_index": model_params["learnt_world_model"]["task_index"],
                        "train_buffer_size": model_params["learnt_world_model"]["train_buffer_size"],
                        "epochs": model_params["learnt_world_model"]["epochs"],
                        "hypernet_old": model_params["learnt_world_model"]["hypernet_old"],
                        "batch_size": model_params["learnt_world_model"]["batch_size"],
                        "no_of_updates": model_params["learnt_world_model"]["no_of_updates"],
                        "train_test_ratio":model_params["learnt_world_model"]["train_test_ratio"],
                        "no_of_synthetic_samples": model_params["learnt_world_model"]["no_of_synthetic_samples"],
                        "hidden_size":model_params["learnt_world_model"]["hidden_size"],
                        "num_layers": model_params["learnt_world_model"]["num_layers"]    ,
                        "no_of_models": model_params["learnt_world_model"]["no_of_models"],
                        "save_model_path":  model_params["learnt_world_model"]["save_model_path"],
                        "exp_name": model_params["exp_name"] ,
                        "load_model_path": model_params["learnt_world_model"]["load_model_path"],
                        "resume_from_episode": model_params["learnt_world_model"]["resume_from_episode"] 
                        }
    
    
    return  env,  real_env_attributes, agent_attributes , learnt_env_attributes

    
    
    
    
if __name__ == "__main__":    
    env, env_attributes  = bestest_hydronic_heat_pump()
    

    
