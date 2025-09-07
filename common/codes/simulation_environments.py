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
         
     # Summer period (from June 21st till September 22nd). 
     # Excluded since no heating during this period (nothing to learn).
    #excluding_periods.append((173*24*3600, 266*24*3600))  # June 23 to Sept 26 its hot so we dont need to learn anything since heating system is off
    
    excluding_periods.append( data_params["excluding_periods"] )
    
    n_bins_act = data_params["n_bins_act"]  #10
    
    n_training_episodes = model_params["n_training_episodes"] #500
    
    
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
        
        
    obs_dim = data_params["obs_dim"] #13
    
    excluded_states = data_params["excluded_states"] #["reaTSetHea_y","reaTSetCoo_y"] #
    
    state_masks = []
    
    for observation in env.observations:
        if [ excluded_state for excluded_state in excluded_states if excluded_state in observation ]:
            state_masks.append(False)
        else:
            state_masks.append ( True )
        
    #env_attributes should have sample and batch size seperately. sample size can change based on if synthetic data is present or not. batch size will be constant.
    env_attributes = {
                         "state_space": len([ state_mask for state_mask in state_masks if state_mask ]) , #str( obs_dim  - len(excluded_states) ) ,  
                         "actions": actions,
                         "action_space": str(env.action_space.n),
                         "action_bins": str(n_bins_act + 1),
                         "points":points,
                         "agent_h_size": str(model_params["agent_h_size"]) ,#str(100), 
                         "n_training_episodes": str(n_training_episodes),
                         "max_t": str(max_episode_length // step_period), 
                         "gamma": str(model_params["gamma"]), #str(0.99),
                          "actor_lr": model_params["actor_lr"], #  0.00005,
                          "critic_lr": model_params["critic_lr"], #0.0002 ,
                          "no_of_action_types":len(actions),
                          "buffer_size": model_params["buffer_size"], #35000,
                          "multiplicative_factor": model_params["multiplicative_factor"], #1,
                          "batch_size": model_params["batch_size"],  #1024,
                          "no_of_updates": model_params["no_of_updates"], #1, 
                           "state_mask": state_masks,
                           "action_mask": [True if d == "True" else False for d in data_params["action_mask"]], #[True, False, False],
                           "type": data_params["env_type"]
                        } 
    
    if model_params["algorithm_type"]=="vanilla_sac":
        return env,  env_attributes
    
    env_model_attributes = {
                        "num_tasks": 3,
                        "num_layers":3,
                        "real_zone_input": ( obs_dim - 3 ) + 3, #-3 is because we donot use time, "reaTSetHea_y","reaTSetCoo_y" to predict next state, +3 is for actions
                        "dry_bulb_input": ( obs_dim - 4 ), #-4 is because we donot use time, "reaTZon_y", "reaTSetHea_y","reaTSetCoo_y" to predict next state
                        "reward_model_input": 6, # 3 beacuse we use  "reaTZon_y", "reaTSetHea_y","reaTSetCoo_y" and another 3 for actions 
                        "hidden_layers": 32,
                         "lr": 0.0001,  
                         "real_zone_output": 1,
                         "dry_bulb_output": 1 , 
                         "reward_model_output": 1,
                         "task_index": 0,
                         "epochs": 301, #was 501
                         "hypernet_old":None,
                         "buffer_size": 4035, #was 4000, 5000, 1400 is about 1 episode, to keep 4035 (slightly above 1344*3 )
                         "batch_size": 20, #was 20,
                         "train_test_ratio":0.80,

                       }
    
    return  env,  env_attributes , env_model_attributes

    
    
    
    
if __name__ == "__main__":    
    env, env_attributes  = bestest_hydronic_heat_pump()
    

    


