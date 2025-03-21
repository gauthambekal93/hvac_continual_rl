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
import os

import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  

#import math
#import json

#import requests
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper, DiscretizedObservationWrapper, NormalizedObservationWrapper
from boptestGymEnv import BoptestGymEnvRewardClipping, BoptestGymEnvRewardWeightDiscomfort

#import random
#from IPython.display import clear_output
#from collections import namedtuple
#from itertools import count
#from collections import deque  
#import time  

#import numpy as np

#from collections import deque

#import matplotlib.pyplot as plt
# %matplotlib inline

# PyTorch
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.distributions import Categorical

#from examples.test_and_plot import test_agent, plot_results
# Decide the state-action space of your test case
#import random

# Seed for random starting times of episodes
#seed = 123456
#random.seed(seed)
# Seed for random exploration and epsilon-greedy schedule
#np.random.seed(seed)

from collections import OrderedDict

#with open('Progressive_neural_nets/all_paths.json', 'r') as openfile:
#    paths = json.load(openfile)

url = 'http://127.0.0.1:5000' 
operational_data = 'http://127.0.0.1:5000/initialize'


warmup_period = 24*3600
episode_length_test = 14*24*3600 #was 7*24*3600
warmup_period_test  = 7*24*3600

start_time_train = 5173160
#Jan 17. April 19, Nov 15, Dec 08 all for 2023
start_time_tests    = [(23-7)*24*3600, (115-7)*24*3600] #(325-7)*24*3600, (348-7)*24*3600]  # (Jan 16 to Jan 30) and (April 18 to May 02) 

#start_time_tests    = [(325-7)*24*3600, (348-7)*24*3600]  

max_episode_length = 14*24*3600 #was 7*24*3600
random_start_time = True # was False 
step_period = 900
render_episodes = False # False

predictive_period =  2*3600   #2700 #was 24*3600 #None # 2*3600  
regressive_period = None #None #2700 #was 6*3600 #None



def bestest_hydronic_heat_pump():
    

    
  
    actions = ['oveHeaPumY_u', 'oveFan_u', 'ovePum_u'] 
    #no_of_action_types = len(actions)
    points = ["reaTZon_y","reaTSetHea_y","reaTSetCoo_y","oveHeaPumY_u", 'oveFan_u','ovePum_u', "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y"]
    

    #o'nly time and realTZon_y was used
    '''
    observations          = OrderedDict([('time',(0,604800)),
                             ('reaTZon_y',(280.,310.)),
                             ('TDryBul',(265,303)),
                             ('HDirNor',(0,862)),
                             ('InternalGainsRad[1]',(0,219)),
                             ('PriceElectricPowerConstant',(-0.4,0.4)),
                             ('LowerSetp[1]',(280.,310.)),
                             ('UpperSetp[1]',(280.,310.))])
    '''
    observations          = OrderedDict(
                            [('time',(0,604800)),
                             ('reaTZon_y',(280.,310.)),
                             ('TDryBul',(265,303)),
                             ('reaTSetCoo_y', (296, 303) ),
                             ('reaTSetHea_y', (288, 295) ),
                             #('LowerSetp[1]',(280.,310.)),
                             #('UpperSetp[1]',(280.,310.))
                             ])       
    
    excluding_periods = []
    for start_time_test in start_time_tests:
         excluding_periods.append((start_time_test, start_time_test+episode_length_test))
         
     # Summer period (from June 21st till September 22nd). 
     # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  # June 23 to Sept 26 its hot so we dont need to learn anything since heating system is off
    
    n_bins_act = 10 #was 1 #was 10 #was 1
    
        
    n_training_episodes = 500
    
    
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
        

   #obs = env.reset(options = 1)[0] 
        
    obs_dim = 13 #obs.shape[0]
    
    excluded_states = ["reaTSetHea_y","reaTSetCoo_y"] # "LowerSetp[1]", "UpperSetp[1]" 
    
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
                         "agent_h_size": str(100), #was str(int((obs_dim//2))),  #was 50, was 5, was 100, was 100
                         "n_training_episodes": str(500), #was 55, 200
                         "max_t": str(max_episode_length // step_period), #was 2000   
                         "gamma": str(0.99),
                          "actor_lr": 0.00005,
                          "critic_lr": 0.0002 ,
                          "no_of_action_types":len(actions),
                          "buffer_size": 35000,
                          "multiplicative_factor": 1, #was 1
                          "batch_size": 1024,
                          "no_of_updates": 1, 
                           "state_mask": state_masks,
                           "action_mask": [True, True, True],
                           "type": "bestest_hydronic_heat_pump"
                        } 
    
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




def twozone_commercial_hydronic():
   
    actions = ["hydronicSystem_oveMDayZ_u","hydronicSystem_oveMNigZ_u","hydronicSystem_oveMpumCon_u","hydronicSystem_oveTHea_u"]
    #no_of_action_types = len(actions)
    points=  ["hydronicSystem_oveMDayZ_u", "hydronicSystem_oveMNigZ_u", "thermostatDayZon_oveTsetZon_u", "thermostatNigZon_oveTsetZon_u", "dayZon_reaTRooAir_y", "nigZon_reaTRooAir_y", "dayZon_reaTavgFloHea_y",               "dayZon_reaCO2RooAir_y", "weatherStation_reaWeaTDryBul_y", "weatherStation_reaWeaHDirNor_y", "hydronicSystem_oveMpumCon_u", "hydronicSystem_oveTHea_u"]
    
    observations = {
          'time':(0,604800),
          
          'dayZon_reaTRooAir_y': (273,320), #[K] [min=None, max=None]: Air temperature of zone Bth
          'nigZon_reaTRooAir_y': (273,320),#[K] [min=None, max=None]: Air temperature of zone Liv
          
      #    'dayZon_reaTavgFloHea_y':(273,320),
       #   'nigZon_reaTavgFloHea_y':(273,320),
          
          'dayZon_reaCO2RooAir_y':(400,1000), #[ppm] [min=None, max=None]: Zone air CO2 concentration
          'nigZon_reaCO2RooAir_y':(400,1000), #[ppm] [min=None, max=None]: Zone air CO2 concentration
          
           'Occupancy[Day]':(0,10),
           'Occupancy[Night]':(0,10),
           
           'weatherStation_reaWeaTDryBul_y':(273,320),
           
           'dayZon_reaMFloHea_y':(0, 10),  #the range i put here is arbitary
           'nigZon_reaMFloHea_y':(0, 10),
           
           'dayZon_reaTsupFloHea_y':(273,330),
           'nigZon_reaTsupFloHea_y':(273,330)
        }



    excluding_periods = []
    for start_time_test in start_time_tests:
         excluding_periods.append((start_time_test, start_time_test+episode_length_test))
     # Summer period (from June 21st till September 22nd). 
     # Excluded since no heating during this period (nothing to learn).
    excluding_periods.append((173*24*3600, 266*24*3600))  # June 23 to Sept 26 its hot so we dont need to learn anything since heating system is off


    n_bins_act = 2 #was 1


    env = BoptestGymEnv(      
                url                  = url,
                actions              = actions,
                observations         = observations,
                #predictive_period    = predictive_period,            #was commented
                #regressive_period    = regressive_period,             #was commented
                random_start_time    = random_start_time,   #was True
                excluding_periods    = excluding_periods,    #was commented
                max_episode_length   = max_episode_length, #was  7*24*3600, 
                warmup_period        = warmup_period,
                step_period          = step_period,
                #render_episodes      =  render_episodes
                )  #was False    
        
        
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act)  #action space same as original code
        

    #env = create_train_environment() 
    
    obs = env.reset()[0] 
        
    obs_dim = obs.shape[0]

    env_attributes = {
                      'state_space': str( obs_dim),
                      "actions":actions,
                      "action_space":str(env.action_space.n),
                      "action_bins": n_bins_act + 1,
                      "points":points,
                      "h_size":str(100),
                      "n_training_episodes": str(1000),
                      "n_evaluation_episodes": (10),
                      "max_t": str(1000),   
                      "gamma": str(0.99),
                      "lr": str(0.001)

                      }
    return  env,  env_attributes 







def bestest_hydronic():

    
    actions = ["oveTSetSup_u", "ovePum_u"]
    points = ["reaTRoo_y","oveTSetHea_u","oveTSetCoo_u","ovePum_u","oveTSetSup_u", "weaSta_reaWeaTDryBul_y","weaSta_reaWeaHDirNor_y"]
    
    '''
    observations = { 'time':(0,604800),
                    # 'reaCO2RooAir_y':(0,1000),   #(-100000,100000)  #was 0, 100
                     'reaTRoo_y':(280,310),  #(-1000,1000)
                     #'weaSta_reaWeaTDryBul_y':(275, 330),
                     #'weaSta_reaWeaTWetBul_y':(275,330),
                     'TDryBul':(265,303),   #'Outside drybulb temperature measurement
                     'HDirNor':(0,862),      #Direct normal radiation  'measurement
                     'InternalGainsRad[1]':(0,219),    #forecasting variable
                     'PriceElectricPowerHighlyDynamic':(-0.4,0.4),  #forecasting variable
                     'LowerSetp[1]':(280.,310.),
                     'UpperSetp[1]':(280.,310.)  
                     }
    '''
    
    observations          = OrderedDict(
                            [('time',(0,604800)),
                             ('reaTRoo_y',(280.,310.)),
                             ('TDryBul',(265,303))
                             ])       
    
    excluding_periods = []
    for start_time_test in start_time_tests:
         excluding_periods.append((start_time_test, start_time_test+episode_length_test))
     # Summer period (from June 21st till September 22nd). 
     # Excluded since no heating during this period (nothing to learn).
    #excluding_periods.append((79*24*3600, 355*24*3600))  # June 23 to Sept 26 its hot so we dont need to learn anything since heating system is off
    
    
    n_bins_act = 10    

    env = BoptestGymEnv(      
                url                  = url,
                actions              = actions,
                observations         = observations,
                predictive_period    = predictive_period,            #was commented
                regressive_period    = regressive_period,             #was commented
                random_start_time    = random_start_time,   #was True
                excluding_periods     = excluding_periods,    #was commented
                max_episode_length   = max_episode_length, #was  7*24*3600, 
                warmup_period        = warmup_period,
                step_period          = step_period
               # render_episodes      =  render_episodes,
                )  
        
        
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act)  #action space same as original code
        
    

    obs,_ = env.reset() 
        
    obs_dim = obs.shape[0]

    env_attributes = {
                         "state_space": str( obs_dim),
                         "actions": actions,
                         "action_space": str(env.action_space.n),
                         "action_bins": str(n_bins_act + 1),
                         "points":points,
                         "agent_h_size": str(100), #was str(int((obs_dim//2))),  #was 50, was 5, was 100, was 100
                         "env_h_size": str(26), 
                         "n_training_episodes": str(200), #was 55
                        # "n_evaluation_episodes": str(10),
                         "max_t": str(max_episode_length // step_period), #was 2000   
                         "gamma": str(0.99),
                          "actor_lr": str( 0.0001),  #was  0.0001  
                          "critic_lr":str(0.0005),   #was 0.0005
                          "baseline_lr":str(0.001),
                          "no_of_action_types":len(actions),
                          "oveTSetCoo_u": (296.15, 303.15)  ,
                           "oveTSetHea_u": (288.15, 296.15) ,
                           "num_tasks": 3,
                           "env_model_num_layers":3
                        }
      
    return  env,  env_attributes 
    
    
    
    
if __name__ == "__main__":    
    env, env_attributes  = bestest_hydronic_heat_pump()
    

    


