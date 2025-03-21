# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:45:44 2024

@author: gauthambekal93
"""

'''
Common functionality to test and plot an agent

'''
import os


import numpy as np
import torch
import random

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  


import matplotlib.pyplot as plt
from scipy import interpolate
from gymnasium.core import Wrapper
import matplotlib.dates as mdates
import pandas as pd
import requests
import json



from simulation_environments import start_time_train, random_start_time
 


with open('all_paths.json', 'r') as openfile:           json_data = json.load(openfile)

exp_path = json_data['experiment_path']

def test_agent(env, state_mask, actor, start_time, episode_length, warmup_period,
               log_dir=os.getcwd(), 
               save_to_file=True, plot=False,  points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u',
                                                      'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'], testcase='bestest_hydronic', i_episode=0, cid = 0,  compress_features = None):
    ''' Test model agent in env.
    
    '''
        
    # Set a fixed start time
    if isinstance(env,Wrapper): 
        env.unwrapped.random_start_time   = False
        env.unwrapped.start_time          = start_time
        env.unwrapped.max_episode_length  = episode_length
        env.unwrapped.warmup_period       = warmup_period
    else:
        env.random_start_time   = False
        env.start_time          = start_time
        env.max_episode_length  = episode_length
        env.warmup_period       = warmup_period
    
    # Reset environment
    state, _ = env.reset()
    
    # Simulation loop
    done = False
    observations = []
    observations.append(state)

    actions = []
    rewards = []
    results = []
    print('Test Simulating...')
    while not done:
        
        _, discrete_action, _ = actor.select_action(state[state_mask ])
        

        state, reward, truncated, terminated, res = env.step(discrete_action)   
        observations.append(state)
        actions.append(discrete_action)
        rewards.append(reward)
        results.append(res)
        done = truncated or terminated
        
        
    kpis = env.get_kpis()
 
    if plot:
         day_of_year = plot_results(env, rewards, points, save_to_file=save_to_file, log_dir=log_dir,  testcase=testcase, i_episode=i_episode, data_type ='test')
    
    # If we want random start date for training, we need to make this True. If False, it uses a specific start date as per the initialized values 
    
    if random_start_time:
        
        if isinstance(env,Wrapper): 
            env.unwrapped.random_start_time = True
        else:
            env.random_start_time = True    
    else:
        if isinstance(env,Wrapper): 
            env.unwrapped.start_time  = start_time_train
        else:
            env.start_time = start_time_train
    
    
    return observations, actions, rewards, kpis,  day_of_year, results


def plot_results(env, rewards, points=['reaTZon_y','reaTSetHea_y','reaTSetCoo_y','oveHeaPumY_u','oveTSet_u',
                                       'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'],
                 log_dir=os.getcwd(), save_to_file=True, testcase ='bestest_hydronic_heat_pump', i_episode = 0, data_type = 'train'):
    
    if testcase == 'bestest_hydronic_heat_pump':
        if points is None:
            points = list(env.all_measurement_vars.keys()) + \
                     list(env.all_input_vars.keys())
            
        # Retrieve all simulation data
        # We use env.start_time+1 to ensure that we don't return the last 
        # point from the initialization period to don't confuse it with 
        # actions taken by the agent in a previous episode. 
        res = requests.put('{0}/results'.format(env.url), 
                            json={'point_names':points,
                                  'start_time':env.start_time+1, 
                                  'final_time':3.1536e7}).json()['payload']
        
        df = pd.DataFrame(res)
        df = create_datetime_index(df)
        df.dropna(axis=0, inplace=True)
        scenario = env.scenario
    
  
        # Project rewards into results index
        rewards_time_days = np.arange(df['time'][0], 
                                      env.start_time+env.max_episode_length,
                                      env.step_period)/3600./24.
     
        f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                                 fill_value='extrapolate')
        
        res_time_days = np.array(df['time'])/3600./24.
        rewards_reindexed = f(res_time_days)
        
        if not plt.get_fignums():
            # no window(s) are open, so open a new window. 
            _, axs = plt.subplots(6, sharex=True, figsize=(12,10))
        else:
            # There is a window open, so get current figure. 
            # Combine this with plt.ion(), plt.figure()
            fig = plt.gcf()
            axs = fig.subplots(nrows=5, ncols=1, sharex=True)
                
        x_time = df.index.to_pydatetime()
    
        axs[0].plot(x_time, df['reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
        axs[0].plot(x_time, df['reaTSetHea_y'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
        axs[0].plot(x_time, df['reaTSetCoo_y'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
        axs[0].set_yticks(np.arange(15, 31, 5))
        axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
        
        axs[1].plot(x_time, df['oveHeaPumY_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[1].set_ylabel('Heat pump\nmodulation\nsignal\n( - )')
        
        axs[2].plot(x_time, df['oveFan_u'],   color='green',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[2].set_ylabel('Fan\nsignal', fontsize=8, labelpad=17)
        
        axs[3].plot(x_time, df['ovePum_u'],   color='green',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[3].set_ylabel('Circuit\npump', fontsize=8, labelpad=17)
        
        axs[4].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
        axs[4].set_ylabel('Rewards\n(-)')
        
        axs[5].plot(x_time, df['weaSta_reaWeaTDryBul_y'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
        axs[5].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
        axs[5].set_yticks(np.arange(-5, 16, 5))
        axt = axs[5].twinx()
        
        axt.plot(x_time, df['weaSta_reaWeaHDirNor_y'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
        axt.set_ylabel('Solar\nirradiation\n($W$)')
        
        #axs[3].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='RL')
        #axs[3].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
        axs[3].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
        axs[3].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
        axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
        
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        plt.tight_layout()
         
        if save_to_file:
            
            plt.savefig(exp_path + '/Results/'  + data_type + '_episode_' + str(i_episode) +'_'+str(int(res['time'][0]/3600/24))+ '.png',  bbox_inches='tight')
        
        plt.show()
        
        return res['time'][0]/3600/24    
        

    elif testcase =='bestest_hydronic':
        if points is None:
            points = list(env.all_measurement_vars.keys()) + \
                     list(env.all_input_vars.keys())
            
        # Retrieve all simulation data.  # We use env.start_time+1 to ensure that we don't return the last 
        # point from the initialization period to don't confuse it with actions taken by the agent in a previous episode. 
        
        res = requests.put('{0}/results'.format(env.url), 
                            json={'point_names':points,
                                  'start_time':env.start_time+1, 
                                  'final_time':3.1536e7}).json()['payload']

        df = pd.DataFrame(res)
        df = create_datetime_index(df)
        df.dropna(axis=0, inplace=True)
        scenario = env.scenario
    

        # Project rewards into results index
        rewards_time_days = np.arange(df['time'][0], 
                                      env.start_time+env.max_episode_length,
                                      env.step_period)/3600./24.
    
        f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                                 fill_value='extrapolate')
        
        res_time_days = np.array(df['time'])/3600./24.
        rewards_reindexed = f(res_time_days)
        
        if not plt.get_fignums():
            # no window(s) are open, so open a new window. 
            _, axs = plt.subplots(5, sharex=True, figsize=(6,4)) #was 8, 6
        else:
            # There is a window open, so get current figure. 
            # Combine this with plt.ion(), plt.figure()
            fig = plt.gcf()
            axs = fig.subplots(nrows=4, ncols=1, sharex=True)
    
        x_time = df.index.to_pydatetime()
        

        axs[0].plot(x_time, df['reaTRoo_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
        axs[0].plot(x_time, df['oveTSetHea_u'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
        axs[0].plot(x_time, df['oveTSetCoo_u'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
        axs[0].set_yticks(np.arange(15, 31, 5))
        axs[0].set_ylabel('Operative\ntemperature \n($^\circ$C)', fontsize=8 )
        
        axs[1].plot(x_time, df['oveTSetSup_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[1].set_ylabel('Temperature\nsetpoint', fontsize=8, labelpad=17)
        
        axs[2].plot(x_time, df['ovePum_u'],   color='green',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[2].set_ylabel('Pump\n Modulation', fontsize=8)
        
        axs[3].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
        axs[3].set_ylabel('Rewards\n(-)', fontsize=8)
        
       
        axs[4].plot(x_time, df['weaSta_reaWeaTDryBul_y'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
        axs[4].set_ylabel('Ambient\ntemperature\n($^\circ$C)', fontsize=8)
        axs[4].set_yticks(np.arange(-5, 16, 5))
        axt = axs[4].twinx()
        
        axt.plot(x_time, df['weaSta_reaWeaHDirNor_y'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
        axt.set_ylabel('Solar\nirradiation\n($W$)', fontsize=8)
        

        axs[4].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
        axs[4].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
        axs[4].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
        
        axs[4].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        axs[4].tick_params(axis='x', labelsize=8) 
        
        plt.tight_layout()
        
        if save_to_file:
            # showing and saving to file are incompatible
            plt.savefig(exp_path + '/Results/' +'/' +data_type + '_episode_' + str(i_episode) + '_'+str(int(res['time'][0]/3600/24))+ '.png',  bbox_inches='tight')
        
        plt.show()      
        
        return res['time'][0]/3600/24
    
    
    elif testcase =='singlezone_commercial_hydronic':
        if points is None:
            points = list(env.all_measurement_vars.keys()) + \
                     list(env.all_input_vars.keys())
            
        # Retrieve all simulation data
        # We use env.start_time+1 to ensure that we don't return the last 
        # point from the initialization period to don't confuse it with 
        # actions taken by the agent in a previous episode. 
        res = requests.put('{0}/results'.format(env.url), 
                            json={'point_names':points,
                                  'start_time':env.start_time+1, 
                                  'final_time':3.1536e7}).json()['payload']
        

        df = pd.DataFrame(res)
        df = create_datetime_index(df)
        df.dropna(axis=0, inplace=True)
        scenario = env.scenario
        
      
        # Project rewards into results index
        rewards_time_days = np.arange(df['time'][0], 
                                      env.start_time+env.max_episode_length,
                                      env.step_period)/3600./24.
     #   try:
        f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                                 fill_value='extrapolate')
        
        res_time_days = np.array(df['time'])/3600./24.
        rewards_reindexed = f(res_time_days)
        
        if not plt.get_fignums():
            # no window(s) are open, so open a new window. 
            _, axs = plt.subplots(4, sharex=True, figsize=(8,6))
        else:
            # There is a window open, so get current figure. 
            # Combine this with plt.ion(), plt.figure()
            fig = plt.gcf()
            axs = fig.subplots(nrows=4, ncols=1, sharex=True)
                
        x_time = df.index.to_pydatetime()
        
    
        axs[0].plot(x_time, df['reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
        axs[0].plot(x_time, df['oveTZonSet_u'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
        #axs[0].plot(x_time, df['oveTSetCoo_u'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
        #axs[0].plot(x_time, df['reaCO2RooAir_y']  -273.15, color='green',   linestyle='-', linewidth=1, label='_nolegend_')
        axs[0].set_yticks(np.arange(15, 31, 5))
        axs[0].set_ylabel('Zone temperature & heating setpoint \n($^\circ$C)')
        
        axs[1].plot(x_time, df['oveValCoi_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
        #axs[1].plot(x_time, df['ovePum_u'],   color='green',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[1].set_ylabel('heating coil valve control signal')
        
        axs[2].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
        axs[2].set_ylabel('Rewards\n(-)')

        
        axs[3].plot(x_time, df['reaOcc_y'] , color='green', linestyle='-', linewidth=1, label='_nolegend_')
        axs[3].set_ylabel('Occupant Count')
        
        
        axs[3].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='RL')
        axs[3].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
        axs[3].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
        axs[3].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
        axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
        
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        plt.tight_layout()
      
        if save_to_file:
            plt.savefig(exp_path + '/Results/' + data_type + '_episode_' + str(i_episode) +'_'+str(int(res['time'][0]/3600/24))+ '.png',  bbox_inches='tight')
            
        plt.show()    
          
        return res['time'][0]/3600/24
    
    if testcase == 'multizone_residential_hydronic':
        if points is None:
            points = list(env.all_measurement_vars.keys()) + \
                     list(env.all_input_vars.keys())
            
        # Retrieve all simulation data
        # We use env.start_time+1 to ensure that we don't return the last 
        # point from the initialization period to don't confuse it with 
        # actions taken by the agent in a previous episode. 
        res = requests.put('{0}/results'.format(env.url), 
                            json={'point_names':points,
                                  'start_time':env.start_time+1, 
                                  'final_time':3.1536e7}).json()['payload']
        
        df = pd.DataFrame(res)
        df = create_datetime_index(df)
        df.dropna(axis=0, inplace=True)
        scenario = env.scenario
        

        # Project rewards into results index
        rewards_time_days = np.arange(df['time'][0], 
                                      env.start_time+env.max_episode_length,
                                      env.step_period)/3600./24.
     #   try:
        f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                                 fill_value='extrapolate')
        
        res_time_days = np.array(df['time'])/3600./24.
        rewards_reindexed = f(res_time_days)
        
        if not plt.get_fignums():
            # no window(s) are open, so open a new window. 
            _, axs = plt.subplots(4, sharex=True, figsize=(8,6))
        else:
            # There is a window open, so get current figure. 
            # Combine this with plt.ion(), plt.figure()
            fig = plt.gcf()
            axs = fig.subplots(nrows=4, ncols=1, sharex=True)
                
        x_time = df.index.to_pydatetime()
    
        axs[0].plot(x_time, df['conHeaLiv_reaTZon_y']  -273.15, color='darkorange',   linestyle='-', linewidth=1, label='_nolegend_')
        axs[0].plot(x_time, df['conHeaBth_reaTZon_y'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
        axs[0].plot(x_time, df['conHeaRo1_reaTZon_y'] -273.15, color='gray',       linewidth=1, label='_nolegend_')
        axs[0].set_yticks(np.arange(15, 31, 5))
        axs[0].set_ylabel('Operative\ntemperature\n($^\circ$C)')
        
        axs[1].plot(x_time, df['conHeaLiv_oveActHea_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[1].set_ylabel('Heat \nsignal\n( - )')
        
        axs[2].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
        axs[2].set_ylabel('Rewards\n(-)')
        
        axs[3].plot(x_time, df['weatherStation_reaWeaTDryBul_y'] - 273.15, color='royalblue', linestyle='-', linewidth=1, label='_nolegend_')
        axs[3].set_ylabel('Ambient\ntemperature\n($^\circ$C)')
        axs[3].set_yticks(np.arange(-5, 16, 5))
        axt = axs[3].twinx()
        
        axt.plot(x_time, df['weatherStation_reaWeaHDirNor_y'], color='gold', linestyle='-', linewidth=1, label='$\dot{Q}_rad$')
        axt.set_ylabel('Solar\nirradiation\n($W$)')
        
        axs[3].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='RL')
        axs[3].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
        axs[3].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
        axs[3].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
        axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
        
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        plt.tight_layout()
        
        if save_to_file:
            
            plt.savefig(exp_path + '/Results/' + data_type + '_episode_' + str(i_episode) +'_'+str(int(res['time'][0]/3600/24))+ '.png',  bbox_inches='tight')
        
        plt.show()       
        return res['time'][0]/3600/24
    
    
    if testcase == 'twozone_apartment_hydronic':
        if points is None:
            points = list(env.all_measurement_vars.keys()) + \
                     list(env.all_input_vars.keys())
            
        # Retrieve all simulation data
        # We use env.start_time+1 to ensure that we don't return the last 
        # point from the initialization period to don't confuse it with 
        # actions taken by the agent in a previous episode. 
        res = requests.put('{0}/results'.format(env.url), 
                            json={'point_names':points,
                                  'start_time':env.start_time+1, 
                                  'final_time':3.1536e7}).json()['payload']
        
        df = pd.DataFrame(res)
        df = create_datetime_index(df)
        df.dropna(axis=0, inplace=True)
        scenario = env.scenario
        

        # Project rewards into results index
        rewards_time_days = np.arange(df['time'][0], 
                                      env.start_time+env.max_episode_length,
                                      env.step_period)/3600./24.
     #   try:
        f = interpolate.interp1d(rewards_time_days, rewards, kind='zero',
                                 fill_value='extrapolate')
        
        res_time_days = np.array(df['time'])/3600./24.
        rewards_reindexed = f(res_time_days)
        
        if not plt.get_fignums():
            # no window(s) are open, so open a new window. 
            _, axs = plt.subplots(4, sharex=True, figsize=(8,6))
        else:
            # There is a window open, so get current figure. 
            # Combine this with plt.ion(), plt.figure()
            fig = plt.gcf()
            axs = fig.subplots(nrows=4, ncols=1, sharex=True)
                
        x_time = df.index.to_pydatetime()
    
        axs[0].plot(x_time, df['dayZon_reaTRooAir_y']  -273.15, color='orange',   linestyle='-', linewidth=1, label='_nolegend_')
        axs[0].plot(x_time, df['thermostatDayZon_oveTsetZon_u'] -273.15, color='gray',       linewidth=1, label='Comfort setp.')
        axs[0].plot(x_time, df['thermostatNigZon_oveTsetZon_u'] -273.15, color='black',       linewidth=1, label='_nolegend_')
        axs[0].set_yticks(np.arange(15, 31, 5))
        axs[0].set_ylabel('Day Operative\ntemperature\n($^\circ$C)')
        
        axs[1].plot(x_time, rewards_reindexed, 'b', linewidth=1, label='rewards')
        axs[1].set_ylabel('Rewards\n(-)')
        
        axs[2].plot(x_time, df['hydronicSystem_oveMDayZ_u'],   color='darkorange',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[2].set_ylabel('Signal Day zone valve')
       
        axs[3].plot(x_time, df['hydronicSystem_oveMNigZ_u'],   color='gold',     linestyle='-', linewidth=1, label='_nolegend_')
        axs[3].set_ylabel('Signal Night zone valve')
        
       
        
  
        axs[3].plot([],[], color='darkorange',  linestyle='-', linewidth=1, label='RL')
        axs[3].plot([],[], color='dimgray',     linestyle='dotted', linewidth=1, label='Price')
        axs[3].plot([],[], color='royalblue',   linestyle='-', linewidth=1, label='$T_a$')
        axs[3].plot([],[], color='gold',        linestyle='-', linewidth=1, label='$\dot{Q}_{rad}$')
        axs[3].legend(fancybox=True, ncol=6, bbox_to_anchor=(1.06, -0.3)) 
        
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

        if save_to_file:
            
            plt.savefig(exp_path + '/Results/' + testcase + '/' + data_type + '_episode_' + str(i_episode) +'_'+str(int(res['time'][0]/3600/24))+ '.png',  bbox_inches='tight')
        
        plt.show()     
        return res['time'][0]/3600/24
     
        
def reindex(df, interval=60, start=None, stop=None):
    '''
    Define the index. Make sure last point is included if 
    possible. If interval is not an exact divisor of stop,
    the closest possible point under stop will be the end 
    point in order to keep interval unchanged among index.
    
    ''' 
    
    if start is None:
        start = df['time'][0]
    if stop is None:
        stop  = df['time'][-1]  
    index = np.arange(start,stop+0.1,interval).astype(int)
    df_reindexed = df.reindex(index)
    
    # Avoid duplicates from FMU simulation. Duplicates lead to 
    # extrapolation errors
    df.drop_duplicates('time',inplace=True)
    
    for key in df_reindexed.keys():
        # Use linear interpolation 
        f = interpolate.interp1d(df['time'], df[key], kind='linear',
                                 fill_value='extrapolate')
        df_reindexed.loc[:,key] = f(index)
        
    return df_reindexed


def create_datetime_index(df):
    '''
    Create a datetime index for the data
    
    '''
    
    datetime = []
    for t in df['time']:
        datetime.append(pd.Timestamp('2023/1/1') + pd.Timedelta(t,'s'))
    df['datetime'] = datetime
    df.set_index('datetime', inplace=True)
    
    return df
    






