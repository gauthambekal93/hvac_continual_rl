# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 10:56:11 2025

@author: gauthambekal93
"""

import os

import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn

import torch.optim as optim
import torch.distributions as dist

from collections import deque  
import time
from operator import itemgetter

seed = 42
np.random.seed(seed)
torch.manual_seed(seed) 
random.seed(seed)  


class Agent_Memory:
    def __init__(self, buffer_size = 10000):  
        
        self.states = deque(maxlen=buffer_size) 
        self.actions = deque(maxlen=buffer_size) 
        self.discrete_actions = deque(maxlen=buffer_size) 
        self.rewards = deque(maxlen=buffer_size) 
        self.next_states = deque(maxlen=buffer_size) 
        self.done = deque(maxlen=buffer_size)
        
                                 
        
    def remember(self, state, action, discrete_action, reward, next_state, done ):
        
        self.states.append(  torch.tensor( state ).reshape(1,-1) )
    
        self.actions.append(  action.detach().clone()    )
        
        self.discrete_actions.append( torch.tensor(discrete_action).reshape(1,-1) )
        
        self.rewards.append(torch.tensor(reward).reshape(1,-1))
        
        self.next_states.append(torch.tensor(next_state ).reshape(1,-1) )
        
        self.done.append(torch.tensor(done).reshape(1,-1))
        
        
    def sample_memory(self,  sample_size = 64 , last_element = False ):  
        
        if last_element is False:
           #random_numbers = torch.randint(0, torch.cat(list(self.states), dim = 0).shape[0] , (sample_size,))   
           random_numbers = [random.randint(0, len(self.states) - 1) for _ in range(sample_size)]
        else:
            last_index = self.memory_size() - 1
            
            random_numbers = [last_index]
            
        
        return (
                torch.cat( list(itemgetter(*random_numbers)(self.states)) , dim = 0),
                torch.cat( list(itemgetter(*random_numbers)(self.actions)) , dim = 0),
                torch.cat( list(itemgetter(*random_numbers)(self.discrete_actions)) , dim = 0),
                torch.cat( list(itemgetter(*random_numbers)(self.rewards)) , dim = 0),
                torch.cat( list(itemgetter(*random_numbers)(self.next_states)) , dim = 0),
                torch.cat( list(itemgetter(*random_numbers)(self.done)) , dim = 0)
                )
        
       
    def memory_size(self):
         return len(self.states)



class Synthetic_Memory:
    def __init__(self, buffer_size = 10000):  
        
        self.states = deque(maxlen=buffer_size) 
        self.actions = deque(maxlen=buffer_size) 
        self.rewards = deque(maxlen=buffer_size) 
        self.next_states = deque(maxlen=buffer_size) 
        self.done = deque(maxlen=buffer_size)
        self.uncertanities = deque(maxlen=buffer_size)                        
        
    def remember(self, state, action, reward, next_state, done , uncertanity ):
        
        self.states.append(  state.reshape(1,-1) )
        self.actions.append( action.reshape(1,-1) )
        self.rewards.append( reward.detach().clone().reshape(1,-1) )
        self.next_states.append( next_state.detach().clone().reshape(1,-1) )
        self.done.append( done.reshape(1,-1) )
        self.uncertanities.append( uncertanity.reshape(1,-1) )
        
    def sample_memory(self,  sample_size = 64 ):  
        
        random_numbers = [random.randint(0, len(self.states) -1 ) for _ in range(sample_size)]
        
        return (
            torch.cat( list(itemgetter(*random_numbers)(self.states)) , dim = 0),
            torch.cat( list(itemgetter(*random_numbers)(self.actions)) , dim = 0),
            torch.cat( list(itemgetter(*random_numbers)(self.rewards)) , dim = 0),
            torch.cat( list(itemgetter(*random_numbers)(self.next_states)) , dim = 0),
            torch.cat( list(itemgetter(*random_numbers)(self.done)) , dim = 0), 
            torch.cat( list(itemgetter(*random_numbers)(self.uncertanities)) , dim = 0)
            )
        

            
    def memory_size(self):
         return len(self.states)
        
        
class Environment_Memory_Train():
    
    def __init__(self, buffer_size , train_test_ratio ): #was 200   
    
        self.name = "Train"
        self.train_test_ratio = train_test_ratio
        
        self.train_buffer_size = int(buffer_size * train_test_ratio )
        self.validation_buffer_size = int(buffer_size * (1 - train_test_ratio ))
        
        self.states_train = deque(maxlen = self.train_buffer_size) 
        self.actions_train = deque(maxlen = self.train_buffer_size) 
        self.rewards_train = deque(maxlen = self.train_buffer_size) 
        self.next_states_train = deque(maxlen = self.train_buffer_size) 
        
        self.states_validation = deque(maxlen = self.validation_buffer_size) 
        self.actions_validation = deque(maxlen = self.validation_buffer_size) 
        self.rewards_validation = deque(maxlen = self.validation_buffer_size) 
        self.next_states_validation = deque(maxlen = self.validation_buffer_size) 
        
        self.time_diff =  None
        
        self.buffer_size = buffer_size
        
        #self.train_index = deque(maxlen = int(buffer_size * train_test_ratio ) )  
        
        #self.validation_index = deque(maxlen = int(buffer_size * (1 - train_test_ratio )) ) 
        
        
    def remember(self, agent_actual_memory):
    
        
        #last_index = len(self.states) - 1
        
        if random.random() <= self.train_test_ratio:  
            #self.train_index.append(last_index)
            
            self.states_train.append( agent_actual_memory.states[-1] )
            
            self.actions_train.append( agent_actual_memory.actions[-1] )
            
            self.next_states_train.append( agent_actual_memory.next_states[-1] )
            
            self.rewards_train.append( agent_actual_memory.rewards[-1] )
            
        else:
            #self.validation_index.append(last_index)
            
            self.states_validation.append( agent_actual_memory.states[-1] )
            
            self.actions_validation.append( agent_actual_memory.actions[-1] )
            
            self.next_states_validation.append( agent_actual_memory.next_states[-1] )
            
            self.rewards_validation.append( agent_actual_memory.rewards[-1] )
            
    
    def memory_size(self):
         return len(self.states_train) 
     
        
    def is_full(self):
        return self.train_buffer_size <= len(self.states_train)  #the greater symbol is just in case there is some anomaly and more data is stored


    def clear_buffer(self):
        
        self.states_train.clear()
        self.action_train.clear()  
        self.rewards_train.clear()
        self.next_states_train.clear()
        
        self.states_validation.clear()
        self.action_validation.clear()  
        self.rewards_validation.clear()
        self.next_states_validation.clear()
        
        #self.train_index =[]
        #self.validation_index = []
        

