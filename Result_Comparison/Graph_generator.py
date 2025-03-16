# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 10:48:03 2025

@author: gauthambekal93
"""

import pandas as pd
import os
os.chdir("C:/Users/gauthambekal93/Research/hvac_continual_rl/Result_Comparison")

import matplotlib.pyplot as plt

model_based_rl_stage_1 = pd.read_csv("model_based_rl_stage_1.csv")

model_free_rl_stage_1 = pd.read_csv("model_free_rl_stage_1.csv")


model_based_rl_stage_2 = pd.read_csv("model_based_rl_stage_2.csv")

model_free_rl_stage_2 = pd.read_csv("model_free_rl_stage_2.csv")


model_based_rl_stage_3 = pd.read_csv("model_based_rl_stage_3.csv")


model_free_rl_stage_1_jan =  model_free_rl_stage_1.loc[ (model_free_rl_stage_1["episode"]<=15 ) & ((model_free_rl_stage_1["Date"]=='January 16, 2024' ))]

model_free_rl_stage_1_apr = model_free_rl_stage_1.loc[ (model_free_rl_stage_1["episode"]<=15 ) & ((model_free_rl_stage_1["Date"]=='April 17, 2024' ))]

model_based_rl_stage_1_jan = model_based_rl_stage_1.loc[ (model_based_rl_stage_1["episode"]<=15 ) & ((model_based_rl_stage_1["Date"]=='January 16, 2024' ))]

model_based_rl_stage_1_apr =  model_based_rl_stage_1.loc[ (model_based_rl_stage_1["episode"]<=15 ) & ((model_based_rl_stage_1["Date"]=='April 17, 2024' ))]


model_free_rl_stage_2_jan =  model_free_rl_stage_2.loc[ (model_free_rl_stage_2["episode"]<=15 ) & ((model_free_rl_stage_2["Date"]=='January 16, 2024' ))]

model_free_rl_stage_2_apr = model_free_rl_stage_2.loc[ (model_free_rl_stage_2["episode"]<=15 ) & ((model_free_rl_stage_2["Date"]=='April 17, 2024' ))]

model_based_rl_stage_2_jan = model_based_rl_stage_2.loc[ (model_based_rl_stage_2["episode"]<=15 ) & ((model_based_rl_stage_2["Date"]=='January 16, 2024' ))]

model_based_rl_stage_2_apr =  model_based_rl_stage_2.loc[ (model_based_rl_stage_2["episode"]<=15 ) & ((model_based_rl_stage_2["Date"]=='April 17, 2024' ))]



model_based_rl_stage_3_jan = model_based_rl_stage_3.loc[ (model_based_rl_stage_3["episode"]<=15 ) & ((model_based_rl_stage_3["Date"]=='January 16, 2024' ))]

model_based_rl_stage_3_apr =  model_based_rl_stage_3.loc[ (model_based_rl_stage_3["episode"]<=15 ) & ((model_based_rl_stage_3["Date"]=='April 17, 2024' ))]




""" ----- STAGE 1 ----"""

episodes= list(model_based_rl_stage_1_jan["episode"])

plt.plot(episodes, list(model_free_rl_stage_1_jan["extrinsic_reward"] ), label="Model Free Rewards", marker='o')
plt.plot(episodes, list( model_based_rl_stage_1_jan["extrinsic_reward"] ) , label="Model Based Rewards", marker='s')

plt.xlabel("episodes")
plt.ylabel("Metric Value")
plt.title("Stage 1 January Test Results")
plt.legend()
plt.grid(True)
plt.show()


episodes= list(model_based_rl_stage_1_apr["episode"])

plt.plot(episodes, list(model_free_rl_stage_1_apr["extrinsic_reward"] ), label="Model Free Rewards", marker='o')
plt.plot(episodes, list( model_based_rl_stage_1_apr["extrinsic_reward"] ) , label="Model Based Rewards", marker='s')

plt.xlabel("episodes")
plt.ylabel("Metric Value")
plt.title("Stage 1 April Test Results")
plt.legend()
plt.grid(True)
plt.show()


""" ----- STAGE 2 ----"""

episodes= list(model_based_rl_stage_2_jan["episode"])

plt.plot(episodes, list(model_free_rl_stage_2_jan["extrinsic_reward"] ), label="Model Free Rewards", marker='o')
plt.plot(episodes, list( model_based_rl_stage_2_jan["extrinsic_reward"] ) , label="Model Based Rewards", marker='s')

plt.xlabel("episodes")
plt.ylabel("Metric Value")
plt.title("Stage 2 January Test Results")
plt.legend()
plt.grid(True)
plt.show()


episodes= list(model_based_rl_stage_2_apr["episode"])

plt.plot(episodes, list(model_free_rl_stage_2_apr["extrinsic_reward"] ), label="Model Free Rewards", marker='o')
plt.plot(episodes, list( model_based_rl_stage_2_apr["extrinsic_reward"] ) , label="Model Based Rewards", marker='s')

plt.xlabel("episodes")
plt.ylabel("Metric Value")
plt.title("Stage 2 April Test Results")
plt.legend()
plt.grid(True)
plt.show()


""" ----- STAGE 3 ----"""

episodes= list(model_based_rl_stage_3_jan["episode"])

plt.plot(episodes, list(model_free_rl_stage_1_jan["extrinsic_reward"] ), label="Model Free Rewards", marker='o')
plt.plot(episodes, list( model_based_rl_stage_3_jan["extrinsic_reward"] ) , label="Model Based Rewards", marker='s')

plt.xlabel("episodes")
plt.ylabel("Metric Value")
plt.title("Stage 3 January Test Results")
plt.legend()
plt.grid(True)
plt.show()


episodes= list(model_based_rl_stage_3_apr["episode"])

plt.plot(episodes, list(model_free_rl_stage_1_apr["extrinsic_reward"] ), label="Model Free Rewards", marker='o')
plt.plot(episodes, list( model_based_rl_stage_3_apr["extrinsic_reward"] ) , label="Model Based Rewards", marker='s')

plt.xlabel("episodes")
plt.ylabel("Metric Value")
plt.title("Stage 3 April Test Results")
plt.legend()
plt.grid(True)
plt.show()


