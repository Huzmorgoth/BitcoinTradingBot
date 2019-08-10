#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 17:33:16 2019

@author: huzmorgoth
"""
#import json
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from Environment import BitcoinTradingEnv

df = pd.read_csv('/Users/huzmorgoth/Gigs/BTCBot/Dataset/BTCds.csv')
df = df.sort_values('Date')

slice_point = int(len(df)*0.7)
train_df = df[:slice_point]
test_df = df[slice_point:]

# The algorithms require a vectorized environment to run
train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train_df, 
                         commission=0, serial=False)])
test_env = DummyVecEnv([lambda: BitcoinTradingEnv(test_df, 
                        commission=0, serial=True)])

#Employing Proximal Policy Optimization
model = PPO2(MlpPolicy,
             train_env,
             verbose=1, 
             tensorboard_log="./tensorboard/")

model.learn(total_timesteps=50000)

obsTrain = train_env.reset()
obsTest = test_env.reset()

# Predicting and aligning on the Train Data
for i in range(2000):
  action, _states = model.predict(obsTrain)
  obs, rewards, done, info = train_env.step(action)
  train_env.render()
  
# Predicting the Test Data
for i in range(2000):
  action, _states = model.predict(obsTest)
  obs, rewards, done, info = test_env.step(action)
  test_env.render()  