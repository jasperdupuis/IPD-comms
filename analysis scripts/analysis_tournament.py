# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:57:08 2021

@author: Jasper
"""

import numpy as np
import matplotlib.pyplot as plt

EXPECTED_GAME_LENGTH = 1000 #meta knowledge
DQN_RESULTS_OUTPUT_FILE = "./output/csvs/rewards.csv"

def get_reward_lists(f):
    """
    Excludes where learners played against learners
    """
    with open(f, 'r',newline='') as f:
        lines = f.readlines()
    return lines

def separate_expected_actual(lines):
    exp = []
    act = []
    for l in lines:
        if l[0] == 'E': 
            exp.append(l)
        if l[0] == 'A': 
            act.append(l)
    return act,exp

def list_to_array(lines):
    #need to get accurate n
    if lines[0][0] == "A":
        n = len(list(filter(lambda item:len(item)>1,lines[0].split(',')[1:])))    
    if lines [0][0] == "E":
        n = len(list(filter(lambda item:len(item)>3,lines[0].split(',')[1:])))    
    result = np.zeros((len(lines),n))
    counter = 0
    list_nums = []
    for l in lines:        
        l_as_list = l.split(',')
        if l_as_list[0] == 'Expected':
            list_nums = list(filter(lambda item:len(item)>3,l_as_list[1:]))
        if l_as_list[0] == 'Actual':
            list_nums = list(filter(lambda item:len(item)>1,l_as_list[1:]))
        arr = np.array(list_nums)
        arr = arr.astype(float)
        result[counter,:] = arr
        counter = counter + 1
    return result
   
def cumulative_sum(single_d_vector):    
    result = np.zeros(len(single_d_vector))
    index = 0
    for entry in result:
        result[index] = np.sum(single_d_vector[:index])
        index = index + 1
    return result

lines = get_reward_lists(DQN_RESULTS_OUTPUT_FILE)
act,exp = separate_expected_actual(lines)

aa = list_to_array(act)
ee = list_to_array(exp)

x = ee.flatten()
y = aa.flatten()

cc = cumulative_sum(y)
plt.plot(cc)

ideal = np.arange(len(x))*2.5

plt.plot(ideal);plt.plot(cc)


