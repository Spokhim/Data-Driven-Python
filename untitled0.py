# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 11:11:58 2020

@author: Artemio Soto-Breceda [artemios]
"""
# Libraries
import numpy as np
import mat73 # To load Matlab v7.3 and later data files

# Local functions
from set_params import set_params
 
mat = mat73.loadmat('./data/Seizure_1.mat') # Load the data. Change this file for alternative data
Seizure = mat["Seizure"] # 16 Channels

# Chose a channel, or loop through the 16
iCh = 1;    # Setting channel manually
print('Channel %02d ...' % iCh) # Print current channel

# Initialize input
input = 300
input_offset = np.empty(0)

# Generate some data
time = 5;
Fs = 0.4e3;
#TODO: Call set_params, define it first, obviously
out = set_params(1,2,3,4)

