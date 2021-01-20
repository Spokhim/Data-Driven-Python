# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:21:58 2020

@author: Artemio Soto-Breceda [artemios]

Estimation
----------
Runs the state/parameter estimation and plots results for a single channel 
at a time

Written for MATLAB by:
    Dean Freestone, Philippa Karoly 2016

This code is licensed under the MIT License 2018

"""
import numpy as np
import mat73 # To load Matlab v7.3 and later data files
import matplotlib.pyplot as plt
from tqdm import tqdm
 
# Local functions
from nmm import set_params, g, propagate_metrics

# Load data from .mat file (real ECoG data) 
#
# mat = mat73.loadmat('./data/Seizure_1.mat') # Load the data. Change this file for alternative data
# Seizure = mat["Seizure"] # 16 Channels
# Fs_from_data = False

# Load data from simulation (.csv or .txt)
#
# data = np.loadtxt('data.csv', delimiter=','); Seizure = data[1:]; Fs = data[0]
# data = np.loadtxt('C:/Users/artemios/hnn_out/data/AlphaAndBeta/dpl.txt', delimiter = '\t'); Seizure = data[:,1]; Fs = 1000/(data[1,0]-data[0,0])
# data = np.loadtxt('C:/Users/artemios/hnn_out/data/gamma_L5ping_L2ping/dpl.txt', delimiter = '\t'); Seizure = data[:,1]; Fs = 1000/(data[1,0]-data[0,0])
data = np.loadtxt('C:/Users/artemios/hnn_out/data/gamma_rhythmic_drive/dpl.txt', delimiter = '\t'); Seizure = data[:,1]; Fs = 1000/(data[1,0]-data[0,0])
Seizure = np.reshape(Seizure, [Seizure.size, 1])
Fs_from_data = True

# Chose a channel, or loop through the 16
iCh = 1 # Setting channel manually
print('Channel %02d ...' % iCh) # Print current channel
# Check channel exists
if Seizure.shape[1] < iCh:
    raise ValueError('iCh is larger than the number of channels in "Seizure".')

# Initialize input
ext_input = 10#300 # External input
input_offset = np.empty(0)

if not(Fs_from_data):
    # Generate some data
    time = 5
    Fs = 0.4e3
else:
    time = Seizure.size/Fs

# Parameter initialization
params = set_params(ext_input, input_offset, time,Fs)

if input_offset.size > 0:
    # Reset the offset
    my = np.mean(params['y'])
    input_offset = np.array([-my/0.0325]) # Scale (mV) to convert constant input to a steady-state effect on pyramidal membrane. NB DIVIDE BY 10e3 for VOLTS
    params = set_params(ext_input, input_offset)
    
# Retrive parameters into single variables
A = params['A']
B = params['B']
C = params['C']
H = params['H']
N_inputs = params['N_inputs']
N_samples = params['N_samples']
N_states = params['N_states']
N_syn = params['N_syn']
Q = params['Q']
R = params['R']
v0 = params['v0']
varsigma = params['varsigma']
xi = params['xi']
y = params['y']

xi_hat_init = np.mean( params['xi'][:, np.int(np.round(N_samples/2))-1:] , axis = 1)
P_hat_init = 10 * np.cov(params['xi'][:, np.int(np.round(N_samples/2))-1:])
P_hat_init[2*N_syn:, 2*N_syn:] = np.eye(N_syn + N_inputs) * 10e-2

# Set initial conditions for the Kalman Filter
xi_hat = np.zeros([N_states, N_samples])
P_hat = np.zeros([N_states, N_states, N_samples])
P_diag = np.zeros([N_states, N_samples])
                   
xi_hat[:,0] = xi_hat_init
P_hat[:,:,0] = P_hat_init

anneal_on = 1 # Nice!
kappa_0 = 10000
T_end_anneal = N_samples/20

# Get one channel at a time
# NB - portal data is inverted. We need to scale it to some 'reasonable'
# range for the model, but still capture amplitude differences between
# seizures
y = -0.5 * Seizure[:, iCh-1:iCh]
N_samples = y.size

# Redefine xi_hat and P because N_samples changed:
#   Set initial conditions for the Kalman Filter
xi_hat = np.zeros([N_states, N_samples])
P_hat = np.zeros([N_states, N_states, N_samples])
P_diag = np.zeros([N_states, N_samples])
xi_hat[:,0] = xi_hat_init
P_hat[:,:,0] = P_hat_init

for t in tqdm(range(1,N_samples)):
    
    xi_0p = xi_hat[:, t-1].squeeze()
    P_0p = P_hat[:, :, t-1].squeeze()
    
    # Predict
    metrics = propagate_metrics(N_syn, N_states, N_inputs, A, B, C, P_0p, xi_0p, varsigma, v0, Q)
    xi_1m = metrics['xi_1m']
    P_1m = metrics['P_1m']
    
    if (t <= T_end_anneal) & (anneal_on):
        kappa = pow(kappa_0, (T_end_anneal-t-1)/(T_end_anneal-1))
    else:
        kappa = 1
        
    # K = P_1m*H'/(H*P_1m*H' + kappa*R);
    K = np.divide(np.matmul(P_1m, np.transpose(H)), np.matmul(H, np.matmul(P_1m, np.transpose(H))) + kappa*R)
    
    # Correct
    xi_1m = np.reshape(xi_1m, [xi_1m.size, 1])
    xi_hat[:, t:t+1] = xi_1m + K*(y[t] - np.matmul(H, xi_1m))
    
    P_hat[:,:,t] = np.matmul((np.identity(N_states) - np.matmul(K,H)), P_1m)
    P_diag[:,t] = np.diag(P_hat[:,:,t])
 
    
#----------------------------------------------------------------------------#
#    Plotting results:                                                       #
#----------------------------------------------------------------------------#
plt.close('all')    
    
T = np.array([range(0,N_samples)])/Fs # Horizontal axis vector
T = np.reshape(T, [T.size, 1])

# Plot the estimated ECoG
plt.figure('ECoG')
y_ = np.transpose(np.matmul(H,xi_hat)) # ECoG
plt.plot(T, y_, linewidth = 0.5)
plt.xlabel('Time (s)')
plt.ylabel('ECoG (mV)')

# Plot the membrane potential estimates --------------------------------------
scale = 50 # Scale comes from set_params (used for numerical stabiliy)
plt.figure('Membrane potential estimates')
ax = plt.subplot(411)
plt.plot(T, xi_hat[0,:]/scale, linewidth = 0.5)
plt.title('Inhibitory -> Pyramidal')
# plt.axis([]) # replaces xlim and ylim
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)

ax = plt.subplot(412)
plt.plot(T, xi_hat[2,:]/scale, linewidth = 0.5)
plt.ylabel('Post-synaptic membrane potential (mV)')
plt.title('Pyramidal -> Inhibitory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(413)
plt.plot(T, xi_hat[4,:]/scale, linewidth = 0.5)
plt.title('Pyramidal -> Excitatory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(414)
plt.plot(T, xi_hat[6,:]/scale, linewidth = 0.5)
plt.xlabel('Time (s)')
plt.title('Excitatory -> Pyramidal')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

# Plot parameter estimates (parameters = synaptic strenghts)------------------
plt.figure('Parameter estimates')
ax = plt.subplot(511)
plt.plot(T, xi_hat[8,:], linewidth = 0.5)
plt.title('Input')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(512)
plt.plot(T, xi_hat[9,:], linewidth = 0.5)
plt.title('Inhibitory -> Pyramidal')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(513)
plt.plot(T, xi_hat[10,:], linewidth = 0.5)
plt.ylabel('Connectivity strength')
plt.title('Pyramidal -> Inhibitory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(514)
plt.plot(T, xi_hat[11,:], linewidth = 0.5)
plt.title('Pyramidal -> Excitatory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(515)
plt.plot(T, xi_hat[12,:], linewidth = 0.5)
plt.xlabel('Time (s)')
plt.title('Excitatory -> Pyramidal')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off
