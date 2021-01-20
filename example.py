# -*- coding: utf-8 -*-
"""
Simulates data using the Neural Mass Model defined in nmm.py

Created on Wed Aug  5 16:11:34 2020

@author: Artemio Soto-Breceda [artemios]
"""
import numpy as np
import matplotlib.pyplot as plt
from nmm import set_params

# Simulates data from the Neural Mass Model
time = 30
Fs = 1e3
sigma_R = 0 # 1e-4
inpt = 50

params = set_params(inpt, [], time, Fs, sigma_R)
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

# Plot
T = np.array([range(0,round(time*Fs))]) # x axis vector
T = np.reshape(T, [T.size, 1])/Fs # time vector in samples per second
y_ = np.transpose(y) # Fix y's dimension for plot
plt.figure('Simulated ECoG')
plt.plot(T, y_, 'r', linewidth = 0.5)
plt.xlabel('Time (s)')
plt.ylabel('ECoG (mV)')

# Save data <uncomment to save>
Fs_ = np.reshape(np.array(Fs), [1,1])
yy_=np.concatenate((Fs_, y_))
np.savetxt('data.csv', (yy_), delimiter=',')
np.savetxt('params.csv', (xi), delimiter=',')

# Plot the parameters --------------------------------------------------------
xi_hat = xi # np.diff(xi, axis = 1)
N_samples_ = xi_hat.shape[1]
# Compute time vector
T = np.array([range(0,N_samples_)])/Fs # Horizontal axis vector
T = np.reshape(T, [T.size, 1])

# Plot the membrane potential estimates --------------------------------------
scale = 50 # Scale comes from set_params (used for numerical stabiliy)
plt.figure('Membrane potential from simulation')
ax = plt.subplot(411)
plt.plot(T, xi_hat[0,:]/scale, 'r', linewidth = 0.5)
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
plt.plot(T, xi_hat[2,:]/scale, 'r', linewidth = 0.5)
plt.ylabel('Post-synaptic membrane potential (mV)')
plt.title('Pyramidal -> Inhibitory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(413)
plt.plot(T, xi_hat[4,:]/scale, 'r', linewidth = 0.5)
plt.title('Pyramidal -> Excitatory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(414)
plt.plot(T, xi_hat[6,:]/scale, 'r', linewidth = 0.5)
plt.xlabel('Time (s)')
plt.title('Excitatory -> Pyramidal')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off


# Plot parameters (parameters = synaptic strenghts)---------------------------
plt.figure('Parameters (true)')
ax = plt.subplot(511)
plt.plot(T, xi_hat[8,:], 'r', linewidth = 0.5)
plt.title('Input')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(512)
plt.plot(T, xi_hat[9,:], 'r', linewidth = 0.5)
plt.title('Inhibitory -> Pyramidal')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(513)
plt.plot(T, xi_hat[10,:], 'r', linewidth = 0.5)
plt.ylabel('Connectivity strength')
plt.title('Pyramidal -> Inhibitory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(514)
plt.plot(T, xi_hat[11,:], 'r', linewidth = 0.5)
plt.title('Pyramidal -> Excitatory')
plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

ax = plt.subplot(515)
plt.plot(T, xi_hat[12,:], 'r', linewidth = 0.5)
plt.xlabel('Time (s)')
plt.title('Excitatory -> Pyramidal')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off
