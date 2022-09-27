# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 14:43:53 2020

@author: Artemio Soto-Breceda [artemios]

Estimation
----------
Runs the state/parameter estimation and calculates the Bayesian Cramer-Rao lower bound

Adapted from MATLAB version by:
    Dean Freestone, Philippa Karoly 2016

    Done some further hacking things to turn into function and what-not. 

This code is licensed under the MIT License 2018

"""


import numpy as np
import mat73 # To load Matlab v7.3 and later data files
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import io 
 

# Local functions
from Data_Driven_Python.nmm import set_params, propagate_metrics
from Data_Driven_Python.crb import compute_pcrb, model_equations

def estimation_crb_func(PATH,FILE_TYPE): 

    # Options
    #FILE_TYPE = False
    TRUE_INITIAL_PARAMS = False
    EXT_INPUT = 50
    SIGMA_R = 0
    OFFSET = np.empty(0)

    # Load data
    if FILE_TYPE == "csv":
        # Load data from .csv (simulation)
        data = np.loadtxt(PATH, delimiter=',') # 1 channel only
        true_xi = np.loadtxt('params.csv', delimiter=',') # 1 channel only
        Seizure = data[1:]
        Fs = data[0]
        Seizure = np.reshape(Seizure, [Seizure.size, 1])
        Fs_from_data = True
    elif FILE_TYPE == "mat73":
        # Load data from .mat file (real ECoG data) 
        mat = mat73.loadmat(PATH) # Load the data. Change this file for alternative data
        Seizure = mat["Seizure"] # 16 Channels
        Fs_from_data = False
        TRUE_INITIAL_PARAMS = False
    else: 
        # Load from .mat file which is not mat73 and uses io package from scipy 
        mat = io.loadmat(PATH) # Load the data. Change this file for alternative data
        Seizure = mat["Data"] # 16 Channels
        Fs_from_data = False
        TRUE_INITIAL_PARAMS = False

        # Noting that there are NANs in the files, need to remove rows with NANs
        Seizure = Seizure[~np.isnan(Seizure).any(axis=1), :]
    



    # Chose a channel, or loop through the 16
    iCh = 1 # Setting channel manually
    print('Channel %02d ...' % iCh) # Print current channel
    # Check channel exists
    if Seizure.shape[1] < iCh:
        raise ValueError('iCh is larger than the number of channels in "Seizure".')

    # Initialize input
    ext_input = EXT_INPUT # External input
    input_offset = OFFSET # np.empty(0)

    if not(Fs_from_data):
        # Generate some data
        time = 5
        Fs = 0.4e3
    else:
        time = Seizure.size/Fs

    # Parameter initialization
    if TRUE_INITIAL_PARAMS:
        params = set_params(ext_input, input_offset, time, Fs, SIGMA_R)
    else:
        params = set_params(300, input_offset, time, Fs)

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
    if FILE_TYPE=="csv":
        y = Seizure[:, iCh-1:iCh]
    else:
        y = -0.5 * Seizure[:, iCh-1:iCh]    
    N_samples = y.size

    # Redefine xi_hat and P_hat because N_samples changed:
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

    # Compute time vector
    T = np.array([range(0,N_samples)])/Fs # Horizontal axis vector
    T = np.reshape(T, [T.size, 1])

    # Compute the Posterior Cramer-Rao bound
    f = model_equations; f.params = params; f.ext_input = ext_input; f.mode = 'transition' # transition function
    # F = model_equations; F.params = params; F.ext_input = ext_input; F.mode = 'jacobian' # transition matrix function

    # M = 100 # Number of Monte Carlo samples
    # R_ = np.reshape(R, [1,1])
    # pcrb = compute_pcrb(T, f, F, H, Q, R_, xi_0p, P_0p, M)
        
    #----------------------------------------------------------------------------#
    #    Plotting results:                                                       #
    #----------------------------------------------------------------------------#

    # Plot the Mean field potential (not sure if it is the MFP, but I think it is)
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
    if Fs_from_data: plt.plot(T, true_xi[0,:]/scale, linewidth = 0.5)
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
    if Fs_from_data: plt.plot(T, true_xi[2,:]/scale, linewidth = 0.5)
    plt.ylabel('Post-synaptic membrane potential (mV)')
    plt.title('Pyramidal -> Inhibitory')
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

    ax = plt.subplot(413)
    plt.plot(T, xi_hat[4,:]/scale, linewidth = 0.5)
    if Fs_from_data: plt.plot(T, true_xi[4,:]/scale, linewidth = 0.5)
    plt.title('Pyramidal -> Excitatory')
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

    ax = plt.subplot(414)
    plt.plot(T, xi_hat[6,:]/scale, linewidth = 0.5)
    if Fs_from_data: plt.plot(T, true_xi[6,:]/scale, linewidth = 0.5)
    plt.xlabel('Time (s)')
    plt.title('Excitatory -> Pyramidal')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

    # Plot parameter estimates (parameters = synaptic strenghts)------------------
    plt.figure('Parameter estimates')
    ax = plt.subplot(511)
    plt.plot(T, xi_hat[8,:], linewidth = 0.5)
    if Fs_from_data: plt.plot(T, true_xi[8,:], linewidth = 0.5)
    plt.title('Input')
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

    ax = plt.subplot(512)
    plt.plot(T, xi_hat[9,:], linewidth = 0.5)
    if Fs_from_data: plt.plot(T, true_xi[9,:], linewidth = 0.5)
    plt.title('Inhibitory -> Pyramidal')
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

    ax = plt.subplot(513)
    plt.plot(T, xi_hat[10,:], linewidth = 0.5)
    if Fs_from_data: plt.plot(T, true_xi[10,:], linewidth = 0.5)
    plt.ylabel('Connectivity strength')
    plt.title('Pyramidal -> Inhibitory')
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

    ax = plt.subplot(514)
    plt.plot(T, xi_hat[11,:], linewidth = 0.5)
    if Fs_from_data: plt.plot(T, true_xi[11,:], linewidth = 0.5)
    plt.title('Pyramidal -> Excitatory')
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off

    ax = plt.subplot(515)
    plt.plot(T, xi_hat[12,:], linewidth = 0.5)
    if Fs_from_data: plt.plot(T, true_xi[12,:], linewidth = 0.5)
    plt.xlabel('Time (s)')
    plt.title('Excitatory -> Pyramidal')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False) # Equivalent to box off
