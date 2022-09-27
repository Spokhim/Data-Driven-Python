# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:30:58 2020

@author: Artemio Soto-Breceda [artemios]
"""

import numpy as np
# import nmm  #hacky way of importing

# Bayesian Cramer-Rao Lower Bound
def compute_pcrb(t, f, F, H, Q, R, m0, P0, M):
    """
    Compute the Posterior CRB using Bergman iteration    
    
    Adapted from: Kelvin Layton - Feb 2013 (MATLAB version)
 
    Parameters
    ----------
    t : 
        Time vector.
    f : 
        Transition function.
    F:
        Transition matrix function. Takes the current state and returns the Jacobian.
    H : 
        Observation matrix.
    Q : 
        Process covariance.
    R : 
        Measurement covariance.
    m0 : 
        Mean of prior distribution.
    P0 : 
        Covariance of prior distribution.
    M : 
        Number of Monte Carlo samples.

    Returns
    -------
    pcrb :
        Posterior Crammer-Rao Bound

    """
    N = t.size
    N_states = m0.size
    
    # Initialize variables
    P = np.zeros([N_states, N_states, N])
    P[:,:,0] = P0
    pcrb = np.zeros([N_states, N])
    
    # Initializa trajectories
    xk = np.random.multivariate_normal(mean = m0, cov = P0, size = M)
    xk = np.transpose(xk)
    Rinv = np.linalg.inv(R)
    
    # Compute the PCRB using a Monte Carlo approximation
    for k in range(1,N):
        F_hat = np.zeros([N_states, N_states])
        Rinv_hat = np.zeros([N_states, N_states])
        
        # Additive gaussian noise
        v = np.random.multivariate_normal(mean = np.zeros(N_states), cov = Q, size = M)
        v = np.transpose(v)
        
        for i in range(0,M):
            # Sample the next time point for the current trajectory realisation
            # To do: fix the transition matrix (13 states, not 2)
            xk[:, i] = f(xk[:,i], f.delta, f.mode) + v[:,i]
            
            # Compute the PCRB ters for the current trajectory realisation
            F_hat = F_hat + F(xk[:,i], F.delta, F.mode)
            
            H_mat = H # H(xk[:,i])
            HRH = np.matmul(np.matmul(np.transpose(H_mat), Rinv), H_mat)
            Rinv_hat = Rinv_hat + HRH
        
        F_hat = F_hat/M
        Rinv_hat = Rinv_hat/M
        
        # Recursively compute the Fisher information matrix
        FPF = np.matmul(np.matmul(F_hat, P[:,:,k-1]), np.transpose(F_hat))
        FPFinv = np.linalg.inv(FPF + Q)
        P[:,:,k] = np.linalg.inv(FPFinv + Rinv_hat)
        
        # Compute PCRB at current time
        pcrb[:, k:k+1] = np.diag(P[:,:,k:k+1])
        
    return pcrb

def model_equations(mode, params, ext_input, dt):
    """
    This function implements the state space representation of the nonlinear pendulum equations.

    Parameters
    ----------
    mode : str <ignore case>
        'transition' returns the new state. 'jacobian' returns the Jacobian at the current state.
    params : 
        Parameters: xo (current state), dt (time step), e_0, R, v0, A, B, mu

    Returns
    -------
    out : 
        New state or Jacobian of current state, depending on 'mode'.

    """
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
    x = params['xi']
    y = params['y']
  
    mu = ext_input
    
    # Connectivity
    ConnectivityConst = 270     # Jansen and Rit connectivity parameters. Either 135, 270 or 675
    C1 = ConnectivityConst
    C2 = 0.8*ConnectivityConst
    C3 = 0.25*ConnectivityConst
    C4 = 0.25*ConnectivityConst
    
    # Synaptic kernel time constants
    dte = 0.010  # excitatory synaptic time constant (s)
    dti = 0.020  # inhibitory synaptic time constant (s)
    
    # States x = [ve, ze, vp1, zp1, vp2, zp2, vi, zi]
    ve = x[0:1]
    ze = x[1:2]
    
    vp1 = x[2:3]
    zp1 = x[3:4]
    
    vp2 = x[4:5]
    zp2 = x[5:6]
    
    vi = x[6:7]
    zi = x[7:8]
    
    aep = x[9:10]
    ape = x[10:11]
    api = x[11:12]
    aip = x[12:13]
    
    # Linear component (8x8 matrix)
    F = np.array([[1             , dte        , 0             , 0          , 0             , 0          , 0             , 0],
                  [-(aep**2)*dte , 1-2*aep*dte, 0             , 0          , 0             , 0          , 0             , 0],
                  [0             , 0          , 1             , dte        , 0             , 0          , 0             , 0],
                  [0             , 0          , -(ape**2)*dte , 1-2*ape*dte, 0             , 0          , 0             , 0],
                  [0             , 0          , 0             , 0          , 1             , dte        , 0             , 0],
                  [0             , 0          , 0             , 0          , -(api**2)*dte , 1-2*api*dte, 0             , 0],
                  [0             , 0          , 0             , 0          , 0             , 0          , 1             , dti],
                  [0             , 0          , 0             , 0          , 0             , 0          , -(aip**2)*dti , 1-2*aip*dti]])
    
    # Sigmoid functions
    fe = nmm.g(ve, v0, varsigma) # inhibitory population firing rate
    fi = nmm.g(vi, v0, varsigma) # excitatory population firing rate
    fp1 = nmm.g(vp1, v0, varsigma) # pyramidal population firing rate
    fp2 = nmm.g(vp2, v0, varsigma) # pyramidal population firing rate
    
    if mode.lower() == 'transition':
        # Nonlinear component
        gx = np.array([[0],
                       [],
                       [0],
                       [],
                       [0],
                       [],
                       [0],
                       []])
        
        out = F*x + gx
                
    elif mode.lower() == 'jacobian':
        # Jacobian
        out = np.array([[1, delta],
                        [-100*np.cos(x[0:1])*delta, 1]])
        
    else:
        raise ValueError('Wrong "mode" selection. Options are "transition" and "jacobian"')
    
    return out