"""
Create stim cues and hold/go

    
Harsha Gurnani, 2023
"""
import numpy as np
from math import sqrt

def stim_gen_simple( theta, delay=100, dt=0.1, T=500, stim_delay=20 ):
    """
    Generate a 3-dimensional stimulus input. 
    - First 2 inputs are constant and cue the x, y target coordinates.
    - Dim 3 is a hold/go signal that is zero until delay, then 0.5.
    -  ------- Alternatively could it be a pulse go-cue?
    """

    nT = np.int_(T/dt)   # no. of timepoints - redundant since i always use dt=1 and just scale tau
    nS = np.int_(stim_delay/dt)

    stim = np.zeros( shape=(nT,3) )
    stim[nS:,0] = np.cos(theta)
    stim[nS:,1] = np.sin(theta)
    stim[np.int_(delay/dt):,2] = .5

    return stim

def perturb_bump( dt, T, amp=.1, sigma=10, bump_delay=150):
    """
    Generate a 2D perturbation as a Gaussian bump of half-width sigma , amplitude amp and at delay bump_delay
    """
    nT = np.int_(T/dt)
    sigma = np.int_(sigma/dt)
    bump_delay = np.int_(bump_delay/dt)

    noise = np.zeros(shape=(nT,2))
    t = np.arange(nT)
    ft = amp * np.exp(-((t-bump_delay)/sigma)**2)
    ft[ft<amp/5]=0
    if np.random.random()<0.5:
        noise[:,np.random.randint(2)] = ft
    else:
        noise[:,np.random.randint(2)] = -ft

    return noise



def target_gen_simple( theta, delay, dt, T ):
    """
    Generate a 2D position signal depending on target theta and delay to go
    """

    nT = np.int_(T/dt)

    vel= np.zeros( shape=(nT,2))
    nGo = nT - np.int_(delay/dt)

    t = np.arange(nGo)*dt  # ms
    
    k = 50/1000 #  /ms speed scaling
    shift = 6/k   # shift such that at t=delay, f(t)~0
    ft = (1/(1+np.exp(-((t-shift)*k))) ) # Sigmoidal trajectory

    vel[-nGo:,0] = np.cos(theta) * ft #[-nGo:]
    vel[-nGo:,1] = np.sin(theta) * ft #[-nGo:]

    return vel


def target_gen_scaled( theta, delay, dt, T, k=0.050 ):
    """
    Generate a 2D position signal depending on target theta and delay to go
    """

    nT = np.int_(T/dt)

    vel= np.zeros( shape=(nT,2))
    nGo = nT - np.int_(delay/dt)

    t = np.arange(nGo)*dt  # ms
    
    shift = 6/k   # shift such that at t=delay, f(t)~0
    ft = (1/(1+np.exp(-((t-shift)*k))) ) # Sigmoidal trajectory

    vel[-nGo:,0] = np.cos(theta) * ft #[-nGo:]
    vel[-nGo:,1] = np.sin(theta) * ft #[-nGo:]

    return vel



def gen_data( nSamples, dt=0.05, maxT=600, fixedDelay=True, useDelay=100, delays=[50,200], stim_delay=20, add_hold=False, vel=0.050 ):
    '''
    Produce stimulus, desired trajectory(target), and end target (hold)

    velocity scale = cm/ms
    '''

    inp_dim=3
    out_dim=2
    nT = np.int_(maxT/dt)

    stim = np.zeros( shape=(nSamples,nT, inp_dim))
    target = np.zeros( (nSamples, nT, out_dim) )
    hold = np.zeros( (nSamples, nT, out_dim) )

    if not fixedDelay:
        useDelay = np.random.rand(nSamples)*(delays[1]-delays[0]) + delays[0]
    else:
        useDelay = np.ones( nSamples)* useDelay

    
    for jj in range(nSamples):
        theta=np.random.rand(1)*2*np.pi
        stim[jj,:,:] = stim_gen_simple( theta, useDelay[jj], dt, maxT, stim_delay=stim_delay )
        target[jj,:,:] = target_gen_scaled( theta, useDelay[jj], dt, maxT, k=vel )
        go_idx = np.int_( useDelay[jj]/dt)
        hold[jj, go_idx:, 0] = np.cos(theta)
        hold[jj, go_idx:, 1] = np.sin(theta)

    if add_hold:
        return stim, target, useDelay, hold
    else:
        return stim, target, useDelay


def gen_data_2( nSamples, dt=0.05, maxT=600, fixedDelay=True, useDelay=100, delays=[50,200], stim_delay=20, vel=0.050):
    inp_dim=3
    out_dim=2
    nT = np.int_(maxT/dt)

    stim = np.zeros( shape=(nSamples,nT, inp_dim))
    target = np.zeros( (nSamples, nT, out_dim) )
    holdtarget = np.zeros( (nSamples, nT, out_dim) )

    if not fixedDelay:
        useDelay = np.random.rand(nSamples)*(delays[1]-delays[0]) + delays[0]
    else:
        useDelay = np.ones( nSamples)* useDelay

    
    for jj in range(nSamples):
        theta=np.random.rand(1)*2*np.pi
        stim[jj,:,:] = stim_gen_simple( theta, useDelay[jj], dt, maxT, stim_delay=stim_delay )
        target[jj,:,:] = target_gen_scaled( theta, useDelay[jj], dt, maxT, k=vel  )
        go_idx = np.int_( useDelay[jj]/dt)
        holdtarget[jj, go_idx:, 0] = np.cos(theta)
        holdtarget[jj, go_idx:, 1] = np.sin(theta)
        


    return stim, target, useDelay, holdtarget



def gen_data_discrete( nSamples, useTheta=None, dt=0.05, maxT=600, fixedDelay=True, useDelay=100, delays=[50,200], stim_delay=20, add_hold=False, vel=0.050):
    '''
    target locations can be assigned randomly from the useTheta set 
    or be sampled uniformly from [0,2*pi]
    '''
    inp_dim=3
    out_dim=2
    nT = np.int_(maxT/dt)

    stim = np.zeros( shape=(nSamples,nT, inp_dim))
    target = np.zeros( (nSamples, nT, out_dim) )
    hold = np.zeros( (nSamples, nT, out_dim) )

    if not fixedDelay:
        useDelay = np.random.rand(nSamples)*(delays[1]-delays[0]) + delays[0]
    else:
        useDelay = np.ones( nSamples)* useDelay

    if useTheta is not None:
        for jj in range(nSamples):
            theta=np.random.choice(useTheta)
            stim[jj,:,:] = stim_gen_simple( theta, useDelay[jj], dt, maxT, stim_delay=stim_delay )
            target[jj,:,:] = target_gen_scaled( theta, useDelay[jj], dt, maxT, k=vel  )
            go_idx = np.int_(useDelay[jj]/dt)
            hold[jj, go_idx:, 0] = np.cos(theta)
            hold[jj, go_idx:, 1] = np.sin(theta)

    else:
        for jj in range(nSamples):
            theta=np.random.rand(1)*2*np.pi
            stim[jj,:,:] = stim_gen_simple( theta, useDelay[jj], dt, maxT, stim_delay=stim_delay )
            target[jj,:,:] = target_gen_scaled( theta, useDelay[jj], dt, maxT, k=vel  )
            go_idx = np.int_(useDelay[jj]/dt)
            hold[jj, go_idx:, 0] = np.cos(theta)
            hold[jj, go_idx:, 1] = np.sin(theta)

    
    if add_hold:
        return stim, target, useDelay, hold
    else:
        return stim, target, useDelay

