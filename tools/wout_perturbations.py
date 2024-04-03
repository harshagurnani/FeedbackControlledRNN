'''
various assortments of perturbed readouts: generators and analysers of the perturbations

'WMP' = within-manifold perturbation; obtained by permuting regression coefficients of factors/PCs
'OMP' = outside-manifold perturbation; obtained by permuting loadings of factors/PCs

Harsha; last revised Aug 2023
'''
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import tools.analysis_helpers as ah
import scipy.linalg as ll
import numpy as np
import math
from sympy.combinatorics import Permutation

from sympy import init_printing



def get_omp( allgps, n_groups0, lin_coef, U, Wout_orig, theta_range=[20,75], gp_order=None ):
    # OMP - Instead of shuffling neurons (onto loadings) randomly, we limit the number of permutations
    # by keeping neurons into groups and move exact groups entirely

    if gp_order is None:
        gp_order = np.random.permutation(n_groups0) # shuffle the full groups
    n_gps = len(allgps)
    if n_gps>n_groups0:
        gp_order=np.hstack((gp_order,n_gps-1))
    idx = allgps[gp_order[0]]                       # take the neurons from the first group
    for gg in range(1,n_gps):
        idx = np.hstack((idx, allgps[gp_order[gg]])) # stack neuron ids from other groups


    U_new = U[:,idx]
    Wout_full = lin_coef @ U_new        # new readout

    # don't keep if angles too small/large
    theta = np.rad2deg(ll.subspace_angles(Wout_full.T, Wout_orig.T))
    keep_p = (min(theta)>theta_range[0] and max(theta)<theta_range[1])

    return idx, Wout_full, keep_p


def get_wmp( nDim, lin_coef, U, Wout_orig, theta_range=[20,75], idx=None ):
    # shuffle coefficients (lin_coef) mapping PCs (U) to velocity

    if idx is None:
        idx = np.random.permutation(nDim)
    
    w_new = lin_coef[:,idx] #* np.sqrt(px_1.explained_variance_[idx]) / np.sqrt(px_1.explained_variance_)
    Wout_full = w_new @ U   # new readout

    # angle with velocity decoder
    theta = np.rad2deg(ll.subspace_angles(Wout_full.T, Wout_orig.T))
    
    # don't keep if angles too small/large
    keep_p = (min(theta)>theta_range[0] and max(theta)<theta_range[1])
    
    return idx, Wout_full, keep_p

def get_rmp( nDim, lin_coef, U, Wout_orig, theta_range=[20,75], idx=None ):
    #random decoder perturbation - sample output weights randomly

    if idx is None:
        idx = np.random.randint(40000)
    
    np.random.seed( idx )
    Wout_full = np.random.randn( Wout_orig.shape[0], Wout_orig.shape[1] )
    for d in range(Wout_full.shape[0]):
        Wout_full[d,:] = Wout_full[d,:]*ll.norm(Wout_orig[d,:])/ll.norm(Wout_full[d,:])
    
    # angle with velocity decoder
    theta = np.rad2deg(ll.subspace_angles(Wout_full.T, Wout_orig.T))
    
    # don't keep if angles too small/large
    keep_p = (min(theta)>theta_range[0] and max(theta)<theta_range[1])
    
    return idx, Wout_full, keep_p


def dist_neurons( neu_order, n_groups0=10 ):
    nNeu = max(neu_order)+1

    gp_sz = np.int_(np.floor(nNeu/n_groups0))   # number of neurons in each full group
    n_gps = np.int_(np.ceil(nNeu/gp_sz)) # will be 10 or 11  - with leftover neurons

    allgps = [ [] for jj in range(n_gps) ]

    # distribute the top ngroups*gp_sz into n_groups randomly
    gp_id = np.random.permutation(n_groups0)
    for gg in range(1, gp_sz):              # assign every set of gp_sz neurons a group number
        gp_id = np.hstack((gp_id, np.random.permutation(n_groups0)))   
#    gp_id = np.repeat(np.arange(10), gp_sz, axis=0 )
#    gp_id = np.random.permutation(gp_id)
    if n_gps>n_groups0:                    # add leftover neurons
        gp_id = np.hstack( (gp_id, np.repeat(n_groups0,nNeu-n_groups0*gp_sz)) )
    for gg in range(n_gps):
        allgps[gg] = neu_order[gp_id==gg]           # list of neurons in each group

    return allgps, n_groups0



def get_random_permutations( nX, nperms=None ):
    '''
    Get a list of unique permutations of ngps
    '''
    p = Permutation( [jj for jj in range(nX)])
    if nperms is None:
        z = np.random.permutation(math.factorial(nX))
    else:
        z = np.random.choice(math.factorial(nX), nperms, replace=False )

    return p, z

def get_perm_order( p, rank ):
    q = p.unrank_lex( p.size, rank )
    idx = [jj^q for jj in range(p.size) ]

    return idx


def angle_rowise(A, B):
    p1 = np.einsum('ij,ij->i',A,B)
    p2 = np.einsum('ij,ij->i',A,A)
    p3 = np.einsum('ij,ij->i',B,B)
    p4 = p1 / np.sqrt(p2*p3)
    return np.arccos(np.clip(p4,-1.0,1.0))





# -- CHECK MANY PERTURBATIONS

def select_wmp_wout( nDim, lin_coef, U, wout0, theta_range=[20,75], p=None, maps=None  ):
    # shuffle coefficients (lin_coef) mapping PCs (U) to velocity

    if maps is None:
        nMaps = 100
        p2, maps = get_random_permutations( nDim, nperms=nMaps)
    else:
        nMaps = len(maps)
    
    if p is None:
        p=p2

    
    theta = np.zeros( (nMaps, wout0.shape[1]) )     # wout0 = nNeurons x nOut
    for jj in range(nMaps):
        # angle with velocity decoder
        idx = get_perm_order( p, maps[jj] )
        w_new = lin_coef[:,idx] #* np.sqrt(px_1.explained_variance_[idx]) / np.sqrt(px_1.explained_variance_)
        Wout_full = w_new @ U
        theta[jj,:] = np.rad2deg(ll.subspace_angles(Wout_full.T, wout0))

    keep_p =  (theta[:,-1]> theta_range[0]) * (theta[:,0]<theta_range[1])            # compare smallest and largest angle

    return keep_p



def select_wmp_openloopv( X, wout0, maps, lin_coef, U, use_p=True, p=None, ratio_lim=5, stim=None, limA=[20,80] , moveT=200, useTm=None ):
    ''' 
    Send list of maps or permutation ranks
    '''
    nDim = U.shape[0]
    if (p is None) and use_p:
        p = Permutation( [jj for jj in range(nDim)])

    
    # original decoder
    v0 = X @ wout0                          # trial x time x nOut
    mean_v0 = np.mean( ll.norm(v0,axis=-1) ) # speed

    if stim.shape[2]>1:
        stim = np.squeeze(stim[:,-1,:])

    if stim is not None:
        target = np.arctan2( stim[:,1], stim[:,0] )     # stim is  trial x 2 : (cos(target), sin(target))
        all_target = np.unique(target)
        use_trials = [ (target==ct) for ct in all_target ]
    else:
        use_trials = [ np.arange(X.shape[0]) ]  # all trials treated together

    if useTm is None:
        useTm = np.arange(start=220,stop=400)

    v0_list = [ v0[np.ix_(use_trials[tgt],useTm)] for tgt in range(len(use_trials)) ]

    nMaps = len(maps)
    ratio = np.zeros( nMaps )
    angles = np.zeros( (nMaps,2) )  # mean and median
    for map in range(nMaps):
        if use_p:
            idx = get_perm_order( p, maps[map] )
            w_new = lin_coef[:,idx] #* np.sqrt(px_1.explained_variance_[idx]) / np.sqrt(px_1.explained_variance_)
            wout1 = U.T @ w_new.T
        else:
            wout1 = maps[map]
        v1 = X @ wout1
        mean_v1 = np.mean( ll.norm(v1,axis=-1) ) #speed

        # check ratio of mean speeds
        ratio[map] = mean_v1/mean_v0

        # check angle between open loop velocities for all targets - in move period
        mh = np.zeros(len(use_trials))  # mean angular difference per target set
        for tgt in range(len(use_trials)):
            v1_tgt = v1[np.ix_(use_trials[tgt],useTm)]
            ntr,ntm,nout = v1_tgt.shape
            v1_tgt = np.reshape(v1_tgt, (ntr*ntm, nout) )
            v0_tgt = np.reshape( v0_list[tgt], (ntr*ntm, nout) )
            vel_angles = np.rad2deg( angle_rowise( v1_tgt, v0_tgt ) )
            mh[tgt] = np.mean( np.abs(vel_angles) )
        #print(mh)
        angles[map,:] = np.array( np.mean(mh), np.median(mh) )
        

    speed_check = (ratio >  ratio_lim[0]) * (ratio<ratio_lim[1])
    angle_check_mean = (angles[:,0] > limA[0]) * (angles[:,0] < limA[1])
    angle_check_med = (angles[:,1] > limA[0]) * (angles[:,1] < limA[1])


    return speed_check, angle_check_mean, angle_check_med, ratio



def mean_speeds(  X, stim=None, dt=1, useTm=None ):
    v0 = (X[:,1:,:] - X[:,:-1,:])*1000/dt
    nT = v0.shape[1]
    #if useTm is None:
    #    useTm = np.arange(start=200,stop=min(nT,600))
    useTm = np.arange(start=200,stop=nT)

    #v0 = vel[:,useTm,:]
    #mean_v0 = np.mean( ll.norm(v0,axis=-1) ) # speed

    if stim.shape[2]>1:
        stim = np.squeeze(stim[:,-1,:])

    if stim is not None:
        target = np.arctan2( stim[:,1], stim[:,0] )     # stim is  trial x 2 : (cos(target), sin(target))
        all_target = np.unique(target)
        use_trials = [ (target==ct) for ct in all_target ]
    else:
        use_trials = [ np.arange(v0.shape[0]) ]  # all trials treated together

    mh = np.zeros(len(use_trials))
    #print(use_trials[0])
    for tgt in range(len(use_trials)):
        v0_tgt = v0[np.ix_(use_trials[tgt],useTm)]
        speeds = ll.norm(v0_tgt,axis=-1) # </.....
        #mh[tgt] = np.mean( ll.norm(v0_tgt,axis=-1) )
        mh[tgt] = np.mean( speeds[speeds>0.05] )

   
    return mh


def select_omp_wout( nDim, lin_coef, U, wout0, stdR, theta_range=[20,75], p=None, maps=None  ):
    # OMP - Instead of shuffling neurons (onto loadings) randomly, we limit the number of permutations
    # by keeping neurons into groups and move exact groups entirely

    allgps = split_neurons_into_groups( nDim, stdR )
    n_gps = len(allgps)

    if maps is None:
        nMaps = 100
        p2, maps = get_random_permutations( nDim, nperms=nMaps)
    else:
        nMaps = len(maps)
    
    if p is None:
        p=p2

    theta = np.zeros( (nMaps, wout0.shape[1]) )     # wout0 = nNeurons x nOut
    for jj in range(nMaps):
        # get group order then reorder neurons
        gp_order = get_perm_order( p, maps[jj] )
        if n_gps>nDim:
            gp_order=np.hstack((gp_order,n_gps-1))
        idx = allgps[gp_order[0]]
        for gg in range(1,n_gps):
            idx = np.hstack((idx, allgps[gp_order[gg]]))

        U_new = U[:,idx]
        Wout_full = lin_coef @ U_new        # new readout
        # angle with velocity decoder
        #theta[jj,:] = np.rad2deg(ll.subspace_angles(Wout_full.T, wout0)) # or U.T
        theta_U = np.rad2deg(ll.subspace_angles(Wout_full.T, U.T))      # with intrinsic manifold instead?
        #print(theta_U)
        theta[jj,:] = np.array([theta_U[0], theta_U[-1]]) 

    keep_p =  (theta[:,-1]> theta_range[0]) * (theta[:,0]<theta_range[1])            # compare smallest and largest angle

    return keep_p, allgps


def split_neurons_into_groups( nDim, stdR ):
    ''' where stdR should be the amount of modulation - depends on user's choice'''
    nNeu = len(stdR)
    eps = 1e-6    
    fx = np.argsort(1/(stdR+eps)) # order by their variance/level of modulation (that should account for bein in top PCs anyway)

    # and distribute the top km into nDim=k groups randomly
    gp_sz = np.int_(np.floor(nNeu/nDim))
    n_gps = np.int_(np.ceil(nNeu/gp_sz))    # will be nDim or nDim+1
    allgps = [ [] for jj in range(n_gps) ]

    # make sure variance in each group roughly similar
    gp_id = np.random.permutation(nDim)         # distribute into nDim groups
    for gg in range(1, gp_sz):                  # do this block wise so that neurons evenly distributed based on variance
        gp_id = np.hstack((gp_id, np.random.permutation(nDim)))
    if n_gps>nDim:
        gp_id = np.hstack( (gp_id, np.repeat(nDim,nNeu-nDim*gp_sz)) )   # put the rest in the last group
    for gg in range(n_gps):
        allgps[gg] = fx[gp_id==gg]

    return allgps




def select_map_openloopv( X, wout0, maps, lin_coef, U, use_p=True, p=None, ratio_lim=5, stim=None, limA=[20,80] , moveT=200, useTm=None , mapgen='wmp', neuron_gps=None ):
    ''' 
    Send list of maps or permutation ranks, and map generator type; for custom generator, send maps themselves instead of permutation index
    - X:        Neural activity 
    - wout0:    Original/Intuitive decoder (for comparing open loop velocities) [Neurons x ID]
    - maps:     Either indices for permutation group (p) or readout weights themselves (first index is map#)
    - lin_coef: Regression coefficients (for defining intuitive decoder)
    - U:        Top PC space (used to generate intuitive decoder)
    - use_p:    Use permutation group 
    - p:        Permutation group (if None, generated using dimensionality of U)
    - ratio_lim:Lower and upper limit for ratio between open loop velocities (using X) for original and perturbed readouts
    - stim:     Stimulus series for each trial/time in X, used to define target ( angular difference between open loop V per target )
    - limA:     Limits for angular difference between open loop velocities thru original and perturbed readout
    - moveT:    Time of movement period start (to calculate open loop V for)
    - useTm:    Alternative all time indices to use to calculate open loop V
    - mapgen:   Map generator type: 'wmp' or 'omp'
    - neuron_gps: IDs of neurons in each of the groups that are permuted for OMPs
    '''

    nDim = U.shape[0]
    if (p is None) and use_p:
        p = Permutation( [jj for jj in range(nDim)])

    
    # original decoder
    v0 = X @ wout0                          # trial x time x nOut
    mean_v0 = np.mean( ll.norm(v0,axis=-1) ) # speed

    if stim.shape[2]>1:
        stim = np.squeeze(stim[:,-1,:])

    if stim is not None:
        target = np.arctan2( stim[:,1], stim[:,0] )     # stim is  trial x 2 : (cos(target), sin(target))
        all_target = np.unique(target)
        use_trials = [ (target==ct) for ct in all_target ]
    else:
        use_trials = [ np.arange(X.shape[0]) ]  # all trials treated together

    if useTm is None:
        useTm = np.arange(start=220,stop=400)

    v0_list = [ v0[np.ix_(use_trials[tgt],useTm)] for tgt in range(len(use_trials)) ]

    nMaps = len(maps)
    ratio = np.zeros( nMaps )
    angles = np.zeros( (nMaps,2) )  # mean and median
    for map in range(nMaps):
        if use_p:
            # need to send wout as nOut x nNeurons
            if mapgen == 'wmp':
                p_order = get_perm_order( p, maps[map] )
                _, wout1, _ = get_wmp( nDim, lin_coef, U, wout0.T, idx=p_order )
            elif mapgen == 'omp':
                p_order = get_perm_order( p, maps[map] )
                _, wout1, _ = get_omp( allgps=neuron_gps, n_groups0=nDim, lin_coef=lin_coef, U=U, Wout_orig=wout0.T, gp_order=p_order )
            elif mapgen == 'rmp':
                wout1 = maps[map].T    #get_rmp( nDim, lin_coef, U,  wout0.T ,    idx=maps[map]    )
            wout1 = wout1.T
        else:
            wout1 = maps[map]
        v1 = X @ wout1
        mean_v1 = np.mean( ll.norm(v1,axis=-1) ) #speed

        # check ratio of mean speeds
        ratio[map] = mean_v1/mean_v0

        # check angle between open loop velocities for all targets - in move period
        mh = np.zeros(len(use_trials))  # mean angular difference per target set
        for tgt in range(len(use_trials)):
            v1_tgt = v1[np.ix_(use_trials[tgt],useTm)]
            ntr,ntm,nout = v1_tgt.shape
            v1_tgt = np.reshape(v1_tgt, (ntr*ntm, nout) )
            v0_tgt = np.reshape( v0_list[tgt], (ntr*ntm, nout) )
            vel_angles = np.rad2deg( angle_rowise( v1_tgt, v0_tgt ) )
            mh[tgt] = np.mean( np.abs(vel_angles) )
        #print(mh)
        angles[map,:] = np.array( np.mean(mh), np.median(mh) )
        

    speed_check = (ratio >  ratio_lim[0]) * (ratio<ratio_lim[1])
    angle_check_mean = (angles[:,0] > limA[0]) * (angles[:,0] < limA[1])
    angle_check_med = (angles[:,1] > limA[0]) * (angles[:,1] < limA[1])


    return speed_check, angle_check_mean, angle_check_med, ratio



def select_rmp_wout( nDim, lin_coef, U, wout0, theta_range=[20,75], p=None, maps=None  ):
    # shuffle coefficients (lin_coef) mapping PCs (U) to velocity

    if maps is None:
        nMaps = 100
        p2, maps = get_random_permutations( nDim, nperms=nMaps)
    else:
        nMaps = len(maps)
    
    if p is None:
        p=p2

    new_maps = [ None for jj in range(nMaps) ]    
    theta = np.zeros( (nMaps, wout0.shape[1]) )     # wout0 = nNeurons x nOut
    for jj in range(nMaps):
        # angle with velocity decoder
        np.random.seed( maps[jj] )
        Wout_full = np.random.randn( wout0.shape[0], wout0.shape[1] )
        for d in range(Wout_full.shape[1]):
            Wout_full[:,d] = Wout_full[:,d]*ll.norm(wout0[:,d])/ll.norm(Wout_full[:,d])
        new_maps[jj] = Wout_full
        theta[jj,:] = np.rad2deg(ll.subspace_angles(Wout_full, wout0))

    keep_p =  (theta[:,-1]> theta_range[0]) * (theta[:,0]<theta_range[1])            # compare smallest and largest angle

    return keep_p, new_maps