#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as ll
from matplotlib import pyplot as pp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter1d


'''various assortments of functions for simple calculations'''


# Calculate performance error - specifically in hold period and end period by default
def get_activity_diff( output, target, t_eval=None, use_trials=None):

    error = 0
    ntrials, nT, nOut = output.shape
    if t_eval is None:
        t_eval = np.concatenate((np.arange(start=50, stop=100), np.arange(start=nT-100, stop=nT)))
    if use_trials is None:
        use_trials = np.arange(50)

    dOut = output[np.ix_(use_trials, t_eval)]-target[np.ix_(use_trials, t_eval)]
    error= np.mean(ll.norm(dOut,axis=2))

    return error


# change in weight matrix
def get_fbk_change( model, wtype='W_fbk_0' ):
    '''
    model has pre-('params0') and post-('params1') model parameters
    compare change in norms and angle with readout.
    Optional: wtype as key - for a different wt matrix than W_fbk_0
    '''

    p0 = model['params0']
    p1 = model['params1']
    wt0 = model['params0'][wtype]
    wt1 = model['params1'][wtype]

    # Norms:
    wt_n0 = ll.norm(wt0)
    wt_n1 = ll.norm(wt1)

    wt_neu_n0 = np.mean(ll.norm(wt0, axis=0))   # input to individual neuron
    wt_neu_n1 = np.mean(ll.norm(wt1, axis=0))   # input to individual neuron

    # angle with readout wts
    ori_0 = ll.subspace_angles( wt0.T, p0['W_out_0'] )
    ori_1 = ll.subspace_angles( wt1.T, p1['W_out_0'] )

    # properties of delta W
    delW = wt1-wt0
    delwt_nm = ll.norm(delW)
    delwt_neu_nm = np.mean(ll.norm(delW, axis=0))
    ori_del = ll.subspace_angles( delW.T, p1['W_out_0'] )

    wt_res = { 'normWfb_0':wt_n0, 'neuWfb_0':wt_neu_n0, 'ori_0': ori_0, 
                'normWfb_1':wt_n1, 'neuWfb_1':wt_neu_n1, 'ori_1': ori_1, 
                'delW_nm': delwt_nm, 'delW_neu': delwt_neu_nm, 'ori_delW': ori_del
            }

    return wt_res


# Get activity pcs
def get_rnn_pcs( activity, npc=8, t_eval=None, use_trials=None ):
    ntrials, nT, nNeu = activity.shape
    
    if use_trials is None:
        use_trials=np.arange(min(ntrials,50))

    xx = activity[use_trials, 30:, :]
    nTr, nT, nNeu = xx.shape
    xx = np.reshape( xx, (nTr*nT, nNeu))

    dopc = PCA(n_components=npc)
    dopc.fit(xx)
    return dopc

# pcs but given output dictionary rather than matrix, also returns reduced matrix
def analyze_activity( a , nC=50 , nop=True, use_trials=None  ):
    Xtest = a['res']['activity1']
    ntrials, nT, nNeu = Xtest.shape

    if nop:
        ptrials = 50
    else:
        ptrials = ntrials

    starttm = 20
    if use_trials is None:
        X_use = Xtest[np.ix_(use_trials,np.arange(start=starttm,stop=nT))]
    else:
        X_use = Xtest[:ptrials,starttm:,:]

    tr,tm,nneu = X_use.shape
    X_use = np.reshape( X_use, [tr*tm, nneu])

    px1 = PCA(n_components=nC)  # no scaling
    px1.fit(X_use)

    X_red = px1.transform( X_use )
    X_red = np.reshape(X_red, [tr,tm,nC])

    return px1, X_red



# changes due to perturbation on hold period activity and latent factors -
def hold_variability( activity , nPC=5, tp=50, endtm = 30 ):
    '''
    use activity only from endtm to end 
    '''

    holdX = activity[:,-endtm:, :]
    tr, tm, nneu = holdX.shape

    holdX = np.reshape( holdX, [tr*tm, nneu])
    holdX = holdX - np.mean(holdX, axis=0)
    
    # only for non-perturbed trials
    nSets = np.int_(tr/tp)
    nop_X = holdX[:tp*tm,:]

    px1 = PCA( n_components=nPC )     # no scaling
    px1.fit(nop_X)

    holdPC = px1.transform(holdX)
    holdPC = np.reshape( holdPC, [tr, tm, nPC])
    holdX = np.reshape( holdX, [tr, tm, nneu] )
    
    pcdic = { 'holdPC':holdPC, 'holdX':holdX, 'px1':px1 }

    normX = np.zeros(nSets-1)
    normPC = np.zeros(nSets-1)
    meanX = np.zeros(nSets-1)
    meanPC = np.zeros(nSets-1)

    for jj in range(nSets-1):
        startt = tp*(jj+1)
        diffX = holdX[startt:startt+tp,:,:] - holdX[:tp,:,:]
        diffPC = holdPC[startt:startt+tp,:,:] - holdPC[:tp,:,:]

        normX[jj] = np.mean(ll.norm(diffX, axis=2))
        meanX[jj] = np.mean(ll.norm(holdX[:tp,:,:], axis=2))
        normPC[jj] = np.mean(ll.norm(diffPC, axis=2))
        meanPC[jj] = np.mean(ll.norm(holdPC[:tp,:,:]))

    diffdic = {'normX':normX, 'normPC':normPC, 'meanX':meanX, 'meanPC':meanPC}

    return pcdic, diffdic


def vec_angles( v0, v1):
    '''angle between 2 vectors - in the 2d space spanned by them '''
    v0 = v0/ll.norm(v0) # normalise
    v1 = v1/ll.norm(v1)

    x = np.dot( v0, v1)
    y = ll.norm(v1 - x*v0)

    return np.arctan2( y, x)

        
        
def get_transform_angles( W1 ):
    # W1 = [[c,s],[c,-s]]
    a0 = np.arctan2( -W1[1,0], W1[0,0] )
    a1 = np.arctan2(  W1[0,1], W1[1,1] )

    return a0, a1


# Calculate mean or maximal perpendicular displacement in specified period:
def get_perp_diff( output, target, t_eval=None, use_trials=None):

    error = 0
    ntrials, nT, nOut = output.shape
    if t_eval is None:
        t_eval = np.arange(start=200, stop=nT)
    if use_trials is None:
        use_trials = np.arange(50)

    vec = np.zeros((2,1))
    dP = np.zeros((len(use_trials), len(t_eval)))
    for jj in use_trials:
        vec[0] = - target[jj,-1,1]  # ccw = [[0,-1],[1,0]]
        vec[1] =   target[jj,-1,0]
        dP[jj:jj+1,:] = (output[jj,t_eval,:] @ vec).T
    #dOut = output[np.ix_(use_trials, t_eval)]-target[np.ix_(use_trials, t_eval)]
    #error= np.mean(ll.norm(dOut,axis=2))
    error = np.mean(np.abs(dP))

    return error



################ Functions for new Readout Selection

def get_vel_decoder( mod, use_tm=None, noise_x=0.01, scale=False, nPC=10, fitPC=True, 
                    nTest=90, testfrac=0.10, add_bias=False, use_vel=True ):
    ''' find an 'intuitive mapping' based on observed data '''
    ''' requires an object of class mainModel (from perturb_rnn) OR postModel from '''

    nT = mod.adapt_params['test_maxT']
    if use_tm is None:
        use_tm =  np.arange(start=50, stop=min(600,nT-1))
    mod.adapt_params['testing_perturb']=nTest
    mod.adapt_params['jump_amp']=0.05

    # get observations
    dic = mod.test_model()
    newp = mod.model.save_parameters()
    wout = newp['W_out_0']
    pos = dic['res']['output']
    if use_vel:
        vel = (pos[:,1:,:] - pos[:,:-1,:])*1000 # to set up vel as cm/s from /ms
    else:
        vel = pos[:,1:,:]  # position decoder

    #only use move period
    y = vel[:nTest,use_tm,:]
    nTr, nTm, nOut = y.shape
    y = np.reshape(y,(nTr*nTm,nOut))

    X = dic['res']['activity1']
    X = X[:,1:,:]
    X = X[:nTest,use_tm,:]
    nNeu = X.shape[2]
    X = np.reshape(X,(nTr*nTm,nNeu))
    X1 = X+np.random.randn(X.shape[0],X.shape[1])*noise_x  # jitter for non-active neurons -> to make sure nonzero variance in case of scaling

    if scale:
        # zscore activity
        scaler = StandardScaler()
        scaler.fit(X1)
        X_scaled= scaler.transform(X1)
    else:
        # just centre data
        X_scaled = X1 - np.mean(X1,axis=0)

    if fitPC:
        px_1 = PCA(n_components=nPC)
        px_1.fit(X_scaled)
        X_red = px_1.transform(X_scaled)
    else:
        px_1=None
        X_red = X_scaled
    
    # split data into train/test
    trialTrain = nTr - np.int_(nTr*testfrac)
    x_train = X_red[:trialTrain*nTm,:]
    x_test = X_red[trialTrain*nTm:,:]
    y_train = y[:trialTrain*nTm,:]
    y_test = y[trialTrain*nTm:,:]

    if add_bias:
        reg = LinearRegression(fit_intercept=True).fit(x_train, y_train )
    else:
        reg = LinearRegression(fit_intercept=False).fit(x_train, y_train )
    #print(reg.coef_)
    #print(reg.score(x_test,y_test))
    #print(reg.intercept_)

    return reg, px_1


def get_performance( dic, nTest=90, use_tm=None, thresh=0.2, use_trials=None ):
    ''' compute performance metrics '''

    dist_from_target = dic['res']['output'] - dic['res']['end_target']
    if np.isnan(dist_from_target).any():
        # check if sim failed
        res = {'rdist':np.nan, 'min_rdist':np.nan, 'success':np.nan, 'traj_error':np.nan, 'acq_time':np.nan}
    else:

        nTr, nTm, nx = dist_from_target.shape
        if use_tm is None:
            use_tm = np.arange(start=200, stop=nTm-1)
        if use_trials is None:
            use_trials = np.arange(nTest)
        nTr = len(use_trials)
        usedist = dist_from_target[np.ix_(use_trials,use_tm,np.arange(nx))]
        rdist = ll.norm(usedist,axis=2)
        min_rdist = np.mean(np.min(rdist,axis=1))

        mind = (rdist<thresh)       # hit?
        # trials x time
        acq_time=[]
        for tr in range(nTr):
            if sum(mind[tr,:])>0:
                acq_time.append( use_tm[mind[tr,:]][0] )
        if len(acq_time)==0:
            acq_time=1e5 # large number
        mind = np.sum(mind,axis=1)
        success = np.sum(mind>0)/nTr


        #use_trials = np.arange(start=0, stop=nTest-1)
        traj_error = get_activity_diff( dic['res']['output'] , dic['res']['target'], t_eval=use_tm, use_trials=use_trials)

        # oscillations? variance of mindist in last 100 timepoints
        usedist_100 = usedist[:,-100:,:]
        std_dist = np.std( usedist_100, axis=1) # along timepoints - in either x or y direction
        std_dist = np.mean(np.sum(std_dist,axis=-1))

        res = {'rdist':rdist, 'min_rdist':min_rdist, 'success':success, 'traj_error':traj_error, 'acq_time':acq_time, 'std_dist':std_dist }

    return res


def vec_angles_n( v0, v1):

    x = np.dot( v0, v1)/(ll.norm(v0)*ll.norm(v1))
    x = max(min(x, 1), -1) # to clamp minor numerical errors
    return np.arccos( x)



##  angular change in span of weights
def get_wt_change( wmat_1, wmat_2 , K2=5):
    """
    assumes wmat1 and wmat2 are k x N matrices, where N is dim of neural space
    if recurrent matrix, angle is calculated between K2 right eigenvectors instead
    """

    K, nNeu = wmat_1.shape

    if K<nNeu:
        angle_neu = np.zeros((nNeu))
        for nn in range(nNeu):
            angle_neu[nn] = vec_angles_n(wmat_1[:,nn], wmat_2[:,nn])

        angle_inp = np.zeros((K))
        for inp in range(K):
            angle_inp[inp] = vec_angles_n(wmat_1[inp,:], wmat_2[inp,:])

        del_norm = ll.norm( wmat_2-wmat_1, axis=1)

        return angle_inp, angle_neu, del_norm

    else:
        # recurrent matrix - change in eigenvectors
        #K2=5
        ev1, vr1 = ll.eig(wmat_1 )
        ev2, vr2 = ll.eig(wmat_2 )#vr[:,i]
        
        angle_neu = np.zeros((nNeu))
        
        for nn in range(nNeu):
            angle_neu[nn] = vec_angles_n(vr1[nn,:K2], vr2[nn,:K2])

        angle_eig = ll.subspace_angles( vr1[:,:K2], vr2[:,:K2] ) # angle between top K2- eigenvectors
        del_norm = ll.norm( wmat_2-wmat_1, axis=1)

        return angle_eig, angle_neu, del_norm, ev1, ev2


    

# compare open loop velocity through 2 readouts
def open_loop_vel( dic, wout0, wout1, nTest, ratio_lim=5, limA=[20,80] , moveT=200 ):
    X = dic['res']['activity1'][:nTest,20:,:]
    moveT = moveT-20

    X = np.reshape(X, (X.shape[0]*X.shape[1],X.shape[2]))
    vel = ll.norm(X @ wout0, axis=1)        # calculate speed
    zm = np.mean( vel )
    zr = [np.min(vel), np.max(vel)]
    zv = np.std( vel )

    vel2 = ll.norm(X @ wout1, axis=1)
    zm2 = np.mean( vel2 )
    zr2 = [np.min(vel2), np.max(vel2)]
    zv2 = np.std( vel2 )

    # check magnitude of speeds
    use_w = True
    print('zm2/zm1 = ', zm2/zm)
    ratio = (zm2/zm)
    if ratio<1/ratio_lim or ratio>ratio_lim:
        use_w = False


    # check angle between open loop velocities for all targets - in move period
    stim = dic['res']['stimulus'][:nTest,-1,:2] # take 2 inputs
    theta = np.arctan2( stim[:,1], stim[:,0] )
    all_theta = np.unique(theta)

    X2 = dic['res']['activity1'][:nTest,220:400,:]
    mh = np.ones(len(all_theta))*180
    ctr=0
    for ang in all_theta:
        tr = (theta==ang)
        # only use activity after go - since can be zero for map 1 before that
        Y = np.reshape( X2[tr,:,:], (sum(tr)*(X2.shape[1]), X2.shape[2]))        
        
        v1 = wout0.T @ Y.T      # xy x time
        v2 = wout1.T @ Y.T
        # normalise to compute dirn (ignore speed, take dirn)
        v1 = v1/ll.norm(v1,axis=0)
        v2= v2/ll.norm(v2, axis=0)

        hh  = np.array([np.rad2deg(vec_angles(v1[:,jj], v2[:,jj])) for jj in range(Y.shape[0])])
        mh[ctr] = np.mean(abs(hh))
        ctr=ctr+1

    print(mh)

    if np.mean(mh)<limA[0] or np.mean(mh)>limA[1]:
        use_w=False
    print(np.median(mh))
    if np.median(mh)<limA[0] or np.median(mh)>limA[1]:
        use_w=False
    
    return use_w




def get_vec_amp( weights, vectors, k=None, nShuf=300 ):
    nVec, nX = vectors.shape
    # weights should be nX * nX

    ev = ll.eigvals(weights)

    # check vectors normalized:
    vectors = vectors.T/ll.norm(vectors.T, axis=0)
    vectors = vectors.T

    new_vec = vectors @ weights
    new_norm = ll.norm( new_vec, axis=1)

    
    random_vec = np.random.random((nX, nShuf))-0.5
    random_vec = random_vec/ll.norm(random_vec, axis=0)
    random_vec = random_vec.T

    rand_out = random_vec @ weights
    rand_norm = ll.norm(rand_out, axis=1)

    if k is not None:
        ratio = new_norm/ev[k]
    else:
        ratio = new_norm/ev[0] # fraction of largest eigenvalue

    return ev, new_norm, ratio, rand_norm





def get_vel_decoder_0( dic, use_tm=None, noise_x=0.01, scale=False, nPC=10, fitPC=True, nTest=50, testfrac=0.10 ):
    
    wout = dic['params1']['W_out_0']
    vel = dic['res']['activity1'] @ wout
    X = dic['res']['activity1']
    nTr, nT, nNeu = X.shape
    if use_tm is None:
        use_tm =  np.arange(start=50, stop=min(600,nT))
    
    
    #only use move period
    y = vel[:nTest,use_tm,:]
    nTr, nTm, nOut = y.shape
    y = np.reshape(y,(nTr*nTm,nOut))
    
    X = X[:nTest,use_tm,:]
    X = np.reshape(X,(nTr*nTm,nNeu))
    X1 = X+np.random.randn(X.shape[0],X.shape[1])*noise_x  # jitter for non-active neurons

    if scale:
        # zscore activity
        scaler = StandardScaler()
        scaler.fit(X1)
        X_scaled= scaler.transform(X1)
    else:
        # just centre data
        X_scaled = X1 - np.mean(X1,axis=0)

    if fitPC:
        px_1 = PCA(n_components=nPC)
        px_1.fit(X_scaled)
        X_red = px_1.transform(X_scaled)
    else:
        px_1=None
        X_red = X_scaled
    
    # split data into train/test
    trialTrain = nTr - np.int_(nTr*testfrac)
    x_train = X_red[:trialTrain*nTm,:]
    x_test = X_red[trialTrain*nTm:,:]
    y_train = y[:trialTrain*nTm,:]
    y_test = y[trialTrain*nTm:,:]

    reg = LinearRegression(fit_intercept=False).fit(x_train, y_train )

    return reg, px_1



# ----------------------------------------------- #
#     REWRITTEN FUNCTIONS FOR NEW FRAMEWORK       #
###################################################


# change in weight matrix
def get_weight_change( wt0, wt1, wout ):
    '''
    compare change in norms and angle with readout.
    '''
    
    # Norms:
    wt_n0 = ll.norm(wt0)
    wt_n1 = ll.norm(wt1)

    wt_neu_n0 = np.mean(ll.norm(wt0, axis=0))   # input to/output from individual neuron
    wt_neu_n1 = np.mean(ll.norm(wt1, axis=0))   # input to/output from individual neuron

    # angle with readout wts
    ori_0 = ll.subspace_angles( wt0.T, wout )
    ori_1 = ll.subspace_angles( wt1.T, wout )
        
    # properties of delta W
    delW = wt1-wt0
    delwt_nm = ll.norm(delW)
    delwt_neu_nm = np.mean(ll.norm(delW, axis=0))
    ori_del = ll.subspace_angles( delW.T, wout )

    wt_res = { 'norm_0':wt_n0, 'neunorm_0':wt_neu_n0, 'ori_0': ori_0, 
            'norm_1':wt_n1, 'neunorm_1':wt_neu_n1, 'ori_1': ori_1, 
            'delW_nm': delwt_nm, 'delW_neu': delwt_neu_nm, 'ori_delW': ori_del
            }

    return wt_res


# get principal components and project data on nC components
def get_X_pcs( Xtest, nC=10, use_trials=None, starttm=20, stoptm=None ):
    ''' start time is 20 - to eliminate transients during model run'''
    
    ntrials, nT, _ = Xtest.shape
    if stoptm is None:
        stoptm = nT

    if use_trials is not None:
        Xtest = Xtest[np.ix_(use_trials,np.arange(start=starttm,stop=stoptm))]
    else:
        Xtest = Xtest[0:ntrials,starttm:stoptm,:]

    tr,tm,nneu = Xtest.shape
    Xtest = np.reshape( Xtest, [tr*tm, nneu])

    #px1 = PCA(n_components=nC, svd_solver='full')  # no scaling, only centering #<------------------------
    px1 = PCA(n_components=nC, n_oversamples=np.int_(1.3*nneu))       # randomized svd
    
    px1.fit(Xtest)

    X_red = px1.transform( Xtest )
    X_red = np.reshape( X_red, [tr,tm,nC] )

    return px1, X_red



# compute velocity decoder from given timeseries:
def intuitive_decoder( res, noise_x=0.01, scaleX=False, nPC=10, fitPC=True, 
                        dt=1, use_tm=None,  nTest=90, testfrac=0.10, add_bias=True, use_vel=True, scaleCoef=True ):
    ''' find an 'intuitive mapping' based on observed data
        requires dictionary with results from model run ('activity1', 'output')  and model parameters
    '''
    
        
    pos = res['output']         # cursor position
    if use_vel:
        vel = (pos[:,1:,:] - pos[:,:-1,:])*1000/dt # to set up vel as cm/s from /ms
    else:
        vel = pos[:,1:,:]    # position decoder

    if use_tm is None:
        use_tm = np.arange(start=20,stop=vel.shape[1])


    # exclude initial transient or custom period
    nTest = min(nTest, vel.shape[0] )   # trials to use
    use_trials = np.random.choice(vel.shape[0], nTest, replace=False)
    y = vel[ np.ix_(use_trials, use_tm) ]
    print(y.shape)
    nTr, nTm, nOut = y.shape
    y = np.reshape(y,(nTr*nTm,nOut))

    # neural activity as regressor
    X = res['activity1']
    X = X[:,1:,:]
    X = X[ np.ix_(use_trials, use_tm)]
    #X = X[:nTest,use_tm,:]
    nNeu = X.shape[2]
    X = np.reshape(X,(nTr*nTm,nNeu))
    X1 = X+np.random.randn(X.shape[0],X.shape[1])*noise_x  # jitter for non-active neurons -> to make sure nonzero variance in case of scaling

    if scaleX:
        # zscore activity
        scaler = StandardScaler()
        scaler.fit(X1)
        X_scaled= scaler.transform(X1)
    else:
        # just centre data
        X_scaled = X1 - np.mean(X1,axis=0)

    if fitPC:
        #px_1 = PCA(n_components=nPC, svd_solver='full') #<--------------
        px_1 = PCA(n_components=nPC, n_oversamples=np.int_(1.3*nNeu))   # randomized svd
        px_1.fit(X_scaled)
        X_red = px_1.transform(X_scaled)
    else:
        px_1=None
        X_red = X_scaled
    
    # can fit linear model to centred data because the intercept will reappear when true rates are multiplied by the weights
    # assume X = X0 + m,    
    # fit: vel = (X0)*W + b,        vel_test = X*W = (X0+m)*W = X0*W + b'
    # if good fit, b~b'
    
    # split data into train/test
    trialTrain = nTr - np.int_(nTr*testfrac)
    x_train = X_red[:trialTrain*nTm,:]
    x_test = X_red[trialTrain*nTm:,:]
    y_train = y[:trialTrain*nTm,:]
    y_test = y[trialTrain*nTm:,:]

    reg = LinearRegression(fit_intercept=add_bias).fit(x_train, y_train )

    score = reg.score( x_test, y_test )
    print(score)

    if scaleCoef:
        reg.coef_ = reg.coef_/ sum(px_1.explained_variance_ratio_)      # to get similar amplitude velocity



    return reg, px_1



def get_progress_ratios(  X, stim, Wout, px, use_tm=None, dt=1 ):

    if use_tm is None:
        use_tm = np.arange(start=200, stop=500)

    if len(X.shape)>2:
        X = X[:,use_tm,:]
        X = np.reshape( X, (X.shape[0]*X.shape[1], X.shape[2]))
        stim = stim[:,use_tm,:]
        stim = np.reshape( stim, (stim.shape[0]*stim.shape[1], stim.shape[2]))#np.squeeze(stim[:,-1,:2])


    X_inside = px.transform( X )        # get inside manifold component
    U = px.components_
    X_inside = X_inside@ U
    
    v_target = stim[:,:2]   # (cos-theta, sin-theta)
    vel_X = X @ Wout
    vel_inside = X_inside @ Wout
    p_total = np.zeros((v_target.shape[0],1))
    p_inside = np.zeros((v_target.shape[0],1))
    for tm in range( v_target.shape[0] ):
        p_total[tm] = np.dot(vel_X[tm,:], v_target[tm,:])
        p_inside[tm] = np.dot(vel_inside[tm,:], v_target[tm,:])

    
    p_outside = p_total-p_inside

    p_total_mean = np.mean(p_total)
    p_inside_mean = np.mean(p_inside)
    p_outside_mean = np.mean(p_outside)

    p_total_sd = np.std( p_total )
    p_inside_sd = np.std( p_inside )
    p_outside_sd = np.std( p_outside )

    
    target = np.arctan2( stim[:,1], stim[:,0] )     # stim is  trial x 2 : (cos(target), sin(target))
    all_target = np.unique(target)
    use_trials = [ (target==ct) for ct in all_target ]

    p_total_target = np.zeros((len(all_target),1))
    p_inside_target = np.zeros((len(all_target),1))
    p_outside_target = np.zeros((len(all_target),1))
    for tgt in range(len(use_trials)):
        p_total_target[tgt] = np.mean(p_total[use_trials[tgt]])
        p_inside_target[tgt] = np.mean(p_inside[use_trials[tgt]])
        p_outside_target[tgt]  = p_total_target[tgt]-p_inside_target[tgt]
    
    asymm = (max(p_total_target)  - min(p_total_target))/np.median(p_total_target)

    progress  = {'total_m': p_total_mean, 'inside_m':p_inside_mean, 'outside_m':p_outside_mean,
                 'total_sd': p_total_sd, 'inside_sd':p_inside_sd, 'outside_sd':p_outside_sd, 'asymm':asymm[0],
                 'all_target':all_target, 'p_total_target':p_total_target, 'p_inside_target':p_inside_target, 'p_outside_target':p_outside_target}
    
    #ptot=progress
    #pp.scatter(progress['all_target'],progress['p_total_target'],c='k')
    #pp.scatter(ptot['all_target'],ptot['p_outside_target'],c='b')
    #pp.scatter(ptot['all_target'],ptot['p_inside_target'],c='r')
    #pp.savefig('ptot_tgt.png')


    return progress


def get_progress_path(  X, stim, Wout, px, pos, use_tm=None, dt=1 ):
    # use actual vector to target than just cos/sin theta direction

    if use_tm is None:
        use_tm = np.arange(start=200, stop=500)

    if len(X.shape)>2:
        X = X[:,use_tm,:]
        X = np.reshape( X, (X.shape[0]*X.shape[1], X.shape[2]))
        stim = stim[:,use_tm,:]
        stim = np.reshape( stim, (stim.shape[0]*stim.shape[1], stim.shape[2]))#np.squeeze(stim[:,-1,:2])
        pos = pos[:,use_tm,:]
        pos = np.reshape( pos, (pos.shape[0]*pos.shape[1], pos.shape[2]))


    X_inside = px.transform( X )        # get inside manifold component
    U = px.components_
    X_inside = X_inside@ U
    
    v_target = stim[:,:2]   # (cos-theta, sin-theta)
    del_target = stim[:,:2] - pos   # current vector to target
    vel_X = X @ Wout
    vel_inside = X_inside @ Wout
    p_total = np.zeros((v_target.shape[0],1))
    p_inside = np.zeros((v_target.shape[0],1))
    for tm in range( v_target.shape[0] ):
        #p_total[tm] = np.dot(vel_X[tm,:], v_target[tm,:])
        #p_inside[tm] = np.dot(vel_inside[tm,:], v_target[tm,:])
        dt_u  = del_target[tm,:]/ll.norm(del_target[tm,:])  # unit vector
        p_total[tm] = np.dot(vel_X[tm,:], dt_u)
        p_inside[tm] = np.dot(vel_inside[tm,:], dt_u)

    
    p_outside = p_total-p_inside

    p_total_mean = np.mean(p_total)
    p_inside_mean = np.mean(p_inside)
    p_outside_mean = np.mean(p_outside)

    p_total_sd = np.std( p_total )
    p_inside_sd = np.std( p_inside )
    p_outside_sd = np.std( p_outside )

    
    target = np.arctan2( stim[:,1], stim[:,0] )     # stim is  trial x 2 : (cos(target), sin(target))
    all_target = np.unique(target)
    use_timepts = [ (target==ct) for ct in all_target ]

    p_total_target = np.zeros((len(all_target),1))
    p_inside_target = np.zeros((len(all_target),1))
    p_outside_target = np.zeros((len(all_target),1))
    for tgt in range(len(use_timepts)):
        p_total_target[tgt] = np.mean(p_total[use_timepts[tgt]])
        p_inside_target[tgt] = np.mean(p_inside[use_timepts[tgt]])
        p_outside_target[tgt]  = p_total_target[tgt]-p_inside_target[tgt]
    
    asymm = (max(p_total_target)  - min(p_total_target))/np.median(p_total_target)

    progress  = {'total_m': p_total_mean, 'inside_m':p_inside_mean, 'outside_m':p_outside_mean,
                 'total_sd': p_total_sd, 'inside_sd':p_inside_sd, 'outside_sd':p_outside_sd, 'asymm':asymm[0],
                 'all_target':all_target, 'p_total_target':p_total_target, 'p_inside_target':p_inside_target, 'p_outside_target':p_outside_target}
    
    #ptot=progress
    #pp.scatter(progress['all_target'],progress['p_total_target'],c='k')
    #pp.scatter(ptot['all_target'],ptot['p_outside_target'],c='b')
    #pp.scatter(ptot['all_target'],ptot['p_inside_target'],c='r')
    #pp.savefig('ptot_tgt.png')


    return progress



def shuffle_difference( WMP, OMP, niters=500, median=False ):
    if median:
        true_diff = np.nanmedian(WMP) - np.nanmedian(OMP)
    else:
        true_diff = np.nanmean(WMP) - np.nanmean(OMP)

    nWMP = len(WMP)
    nOMP = len(OMP)
    n = nOMP+nWMP
    measures = np.concatenate( (WMP, OMP), axis=0)
    print(n)
    print(measures.shape)
    
    shuffdiff = []
    for kk in range(niters):
        newlist = np.random.permutation(measures)
        newwmp = newlist[0:nWMP]
        newomp = newlist[nWMP:]
        if median:
            di = np.nanmedian(newwmp)-np.nanmedian(newomp)
        else:
            di = np.nanmean(newwmp) - np.nanmean(newomp)
        shuffdiff.append(di) 

    return true_diff, np.array(shuffdiff)
