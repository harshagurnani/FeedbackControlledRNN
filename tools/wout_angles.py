'''
assortment of function to analyse trained model - angles between subspaces and readouts and weights

Harsha Gurnani. Last revised Sep 2023
'''
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


import numpy as np
import tools.analysis_helpers as ah
import scipy.linalg as ll
import control as ctrl

def study_trained_dynamics( params1, res, nPC=10, use_trials=None, starttm=20, stoptm=None, output_dim=2 ):
    '''
    study output wts and dynamics alignment
    - params1: dictionary of weights/parameters
    - res: results with model run
    - nPC, use_trials, starttm, stoptm - params to decide what to use to define top pcs and intuitive decoder
    - output dim - no. of readout dimensions
    '''

    # 1 --- angle with top pcs
    activityX = res['activity1']
    px1, _ = ah.get_X_pcs( Xtest=activityX, nC=nPC, use_trials=use_trials, starttm=starttm, stoptm=stoptm )
    U = px1.components_    
    theta = np.zeros( (nPC, output_dim ) )
    theta.fill( np.nan )
    for nvec in np.arange(start=1,stop=nPC+1):
        nd = min(nvec, output_dim)
        theta[nvec-1,:nd] = ll.subspace_angles( params1['W_out_0'], U[:nvec,:].T )

    # 2 --- get singular vectors of A (may not be the same as driven dynamics):
    LeftSV, SVal, RightSV = ll.svd( params1['W_rec_0'].T )    # left = observed, right = input
    theta_A = np.zeros( (nPC, output_dim ) )
    theta_A.fill( np.nan )
    for nvec in np.arange(start=1,stop=nPC+1):
        nd = min(nvec, output_dim)
        theta_A[nvec-1,:nd] = ll.subspace_angles( params1['W_out_0'], RightSV[:nvec,:].T )      # sensitivity to dynamics

    theta_AL = np.zeros( (nPC, output_dim ) )
    theta_AL.fill( np.nan )
    for nvec in np.arange(start=1,stop=nPC+1):
        nd = min(nvec, output_dim)
        theta_AL[nvec-1,:nd] = ll.subspace_angles( params1['W_out_0'], LeftSV[:,:nvec] )       # amplification along readout

    # 3 --- get angle with feedback weights
    if (ll.norm(params1['W_fbk_0']))>0:
        theta_fbk = ll.subspace_angles( params1['W_out_0'], params1['W_fbk_0'].T )
    else:
        theta_fbk = np.nan

    fbk_dim = params1['W_fbk_0'].shape[0]
    theta_Fb_A = np.zeros( (nPC, fbk_dim ) ) # between fbk wts and amplified dirns of rnn
    theta_Fb_A.fill( np.nan )
    if (ll.norm(params1['W_fbk_0']))>0:
        for nvec in np.arange(start=1,stop=nPC+1):
            nd = min(nvec, fbk_dim)
            theta_Fb_A[nvec-1,:nd] = ll.subspace_angles( params1['W_fbk_0'].T, LeftSV[:,:nvec] )

    theta_Fb_AR = np.zeros( (nPC, fbk_dim ) ) # between fbk wts and sensitive dirns of rnn
    theta_Fb_AR.fill( np.nan )
    if (ll.norm(params1['W_fbk_0']))>0:
        for nvec in np.arange(start=1,stop=nPC+1):
            nd = min(nvec, fbk_dim)
            theta_Fb_AR[nvec-1,:nd] = ll.subspace_angles( params1['W_fbk_0'].T, RightSV[:nvec,:].T  )    # sensitivity to dynamics

    # 4 - get angle with top pcs and intrinsic dynamics

    theta_ID = np.zeros( (nPC, 2 ) )
    theta_ID.fill( np.nan )
    # 3 --- get angle between pcs and sensitive dynamics:
    for nvec in np.arange(start=1,stop=nPC+1):
        tmp = ll.subspace_angles( U[:nvec,:].T, RightSV[:nvec,:].T )        # has 'jj' angles
        theta_ID[nvec-1,:] = np.array( [tmp[0], tmp[-1]] )  # keep largest and smallest angle


    theta_ID_Left = np.zeros( (nPC, 2 ) )
    theta_ID_Left.fill( np.nan )
    # 3 --- get angle between pcs and amplified dynamics:
    for nvec in np.arange(start=1,stop=nPC+1):
        tmp = ll.subspace_angles( U[:nvec,:].T, LeftSV[:,:nvec] ) # has 'jj' angles
        theta_ID_Left[nvec-1,:] = np.array( [tmp[0], tmp[-1]] )   # keep largest and smallest angle


    

    theta_dic = {'Out_PCs':theta, 'Out_RightSV': theta_A, 'ID_PCs':theta_ID, 'ID_PCs_LeftSV':theta_ID_Left,
                 'Out_Fbk':theta_fbk , 'Fbk_LeftSV':theta_Fb_A , 'Fbk_RightSV':theta_Fb_AR, 'Out_LeftSV':theta_AL }


    return theta_dic


def study_intuitive_dynamics( params1, res, intuitive, output_dim=2 ):

    theta_base = ll.subspace_angles( intuitive['wout'], params1['W_out_0'] )

    U = intuitive['PCRes'].components_
    nPC = U.shape[0]
    theta_IM = np.zeros((nPC,output_dim))
    theta_IM.fill(np.nan)

    for nvec in np.arange(start=1,stop=nPC+1):
        nd = min(nvec, output_dim)
        theta_IM[nvec-1,:nd] = ll.subspace_angles( U[:nvec,:].T, params1['W_out_0'] )

    theta_dic = {'Out_Intuitive': theta_base, 'Out_Intrinsic':theta_IM}
    return theta_dic


def get_all_wchange( params0, params1, usew=None ):
    if usew is None:
        usew = ['W_rec_0', 'W_fbk_0', 'W_in_0']
    
    wres = {}
    for wtype in usew:
        wres[wtype] = ah.get_weight_change( params0[wtype], params1[wtype], params1['W_out_0'] )
        
    return wres


def study_angle_delW( params0, params1, nPC=10, output_dim=2 ):

    del_Wrec = params1['W_rec_0'] - params0['W_rec_0']
    LeftSV, SVal, RightSV = ll.svd( del_Wrec.T )  
    
    theta_AR = np.zeros( (nPC, output_dim ) )
    theta_AR.fill( np.nan )
    for nvec in np.arange(start=1,stop=nPC+1):
        nd = min(nvec, output_dim)
        theta_AR[nvec-1,:nd] = ll.subspace_angles( params1['W_out_0'], RightSV[:nvec,:].T )      # sensitivity to dynamics

    theta_AL = np.zeros( (nPC, output_dim ) )
    theta_AL.fill( np.nan )
    for nvec in np.arange(start=1,stop=nPC+1):
        nd = min(nvec, output_dim)
        theta_AL[nvec-1,:nd] = ll.subspace_angles( params1['W_out_0'], LeftSV[:,:nvec] )       # amplification along readout

    #
    fbk_dim = params1['W_fbk_0'].shape[0]
    
    theta_Fb_A = np.zeros( (nPC, fbk_dim ) ) # between fbk wts and amplified dirns of rnn
    theta_Fb_A.fill( np.nan )

    if (ll.norm(params1['W_fbk_0']))>0:
        for nvec in np.arange(start=1,stop=nPC+1):
            nd = min(nvec, fbk_dim)
            #print(nd)
            #print(ll.subspace_angles( params1['W_fbk_0'].T, LeftSV[:,:nvec] ))
            theta_Fb_A[nvec-1,:nd] = ll.subspace_angles( params1['W_fbk_0'].T, LeftSV[:,:nvec] )

    theta_Fb_AR = np.zeros( (nPC, fbk_dim ) ) # between fbk wts and sensitive dirns of rnn
    theta_Fb_AR.fill( np.nan )
    if (ll.norm(params1['W_fbk_0']))>0:
        for nvec in np.arange(start=1,stop=nPC+1):
            nd = min(nvec, fbk_dim)
            theta_Fb_AR[nvec-1,:nd] = ll.subspace_angles( params1['W_fbk_0'].T, RightSV[:nvec,:].T  )    # sensitivity to dynamics

    theta_dic = {'Out_RightSV': theta_AR, 'Out_LeftSV':theta_AL ,
                  'Fbk_LeftSV':theta_Fb_A , 'Fbk_RightSV':theta_Fb_AR}


    return theta_dic



def get_gen_correlation( params1, res, use_trials=None, starttm=20, stoptm=500, output_dim=2):
    X_all = res['activity1']
    nTr, nTm, nNeu = X_all.shape
    if use_trials is None:
        use_trials = np.arange(nTr)
    use_time = np.arange(start=starttm, stop=stoptm)
    useX = X_all[np.ix_(use_trials,use_time)]
    useX = np.reshape( useX, (useX.shape[0]*useX.shape[1], useX.shape[2]))
    
    numerator = useX @ params1['W_out_0']
    ratio = ll.norm( numerator)/(ll.norm(params1['W_out_0'])*ll.norm(useX)) # dividing by wout norm should be fine then?
    
    return ratio



def get_gen_corr_ratios( params1, res, use_trials=None, starttm=20, stoptm=500, nPC=30, nexclude=15, output_dim=2 ):
    X_all = res['activity1']

    px1, _ = ah.get_X_pcs( Xtest=X_all, nC=nPC, use_trials=use_trials, starttm=starttm, stoptm=stoptm )
    PC1 = px1.components_[0,:]

    nTr, nTm, nNeu = X_all.shape
    if use_trials is None:
        use_trials = np.arange(nTr)
    use_time = np.arange(start=starttm, stop=stoptm)
    useX = X_all[np.ix_(use_trials,use_time)]
    useX = np.reshape( useX, (useX.shape[0]*useX.shape[1], useX.shape[2]))

    xnorm=ll.norm(useX)

    numerator = useX @ params1['W_out_0']
    ratio = ll.norm( numerator)/(ll.norm(params1['W_out_0'])*ll.norm(useX)) # dividing by wout norm should be fine then?
    
    numerator = useX @ PC1.T
    ratio_PC = ll.norm( numerator)/(ll.norm(PC1.T)*xnorm)

    allU = px1.components_[nexclude:,:]
    nvec = allU.shape[0]
    ratio_otherPCs = 0
    for vv in range(nvec):
        vec = allU[vv,:].T
        numerator = useX @ vec
        ratio_otherPCs += ll.norm( numerator)/(ll.norm(vec)*ll.norm(useX))
        
    ratio_otherPCs=ratio_otherPCs/nvec
    '''
    ratio=0
    for dim in range(output_dim):
        wvec = params1['W_out_0'][:,dim]
        numerator = useX @ wvec
        ratio += ll.norm( numerator)/(ll.norm(wvec)*ll.norm(useX))
        
    ratio = ratio/output_dim    # averaged across output dim -> to compare to single vectors


    numerator = useX @ PC1.T
    ratio_PC = ll.norm( numerator)/(ll.norm(PC1.T)*xnorm)

    allU = px1.components_[nexclude:,:]
    nvec = allU.shape[0]
    ratio_otherPCs = 0
    for nv in range(nvec):
        vec = allU[nv,:].T
        numerator = useX @ vec
        ratio_otherPCs += ll.norm( numerator)/(ll.norm(vec)*ll.norm(useX))
    ratio_otherPCs=ratio_otherPCs/nvec
    '''

    allRatio={'W_out_0':ratio, 'PC1':ratio_PC, 'OtherPCs':ratio_otherPCs }



    return allRatio



def study_alignment( wout, wfbk, wrec, intuitive=None, nPC=8, nAngles=30, output_dim=2 ):
    '''
    study output wts and dynamics alignment
    '''

    #wout = params1['W_out_0']       # as columns
    out_dim = wout.shape[1]
    wnorm = ll.norm( wout, axis=0 )

    #wfbk = params1['W_fbk_0'].T     # transpose
    wfbk = wfbk.T                    # transpose
    fbk_dim =  wfbk.shape[1]
    fbnorm = ll.norm(wfbk, axis=0 )

    # 1 --- get singular vectors of A (may not be the same as driven dynamics):
    LeftSV, SVal, RightSV = ll.svd( wrec.T )  
    Sensitive = RightSV.T
    Amplified = LeftSV


    # 2 - Angle with Output  
    theta_Out_Sensitive = np.zeros( (nAngles, out_dim ) )
    theta_Out_Sensitive.fill( np.nan )
    Out_Sensitivity = np.zeros( (nAngles))
    for nvec in np.arange(start=1,stop=nAngles+1):
        nd = min(nvec, out_dim)
        theta_Out_Sensitive[nvec-1,:nd] = ll.subspace_angles( wout, Sensitive[:,:nvec] )
        for od in range(out_dim):
            terms = np.array( [SVal[jj]*np.dot(wout[:,od],  Sensitive[:,jj] )/wnorm[od] for jj in range(nvec-1) ] )
            Out_Sensitivity[nvec-1] += np.sum(terms)/out_dim

    theta_Out_Amplified = np.zeros( (nAngles, output_dim ) )
    theta_Out_Amplified.fill( np.nan )
    Out_Amplification = np.zeros( (nAngles))
    for nvec in np.arange(start=1,stop=nAngles+1):
        nd = min(nvec, fbk_dim)
        theta_Out_Amplified[nvec-1,:nd] = ll.subspace_angles( wout, Amplified[:,:nvec] )
        for od in range(out_dim):
            terms = np.array( [SVal[jj]*np.dot(wout[:,od],  Amplified[:,jj] )/wnorm[od] for jj in range(nvec-1) ] )
            Out_Amplification[nvec-1] += np.sum(terms)/out_dim



    
    # 3 - Angle with Feedback  
    theta_Fbk_Sensitive = np.zeros( (nAngles, fbk_dim ) )
    theta_Fbk_Sensitive.fill( np.nan )
    Fbk_Sensitivity = np.zeros( (nAngles))
    if (ll.norm(wfbk))>0:
        for nvec in np.arange(start=1,stop=nAngles+1):
            nd = min(nvec, fbk_dim)
            theta_Fbk_Sensitive[nvec-1,:nd] = ll.subspace_angles( wfbk, Sensitive[:,:nvec] )
            for fd in range(fbk_dim):
                terms = np.array( [SVal[jj]*np.dot(wfbk[:,fd],  Sensitive[:,jj] )/wfbk[fd] for jj in range(nvec-1) ] )
                Fbk_Sensitivity[nvec-1] += np.sum(terms)/fbk_dim

    theta_Fbk_Amplified = np.zeros( (nAngles, fbk_dim ) )
    theta_Fbk_Amplified.fill( np.nan )
    Fbk_Amplification = np.zeros( (nAngles))
    if (ll.norm(wfbk))>0:
        for nvec in np.arange(start=1,stop=nAngles+1):
            nd = min(nvec, fbk_dim)
            theta_Fbk_Amplified[nvec-1,:nd] = ll.subspace_angles( wfbk, Amplified[:,:nvec] )
            for fd in range(fbk_dim):
                terms = np.array( [SVal[jj]*np.dot(wfbk[:,fd],  Amplified[:,jj] )/wfbk[fd] for jj in range(nvec-1) ] )
                Fbk_Amplification[nvec-1] += np.sum(terms)/fbk_dim

    if (ll.norm(wfbk))>0:
        theta_out_fbk = ll.subspace_angles(wout, wfbk )
    else:
        theta_out_fbk = np.nan

    theta_dic = {'theta_Out_Sensitive': theta_Out_Sensitive, 'theta_Out_Amplified':theta_Out_Amplified , 'Out_Sensitivity':Out_Sensitivity, 'Out_Amplification':Out_Amplification,
                'theta_Fbk_Sensitive': theta_Fbk_Sensitive, 'theta_Fbk_Amplified':theta_Fbk_Amplified , 'Fbk_Sensitivity':Fbk_Sensitivity, 'Fbk_Amplification':Fbk_Amplification,
                'theta_Out_Fbk':theta_out_fbk}


    return theta_dic


def Henrici_index( A ):
    #A = params1['W_rec_0']
    n = ll.norm( A, ord='fro')
    e, v = ll.eig(A)

    dfn = np.sqrt( n**2 - np.sum( np.abs(e)**2 ) )
    return dfn

def return_controls( params ):
    Wrec = params['W_rec_0'].T 
    A = Wrec - np.eye(Wrec.shape[0])

    Wout = params['W_out_0'].T # 2 x N
    Wfbk = params['W_fbk_0'].T # N x 2
    Win = params['W_in_0'].T   # N x 3
    # U = # N x npc
    # solve the lyap eqn for observability
    # solve the lyap eqn for controllability
    CtrlGram_FF = ctrl.lyap( A, Win@Win.T )
    CtrlGram_FB = ctrl.lyap( A, Wfbk@Wfbk.T )
    # compute controllability of readout
    npc=2
    Wout_norm = Wout.T/ll.norm(Wout.T,axis=0)
    Wout_norm = Wout_norm.T
    beta_ff = np.trace( Wout_norm @ CtrlGram_FF @ Wout_norm.T)/Win.shape[1]
    beta_fb = np.trace( Wout_norm @ CtrlGram_FB @ Wout_norm.T)/Wfbk.shape[1]

    dic={  'ctrb_ff':beta_ff, 'ctrb_fb':beta_fb}  #'CtrlGram_FF':CtrlGram_FF, 'CtrlGram_FB':CtrlGram_FB,

    return dic