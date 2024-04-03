import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
sys.path.insert(0, parentdir) 


import tools.analysis_helpers as ah
import tools.wout_angles as wang
import scipy.linalg as ll
import control as ctrl
import control.matlab as mt
from control.statesp import StateSpace
import control.modelsimp as msimp
import tools.measures as mm
from scipy.ndimage import uniform_filter1d
import tools.perturb_network_ as pnet

def plot_training_results( file='trained_wmp_round2_.npy', folder='wmp/relu_/Model_8_movePC_new__PC8/' , savfolder='wmp/relu_/Model_8_movePC_new__PC8/' , suffix ='', 
                          simple=True, percp=False, hit_thresh=0.15, new_progress=False, use_mod=None, use_fbwt='W_fbk_0' ):
    # Load some stuff   
    allres = np.load( folder+file, allow_pickle=True ).item()
    train_res = allres['train_res']
    nMaps = len(train_res)
    nReps = len(train_res[0])
    nDim = len(train_res[0][0]['stats']['PC_angle_pre'])

    progress_pre = np.zeros((nMaps,1))
    progress_post = np.zeros((nMaps,1))
    progress_pre_in = np.zeros((nMaps,1))
    progress_post_in = np.zeros((nMaps,1))
    progress_asymm_pre = np.zeros((nMaps,1))
    progress_asymm_post = np.zeros((nMaps,1))
    p_total_target_pre = [np.nan]
    p_total_target_post = [np.nan]
    Xnorm_pre = np.zeros((nMaps,1))
    Xnorm_post = np.zeros((nMaps,1))
    Xnorm_orig = np.zeros((nMaps,1))
    expVar_pre = np.zeros((nMaps,1))
    expVar_post = np.zeros((nMaps,1))
    expVar_decoder = np.zeros((nMaps,1))
    PC_angle_pre = np.zeros((nMaps,nDim))
    PC_angle_post = np.zeros((nMaps,nDim))
    hit_rate_pre =  np.zeros((nMaps,1))
    hit_rate_post =  np.zeros((nMaps,1))
    acq_rate_pre =  np.zeros((nMaps,1))
    acq_rate_post =  np.zeros((nMaps,1))
    im_change = np.zeros((nMaps,1))
    im_angles = np.zeros((nMaps,nDim))
    

    
    angle_res_pre = {'theta_Out_Amplified':np.zeros((nMaps,nDim)), 'theta_Fbk_Amplified':np.zeros((nMaps,nDim)), 
                 'theta_Out_Sensitive':np.zeros((nMaps,nDim)), 'theta_Fbk_Sensitive':np.zeros((nMaps,nDim))}
    sigma_res_pre = {'Out_Amplification':np.zeros((nMaps,nDim)), 'Fbk_Amplification': np.zeros((nMaps,nDim)), 
                 'Out_Sensitivity': np.zeros((nMaps,nDim)), 'Fbk_Sensitivity': np.zeros((nMaps,nDim))}
    angle_res_post = {'theta_Out_Amplified':np.zeros((nMaps,nDim)), 'theta_Fbk_Amplified':np.zeros((nMaps,nDim)), 
                 'theta_Out_Sensitive':np.zeros((nMaps,nDim)), 'theta_Fbk_Sensitive':np.zeros((nMaps,nDim))}
    sigma_res_post = {'Out_Amplification':np.zeros((nMaps,nDim)), 'Fbk_Amplification': np.zeros((nMaps,nDim)), 
                 'Out_Sensitivity': np.zeros((nMaps,nDim)), 'Fbk_Sensitivity': np.zeros((nMaps,nDim))}
    theta_OF_pre = np.zeros((nMaps,1))
    theta_OF_post = np.zeros((nMaps,1))

    theta_intuit_wperturb = np.zeros((nMaps,1))
    theta_base_wperturb = np.zeros((nMaps,1))
    theta_im_wperturb = np.zeros((nMaps,1))
    overlap_base_wperturb = np.zeros( (nMaps,1))
    overlap_intuit_wperturb = np.zeros((nMaps,1))
    overlap_influence = np.zeros((nMaps,1))

    speed = np.zeros((nMaps,1))
    fitted_k = np.zeros((nMaps,1))
    angInp = np.zeros((nMaps,1))
    meanInp = np.zeros((nMaps,1))
    meanFbk = np.zeros((nMaps,1))

    alpha = np.zeros((nMaps,1))
    beta_ff = np.zeros((nMaps,1))
    beta_fb = np.zeros((nMaps,1))
    beta_U_ff = np.zeros((nMaps,1))
    beta_U_fb = np.zeros((nMaps,1))
    beta_bal = np.zeros((nMaps,1))

    dimVF = np.zeros((nMaps,1))
    fracVF = np.zeros((nMaps,1))
    fracVF_ff = np.zeros((nMaps,1))
    fracVF_fb = np.zeros((nMaps,1))
    fracVF_rec = np.zeros((nMaps,1))
    fracOutvel = np.zeros((nMaps,1))
    fracSVF_rec = np.zeros((nMaps,1))
    fracSVF_fb = np.zeros((nMaps,1))
    SVF_tot = np.zeros((nMaps,1))
    SVF_angle = np.zeros((nMaps,1))


    fracTotalVel = np.zeros((nMaps,1))
    fracVar_orig = np.zeros((nMaps,1))
    fracVar_post = np.zeros((nMaps,1))
    fracVar_pre = np.zeros((nMaps,1))

    Fbk_overlap = np.zeros((nMaps,1))
    Out_overlap = np.zeros((nMaps,1))
    
    maxNorm = 25
    maxAsymm = 5
    maxVar = 1

    loss_speed = np.zeros((nMaps,1))
    loss_fitted_k = np.zeros((nMaps,1))

    nTr = train_res[0][0]['posttrain']['lc'].shape[0]
    dim_dw_fbk = np.zeros((nMaps,1))
    corr_dw_fbk = np.zeros((nMaps,1))
    grad_total = np.zeros((nMaps,nTr))

    # concaenate results
    for map in range(nMaps):
        wmp_train = train_res[map][0]
        if new_progress:
            stats  = get_new_progress(wmp_train)
            wmp_train['stats']['progress_pre'] = stats['progress_pre']
            wmp_train['stats']['progress_post'] = stats['progress_post']
        progress_pre[map] = wmp_train['stats']['progress_pre']['total_m']
        progress_post[map] = wmp_train['stats']['progress_post']['total_m']
        progress_pre_in[map] = wmp_train['stats']['progress_pre']['inside_m']
        progress_post_in[map] = wmp_train['stats']['progress_post']['inside_m']
        progress_asymm_pre[map] = abs(wmp_train['stats']['progress_pre']['asymm'])
        progress_asymm_post[map] = abs(wmp_train['stats']['progress_post']['asymm'])
        p_total_target_pre = np.vstack((p_total_target_pre, wmp_train['stats']['progress_pre']['p_total_target']))
        p_total_target_post = np.vstack((p_total_target_post, wmp_train['stats']['progress_post']['p_total_target']))
        Xnorm_pre[map] = wmp_train['stats']['Xnorm_pre']
        Xnorm_post[map] = wmp_train['stats']['Xnorm_post']
        Xnorm_orig[map] = wmp_train['stats']['Xnorm_orig']
        
        PC_angle_pre[map,:] = wmp_train['stats']['PC_angle_pre']
        PC_angle_post[map,:] = wmp_train['stats']['PC_angle_post']
        X_pre = wmp_train['pretrain']['res']['activity1']
        X_post = wmp_train['posttrain']['res']['activity1']    
        wmp_train['stats']['perf_res_pre'] = ah.get_performance( wmp_train['pretrain'], nTest=X_pre.shape[0],thresh=hit_thresh )
        wmp_train['stats']['perf_res_post'] = ah.get_performance( wmp_train['posttrain'], nTest=X_post.shape[0],thresh=hit_thresh )
        if 'perf_res_pre' in wmp_train['stats'].keys():
            hit_rate_pre[map] = wmp_train['stats']['perf_res_pre']['success']
            acq_rate_pre[map] = np.median(wmp_train['stats']['perf_res_pre']['acq_time'])
            hit_rate_post[map] = wmp_train['stats']['perf_res_post']['success']
            acq_rate_post[map] = np.median(wmp_train['stats']['perf_res_post']['acq_time'])

        wout=wmp_train['posttrain']['params1']['W_out_0']
        wrec_pre=wmp_train['posttrain']['params0']['W_rec_0']
        wrec_post=wmp_train['posttrain']['params1']['W_rec_0']
        if simple:
            fb_pre=wmp_train['posttrain']['params0']['W_fbk_0']         
            fb_post=wmp_train['posttrain']['params1']['W_fbk_0']
        dic_pre = wang.study_alignment( wout=wout, wfbk=fb_pre, wrec=wrec_pre, nAngles=nDim )
        dic_post = wang.study_alignment(wout=wout, wfbk=fb_post, wrec=wrec_post, nAngles=nDim )
        for key in angle_res_pre.keys():
            angle_res_pre[key][map,:] = np.rad2deg(np.min(dic_pre[key],axis=1))
            angle_res_post[key][map,:] = np.rad2deg(np.min(dic_post[key],axis=1))
        for key in sigma_res_pre.keys():
            sigma_res_pre[key][map,:] = np.rad2deg(dic_pre[key])
            sigma_res_post[key][map,:] = np.rad2deg(dic_post[key])
        theta_OF_pre[map] = np.rad2deg(np.min(dic_pre['theta_Out_Fbk']))
        theta_OF_post[map] = np.rad2deg(np.min(dic_post['theta_Out_Fbk']))

        wout_intuit = wmp_train['intuitive']['wout']
        upc = wmp_train['intuitive']['PCRes'].components_.T
        theta_intuit_wperturb[map] = np.rad2deg(np.min(ll.subspace_angles(wout,wout_intuit)))#np.rad2deg(np.min(ll.subspace_angles(wout,wout_intuit)))
        theta_im_wperturb[map] = np.rad2deg(np.min(ll.subspace_angles(wout,upc)))#np.rad2deg(np.min(ll.subspace_angles(wout,upc)))
        theta, newExp, dic_orig, wout_base = return_base_angle( wmp_train['file'], wmp_train)
        theta_base_wperturb[map] = theta

        overlap_intuit_wperturb[map]=mm.get_decoder_overlap( wout_intuit, wout )
        overlap_base_wperturb[map]  =mm.get_decoder_overlap( wout_base, wout )

        spd, k, res = return_speed_learning( train_res[map], idx=-1, avg_window=20, use_mod=use_mod, use_fbwt=use_fbwt )
        speed[map] = spd
        fitted_k[map] = k
        angInp[map] = np.mean( np.abs(res['inAngle']) )
        meanInp[map] = res['inDelta'] #res['inNorm1']-res['inNorm0']
        meanFbk[map] = res['fbDelta'] # res['fbNorm1']-res['fbNorm0']

        umap = wmp_train['intuitive']['PCRes'].components_.T    # or should this be pcs of closed-loop operation??
        ctr_dic = return_controls( wmp_train['posttrain']['params0'], U=umap )
        alpha[map] = np.abs(ctr_dic['obsv'])
        beta_ff[map] = np.abs(ctr_dic['ctrb_ff'])
        beta_fb[map] = np.abs(ctr_dic['ctrb_fb'])
        beta_U_ff[map] = np.abs(ctr_dic['ctrb_U_ff'])
        beta_U_fb[map] = np.abs(ctr_dic['ctrb_U_fb'])
        Fbk_overlap[map] = ctr_dic['Fbk_right_overlap']
        Out_overlap[map] = ctr_dic['Out_left_overlap']
        beta_bal[map] = return_balanced_sv( wmp_train['posttrain']['params0'] )

        expVar_pre[map] = mm.get_variance_in_U(X_pre, umap )#wmp_train['stats']['expVar_pre']
        expVar_post[map] = mm.get_variance_in_U(X_post, umap)#wmp_train['stats']['expVar_post']
        expVar_decoder[map] = mm.get_variance_in_U(dic_orig['res']['activity1'], wout )

        a1, a2, a3 = mm.get_change_IM( X_pre, X_post, umap, wout)
        im_change[map] = a1
        im_angles[map,:] = a2
        overlap_influence[map] = a3

        if not percp:
            fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim, y_tot, fracY_fb, fracY_rec, angle_y = mm.get_vectorfld_change_all( wmp_train['posttrain']['res'], wmp_train['posttrain']['params0'], wmp_train['posttrain']['params1'] )
            #fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim = mm.get_vectorfld_in_U( wmp_train['posttrain']['res'], wmp_train['posttrain']['params0'], wmp_train['posttrain']['params1'],  wmp_train['intuitive']['PCRes'].components_)
            velVar, vo, vp, vpre = mm.get_velocity_change( wmp_train['posttrain']['res'], dic_orig['res'], wmp_train['pretrain']['res'], wmp_train['posttrain']['params0'], wmp_train['posttrain']['params1'] )
            fracTotalVel[map] = velVar
            fracVar_orig[map] = vo
            fracVar_post[map] = vp
            fracVar_pre[map]  = vpre
            print('fracZ = '+np.str_(fracZ) + ', fracZ_fb = '+np.str_(fracZ_fb)+ ', velVar = ' +np.str_(velVar)+', dimVF = '+np.str_(dim))
            fracSVF_rec[map] = fracY_rec
            fracSVF_fb[map] = fracY_fb
            SVF_tot[map] = y_tot
            SVF_angle[map] = angle_y
        else:
            fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim = mm.get_vectorfld_prcptron( newExp.model, wmp_train['posttrain']['res'], wmp_train['posttrain']['params0'], wmp_train['posttrain']['params1'])
            velVar, vo, vp, vpre = mm.get_velocity_change( wmp_train['posttrain']['res'], dic_orig['res'], wmp_train['pretrain']['res'], wmp_train['posttrain']['params0'], wmp_train['posttrain']['params1'] )
            #velVar = mm.get_velocity_change( wmp_train['posttrain']['res'], dic_orig['res'], wmp_train['posttrain']['params0'], wmp_train['posttrain']['params1'] )
            fracTotalVel[map] = velVar
            fracVar_orig[map] = vo
            fracVar_post[map] = vp
            fracVar_pre[map]  = vpre
            print('fracZ = '+np.str_(fracZ) + ', fracZ_fb = '+np.str_(fracZ_fb)+ ', velVar = ' +np.str_(velVar)+', dimVF = '+np.str_(dim))
        
        fracVF[map] = fracZ
        fracVF_ff[map] = fracZ_ff
        fracVF_fb[map] = fracZ_fb
        fracOutvel[map] = fracvel
        fracVF_rec[map] = fracZ_rec
        dimVF[map] = dim
        #fracSVF_rec[map] = fracY_rec
        #fracSVF_fb[map] = fracY_fb
        #SVF_tot[map] = y_tot
        #SVF_angle[map] = angle_y

        spd2, k2, res2 = return_speed_loss( train_res[map], idx=0, avg_window=20 )
        loss_speed[map] = spd2
        loss_fitted_k[map] = k2

        grad_dic = return_gradient_info( train_res[map] )
        dim_dw_fbk[map] = grad_dic['dim']
        corr_dw_fbk[map] = grad_dic['corr_fbk']
        grad_total[map,:] = grad_dic['grad']


    Xnorm_pre[Xnorm_pre>maxNorm]=np.nan
    Xnorm_post[Xnorm_post>maxNorm]=np.nan
    Xnorm_orig[Xnorm_orig>maxNorm]=np.nan
    #expVar_post[expVar_post>maxVar]=1#np.nan
    #expVar_pre[expVar_pre>maxVar]=1#np.nan

    progress_asymm_pre[progress_asymm_pre>maxAsymm] = np.nan
    progress_asymm_post[progress_asymm_post>maxAsymm] = np.nan



    # Plotting
    # Progress pre- and post- training:
    

    results = {'progress_pre':progress_pre, 'progress_post':progress_post, 'progress_pre_in':progress_pre_in, 'progress_post_in':progress_post_in,
               'progress_asymm_pre':progress_asymm_pre, 'progress_asymm_post':progress_asymm_post, 'Xnorm_pre':Xnorm_pre, 'Xnorm_post':Xnorm_post , 'Xnorm_orig':Xnorm_orig,
               'p_total_target_pre':p_total_target_pre, 'p_total_target_post':p_total_target_post, 'expVar_pre':expVar_pre, 'expVar_post':expVar_post,  'expVar_decoder':expVar_decoder,
               'PC_angle_pre':PC_angle_pre, 'PC_angle_post':PC_angle_post, 'hit_rate_pre':hit_rate_pre, 'hit_rate_post':hit_rate_post,
               'angle_res_pre':angle_res_pre, 'angle_res_post':angle_res_post, 'sigma_res_pre':sigma_res_pre, 'sigma_res_post':sigma_res_post, 'theta_OF_pre':theta_OF_pre, 'theta_OF_post':theta_OF_post, 
               'theta_intuit_wperturb':theta_intuit_wperturb, 'theta_im_wperturb':theta_im_wperturb,
                'theta_Out_Amplified_pre': angle_res_pre['theta_Out_Amplified'][:,-2:-1],  'theta_Out_Amplified_post':angle_res_post['theta_Out_Amplified'][:,-2:-1] , 
                'theta_Fbk_Sensitive_pre': angle_res_pre['theta_Fbk_Sensitive'][:,-2:-1], 'theta_Fbk_Sensitive_post':angle_res_post['theta_Fbk_Sensitive'][:,-2:-1],
                'speed':speed, 'fitted_k':fitted_k, 'angInp':angInp, 'meanInp':meanInp, 'acq_rate_pre':acq_rate_pre, 'acq_rate_post':acq_rate_post , 'meanFbk':meanFbk,
                'obsv':alpha, 'ctrb_ff':beta_ff, 'ctrb_fb':beta_fb, 'ctrb_obsv_bal':beta_bal, 'theta_wout_wintuit':theta_intuit_wperturb, 'theta_wout_wbase':theta_base_wperturb,
                'fracVF':fracVF, 'fracVF_ff':fracVF_ff, 'fracVF_fb':fracVF_fb, 'fracOutvel':fracOutvel, 'fracVF_rec':fracVF_rec   , 'fracTotalVel':fracTotalVel, 'fracVar_orig':fracVar_orig, 'fracVar_post':fracVar_post, 'fracVar_pre':fracVar_pre,
                'loss_speed':loss_speed,    'loss_fitted_k':loss_fitted_k    , 'Fbk_overlap':Fbk_overlap, 'Out_overlap':Out_overlap   , 
                'dimVF':dimVF , 'fracSVF_rec':fracSVF_rec,   'fracSVF_fb':fracSVF_fb, 'SVF_tot':SVF_tot  ,
                'dim_dw_fbk':dim_dw_fbk,  'corr_dw_fbk':corr_dw_fbk, 'grad_total':grad_total,
                 'ctrb_U_ff':beta_U_ff, 'ctrb_U_fb':beta_U_fb, 'im_change':im_change,  'im_angles':im_angles, 'overlap_intuit_wperturb':overlap_intuit_wperturb, 'overlap_base_wperturb':overlap_base_wperturb, 'overlap_influence':overlap_influence }    

    return results





def get_new_progress( wmp_train ):

    px_base = wmp_train['intuitive']['PCRes']
    W_pert = wmp_train['wout']

    X_pre = wmp_train['pretrain']['res']['activity1']
    stim_pre = wmp_train['pretrain']['res']['stimulus']
    pos_pre = wmp_train['pretrain']['res']['output']
    X_post = wmp_train['posttrain']['res']['activity1']   
    stim_post = wmp_train['posttrain']['res']['stimulus']
    pos_post = wmp_train['posttrain']['res']['output']

    progress_pre = ah.get_progress_path(  X_pre, stim_pre, W_pert, px_base, pos_pre )
    progress_post = ah.get_progress_path(  X_post, stim_post, W_pert, px_base, pos_post )
    stats = {'progress_pre':progress_pre,  'progress_post':progress_post }

    return stats




def return_controls( params, U ):
    Wrec = params['W_rec_0'].T 
    A = Wrec - np.eye(Wrec.shape[0])
    #Ua, sa, Va = ll.svd(A.T, compute_uv=True)
    #Ua = Ua[:,:U.shape[1]]

    Wout = params['W_out_0'].T # 2 x N
    Wfbk = params['W_fbk_0'].T # N x 2
    Win = params['W_in_0'].T   # N x 3
    # U = # N x npc
    # solve the lyap eqn for observability
    ObsGram = ctrl.lyap( A.T, Wout.T@Wout) 
    # calculate observability of intrinsic manifold
    alpha = np.trace(U.T @ ObsGram @ U )/(A.shape[0]-U.shape[1])

    # solve the lyap eqn for controllability
    CtrlGram_FF = ctrl.lyap( A, Win@Win.T )
    CtrlGram_FB = ctrl.lyap( A, Wfbk@Wfbk.T )
    # compute controllability of readout
    npc=2
    Wout_norm = Wout.T/ll.norm(Wout.T,axis=0)
    Wout_norm = Wout_norm.T
    beta_ff = np.trace( Wout_norm @ CtrlGram_FF @ Wout_norm.T)/Win.shape[1]
    beta_fb = np.trace( Wout_norm @ CtrlGram_FB @ Wout_norm.T)/Wfbk.shape[1]
    beta_U_ff = np.trace( U[:,:npc].T @ CtrlGram_FF @ U[:,:npc])/npc
    beta_U_fb = np.trace( U[:,:npc].T @ CtrlGram_FB @ U[:,:npc])/npc
    #beta_ff = np.trace( np.sqrt(np.abs(Wout_norm @ CtrlGram_FF @ Wout_norm.T)))/Win.shape[1]
    #beta_fb = np.trace( np.sqrt(np.abs(Wout_norm @ CtrlGram_FB @ Wout_norm.T)))/Wfbk.shape[1]
    #beta_U_ff = np.trace( np.sqrt(np.abs(U[:,:npc].T @ CtrlGram_FF @ U[:,:npc])))/npc
    #beta_U_fb = np.trace( np.sqrt(np.abs(U[:,:npc].T @ CtrlGram_FB @ U[:,:npc])))/npc
    
    dic={'obsv':alpha, 'ctrb_ff':beta_ff, 'ctrb_fb':beta_fb, 'ctrb_U_ff':beta_U_ff, 'ctrb_U_fb':beta_U_fb}
    print(dic)

    UA, sA, VA = ll.svd( Wout @ll.inv(-A), compute_uv=True )
    topSensitive = VA[:Wout.shape[0],:].T
    overlap = np.sqrt(np.trace( Wfbk.T @ topSensitive @ topSensitive.T @ Wfbk)/Wout.shape[0])

    UA, sA, VA = ll.svd( A, compute_uv=True )
    topAmplified = UA[:,:8]
    overlap_out = np.sqrt(np.trace( Wout @ topAmplified @ topAmplified.T @ Wout.T )/8)

    dic.update({'Fbk_right_overlap':overlap,   'Out_left_overlap':overlap_out})

    return dic


def return_balanced_sv( params ):
    Wrec = params['W_rec_0'].T 
    A = Wrec - np.eye(Wrec.shape[0])
    Wout = params['W_out_0'].T # 2 x N
    Wfbk = params['W_fbk_0'].T # N x 2
    Win = params['W_in_0'].T   # N x 3
    D_fb = np.zeros((Wout.shape[0], Wfbk.shape[1]))
    #D_ff = np.zeros((Wout.shape[0], Win.shape[1]))
    
    # construct state space:
    #fsys_ff = StateSpace(A, Win, Wout, D_ff)
    fsys_fb = StateSpace(A, Wfbk, Wout, D_fb)    
    
    n = 8
    rsys = msimp.balred( fsys_fb, n, 'truncate' )

    try:
        Wc = mt.gram( rsys, 'c' )
        Wo = mt.gram( rsys, 'o' )
        #Wout_norm = Wout.T/ll.norm(Wout.T,axis=0)
        #Wout_norm=Wout_norm.T
        #cb_val = np.trace( Wout_norm @ Wc @ Wout_norm.T)/Wfbk.shape[1]

        ec, evc = ll.eig( Wc )
        #print(ec[:8])
        cb_val = np.sum( ec[:n] )
    except:
        cb_val = np.nan
    
    return cb_val


def return_eig( params ):
    Wrec = params['W_rec_0']
    Wout = params['W_out_0']
    Wfb = params['W_fbk_0']
    nNeu = Wrec.shape[0]
    nOut = Wout.shape[1]
    TotalA = np.zeros((nNeu+nOut, nNeu+nOut))
    TotalA[:nNeu,:nNeu] = Wrec-np.eye(nNeu)
    TotalA[nNeu:,:nNeu] = Wfb
    TotalA[:nNeu,nNeu:] = Wout
    evalue = ll.eig(TotalA, right=False, left=False)
    return evalue


def return_gradient_info( wmp_train ):
    nReps = len(wmp_train)
    dim_fbk = wmp_train[0]['posttrain']['lc'][:,-5]
    alldim = np.zeros((1, nReps))  

    ntr = min(200, dim_fbk.shape[0])
    allcorr = np.zeros((ntr, nReps)) 
    allgrad = np.zeros((dim_fbk.shape[0], nReps)) 
    
    for rep in range(nReps):
        alldim[:,rep] = wmp_train[rep]['posttrain']['lc'][-1,-5]
        allcorr[:,rep] = wmp_train[rep]['posttrain']['lc'][:ntr,-3]  #wmp_train[rep]['posttrain']['lc'][:,-3]
        allgrad[:,rep] = wmp_train[rep]['posttrain']['lc'][:,-2]
    alldim = np.mean(alldim)
    allcorr = np.median(allcorr)#np.median(allcorr)
    allgrad = np.mean(allgrad,axis=1)

    return {'dim':alldim, 'corr_fbk':allcorr, 'grad':allgrad}


def return_base_angle( file, wmp_train ):
    newExp = pnet.postModel( model_path=file, adapt_p={'train_inp' : False, 'train_rec': False, 'train_fbk':False} )
    newExp.load_rnn()
    newExp.restore_model()
    orig_p = newExp.model.save_parameters()
    newExp.params['jump_amp']=0.02
    dic = newExp.test_model()
    wout_base = orig_p['W_out_0']

    wperturb = wmp_train['posttrain']['params1']['W_out_0']

    return np.rad2deg(np.min(ll.subspace_angles(wperturb, wout_base))), newExp, dic, wout_base

def return_speed_loss( wmp_train_map, idx=-1, avg_window=20 ):
    nReps = len(wmp_train_map)
    loss = wmp_train_map[0]['posttrain']['lc'][:,idx]
    allloss = np.zeros((loss.shape[0],nReps))
    for rep in range(nReps):
        allloss[:,rep]= wmp_train_map[rep]['posttrain']['lc'][:,idx]
    allloss = np.mean(allloss, axis=1)

    # fit exponential in the future??
    res = mm.fit_exp( uniform_filter1d(allloss, size=avg_window) )
    speed =  res['speed']
    fitted_k = res['fitted_k']

    return speed, fitted_k, allloss

def return_speed_learning( wmp_train_map, idx=-1, avg_window=20, use_mod=None, use_fbwt='W_fbk_0' ):

    nReps = len(wmp_train_map)
    #print('new map...')
    #print(nReps)
    hitrate = wmp_train_map[0]['posttrain']['lc'][:,idx]
    

    #print(hitrate)
    allh = np.zeros((hitrate.shape[0],nReps))
    for rep in range(nReps):
        allh[:,rep]= wmp_train_map[rep]['posttrain']['lc'][:,idx]
    hitrate = np.mean(allh, axis=1)
    #print(hitrate.shape)

    res = mm.fit_logit( uniform_filter1d(hitrate, size=avg_window) )
    speed =  res['speed']
    fitted_k = res['fitted_k']
    

    

    fb0, fb1, delta = mm.compute_wt_change(  wmp_train_map[0], use_mod=use_mod, usewt=use_fbwt )
    angFbk = mm.compute_wt_angle(  wmp_train_map[0], usewt='W_fbk_0' )
    

    inp0, inp1, deltainp = mm.compute_wt_change(  wmp_train_map[0], usewt='W_in_0' )
    angInp = mm.compute_wt_angle(  wmp_train_map[0], usewt='W_in_0' )

    res.update({ 'fbNorm0':fb0, 'fbNorm1':fb1, 'fbDelta':delta, 'fbAngle':angFbk,
                 'inNorm0':inp0, 'inNorm1':inp1, 'inDelta':deltainp, 'inAngle':angInp})

    return speed, fitted_k, res



