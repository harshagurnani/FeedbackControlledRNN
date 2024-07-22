import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


import torch
import tools.run_network_ as rnet
import tools.perturb_network_ as pnet
import tools.wout_perturbations as wpert
import tools.analysis_helpers as ah
import tools.measures as mm

import matplotlib.pyplot as pp
import matplotlib as mpl
from scipy.ndimage import uniform_filter1d
import scipy.linalg as ll


def train_wmp(rdic, pfolder = 'wmp/', folder = 'relu_rnn_/', resname = 'trained_wmp', pert_params={}, adaptparams={}, device=None, suffix='', suffix_save='', index=1, maptype='wmp' , neuron_gps=None, train_seed=1010203 ):
    
    pfolder=pfolder+folder

    # open cuda connection
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda', index=index)
        else:
            device = torch.device('cpu')

    params = {'nPC':8, 'ntrain':10, 'nReps':1, 'cv_range':[1,5], 'success_range':[0.2,0.7]}
    params.update( pert_params )

    ntrain = params['ntrain']
    nReps = params['nReps']
    nPC = params['nPC']

    cv_range = params['cv_range']
    success_range = params['success_range']
    print(success_range)


    #######
    # Load model

    file = rdic['file']
    print(file)

    # Training parameters:
    # need to have training params set before loading model - because parameters requires_grad value is updated only at model generation!
    n1 = 100
    adapt_p = { 'maxT':800, 'tsteps':800, 'test_maxT':1500, 'lr': 0.001, 'batch_size':1, 'training_epochs':5, 'perturb_epochs':5,
                'train_inp' : True, 'train_rec': False, 'train_fbk':True, 'train_inp_pre': False, 'train_fbk_pre': False, 'train_rec_mf':False,
                'thresh': 0.10, 'loss_mask':None, 'loss_t':300, 'jump_amp': 0.01, 'hit_period': 200, 'sigma_n':0.0 }
    adapt_p.update( adaptparams )
    print(adapt_p)
    newExp = pnet.postModel( model_path=file, adapt_p=adapt_p )
    
    # load model and update rnn size(n1) - dependent parameters:
    newExp.params['device'] = device
    newExp.load_rnn()
    seed = newExp.simparams['rand_seed']
    n1 = newExp.simparams['n1']
    #adapt_p.update({ 'model_type':newExp.simparams['model_type'], 'n1':n1, 'alpha1':5e-2/n1, 'alpha2':5e-2/n1, 'gamma1':5e-2/n1, 'gamma2':5e-2/n1,  'beta1':0.01,'beta2':0.01})
    adapt_p.update({ 'model_type':newExp.simparams['model_type'], 'n1':n1, 'alpha1':adapt_p['alpha1']/n1, 'alpha2':adapt_p['alpha2']/n1, 'gamma1':adapt_p['gamma1']/n1, 'gamma2':5e-2/n1,  'beta1':adapt_p['beta1'],'beta2':0.01})
    adapt_p.update(adaptparams)
    newExp.params.update(adapt_p)
    newExp.reset_training_params(adapt_p)
    newExp.model.sigma_n = adaptparams['sigma_n']       # update activity noise
    
    # set up directories:
    subf = pfolder+ 'Model_'+np.str_(seed)+suffix+'/'
    savfolder = subf+'trained'+suffix_save+'/'
    if not os.path.exists( savfolder ):
        os.mkdir(savfolder)

    # select maps to train: (reset rng seed):
    np.random.seed(train_seed)
    cv = rdic['closedv']
    sx = rdic['perfrate'][:,1]  # rdist , success, mindist, acqtime
    allpass = (cv>cv_range[0])*(cv<cv_range[1])*(sx>success_range[0])*(sx<success_range[1])

    allmaps = rdic['good_maps']
    if maptype=='rmp':
        pass_maps = [allmaps[xx] for xx in range(len(allmaps)) if allpass[xx]]
    else:
        pass_maps = allmaps[allpass]

    nx = min(ntrain, len(pass_maps))
    if maptype=='wmp' or maptype=='omp':
        totrain = np.random.choice( pass_maps, nx, replace=False )  #random sample to train
    else:
        totrain = np.random.choice( np.arange(len(pass_maps)), nx, replace=False ) 

    # Save baseline map
    intuitive = rdic['intuitive']
    w_coef = intuitive['LinModel'].coef_
    UPC = intuitive['PCRes'].components_
    wout_base = intuitive['wout']           # intutive decoder
    if maptype=='omp':
        neuron_gps=rdic['neuron_gps']
    #wout_intuit = w_coef @ UPC

    # model run with intuitive decoder
    newExp.restore_model()
    orig_p = newExp.model.save_parameters()
    orig_p['W_out_0'] = wout_base

    newExp.model.load_parameters(orig_p)


    dic_orig = newExp.test_model()


    train_res = []  # results

    for jj in range(nx):
        map_res = []
        map_id = totrain[jj]
        p,_ = wpert.get_random_permutations( nPC )
        if maptype=='omp' or maptype=='wmp':
            permid = wpert.get_perm_order( p, map_id )
            print(permid)
        else:
            permid = map_id

        newExp.restore_model()
        if maptype=='wmp':
            print('Generating perturbation from permutation ID as WMP')
            _, Wout_full,_ = wpert.get_wmp( nDim=params['nPC'], lin_coef=w_coef, U=UPC,Wout_orig=wout_base.T, idx=permid )
        elif maptype=='omp':
            print('Generating perturbation from permutation ID as OMP')
            _, Wout_full,_ = wpert.get_omp( allgps=neuron_gps, n_groups0=params['nPC'], lin_coef=w_coef, U=UPC, Wout_orig=wout_base.T, gp_order=permid )
        elif maptype=='rmp':
            print('Generating perturbation from saved maps as RMP')
            Wout_full =  pass_maps[map_id].T
        wout = Wout_full.T          # current perturbed readout

        modelW = newExp.model.save_parameters()
        modelW['W_out_0'] = wout                    # current perturbed readout

        for rep in range(nReps):
            newExp.model.load_parameters(modelW)    # load perturbation
            dic = newExp.test_model()               # pre-training data - of perturbed readout
            if rep==0:
                newExp.plot_model_results(dic, savname=savfolder+'test_id_'+np.str_(map_id)+'_rep_'+np.str_(rep) )
            
            # which variables are being trained??
            print('Recurrent and fbk wts grad reqts:')
            print(newExp.model.W_rec.requires_grad)
            print(newExp.model.W_fbk.requires_grad)
            
            tdic = newExp.train_model()
            tdic = newExp.test_model(tdic)          # post-training data (activity + model)
            if rep==0:  # only plot once
                newExp.plot_model_results(tdic, savname=savfolder+'train_id_'+np.str_(map_id)+'_rep_'+np.str_(rep) )
            
            #print(tdic.keys())

            #outp = newExp.model.save_parameters()
            if rep==0:
                wmp_train = {'map_id':map_id, 'file':file, 'rep': rep, 'idx':permid, 'wout':wout, 'intuitive':intuitive, 'pretrain':dic, 'posttrain':tdic, 'maptype':maptype , 'neuron_gps':neuron_gps }
                stats = analyze_training( wmp_train, trained=True, dic_orig=dic_orig, thresh=adapt_p['thresh'] )
                wmp_train.update({'stats':stats})
            else:
                tdic.update({'params0':{}, 'params1':{}, 'res':{}})
                wmp_train = {'map_id':map_id, 'file':file, 'rep': rep, 'idx':permid, 'posttrain':tdic } #'pretrain':dic, 
            map_res.append( wmp_train )

            
            newExp.restore_model()

        train_res.append( map_res )

    train_params = {'nPC':nPC, 'ntrain':ntrain, 'nReps':nReps, 'totrain':totrain, 'cv_range':cv_range, 'success_range':success_range}
    alldic = {'train_params':train_params, 'adapt_p':adapt_p, 'file':file, 'train_res':train_res}

    # # ---------- Save results ------------------
    np.save( subf+resname+suffix_save, alldic )


    # # ---------- Plot results ------------------
    if nx>0:
        # Figure 1: All loss and hit rate curves
        fig = pp.figure(figsize=(10,8))
        ax0 = fig.add_subplot(211)
        ax1 = fig.add_subplot(212)

        # analyse results:
        cm = mpl.colormaps['plasma']
        avg_window=20 #p['avg_window'] = 20

        avg_loss_wmp = np.zeros((train_res[0][0]['posttrain']['lc'].shape[0], nx))
        avg_loss_wmp.fill( np.nan )
        avg_hit_wmp = avg_loss_wmp.copy()

        for jj in range(nx):
            tmp = np.zeros((avg_loss_wmp.shape[0], nReps))
            tmp.fill(np.nan)
            tmp2 = tmp.copy()
            for rep in range(nReps):
                lc = train_res[jj][rep]['posttrain']['lc']
                #loss = lc[:,0]
                hitrate = lc[:,-1]
                yy=(lc[:,0]) # loss
                tmp[:,rep] = yy
                ax0.plot(np.log10(yy),color=cm(jj/nx))
                zz = uniform_filter1d(hitrate, size=avg_window)
                ax1.plot( zz,color=cm(jj/nx))
                tmp2[:,rep] = zz
            avg_loss_wmp[:,jj] = np.nanmean( tmp, axis=1)   # avg loss for each wmp - across repeat training
            avg_hit_wmp[:,jj] = np.nanmean( tmp2, axis=1)   # avg hitrate for each wmp  - across repeat training
        ax1.set_xlabel('Training epoch')
        ax0.set_ylabel('Log training Loss')
        ax1.set_ylabel('Hit rate')
        pp.savefig(subf+f'train_loss_PCs_{nPC}_noscale'+suffix_save+'.png')
        pp.close()


        fig2 = pp.figure(figsize=(10,8))
        ax0 = fig2.add_subplot(211)
        ax1 = fig2.add_subplot(212)
        for jj in range(nx):
            ax0.plot( np.log10(avg_loss_wmp[:,jj]),color=cm(jj/nx))
            ax1.plot( avg_hit_wmp[:,jj],color=cm(jj/nx))
        ax1.set_xlabel('Training epoch')
        ax0.set_ylabel('Mean log training Loss')
        ax1.set_ylabel('Mean hit rate')
        pp.savefig(subf+f'train_loss_WMP_avg_PCs_{nPC}_noscale'+suffix_save+'.png')
        pp.close()


def analyze_training( wmp_train, trained=True, dic_orig={}, thresh=0.10 ):
    '''
    Gather statistics about training:
    - Hit rate at 50, 100, 200 trials (nan if not trained)
    - Progress: p_total, p_U, p_ortho for pre- and post- per target
    - Asymmetry of hit rate and progress
    - Magnitude of activity pre- and post- training
    - (To do) normalized change in wts
    - Top PCs

    OLD CODE _---------- These stats are not used anymore
    '''
    #{'map_id':map_id, 'file':file, 'rep': rep, 'idx':permid, 'wout':wout, 'intuitive':intuitive, 'wout_intuit':wout_intuit, 'pretrain':dic, 'posttrain':tdic , 'wfinal':wfinal }

    W_base = wmp_train['intuitive']['wout']
    px_base = wmp_train['intuitive']['PCRes']
    U_base = px_base.components_
    W_pert = wmp_train['wout']

    X_orig = dic_orig['res']['activity1']
    X_orig_flat = np.reshape(X_orig,(X_orig.shape[0]*X_orig.shape[1],X_orig.shape[2]))

    X_pre = wmp_train['pretrain']['res']['activity1']
    X_pre_flat = np.reshape(X_pre,(X_pre.shape[0]*X_pre.shape[1],X_pre.shape[2]))
    stim_pre = wmp_train['pretrain']['res']['stimulus']
    if trained:
        X_post = wmp_train['posttrain']['res']['activity1']    
        X_post_flat = np.reshape(X_post,(X_post.shape[0]*X_post.shape[1],X_post.shape[2]))
        stim_post = wmp_train['posttrain']['res']['stimulus']
    else:
        X_post=X_pre
        X_post_flat = X_pre_flat
        stim_post = stim_pre
    nTr = X_post.shape[0]
    
    # Activity in Intrinsic manifold - progress toward target
    progress_pre = ah.get_progress_ratios(  X_pre, stim_pre, W_pert, px_base )
    progress_post = ah.get_progress_ratios(  X_post, stim_post, W_pert, px_base )

    # Neural activity norm
    act_norm_orig = np.mean(ll.norm(X_orig, axis=-1))
    act_norm_pre = np.mean(ll.norm(X_pre, axis=-1))
    act_norm_post = np.mean(ll.norm(X_post, axis=-1))

    # Fractional variance in "Intrinsic manifold"
    X_proj_pre = px_base.transform(X_pre_flat)  @ U_base
    X_proj_post = px_base.transform(X_post_flat)  @ U_base
    expVar_pre = mm.get_variance_in_U(X_pre, U_base.T ) #np.sum( np.var(X_proj_pre, axis=-1))/np.sum( np.var(X_pre_flat, axis=-1))
    expVar_post = mm.get_variance_in_U(X_post, U_base.T ) #np.sum( np.var(X_proj_post, axis=-1))/np.sum( np.var(X_post_flat, axis=-1))

    ## Top PCs of closed loop performance?
    px_pre, _ = ah.get_X_pcs( X_pre, nC=U_base.shape[0] )
    px_post, _ = ah.get_X_pcs( X_post, nC=U_base.shape[0] )
    PC_angle_pre = ll.subspace_angles(px_pre.components_.T, U_base.T )
    PC_angle_post = ll.subspace_angles(px_post.components_.T, U_base.T )

    vel_pre = X_orig_flat @ W_pert
    vel_post = X_post_flat @ W_pert
    rn = np.random.randint(1000)
    if np.random.random()<0.1:
        pp.plot( vel_pre[:,0], vel_pre[:,1], c='k')
        pp.plot( vel_post[:,0], vel_post[:,1], c='r')
        pp.savefig('vel_repertoire_wpert_'+np.str_(rn)+'.png')
        pp.close()

    vel_pre = X_orig_flat @ W_base
    vel_post = X_post_flat @ W_base
    if np.random.random()<0.1:
        pp.plot( vel_pre[:,0], vel_pre[:,1], c='k')
        pp.plot( vel_post[:,0], vel_post[:,1], c='b')
        pp.savefig('vel_repertoire_wbase_'+np.str_(rn)+'.png')
        pp.close()


    perf_res_pre = ah.get_performance( wmp_train['pretrain'], nTest=nTr,thresh=thresh )
    perf_res_post = ah.get_performance( wmp_train['posttrain'], nTest=nTr, thresh=thresh )

    stats = {'progress_pre':progress_pre,  'progress_post':progress_post , 'PC_angle_pre':PC_angle_pre, 'PC_angle_post':PC_angle_post, 
             'Xnorm_pre':act_norm_pre, 'Xnorm_post': act_norm_post,  'Xnorm_orig':act_norm_orig, 'expVar_pre':expVar_pre, 'expVar_post':expVar_post
             }
    
    print(stats)
    stats.update({ 'perf_res_pre':perf_res_pre, 'perf_res_post':perf_res_post })
    return stats
