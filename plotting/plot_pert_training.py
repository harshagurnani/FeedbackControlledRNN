import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
sys.path.insert(0, parentdir) 

import matplotlib.pyplot as pp
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)
import seaborn as sns
from scipy import stats


import scipy.linalg as ll
import control as ctrl
import control.matlab as mt
from control.statesp import StateSpace
import control.modelsimp as msimp


def plot_progress( results, savfolder='', suffix='_test' ):
    nMaps = len(results['progress_pre'])

    pp.figure()
    pp.scatter( np.ones((nMaps,1))*1, results['progress_pre'], c='r', alpha=0.3)
    pp.scatter( np.ones((nMaps,1))*2, results['progress_post'], c='k', alpha=0.3)
    pp.scatter( np.ones((nMaps,1))*4, results['progress_pre_in'], c='r', alpha=0.3)
    pp.scatter( np.ones((nMaps,1))*5, results['progress_post_in'], c='k', alpha=0.3)
    pp.bar( x=[1,2,4,5], height=[np.nanmean(results['progress_pre']), np.nanmean(results['progress_post']), np.nanmean(results['progress_pre_in']), np.nanmean(results['progress_post_in']) ], 
           width=0.5, align='center', color=['r','k','r','k'], alpha=0.5)
    pp.xticks([1,2,4,5], labels=['Total: Pre','Post','Inside: Pre','Post'])
    pp.ylabel('Progress towards target')
    pp.savefig(savfolder+'Progress_combined'+suffix+'.png')
    pp.close('all')

    pp.figure()
    pp.scatter(np.ones(len(results['p_total_target_pre']) ),results['p_total_target_pre'], c='r',alpha=0.3 )
    pp.scatter(np.ones(len(results['p_total_target_post']) )*2,results['p_total_target_post'], c='k',alpha=0.3 )
    pp.bar( [1,2], [np.nanmean(results['p_total_target_pre']), np.nanmean(results['p_total_target_post'])], width=0.5, color=['r','k'], alpha=0.5)
    pp.xticks([1,2],labels=['Pre-training', 'Post'])
    pp.ylabel('Progress towards target')
    pp.savefig(savfolder+'Progress_alltarget'+suffix+'.png')
    pp.close('all')

    return None



def plot_all( results, savfolder='', suffix='_test' ):
    # Plot 1:
    # Progress pre- and post- training:
    nMaps = len(results['progress_pre'])
    plot_progress_2( results, savfolder=savfolder, suffix=suffix)

    # Plot 2:
    # Activity Norm 2
    plot_activity_norm_2( results, savfolder=savfolder, suffix=suffix)

    pp.figure()
    pp.scatter(np.ones((nMaps,1)), results['progress_asymm_pre'], c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2, results['progress_asymm_post'], c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(results['progress_asymm_pre']), np.nanmean(results['progress_asymm_post'])], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,5])
    pp.ylabel('Asymmetry of progress (across targets)')
    pp.savefig(savfolder+'Asymmetry_progress'+suffix+'.png')
    pp.close('all')

    pp.figure()
    #for jj in range(nMaps):
    #    pp.plot([1,2], [ results['expVar_pre'][jj],results['expVar_post'][jj]], c='k', alpha=0.3)
    pp.scatter(np.ones((nMaps,1)), results['expVar_pre'], c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2, results['expVar_post'], c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(results['expVar_pre']), np.nanmean(results['expVar_post'])], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,1.05])
    pp.ylabel('Fractional variance within Original top PCs')
    pp.savefig(savfolder+'expVar'+suffix+'.png')
    pp.close('all')

    pp.figure()
    pp.scatter(np.ones((nMaps,1)), np.rad2deg(np.min(results['PC_angle_pre'], axis=-1)), c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2, np.rad2deg(np.min(results['PC_angle_post'], axis=-1)), c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(np.rad2deg(np.min(results['PC_angle_pre'], axis=-1))), np.nanmean( np.rad2deg(np.min(results['PC_angle_post'], axis=-1)))], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,90])
    pp.ylabel('Angle with Original top PCs')
    pp.savefig(savfolder+'minAngle_PCs'+suffix+'.png')
    pp.close('all')

    pp.figure()
    meanA_pre = np.rad2deg(np.arccos(np.nanmean(np.cos(results['PC_angle_pre']), axis=-1)))
    meanA_post = np.rad2deg(np.arccos(np.nanmean(np.cos(results['PC_angle_post']), axis=-1)))
    pp.scatter(np.ones((nMaps,1)), meanA_pre, c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2,meanA_post , c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(meanA_pre), np.nanmean(meanA_post )], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,90])
    pp.ylabel('Mean angle with Original top PCs')
    pp.savefig(savfolder+'meanAngle_PCs'+suffix+'.png')
    pp.close('all')


    pp.figure()
    for jj in range(nMaps):
        pp.plot([1,2], [results['hit_rate_pre'][jj],results['hit_rate_post'][jj]], c='k', alpha=0.3)
    pp.scatter(np.ones((nMaps,1)), results['hit_rate_pre'], c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2, results['hit_rate_post'], c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(results['hit_rate_pre']), np.nanmean(results['hit_rate_post'])], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,1.05])
    pp.ylabel('Hit rate (within 0.1 of target)')
    pp.savefig(savfolder+'Hit_rate'+suffix+'.png')
    pp.close('all')

    pp.figure()
    for jj in range(nMaps):
        pp.plot([1,2], [results['theta_OF_pre'][jj],results['theta_OF_post'][jj]], c='k', alpha=0.3)
    pp.scatter(np.ones((nMaps,1)), results['theta_OF_pre'], c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2, results['theta_OF_post'], c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(results['theta_OF_pre']), np.nanmean(results['theta_OF_post'])], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,90])
    pp.ylabel('Angle_between_Out_Fbk')
    pp.savefig(savfolder+'theta_Out_Fbk'+suffix+'.png')
    pp.close('all')



    pp.figure()
    meanA_pre = results['theta_Out_Amplified_pre']
    meanA_post = results['theta_Out_Amplified_post']
    for jj in range(nMaps):
        pp.plot([1,2], [meanA_pre[jj],meanA_post[jj]], c='k', alpha=0.3)
    pp.scatter(np.ones((nMaps,1)), meanA_pre, c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2,meanA_post , c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(meanA_pre), np.nanmean(meanA_post )], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,90])
    pp.ylabel('Min Out angle with Rec amplification modes')
    pp.savefig(savfolder+'Out_amplified_angle'+suffix+'.png')
    pp.close('all')

    pp.figure()
    meanA_pre = results['theta_Fbk_Sensitive_pre']
    meanA_post = results['theta_Fbk_Sensitive_post']
    #print(meanA_post)
    for jj in range(nMaps):
        pp.plot([1,2], [meanA_pre[jj],meanA_post[jj]], c='k', alpha=0.3)
    pp.scatter(np.ones((nMaps,1)), meanA_pre, c='r', alpha=0.3)
    pp.scatter(np.ones((nMaps,1))*2,meanA_post , c='k', alpha=0.3)
    pp.bar( [1,2], [np.nanmean(meanA_pre), np.nanmean(meanA_post )], width=0.5, color=['r','k'], alpha=0.5 )
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,90])
    pp.ylabel('Min Fbk angle with Rec sensitivity modes')
    pp.savefig(savfolder+'Fbk_sensitive_angle'+suffix+'.png')
    pp.close('all')

    fig = pp.figure()
    ax = fig.add_subplot(111)
    ax.hist(results['speed']/1000.0, bins=np.arange(0, 0.2, 0.025),  lw=3, fc=(1, 0, 0, 0.5))
    pp.ylabel('Count')
    pp.xlabel('Learning speed')
    pp.savefig(savfolder+'learning_speed'+suffix+'.png')

    fig = pp.figure()
    ax = fig.add_subplot(111)
    ax.hist(results['fitted_k'], bins=np.arange(0, 1, 0.05),  lw=3, fc=(1, 0, 0, 0.5))
    pp.ylabel('Count')
    pp.xlabel('Learning rate')
    pp.savefig(savfolder+'learning_k'+suffix+'.png')
    pp.close('all')

    fig = pp.figure()
    ax = fig.add_subplot(111)
    ax.hist(results['hit_rate_pre'], bins=np.arange(0, 1.1, 0.1),  lw=3, fc=(0, 0, 0, 0.3))
    ax.hist(results['hit_rate_post'], bins=np.arange(0, 1.1, 0.1),  lw=3, fc=(1, 0, 0, 0.3))
    pp.ylabel('Count')
    pp.xlabel('Hit rate')
    pp.savefig(savfolder+'hitrate_hist_'+suffix+'.png')
    pp.close('all')

    return None


################### HELPERS ##################################
##############################################################




'''
def return_controls( params ):
    Wrec = params['W_rec_0'].T 
    A = Wrec - np.eye(Wrec.shape[0])
    Wout = params['W_out_0'].T
    Wfbk = params['W_fbk_0'].T
    Win = params['W_in_0'].T
    
    ObsGram = ctrl.lyap( A.T, Wout.T@Wout)   #<---- need to make it A.T
    Wout_orth = ll.null_space(Wout)
    Wout_orth = Wfbk
    alpha = np.trace(Wout_orth.T @ ObsGram @ U )/(A.shape[0]-2)

    CtrlGram_FF = ctrl.lyap( A, Win@Win.T )
    CtrlGram_FB = ctrl.lyap( A, Wfbk@Wfbk.T )
    beta_ff = np.trace( Wout @ CtrlGram_FF @ Wout.T)/2
    beta_fb = np.trace( Wout @ CtrlGram_FB @ Wout.T)/2
    
    dic={'obsv':alpha, 'ctrb_ff':beta_ff, 'ctrb_fb':beta_fb}
    print(dic)

    UA, sA, VA = ll.svd( Wout @ll.inv(-A), compute_uv=True )
    topSensitive = VA[:Wout.shape[0],:].T
    overlap = np.sqrt(np.trace( Wfbk.T @ topSensitive @ topSensitive.T @ Wfbk)/Wout.shape[0])

    UA, sA, VA = ll.svd( A, compute_uv=True )
    topAmplified = UA[:,:8]
    overlap_out = np.sqrt(np.trace( Wout @ topAmplified @ topAmplified.T @ Wout.T )/8)

    dic.update({'Fbk_right_overlap':overlap,   'Out_left_overlap':overlap_out})

    return dic#alpha, beta_ff, beta_fb
'''


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

'''
###### Computing obsv and ctrb using state space module of control toolbox -> only works for stable systems
def return_controls( params, U ):
    Wrec = params['W_rec_0'].T 
    A = Wrec - np.eye(Wrec.shape[0])
    #Ua, sa, Va = ll.svd(A.T, compute_uv=True)
    #Ua = Ua[:,:U.shape[1]]
    Wout = params['W_out_0'].T # 2 x N
    Wfbk = params['W_fbk_0'].T # N x 2
    Win = params['W_in_0'].T   # N x 3
    D_fb = np.zeros((Wout.shape[0], Wfbk.shape[1]))
    D_ff = np.zeros((Wout.shape[0], Win.shape[1]))

    # construct state space:
    fsys_ff = StateSpace(A, Win, Wout, D_ff)
    fsys_fb = StateSpace(A, Wfbk, Wout, D_fb)

    
    # solve the lyap eqn for observability
    #ObsGram = ctrl.lyap( A.T, Wout.T@Wout) 
    
    try:
        ObsGram = mt.gram( fsys_fb, 'o')
        # calculate observability of intrinsic manifold
        alpha = np.trace(U.T @ ObsGram @ U )/(A.shape[0]-U.shape[1])
        
        # solve the lyap eqn for controllability
        #CtrlGram_FF = ctrl.lyap( A, Win@Win.T )
        #CtrlGram_FB = ctrl.lyap( A, Wfbk@Wfbk.T )
        
        CtrlGram_FF = mt.gram(fsys_ff, 'c')
        CtrlGram_FB = mt.gram(fsys_fb, 'c')
        # compute controllability of readout
        beta_ff = np.trace( Wout @ CtrlGram_FF @ Wout.T)/2
        beta_fb = np.trace( Wout @ CtrlGram_FB @ Wout.T)/2

    except:
        print('system unstable... ')
        alpha = 0.01
        beta_ff = 1
        beta_fb = 1
    
    dic={'obsv':alpha, 'ctrb_ff':beta_ff, 'ctrb_fb':beta_fb}
    print(dic)

    UA, sA, VA = ll.svd( Wout @ll.inv(-A), compute_uv=True )
    topSensitive = VA[:Wout.shape[0],:].T
    overlap = np.sqrt(np.trace( Wfbk.T @ topSensitive @ topSensitive.T @ Wfbk)/Wout.shape[0])

    UA, sA, VA = ll.svd( A, compute_uv=True )
    topAmplified = UA[:,:8]
    overlap_out = np.sqrt(np.trace( Wout @ topAmplified @ topAmplified.T @ Wout.T )/8)

    dic.update({'Fbk_right_overlap':overlap,   'Out_left_overlap':overlap_out})

    return dic
'''

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



def plot_progress_2( results, *args, savfolder='', suffix='_test', clrs=None ):

    all_results = [results]
    for new_res in args:
        all_results.append(new_res)

    nres = len(all_results)
    width = 0.5/nres
    allX = np.array([1,2,4,5,7,8])

    cm = mpl.colormaps['jet']

    pp.figure()
    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        pp.scatter( np.ones((nMaps,1))*allX[0]+shift, results['progress_pre'], color=clr, alpha=0.3)
        pp.scatter( np.ones((nMaps,1))*allX[1]+shift, results['progress_post'], color=clr, alpha=0.3)
        pp.scatter( np.ones((nMaps,1))*allX[2]+shift, results['progress_pre_in'], color=clr, alpha=0.3)
        pp.scatter( np.ones((nMaps,1))*allX[3]+shift, results['progress_post_in'], color=clr, alpha=0.3)
        pp.scatter( np.ones((nMaps,1))*allX[4]+shift, np.divide(results['progress_pre_in'],results['progress_pre']), color=clr, alpha=0.3)
        pp.scatter( np.ones((nMaps,1))*allX[5]+shift, np.divide(results['progress_post_in'],results['progress_post']), color=clr, alpha=0.3)
        height=[ np.nanmedian(results['progress_pre']), np.nanmedian(results['progress_post']), 
                np.nanmedian(results['progress_pre_in']), np.nanmedian(results['progress_post_in']), 
                np.nanmedian(np.divide(results['progress_pre_in'],results['progress_pre'])) , np.nanmedian(np.divide(results['progress_post_in'],results['progress_post'])) ]
        pp.bar( x=allX+shift, height=height, width=width, align='center', color=clr, alpha=0.5)
    
    pp.xticks(allX, labels=['Total: Pre','Post','Inside: Pre','Post','fracIn:Pre','Post'])
    pp.ylabel('Progress towards target')
    pp.ylim([-5,3])
    pp.savefig(savfolder+'Progress_combined'+suffix+'.png')
    pp.close('all')

    
    pp.figure()
    allX = np.array([1,2])
    for mm in range(nres):
        results = all_results[mm]
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        pp.scatter(np.ones(len(results['p_total_target_pre']) )*allX[0]+shift,results['p_total_target_pre'], color=clr,alpha=0.3 )
        pp.scatter(np.ones(len(results['p_total_target_post']) )*allX[1]+shift,results['p_total_target_post'], color=clr,alpha=0.3 )
        pp.bar( allX+shift, [np.nanmedian(results['p_total_target_pre']), np.nanmedian(results['p_total_target_post'])], width=width, color=clr, alpha=0.5)
    
    pp.xticks(allX,labels=['Pre-training', 'Post'])
    pp.ylabel('Progress towards target')
    pp.savefig(savfolder+'Progress_alltarget'+suffix+'.png')
    pp.close('all')
    

    return None


def plot_activity_norm_2( results, *args, savfolder='', suffix='_test', clrs=None ):

    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']

    pp.figure()
    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        pp.scatter( np.array([np.ones((nMaps,1)), np.ones((nMaps,1))*2, np.ones((nMaps,1))*3])+shift, [results['Xnorm_pre'], results['Xnorm_post'], results['Xnorm_orig']], color=clr, alpha=0.1)
        pp.bar( np.array([1,2,3])+shift, [np.nanmedian(results['Xnorm_pre']), np.nanmedian(results['Xnorm_post']), np.nanmedian(results['Xnorm_orig'])], width=width, color=clr, alpha=0.5)
        lower_error = [-np.quantile(results['Xnorm_pre'],.05)+np.nanmedian(results['Xnorm_pre']), -np.quantile(results['Xnorm_post'],.05)+np.nanmedian(results['Xnorm_post']), -np.quantile(results['Xnorm_orig'],.05)+np.nanmedian(results['Xnorm_orig'])]
        upper_error = [np.quantile(results['Xnorm_pre'],.95)-np.nanmedian(results['Xnorm_pre']), np.quantile(results['Xnorm_post'],.95)-np.nanmedian(results['Xnorm_post']), np.quantile(results['Xnorm_orig'],.95)-np.nanmedian(results['Xnorm_orig'])]
        asymmetric_error = np.array(list(zip(lower_error, upper_error))).T
        pp.errorbar( np.array([1,2,3])+shift+.1, [np.nanmedian(results['Xnorm_pre']), np.nanmedian(results['Xnorm_post']), np.nanmedian(results['Xnorm_orig'])],yerr=asymmetric_error, color=clr, fmt='o')
    
    pp.xticks([1,2,3], labels=['Pre-training','Post','Baseline'])
    pp.ylabel('Activity norm')
    pp.ylim([0,10])
    pp.savefig(savfolder+'Activity_norm'+suffix+'.png')
    pp.close('all')
    return None


def plot_asymmetry_2( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    pp.figure()
    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        pp.scatter(np.ones((nMaps,1))+shift, results['progress_asymm_pre'], color=clr, alpha=0.08)
        pp.scatter(np.ones((nMaps,1))*2+shift, results['progress_asymm_post'], color=clr, alpha=0.08)
        pp.bar( np.array([1,2])+shift, [np.nanmean(results['progress_asymm_pre']), np.nanmean(results['progress_asymm_post'])], width=width, color=clr, alpha=0.5 )
        pp.errorbar(1+shift, np.nanmean(results['progress_asymm_pre']), yerr=np.nanstd(results['progress_asymm_pre']), fmt='o', color=clr, ecolor=clr)
        pp.errorbar(2+shift, np.nanmean(results['progress_asymm_post']), yerr=np.nanstd(results['progress_asymm_post']), fmt='o', color=clr, ecolor=clr)
    
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,5])
    pp.ylabel('Asymmetry of progress (across targets)')
    pp.savefig(savfolder+'Asymmetry_progress'+suffix+'.png')
    pp.close('all')


    return None


def plot_expvar_2( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    f1 = pp.figure()
    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        pp.scatter(np.ones((nMaps,1))+shift, results['expVar_pre'], color=clr, alpha=0.3)
        pp.scatter(np.ones((nMaps,1))*2+shift, results['expVar_post'], color=clr, alpha=0.3)
        pp.bar( np.array([1,2])+shift, [np.nanmean(results['expVar_pre']), np.nanmean(results['expVar_post'])], width=width, color=clr, alpha=0.5 )
        pp.errorbar([1+shift,2+shift], [np.nanmean(results['expVar_pre']), np.nanmean(results['expVar_post'])], [np.nanstd(results['expVar_pre']), np.nanstd(results['expVar_post'])], fmt='o', color=clr)
    
    pp.xticks([1,2], labels=['Pre-training','Post'])
    pp.ylim([0,1.05])
    pp.ylabel('Fractional variance within Original top PCs')
    pp.savefig(savfolder+'expVar'+suffix+'.png')
    pp.close(f1)

# ############################################ #

def plot_expvar_hist( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    f1 = pp.figure()
    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        pp.hist((results['expVar_post']-results['expVar_pre']), bins=np.arange(start=-0.5,stop=0.2,step=0.05), color=clr, alpha=0.5, lw=3)
        m1 = np.mean(results['expVar_post']-results['expVar_pre'])
        pp.plot([m1,m1],[0,len(results['expVar_post'])/3], color=clr)


    pp.xlabel('Change in variance within original top PCs')
    pp.ylabel('Count')
    pp.savefig(savfolder+'expVar_hist'+suffix+'.png')
    pp.close(f1)




    return None


# ############################################ #

def plot_hitrate_change_2( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig=pp.figure(figsize=[6,10])
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    fig2=pp.figure(figsize=[6,10])
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)

    fig3=pp.figure(figsize=[6,10])
    ax5 = fig3.add_subplot(211)
    ax6 = fig3.add_subplot(212)

    fig4=pp.figure(figsize=[6,10])
    ax7 = fig4.add_subplot(211)
    ax8 = fig4.add_subplot(212)

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        change=(results['hit_rate_post']-results['hit_rate_pre'])/(1-results['hit_rate_pre'])
        change_atime=results['acq_rate_post']-results['acq_rate_pre']
        xmin = min(np.floor(10*change))/10
        minc = min(0, xmin)
        #ax.hist(change, bins=np.arange(0, 1.05, 0.1),  lw=3, color=clr, alpha=0.5)
        ax.hist(change,  lw=3, color=clr, alpha=0.5)
        ax2.hist(change_atime, bins=np.arange(-1000, 500, 50),  lw=3, color=clr, alpha=0.5)
        m1 = np.nanmean(change)
        ax.plot([m1,m1],[0,len(change)/6],color=clr)
        minc = min(0, min(change))
        ax.set_xlim((minc-0.05,1+0.05))

        ax3.hist(results['hit_rate_pre'], bins=np.arange(0, 1.1, 0.1),  lw=3, color=clr, alpha=0.5)
        ax4.hist(results['hit_rate_post'], bins=np.arange(0, 1.1, 0.1),  lw=3, color=clr, alpha=0.5)
        m1 = np.nanmean(results['hit_rate_pre'])
        m2 = np.nanmean(results['hit_rate_post'])
        ax3.plot([m1,m1],[0,len(change)/6],color=clr)
        ax4.plot([m2,m2],[0,len(change)/6],color=clr)

        ax5.hist(results['acq_rate_pre'], bins=np.arange(0, 1500, 50),  lw=3, color=clr, alpha=0.5)
        ax6.hist(results['acq_rate_post'], bins=np.arange(0, 1500, 50),  lw=3, color=clr, alpha=0.5)
        
        sns.kdeplot(change, color=clr, lw=3, ax=ax7, gridsize=21)
        sns.kdeplot(change_atime, color=clr, lw=3, ax=ax8, gridsize=21)
        m1 = np.nanmean(change)
        ax7.plot([m1,m1],[0,len(change)/100],color=clr)



    ax.set_ylabel('Count')
    ax.set_xlabel('Fractional improvement in hit rate')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Change in acquisition time')

    ax3.set_ylabel('Count')
    ax3.set_xlabel('Hit rate pre perturbation')
    ax4.set_ylabel('Count')
    ax4.set_xlabel('Hit rate post retraining')

    ax5.set_ylabel('Count')
    ax5.set_xlabel('Acq time pre perturbation')
    ax6.set_ylabel('Count')
    ax6.set_xlabel('Acq time post retraining')

    ax7.set_ylabel('Density')
    ax7.set_xlabel('Fractional improvement in hit rate')
    ax8.set_ylabel('Density')
    ax8.set_xlabel('Change in acquisition time')

    fig.savefig( savfolder+'change_success_hist_'+suffix+'.png')
    fig2.savefig( savfolder+'success_hist_'+suffix+'.png')
    fig3.savefig( savfolder+'acqtime_hist_'+suffix+'.png')
    fig4.savefig( savfolder+'change_success_density_'+suffix+'.png')
    pp.close('all')


    return None

# ############################################ #

def plot_hitrate_input( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(8,8))
    ax=fig.add_subplot(211)
    ax2=fig.add_subplot(212)

    fig2 = pp.figure(figsize=[8,8])
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)
    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        change=results['hit_rate_post']-results['hit_rate_pre']
        change_atime=results['acq_rate_post']-results['acq_rate_pre']
        ax.scatter(results['meanInp'],change, color=clr, alpha=0.3)
        ax2.scatter(results['meanFbk'], change, color=clr, alpha=0.3)
        ax3.scatter(results['meanInp'],change_atime, color=clr, alpha=0.3)
        ax3.set_ylim(-1000,100)
        ax4.scatter(results['meanFbk'], change_atime, color=clr, alpha=0.3)
        ax4.set_ylim(-1000,100)
    
    ax.set_ylabel('Change in hit rate')
    ax2.set_ylabel('Change in hit rate')
    ax.set_xlabel('Change in input norm')
    ax2.set_xlabel('Change in feedback norm')
    #ax.set_ylim([150,650])


    ax3.set_ylabel('Change in acq time')
    ax4.set_ylabel('Change in acq time')
    ax3.set_xlabel('Change in input norm')
    ax4.set_xlabel('Change in feedback norm')
    fig.savefig(savfolder+'meaninp_hitrate_change_'+suffix+'.png')
    fig2.savefig(savfolder+'meaninp_acqtime_change_'+suffix+'.png')
    pp.close('all')



# ############################################ #

def plot_speed_input( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(8,8))
    ax=fig.add_subplot(211)
    ax2=fig.add_subplot(212)

    fig2 = pp.figure(figsize=[8,8])
    ax3 = fig2.add_subplot(211)
    ax4 = fig2.add_subplot(212)

    fig3 = pp.figure(figsize=[8,8])
    ax5 = fig3.add_subplot(211)
    ax6 = fig3.add_subplot(212)

    fig4 = pp.figure(figsize=[8,8])
    ax7 = fig4.add_subplot(211)
    ax8 = fig4.add_subplot(212)

    fig5 = pp.figure(figsize=[8,8])
    ax9 = fig5.add_subplot(211)
    ax10 = fig5.add_subplot(212)

    fig6 = pp.figure(figsize=[8,8])
    ax11 = fig6.add_subplot(211)
    ax12 = fig6.add_subplot(212)
    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        ax.scatter(results['meanInp'], results['speed'], color=clr, alpha=0.3)
        ax2.scatter(results['meanInp'], results['fitted_k'], color=clr, alpha=0.3)
        ax2.set_ylim(0,0.15)
        ax3.scatter(results['meanFbk'], results['speed'], color=clr, alpha=0.3)
        ax4.scatter(results['meanFbk'], results['fitted_k'], color=clr, alpha=0.3)
        ax4.set_ylim(0,0.15)

        ax5.scatter(results['meanInp']-results['meanFbk'], results['speed'], color=clr, alpha=0.3)
        ax6.scatter(results['meanInp']-results['meanFbk'], results['fitted_k'], color=clr, alpha=0.3)

        ax7.scatter(results['meanInp']/results['meanFbk'], results['speed'], color=clr, alpha=0.3)
        ax8.scatter(results['meanInp']/results['meanFbk'], results['fitted_k'], color=clr, alpha=0.3)

        ax9.scatter(results['theta_wout_wintuit'], results['speed'], color=clr, alpha=0.3)
        ax10.scatter(results['theta_wout_wintuit'], results['fitted_k'], color=clr, alpha=0.3)
        ax10.set_ylim(0,0.15)

        ax11.scatter(results['theta_wout_wbase'], results['speed'], color=clr, alpha=0.3)
        ax12.scatter(results['theta_wout_wbase'], results['fitted_k'], color=clr, alpha=0.3)
        ax12.set_ylim(0,0.15)
    
    
    
    
    ax.set_ylabel('Learning speed')
    ax2.set_ylabel('Fitted learning rate k')
    ax.set_xlabel('Change in input norm')
    ax2.set_xlabel('Change in input norm')
    
    #ax.set_ylim([150,650])

    ax3.set_ylabel('Learning speed')
    ax4.set_ylabel('Fitted learning rate k')
    ax3.set_xlabel('Change in feedback norm')
    ax4.set_xlabel('Change in feedback norm')

    ax5.set_xlabel('Feedforward-feedback')
    ax6.set_xlabel('Feedforward-feedback')
    ax5.set_ylabel('Learning speed')
    ax6.set_ylabel('Fitted learning rate k')

    ax7.set_xlabel('Feedforward/feedback')
    ax8.set_xlabel('Feedforward/feedback')
    ax7.set_ylabel('Learning speed')
    ax8.set_ylabel('Fitted learning rate k')


    ax9.set_xlabel('Angle between perturbed and intuitive decoder')
    ax10.set_xlabel('Angle between perturbed and intuitive decoder')
    ax9.set_ylabel('Learning speed')
    ax10.set_ylabel('Fitted learning rate k')

    ax11.set_xlabel('Angle between perturbed and trained decoder')
    ax12.set_xlabel('Angle between perturbed and trained decoder')
    ax11.set_ylabel('Learning speed')
    ax12.set_ylabel('Fitted learning rate k')

    fig.savefig(savfolder+'Learning_rate_inp_delta_'+suffix+'.png')
    fig2.savefig(savfolder+'Learning_rate_fbk_delta_'+suffix+'.png')
    fig3.savefig(savfolder+'Learning_rate_inp_strategydelta_'+suffix+'.png')
    fig4.savefig(savfolder+'Learning_rate_inp_strategyratio_'+suffix+'.png')
    fig5.savefig(savfolder+'Learning_rate_angle_wintuit_'+suffix+'.png')
    fig6.savefig(savfolder+'Learning_rate_angle_wbase_'+suffix+'.png')
    pp.close('all')



def plot_learning_speed_2( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig=pp.figure(figsize=[6,10])
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        ax.hist(results['speed'], bins=np.arange(0,200,10),  lw=3, color=clr, alpha=0.5)
        ax2.hist(results['fitted_k'], bins=np.arange(0, 0.3,0.025),  lw=3, color=clr, alpha=0.5)
        
        
    ax.set_ylabel('Count')
    ax.set_xlabel('Learning speed')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Fitted_k')
    pp.savefig(savfolder+'learning_speed_hist_'+suffix+'.png')
    pp.close('all')


    return None





# ############################################ #


def plot_speed_controls( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(10,14))
    ax=fig.add_subplot(411)
    ax2=fig.add_subplot(412)
    ax3=fig.add_subplot(413)
    ax3b=fig.add_subplot(414)

    fig2 = pp.figure(figsize=(10,10))
    ax4=fig2.add_subplot(311)
    ax5=fig2.add_subplot(312)
    ax6=fig2.add_subplot(313)

    fig3 = pp.figure(figsize=(8,8))
    ax7=fig3.add_subplot(211)
    ax8=fig3.add_subplot(212)

    fig4 = pp.figure(figsize=(8,8))
    ax9=fig4.add_subplot(211)
    ax10=fig4.add_subplot(212)

    fig5 = pp.figure(figsize=(8,8))
    ax11=fig5.add_subplot(211)
    ax12=fig5.add_subplot(212)

    fig6 = pp.figure(figsize=(8,8))
    ax13=fig6.add_subplot(211)
    ax14=fig6.add_subplot(212)
    

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        ax.scatter(np.log10(results['obsv']), results['speed'], color=clr, alpha=0.3)
        ax2.scatter(np.log10(results['ctrb_ff']), results['speed'], color=clr, alpha=0.3)
        ax3.scatter(np.log10(results['ctrb_fb']), results['speed'], color=clr, alpha=0.3)
        ax3b.scatter(np.log10(results['ctrb_obsv_bal']), results['speed'], color=clr, alpha=0.3)

        ax4.scatter(results['obsv'], results['fitted_k'], color=clr, alpha=0.3)
        ax5.scatter(results['ctrb_ff'], results['fitted_k'], color=clr, alpha=0.3)
        ax6.scatter(results['ctrb_fb'], results['fitted_k'], color=clr, alpha=0.3)

        ax7.scatter(results['ctrb_ff']+results['ctrb_fb'], results['speed'], color=clr, alpha=0.3)
        ax8.scatter(results['ctrb_ff']+results['ctrb_fb'], results['fitted_k'], color=clr, alpha=0.3)

        ax9.scatter(np.log10(results['obsv']/results['ctrb_ff']), results['speed'], color=clr, alpha=0.3)
        ax10.scatter(np.log10(results['obsv']/results['ctrb_fb']), results['fitted_k'], color=clr, alpha=0.3)

        ax11.scatter(np.log10(results['obsv']*results['ctrb_ff']), results['speed'], color=clr, alpha=0.3)
        ax12.scatter(np.log10(results['obsv']*results['ctrb_fb']), results['fitted_k'], color=clr, alpha=0.3)

        ax13.scatter(np.log10(results['ctrb_fb']*results['ctrb_ff']), results['speed'], color=clr, alpha=0.3)
        ax14.scatter(np.log10(results['ctrb_ff']*results['ctrb_fb']), results['fitted_k'], color=clr, alpha=0.3)
    
    

    
    ax.set_ylabel('Learning speed')
    ax2.set_ylabel('Learning speed')
    ax3.set_ylabel('Learning speed')
    ax3b.set_ylabel('Learning speed')
    ax.set_xlabel('Log observability of readout')
    ax2.set_xlabel('Log feedforward controllability of readout')
    ax3.set_xlabel('Log feedback controllability of readout')
    ax3b.set_xlabel('Balanced feedback SV')


    ax4.set_ylabel('Fitted learning rate k')
    ax5.set_ylabel('Fitted learning rate k')
    ax6.set_ylabel('Fitted learning rate k')
    ax4.set_xlabel('observability of readout')
    ax5.set_xlabel('feedforward controllability of readout')
    ax6.set_xlabel('feedback controllability of readout')

    ax7.set_ylabel('Learning speed')
    ax8.set_ylabel('Fitted learning rate k')
    ax7.set_xlabel('ff+fb controllability')
    ax8.set_xlabel('ff+fb controllability')
    
    ax9.set_ylabel('Learning speed')
    ax10.set_ylabel('Fitted learning rate k')
    ax9.set_xlabel('readout observability/ff controllability')
    ax10.set_xlabel('readout observability/fb controllability')

    ax11.set_ylabel('Learning speed')
    ax12.set_ylabel('Fitted learning rate k')
    ax11.set_xlabel('readout observability * ff controllability')
    ax12.set_xlabel('readout observability * fb controllability')

    ax13.set_ylabel('Learning speed')
    ax14.set_ylabel('Fitted learning rate k')
    ax13.set_xlabel('readout fb controllability * ff controllability')
    ax14.set_xlabel('readout ff * fb controllability')

    #ax.set_ylim([150,650])

    fig.savefig(savfolder+'Learning_speed_controls_'+suffix+'.png')
    fig2.savefig(savfolder+'Learning_rate_controls_'+suffix+'.png')
    fig3.savefig(savfolder+'Learning_speed_controlsstrategy_'+suffix+'.png')
    fig4.savefig(savfolder+'Learning_rate_obs_by_controls_'+suffix+'.png')
    fig5.savefig(savfolder+'Learning_rate_obs_times_controls_'+suffix+'.png')
    fig6.savefig(savfolder+'Learning_rate_ff_times_fb_controls_'+suffix+'.png')
    
    pp.close('all')



# ############################################ #

def plot_speed_vectorfld( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(5,20))#pp.figure(figsize=(8,20))
    ax=fig.add_subplot(511)
    ax2=fig.add_subplot(512)
    ax3=fig.add_subplot(513)
    ax3b=fig.add_subplot(514)
    ax3c=fig.add_subplot(515)

    fig2 = pp.figure(figsize=(6,20))#pp.figure(figsize=(8,20))
    ax4=fig2.add_subplot(511)
    ax5=fig2.add_subplot(512)
    ax6=fig2.add_subplot(513)
    ax6b=fig2.add_subplot(514)
    ax6c=fig2.add_subplot(515)

    fig3 = pp.figure(figsize=(5,12))#pp.figure(figsize=(8,12))
    ax7=fig3.add_subplot(211)
    ax8=fig3.add_subplot(212)

    fig4 = pp.figure(figsize=(5,14))#pp.figure(figsize=(8,14))
    ax9=fig4.add_subplot(311)
    ax10=fig4.add_subplot(312)
    ax7a=fig4.add_subplot(313)

    fig5 = pp.figure(figsize=(5,14))#pp.figure(figsize=(8,14))
    ax11=fig5.add_subplot(311)
    ax12=fig5.add_subplot(312)
    ax13=fig5.add_subplot(313)
    

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        if clr=='b':
            cmap='Blues'
        elif clr=='r':
            cmap='Reds'
        else:
            cmap='coolwarm'
        '''
        ax.scatter(np.log10(results['fracVF']), results['speed'], color=clr, alpha=0.3)
        #plot_density( np.log10(results['fracVF']), results['speed'], ax=ax, cmap=cmap )
        #sns.kdeplot(np.log10(results['fracVF']), results['speed'], cmap=cmap, shade=True, shade_lowest=False, ax=ax)
        ax2.scatter(np.log10(results['fracVF_ff']), results['speed'], color=clr, alpha=0.3)
        ax3.scatter(np.log10(results['fracVF_fb']), results['speed'], color=clr, alpha=0.3)
        #plot_density( np.log10(results['fracVF_fb']), results['speed'], ax=ax3, cmap=cmap )
        ax3b.scatter(np.log10(results['fracOutvel']), results['speed'], color=clr, alpha=0.3)
        ax3c.scatter(np.log10(results['fracVF_rec']), results['speed'], color=clr, alpha=0.3)
        '''

        ax.scatter(1/(results['fracVF']), results['speed'], color=clr, alpha=0.3)
        #plot_density( np.log10(results['fracVF']), results['speed'], ax=ax, cmap=cmap )
        #sns.kdeplot(np.log10(results['fracVF']), results['speed'], cmap=cmap, shade=True, shade_lowest=False, ax=ax)
        ax2.scatter(1/(results['fracVF_ff']), results['speed'], color=clr, alpha=0.3)
        ax3.scatter(1/(results['fracVF_fb']), results['speed'], color=clr, alpha=0.3)
        #plot_density( np.log10(results['fracVF_fb']), results['speed'], ax=ax3, cmap=cmap )
        ax3b.scatter(1/(results['fracOutvel']), results['speed'], color=clr, alpha=0.3)
        ax3c.scatter(1/(results['fracVF_rec']), results['speed'], color=clr, alpha=0.3)


        '''
        ax4.scatter(np.log10(results['fracVF']), results['fitted_k'], color=clr, alpha=0.3)
        ax4.set_ylim(0,0.15)
        ax5.scatter(np.log10(results['fracVF_ff']), results['fitted_k'], color=clr, alpha=0.3)
        ax5.set_ylim(0,0.15)
        ax6.scatter(np.log10(results['fracVF_fb']), results['fitted_k'], color=clr, alpha=0.3)
        ax6.set_ylim(0,0.15)
        ax6b.scatter(np.log10(results['fracOutvel']), results['fitted_k'], color=clr, alpha=0.3)
        ax6b.set_ylim(0,0.15)
        ax6c.scatter(np.log10(results['fracVF_rec']), results['fitted_k'], color=clr, alpha=0.3)
        ax6c.set_ylim(0,0.15)
        '''

        ax4.scatter(1/(results['fracVF']), results['fitted_k'], color=clr, alpha=0.3)
        ax4.set_ylim(0,0.15)
        ax5.scatter(1/(results['fracVF_ff']), results['fitted_k'], color=clr, alpha=0.3)
        ax5.set_ylim(0,0.15)
        ax6.scatter(1/(results['fracVF_fb']), results['fitted_k'], color=clr, alpha=0.3)
        ax6.set_ylim(0,0.15)
        ax6b.scatter(1/(results['fracOutvel']), results['fitted_k'], color=clr, alpha=0.3)
        ax6b.set_ylim(0,0.15)
        ax6c.scatter(1/(results['fracVF_rec']), results['fitted_k'], color=clr, alpha=0.3)
        ax6c.set_ylim(0,0.15)

        
        #ax7.hist(results['fracVF'], bins=np.arange(start=0,stop=.5,step=0.025), density=True, lw=3, color=clr, alpha=0.5)
        ax7.hist(results['fracVF'], bins=np.arange(start=0,stop=.8,step=0.050), density=True, lw=3, color=clr, alpha=0.5)
        m1=np.nanmedian(results['fracVF'])
        ax7.plot([m1,m1],[0,len(results['fracVF'])/10],color=clr)
        #sns.kdeplot(results['fracVF'], color=clr, lw=3, ax=ax7, gridsize=21)
        ax8.hist(results['fracOutvel'], bins=np.arange(start=0,stop=5, step=0.25), density=True, lw=3, color=clr, alpha=0.5)
        m1=np.nanmedian(results['fracOutvel'])
        ax8.plot([m1,m1],[0,len(results['fracOutvel'])/100],color=clr)
        #sns.kdeplot(results['fracOutvel'], color=clr, lw=3, ax=ax8, gridsize=21)
        
        ax9.hist(results['fracVF_ff'], bins=np.arange(start=0,stop=.25,step=0.0125), density=True, lw=3, color=clr, alpha=0.5)
        m1=np.nanmedian(results['fracVF_ff'])
        ax9.plot([m1,m1],[0,len(results['fracVF_ff'])/10],color=clr)
        #sns.kdeplot(results['fracVF_ff'], color=clr, lw=3, ax=ax9, gridsize=21)
        ax10.hist(results['fracVF_fb'], bins=np.arange(start=0,stop=.25,step=0.0125), density=True, lw=3, color=clr, alpha=0.5)
        m1=np.nanmedian(results['fracVF_fb'])
        ax10.plot([m1,m1],[0,len(results['fracVF_fb'])/10],color=clr)
        #sns.kdeplot(results['fracVF_fb'], color=clr, lw=3, ax=ax10, gridsize=21)
        ax7a.hist(results['fracVF_rec'], bins=np.arange(start=0,stop=.25,step=0.0125), density=True, lw=3, color=clr, alpha=0.5)
        #ax7a.hist(results['fracVF_rec'], bins=np.arange(start=0,stop=max(results['fracVF_rec'])+0.1,step=0.0125), density=True, lw=3, color=clr, alpha=0.5)
        m1=np.nanmedian(results['fracVF_rec'])
        ax7a.plot([m1,m1],[0,len(results['fracVF_rec'])/10],color=clr)
        #sns.kdeplot(results['fracVF_rec'], color=clr, lw=3, ax=ax7a, gridsize=21)

        ax11.hist(results['fracSVF_fb'], bins=np.arange(start=0,stop=1,step=0.05), density=False, lw=3, color=clr, alpha=0.5)
        sns.kdeplot(results['fracSVF_fb'], color=clr, lw=3, ax=ax11, gridsize=21)
        ax12.hist(results['fracSVF_rec'], bins=np.arange(start=0,stop=1,step=0.05), density=False, lw=3, color=clr, alpha=0.5)
        sns.kdeplot(results['fracSVF_rec'], color=clr, lw=3, ax=ax12, gridsize=21)
        ax13.hist(results['SVF_tot'], bins=np.arange(start=0,stop=1,step=0.05), density=False, lw=3, color=clr, alpha=0.5)
        sns.kdeplot(results['SVF_tot'], color=clr, lw=3, ax=ax13, gridsize=21)
        
    

    
    ax.set_ylabel('Learning speed')
    ax2.set_ylabel('Learning speed')
    ax3.set_ylabel('Learning speed')
    ax3b.set_ylabel('Learning speed')
    ax3c.set_ylabel('Learning speed')
    ax.set_ylim([0,100])
    ax2.set_ylim([0,100])
    ax3.set_ylim([0,100])
    ax3b.set_ylim([0,100])
    ax3c.set_ylim([0,100])
    
    ax.set_xlabel('1/frac VF change')
    ax2.set_xlabel('1/frac VF change - FF')
    ax3.set_xlabel('1/frac VF change - FB')
    ax3b.set_xlabel('1/frac Velocity change')
    ax3c.set_xlabel('1/frac VF change - Rec')
    

    
    
    ax4.set_ylabel('Fitted learning rate k')
    ax5.set_ylabel('Fitted learning rate k')
    ax6.set_ylabel('Fitted learning rate k')
    ax6b.set_ylabel('Fitted learning rate k')
    ax6c.set_ylabel('Fitted learning rate k')
    ax4.set_xlabel('1/frac VF change')
    ax5.set_xlabel('1/frac VF change - FF')
    ax6.set_xlabel('1/frac VF change - FB')
    ax6b.set_xlabel('1/frac Velocity change')
    ax6c.set_xlabel('1/frac VF change - Rec')
    

    
    ax7.set_ylabel('Count')
    ax8.set_ylabel('Count')   
    ax7.set_xlabel('frac VF change')
    ax8.set_xlabel('frac Velocity change')
    
    ax7a.set_ylabel('Count')
    ax9.set_ylabel('Count')
    ax10.set_ylabel('Count')
    ax9.set_xlabel('frac VF change - FF')
    ax10.set_xlabel('frac VF change - FB')
    ax7a.set_xlabel('frac VF change - Rec')


    ax11.set_ylabel('Count')
    ax12.set_ylabel('Count')
    ax13.set_ylabel('Count')
    ax11.set_xlabel('frac state VF change - FB')
    ax12.set_xlabel('frac state VF change - Rec')
    ax13.set_xlabel('state VF change - total')

    
    #ax.set_ylim([150,650])

    fig.savefig(savfolder+'VF_speed_'+suffix+'.png')
    fig2.savefig(savfolder+'VF_fittedk_'+suffix+'.png')
    fig3.savefig(savfolder+'VF_hist_'+suffix+'.png')
    fig4.savefig(savfolder+'VF_FF_FB_Rec_Hist_'+suffix+'.png')
    fig5.savefig(savfolder+'VF_state_FB_Rec_Hist_'+suffix+'.png')
    
    pp.close('all')


# ############################################ #

def plot_loss_speed_controls( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(10,10))
    ax=fig.add_subplot(311)
    ax2=fig.add_subplot(312)
    ax3=fig.add_subplot(313)

    fig2 = pp.figure(figsize=(10,14))
    ax4=fig2.add_subplot(411)
    ax5=fig2.add_subplot(412)
    ax6=fig2.add_subplot(413)
    ax6b=fig2.add_subplot(414)

    fig3 = pp.figure(figsize=(8,8))
    ax7=fig3.add_subplot(211)
    ax8=fig3.add_subplot(212)

    fig4 = pp.figure(figsize=(8,8))
    ax9=fig4.add_subplot(211)
    ax10=fig4.add_subplot(212)

    fig5 = pp.figure(figsize=(8,8))
    ax11=fig5.add_subplot(211)
    ax12=fig5.add_subplot(212)

    fig6 = pp.figure(figsize=(8,8))
    ax13=fig6.add_subplot(211)
    ax14=fig6.add_subplot(212)
    

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        ax.scatter(np.log10(results['obsv']), results['loss_speed'], color=clr, alpha=0.3)
        ax2.scatter(np.log10(results['ctrb_ff']), results['loss_speed'], color=clr, alpha=0.3)
        ax3.scatter(np.log10(results['ctrb_fb']), results['loss_speed'], color=clr, alpha=0.3)

        ax4.scatter(results['obsv'], results['loss_fitted_k'], color=clr, alpha=0.3)
        ax5.scatter(results['ctrb_ff'], results['loss_fitted_k'], color=clr, alpha=0.3)
        ax6.scatter(results['ctrb_fb'], results['loss_fitted_k'], color=clr, alpha=0.3)
        ax6b.scatter(results['ctrb_obsv_bal'], results['loss_fitted_k'], color=clr, alpha=0.3)

        ax7.scatter(results['ctrb_ff']+results['ctrb_fb'], results['loss_speed'], color=clr, alpha=0.3)
        ax8.scatter(results['ctrb_ff']+results['ctrb_fb'], results['loss_fitted_k'], color=clr, alpha=0.3)

        ax9.scatter(np.log10(results['obsv']/results['ctrb_ff']), results['loss_speed'], color=clr, alpha=0.3)
        ax10.scatter(np.log10(results['obsv']/results['ctrb_fb']), results['loss_fitted_k'], color=clr, alpha=0.3)

        ax11.scatter(np.log10(results['obsv']*results['ctrb_ff']), results['loss_speed'], color=clr, alpha=0.3)
        ax12.scatter(np.log10(results['obsv']*results['ctrb_fb']), results['loss_fitted_k'], color=clr, alpha=0.3)

        ax13.scatter(np.log10(results['ctrb_fb']*results['ctrb_ff']), results['loss_speed'], color=clr, alpha=0.3)
        ax14.scatter(np.log10(results['ctrb_ff']*results['ctrb_fb']), results['loss_fitted_k'], color=clr, alpha=0.3)
    
    

    
    ax.set_ylabel('Learning loss_speed')
    ax2.set_ylabel('Learning loss_speed')
    ax3.set_ylabel('Learning loss_speed')
    ax.set_xlabel('Log observability of readout')
    ax2.set_xlabel('Log feedforward controllability of readout')
    ax3.set_xlabel('Log feedback controllability of readout')


    ax4.set_ylabel('Fitted loss recovery rate k')
    ax5.set_ylabel('Fitted loss recovery rate k')
    ax6.set_ylabel('Fitted loss recovery rate k')
    ax6b.set_ylabel('Fitted loss recovery rate k')
    ax4.set_xlabel('observability of readout')
    ax5.set_xlabel('feedforward controllability of readout')
    ax6.set_xlabel('feedback controllability of readout')
    ax6b.set_xlabel('balanced feedback sv')

    ax7.set_ylabel('Learning loss_speed')
    ax8.set_ylabel('Fitted loss recovery rate k')
    ax7.set_xlabel('ff+fb controllability')
    ax8.set_xlabel('ff+fb controllability')
    
    ax9.set_ylabel('Learning loss_speed')
    ax10.set_ylabel('Fitted loss recovery rate k')
    ax9.set_xlabel('readout observability/ff controllability')
    ax10.set_xlabel('readout observability/fb controllability')

    ax11.set_ylabel('Learning loss_speed')
    ax12.set_ylabel('Fitted loss recovery rate k')
    ax11.set_xlabel('readout observability * ff controllability')
    ax12.set_xlabel('readout observability * fb controllability')

    ax13.set_ylabel('Learning loss_speed')
    ax14.set_ylabel('Fitted loss recovery rate k')
    ax13.set_xlabel('readout fb controllability * ff controllability')
    ax14.set_xlabel('readout ff * fb controllability')

    #ax.set_ylim([150,650])

    fig.savefig(savfolder+'Learning_loss_speed_controls_'+suffix+'.png')
    fig2.savefig(savfolder+'Learning_loss_rate_controls_'+suffix+'.png')
    fig3.savefig(savfolder+'Learning_loss_speed_controlsstrategy_'+suffix+'.png')
    fig4.savefig(savfolder+'Learning_loss_rate_obs_by_controls_'+suffix+'.png')
    fig5.savefig(savfolder+'Learning_loss_rate_obs_times_controls_'+suffix+'.png')
    fig6.savefig(savfolder+'Learning_loss_rate_ff_times_fb_controls_'+suffix+'.png')
    
    pp.close('all')




# ############################################ #

def plot_density( m1, m2, ax=None, lim=None, cmap='coolwarm' ):

    if len(m1.shape)>1:
        m1=np.reshape(m1,(len(m1)))
        m2=np.reshape(m2,(len(m2)))
    if lim is None:
        xmin=m1.min()
        xmax = m1.max()
        ymin = m2.min()
        ymax = m2.max()
    else:
        [xmin,xmax,ymin,ymax] = lim

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    if ax is None:
        pp.pcolormesh(Z, cmap=cmap)
    else:
        ax.pcolormesh(Z, cmap=cmap)

    return None

# ############################################ #




def plot_var_dyn( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(10,10))
    ax=fig.add_subplot(311)
    ax2=fig.add_subplot(312)
    ax3=fig.add_subplot(313)

    fig2 = pp.figure(figsize=(10,10))
    ax4=fig2.add_subplot(311)
    ax5=fig2.add_subplot(312)
    ax6=fig2.add_subplot(313)

    fig3 = pp.figure(figsize=(8,8))
    ax7=fig3.add_subplot(211)
    ax8=fig3.add_subplot(212)

    fig4 = pp.figure(figsize=(8,8))
    ax9=fig4.add_subplot(211)
    ax10=fig4.add_subplot(212)

    fig5 = pp.figure(figsize=(8,8))
    ax11=fig5.add_subplot(211)
    ax12=fig5.add_subplot(212)

    fig6 = pp.figure(figsize=(8,8))
    ax13=fig6.add_subplot(211)
    ax14=fig6.add_subplot(212)

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)
        ax.scatter(results['expVar_post'], results['fracVF'], color=clr, alpha=0.3)
        ax2.scatter(results['expVar_post'], results['fracVF_ff'], color=clr, alpha=0.3)
        ax3.scatter(results['expVar_post'], results['fracVF_fb'], color=clr, alpha=0.3)

        ax4.scatter(results['expVar_post'], np.log10(results['obsv']), color=clr, alpha=0.3)
        ax5.scatter(results['expVar_post'], np.log10(results['ctrb_ff']), color=clr, alpha=0.3)
        ax6.scatter(results['expVar_post'], np.log10(results['ctrb_fb']), color=clr, alpha=0.3)

        ax7.scatter(results['expVar_post']-results['expVar_pre'],results['fracVF'], color=clr, alpha=0.3)
        ax8.scatter(results['expVar_post']-results['expVar_pre'], results['fracVF_fb'], color=clr, alpha=0.3)

        ax9.scatter(np.log10(results['obsv']), results['expVar_post']-results['expVar_pre'], color=clr, alpha=0.3)
        ax10.scatter(np.log10(results['ctrb_fb']), np.log10(results['fracVF_fb']), color=clr, alpha=0.3)

        change_hit = (results['hit_rate_post']-results['hit_rate_pre'])/(1-results['hit_rate_pre'])

        ax11.scatter(results['expVar_decoder'], results['speed'], color=clr, alpha=0.3)
        ax12.scatter(results['expVar_decoder'] ,change_hit, color=clr, alpha=0.3)

        ax13.scatter(results['theta_intuit_wperturb'], results['speed'], color=clr, alpha=0.3)
        ax14.scatter(results['theta_intuit_wperturb'] ,change_hit, color=clr, alpha=0.3)
    
    

    
    ax.set_ylabel('Frac. VF')
    ax2.set_ylabel('Frac. VF - feedforward')
    ax3.set_ylabel('Frac. VF - Feedback')
    ax.set_xlabel('ExpVar (post)')
    ax2.set_xlabel('ExpVar (post)')
    ax3.set_xlabel('ExpVar (post)')


    ax4.set_ylabel('IM observability')
    ax5.set_ylabel('ff controllability')
    ax6.set_ylabel('fb controllability')
    ax4.set_xlabel('ExpVar (post)')
    ax5.set_xlabel('ExpVar (post)')
    ax6.set_xlabel('ExpVar (post)')

    ax7.set_ylabel('Frac. VF')
    ax8.set_ylabel('Frac. VF - feedback')
    ax7.set_xlabel('Change in ExpVar')
    ax8.set_xlabel('Change in ExpVar')
    

    ax9.set_ylabel('Change in ExpVar')
    ax10.set_ylabel('Frac. VF - feedback')
    ax9.set_xlabel('Log Observability')
    ax10.set_xlabel('Controllability')

    ax11.set_ylabel('Learning speed')
    ax12.set_ylabel('Amount of learning')
    ax11.set_xlabel('Variance along decoder')
    ax12.set_xlabel('Variance along decoder')

    ax13.set_ylabel('Learning speed')
    ax14.set_ylabel('Amount of learning')
    ax13.set_xlabel('Angle with IM')
    ax14.set_xlabel('Angle with IM')
    
    fig.savefig(savfolder+'expvar_post_vfchange_'+suffix+'.png')
    fig2.savefig(savfolder+'expvar_post_controls_'+suffix+'.png')
    fig3.savefig(savfolder+'expvar_change_vfchange_'+suffix+'.png')
    fig4.savefig(savfolder+'var_obs_dyn_ctrb_'+suffix+'.png')
    fig5.savefig(savfolder+'speed_expvar_decoder_'+suffix+'.png')
    fig6.savefig(savfolder+'speed_expvar_IM_'+suffix+'.png')
 
    pp.close('all')






def plot_success_dyn( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(6,14))
    ax=fig.add_subplot(311)
    ax2=fig.add_subplot(312)
    ax3=fig.add_subplot(313)

    fig2 = pp.figure(figsize=(6,18))
    ax4=fig2.add_subplot(411)
    ax5=fig2.add_subplot(412)
    ax6=fig2.add_subplot(413)
    ax7=fig2.add_subplot(414)


    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)

        change_hit = (results['hit_rate_post'] - results['hit_rate_pre'])/(1 - results['hit_rate_pre'])
        ax.scatter( results['fracVF'], change_hit, color=clr, alpha=0.3)
        ax2.scatter(results['fracVF_ff'], change_hit, color=clr, alpha=0.3)
        ax3.scatter(results['fracVF_fb'], change_hit, color=clr, alpha=0.3)

        ax4.scatter(np.log10(results['obsv']), change_hit, color=clr, alpha=0.3)
        ax5.scatter(np.log10(results['ctrb_ff']), change_hit, color=clr, alpha=0.3)
        ax6.scatter(np.log10(results['ctrb_fb']), change_hit, color=clr, alpha=0.3)
        ax7.scatter(np.log10(results['ctrb_obsv_bal']), change_hit, color=clr, alpha=0.3)


    
    ax.set_xlabel('Frac. VF')
    ax2.set_xlabel('Frac. VF - feedforward')
    ax3.set_xlabel('Frac. VF - Feedback')
    ax.set_ylabel('Rel. hit rate change')
    ax2.set_ylabel('Rel. hit rate change')
    ax3.set_ylabel('Rel. hit rate change')


    ax4.set_xlabel('IM observability')
    ax5.set_xlabel('ff controllability')
    ax6.set_xlabel('fb controllability')
    ax7.set_xlabel('Balanced fb Ctrl-Obs')
    ax4.set_ylabel('Rel. hit rate change')
    ax5.set_ylabel('Rel. hit rate change')
    ax6.set_ylabel('Rel. hit rate change')
    ax7.set_ylabel('Rel. hit rate change')


    
    fig.savefig(savfolder+'changehit_vfchange_'+suffix+'.png')
    fig2.savefig(savfolder+'changehit_controls_'+suffix+'.png')

 
    pp.close('all')




def plot_grads( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    f1 = pp.figure()
    ax = f1.add_subplot(111)

    f2 = pp.figure()
    ax2 = f2.add_subplot(111)

    f3 = pp.figure()
    ax3 = f3.add_subplot(111)

    f4 = pp.figure()
    ax4 = f4.add_subplot(211)
    ax5 = f4.add_subplot(212)

    f5 = pp.figure()
    ax6 = f5.add_subplot(211)
    ax7 = f5.add_subplot(212)

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)

        change_hit = (results['hit_rate_post']-results['hit_rate_pre'])/(1-results['hit_rate_pre'])
        
        ax.scatter(np.ones((nMaps,1))+shift, results['dim_dw_fbk'], color=clr, alpha=0.3)
        ax.bar( 1+shift, np.nanmean(results['dim_dw_fbk']), width=width, color=clr, alpha=0.5 )
        ax.errorbar( 1+shift, np.nanmean(results['dim_dw_fbk']), np.nanstd(results['dim_dw_fbk']),  color=clr, alpha=0.5 )

        ax2.scatter(np.ones((nMaps,1))+shift, results['corr_dw_fbk'], color=clr, alpha=0.3)
        ax2.bar( 1+shift, np.nanmean(results['corr_dw_fbk']), width=width, color=clr, alpha=0.5 )
        ax2.errorbar( 1+shift, np.nanmean(results['corr_dw_fbk']), np.nanstd(results['corr_dw_fbk']), color=clr, alpha=0.5 )

        ax3.plot( results['grad_total'].T, color=clr, alpha=0.5 )
        ax3.plot( np.mean(results['grad_total'],axis=0), color=clr, lw=1.5 )

        ax4.scatter( results['dim_dw_fbk'], change_hit, color=clr, alpha=0.3)
        ax5.scatter( results['corr_dw_fbk'], change_hit, color=clr, alpha=0.3)
    
        ax6.scatter( results['dim_dw_fbk'], results['speed'], color=clr, alpha=0.3)
        ax7.scatter( results['corr_dw_fbk'], results['speed'], color=clr, alpha=0.3)
    
    
    
    ax.set_ylabel('Dim. of Feedback weight updates')
    ax2.set_ylabel('Trial-to-trial dwFbk correlation')
    ax3.set_xlabel('Trial')
    ax3.set_ylabel('Total gradient norm')
    ax4.set_xlabel('Dim. of Feedback weight updates')
    ax5.set_xlabel('Trial-to-trial dwFbk correlation')
    ax4.set_ylabel('Rel. improv. hit rate')
    ax5.set_ylabel('Rel. improv. hit rate')
    ax6.set_ylabel('Learning speed')
    ax7.set_ylabel('Learning speed')

    
    f1.savefig(savfolder+'dim_dw_fbk'+suffix+'.png')
    f2.savefig(savfolder+'corr_dw_fbk'+suffix+'.png')
    f3.savefig(savfolder+'grad_total'+suffix+'.png')
    f4.savefig(savfolder+'change_hit_dim_corr_dw_fbk'+suffix+'.png')
    f5.savefig(savfolder+'learning_speed_dim_corr_dw_fbk'+suffix+'.png')
    pp.close(f1)
    pp.close(f2)
    pp.close(f3)
    pp.close(f4)

# ############################################ #


def plot_ctrb_U( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(6,10))
    ax=fig.add_subplot(211)
    ax2=fig.add_subplot(212)

    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)

        amin = 0
        amax = max(np.log10(results['ctrb_U_fb']))[0]+0.5
        ax.plot([0,amax],[0,amax], color='k')
        ax.scatter( np.log10(results['ctrb_U_fb']), np.log10(results['ctrb_fb']), color=clr, alpha=0.5)
        
        amax = max(np.log10(results['ctrb_U_ff']))[0]+0.5
        ax2.plot([amin,amax],[amin,amax],color='k')
        ax2.scatter( np.log10(results['ctrb_U_ff']), np.log10(results['ctrb_ff']), color=clr, alpha=0.5)
        
    ax.set_ylabel('Feedback controllablity of readout')
    ax2.set_ylabel('Feedback controllablity of readout')    

    ax.set_xlabel('Feedback controllablity of IM')
    ax2.set_xlabel('Feedback controllablity of IM')    

    fig.savefig(savfolder+'ctrb_wout_U'+suffix+'.png')
    fig.savefig(savfolder+'ctrb_wout_U'+suffix+'.png')
    pp.close(fig)


def plot_success_im( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure(figsize=(6,10))
    ax=fig.add_subplot(211)
    ax2=fig.add_subplot(212)

    fig2 = pp.figure(figsize=(6,10))
    ax4=fig2.add_subplot(211)
    ax5=fig2.add_subplot(212)



    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)

        change_hit = (results['hit_rate_post'] - results['hit_rate_pre'])/(1 - results['hit_rate_pre'])
        ax.scatter( results['im_change'], change_hit, color=clr, alpha=0.3)
        ax2.scatter( np.max(results['im_angles'],axis=1), change_hit, color=clr, alpha=0.3)
        
        ax4.scatter(results['im_change'], results['speed'], color=clr, alpha=0.3)
        ax5.scatter( np.max(results['im_angles'],axis=1), results['speed'], color=clr, alpha=0.3)

    
    ax.set_xlabel('Covariance overlap')
    ax.set_ylabel('Rel. hit rate change')
    ax2.set_xlabel('PC change')
    ax2.set_ylabel('Rel. hit rate change')


    ax4.set_xlabel('Covariance overlap')
    ax4.set_ylabel('Learning speed')
    ax5.set_xlabel('PC change')
    ax5.set_ylabel('Learning speed')

    
    fig.savefig(savfolder+'changehit_imchange_'+suffix+'.png')
    fig2.savefig(savfolder+'learningspeed_imchange_'+suffix+'.png')

 
    pp.close('all')






def plot_distrib_im( results, *args, savfolder='', suffix='_test', clrs=None ):
    all_results = [results]
    for new_res in args:
        all_results.append(new_res)
    
    nres = len(all_results)
    width = 0.5/nres

    cm = mpl.colormaps['jet']
    fig = pp.figure()
    ax=fig.add_subplot(111)

    fig2 = pp.figure(figsize=(6,10))
    ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212)
    
    fig3 = pp.figure(figsize=(6,10))
    ax4 = fig3.add_subplot(211)
    ax5 = fig3.add_subplot(212)

    fig4 = pp.figure(figsize=(6,10))
    ax6 = fig4.add_subplot(211)
    ax7 = fig4.add_subplot(212)
    


    for mm in range(nres):
        results = all_results[mm]
        nMaps = len(results['progress_pre'])
        shift=mm*width
        if clrs is not None:
            clr=clrs[mm]
        else:
            clr=cm(mm/nres)

        ax.hist( results['im_change'], bins=np.arange(-1,1.05,0.05), color=clr, alpha=0.3)

        ax2.scatter( results['overlap_intuit_wperturb'], results['im_change'], color=clr, alpha=0.3)
        ax3.scatter( results['overlap_base_wperturb'], results['im_change'], color=clr, alpha=0.3)

        ax4.hist( results['overlap_intuit_wperturb'], bins=np.arange(-1,1,0.05), color=clr, alpha=0.3)
        ax5.hist( results['overlap_base_wperturb'], bins=np.arange(-1,1,0.05), color=clr, alpha=0.3)

        ax6.hist( results['overlap_influence'], bins=np.arange(-1,1,0.05), color=clr, alpha=0.3)
        ax7.scatter( results['overlap_base_wperturb'], results['overlap_influence'], color=clr, alpha=0.3)


        

    ax.set_xlabel('Similarity of covariance structure')
    ax.set_ylabel('Count')

    ax2.set_xlabel('Overlap of Wpert with Wintuit')
    ax2.set_ylabel('Similarity of covariance structure')

    ax3.set_xlabel('Overlap of Wpert with Worig')
    ax3.set_ylabel('Similarity of covariance structure')

    ax4.set_xlabel('Overlap of Wpert with Wintuit')
    ax4.set_ylabel('Count')

    ax5.set_xlabel('Overlap of Wpert with Worig')
    ax5.set_ylabel('Count')

    ax6.set_xlabel('Overlap of PC influence on velocity')
    ax6.set_ylabel('Count')

    ax7.set_xlabel('Overlap of Wpert with Worig')
    ax7.set_ylabel('Overlap of PC influence on velocity')


    
    fig.savefig(savfolder+'covdist_'+suffix+'.png')
    fig2.savefig(savfolder+'covdist_woverlap_'+suffix+'.png')
    fig3.savefig(savfolder+'covdist_woverlap_hist_'+suffix+'.png')
    fig4.savefig(savfolder+'covdist_influence_'+suffix+'.png')

    pp.close('all')


