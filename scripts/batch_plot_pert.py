

import numpy as np
import os, glob, argparse, inspect, sys
import matplotlib.pyplot as pp
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
os.chdir(parentdir)
sys.path.insert(0, parentdir)

import plotting.plot_pert_training as pmt
import tools.pert_analyses as pa
from tools.regression import fit_linear_speed
from scripts.list_of_folders import get_folders

import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Compare and plot results for WMP and OMPs')
    parser.add_argument('-F', '--folder', type=str, default='relu_test_', 
                        help='Option to set up list of folders to compare')
    parser.add_argument('-npro', '--npro', type=int, default=1, help='new progress calculation')
    
    # Set up folders
    ### -------------------------------------------- ###
    args = parser.parse_args()
    main_dic = get_folders(args.folder)
    npro = (args.npro==1)

    #####################################################
    # LOADING DATA
    #####################################################

    # ------- WMP ------------- #
    dic = main_dic['wmp']
    print(dic['savfolder'])


    allfolders = glob.glob(dic['folder']+'Model_*'+dic['suffix'])     # all models
    res = []
    # collect each training result
    print(allfolders)
    for ff in range(len(allfolders)):
        model = allfolders[ff]+'/'
        for file in dic['file']:
            fname = model+file
            if os.path.isfile(fname):
                res.append( pa.get_training_results(file=file,folder=model, savfolder=model, hit_thresh=0.1, new_progress =npro, 
                                                    use_mod=main_dic['use_mod'], use_fbwt=main_dic['use_fbwt'], percp=main_dic['percp']) )
    # combine all results
    results_WMP = res[0].copy()
    for key in res[0].keys():
        for ff in range(len(res)-1):
            results_WMP[key] = np.vstack((results_WMP[key], res[ff+1][key]))

    if not os.path.exists(dic['savfolder']):
        os.makedirs(dic['savfolder'])

    pmt.plot_all(results_WMP, savfolder=dic['savfolder'], suffix=dic['save_suffix'])

    # ------- OMP ------------- #

    dic = main_dic['omp']
    print(dic['savfolder'])
    allfolders = glob.glob(dic['folder']+'Model_*'+dic['suffix'])
    res = []
    for ff in range(len(allfolders)):
        model = allfolders[ff]+'/'
        for file in dic['file']:
            fname = model+file
            if os.path.isfile(fname):
                res.append( pa.get_training_results(file=file,folder=model, savfolder=model, hit_thresh=0.1, 
                                                    new_progress =npro, use_mod=main_dic['use_mod'], use_fbwt=main_dic['use_fbwt'], percp=main_dic['percp'] ) )


    results_OMP = res[0].copy()
    for key in res[0].keys():
        for ff in range(len(res)-1):
            results_OMP[key] = np.vstack((results_OMP[key], res[ff+1][key]))


    if not os.path.exists(dic['savfolder']):
        os.makedirs(dic['savfolder'], exist_ok=True)

    pmt.plot_all(results_OMP, savfolder=dic['savfolder'], suffix=dic['save_suffix'])



   # ------- JOINT ------------- #

    dic = main_dic['joint']
    if not os.path.exists(dic['savfolder']):
        os.makedirs(dic['savfolder'])

    savfolder = dic['savfolder']
    print(savfolder)

    #####################################################
    # SAVING RESULTS
    #####################################################

    svdic = {'results_WMP':results_WMP, 'results_OMP':results_OMP, 'dic':dic }
    np.save(savfolder+'analyses_'+main_dic['save_suffix'], svdic)

    #####################################################
    # PLOTTING RESULTS
    #####################################################

    pmt.plot_progress_2(results_WMP, results_OMP,        savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_activity_norm_2(results_WMP, results_OMP,   savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_asymmetry_2(results_WMP, results_OMP,       savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_expvar_2(results_WMP, results_OMP,          savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])#
    pmt.plot_expvar_hist(results_WMP, results_OMP,       savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])#
    pmt.plot_hitrate_change_2(results_WMP, results_OMP,  savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_hitrate_input(results_WMP, results_OMP,     savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_speed_input(results_WMP, results_OMP,       savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_learning_speed_2(results_WMP, results_OMP,  savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_speed_controls(results_WMP,results_OMP,     savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_grads(results_WMP,results_OMP,              savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_speed_vectorfld(results_WMP,results_OMP,    savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_loss_speed_controls(results_WMP,results_OMP,savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_var_dyn(results_WMP,results_OMP,            savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_success_dyn(results_WMP,results_OMP,        savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_success_im(results_WMP,results_OMP,         savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_distrib_im(results_WMP,results_OMP,         savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])
    pmt.plot_ctrb_U(results_WMP,results_OMP,             savfolder=savfolder, suffix=dic['save_suffix'], clrs=main_dic['clrs'])


    alpha = np.vstack((results_WMP['obsv'], results_OMP['obsv']))
    allff = np.vstack((results_WMP['ctrb_ff'], results_OMP['ctrb_ff']))
    allfb = np.vstack((results_WMP['ctrb_fb'], results_OMP['ctrb_fb']))
    speed = np.vstack((results_WMP['speed'], results_OMP['speed']))

    #####################################################
    # PLOTTING REGRESSIONS
    #####################################################
    fracTrain=0.8
    np.random.seed()

    results_WMP['expVar_pre'][np.isnan(results_WMP['expVar_pre'])]=1
    results_OMP['expVar_pre'][np.isnan(results_OMP['expVar_pre'])]=1
    results_OMP['expVar_post'][np.isnan(results_OMP['expVar_post'])]=1
    results_WMP['expVar_post'][np.isnan(results_WMP['expVar_post'])]=1

    results_WMP['expVar_diff'] = results_WMP['expVar_post']  - results_WMP['expVar_pre']
    results_OMP['expVar_diff'] = results_OMP['expVar_post']  - results_OMP['expVar_pre']

    results_WMP['log_speed'] = np.log10(results_WMP['speed'])
    results_OMP['log_speed'] = np.log10(results_OMP['speed'])

    if results_WMP['fracVF_fb'][0]>0:
        results_WMP['fracVF_fb_inv'] = 1/results_WMP['fracVF_fb']
        results_OMP['fracVF_fb_inv'] = 1/results_OMP['fracVF_fb']
    results_WMP['fracVF_inv'] = 1/results_WMP['fracVF']
    results_OMP['fracVF_inv'] = 1/results_OMP['fracVF']
    results_WMP['log_fracVF'] = np.log10(results_WMP['fracVF'])
    results_OMP['log_fracVF'] = np.log10(results_OMP['fracVF'])
    results_WMP['log_ctrb_fb'] = np.log10(results_WMP['ctrb_fb'])
    results_OMP['log_ctrb_fb'] = np.log10(results_OMP['ctrb_fb'])
    

    
    #### Regression of learning speed
    
    maxy=np.log10(120)
    miny=0
    
    print('speed versus ctrb_fb and fracVF:')
    fit_linear_speed(results_WMP, results_OMP, idy='log_speed',  idx=('ctrb_fb','fracVF'), logX=True, 
                        maxy=maxy, miny=miny, fracTrain=fracTrain, logy=True, savfolder=savfolder, suffix='_logspeed__ctrb_fb' )
    print('speed versus ctrb_ff and fracVF:')
    fit_linear_speed(results_WMP, results_OMP, idy='log_speed',  idx=('ctrb_ff','fracVF'), logX=True, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=True, savfolder=savfolder, suffix='_logspeed__ctrb_ff' )

    
    print('speed versus ctrb_fb and 1/fracVF:')
    fit_linear_speed(results_WMP, results_OMP,idy='log_speed',  idx=('log_ctrb_fb','fracVF_inv'), logX=False, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=True, savfolder=savfolder, suffix='_logspeed__ctrb_fb_fracVF_inv' )
    
    print('speed versus ctrb_fb and 1/fracVF_fb:')
    fit_linear_speed(results_WMP, results_OMP, idy='log_speed',  idx=('log_ctrb_fb','fracVF_fb_inv'), logX=False, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=True, savfolder=savfolder, suffix='_logspeed__ctrb_fb_fracVF_fb_inv' )
    
    print('speed versus ctrb_fb and fracVF_fb:')
    fit_linear_speed(results_WMP, results_OMP, idy='log_speed',  idx=('log_ctrb_fb','fracVF_fb'), logX=False, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=True, savfolder=savfolder, suffix='_logspeed__logctrb_fb_fracVF_fb' )
    
    print('speed versus ctrb_fb and hit rate:')
    fit_linear_speed(results_WMP, results_OMP, idy='log_speed',  idx=('log_ctrb_fb','hit_rate_pre'), logX=False, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=True, savfolder=savfolder, suffix='_logspeed_ctrb_fb_hit_rate_' )
    
    print('speed versus ctrb_fb and expVar_dff:')
    fit_linear_speed(results_WMP, results_OMP, idy='log_speed',  idx=('log_ctrb_fb','expVar_diff'), logX=False, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=True, savfolder=savfolder, suffix='_logspeed_ctrb_fb_exp_Var_' )
    

    #### Regression of logistic rate parameter

    maxy=1.2*max(results_WMP['fitted_k'])
    miny=0

    fit_linear_speed(results_WMP, results_OMP, idy='fitted_k',  idx=('ctrb_fb','fracVF'), logX=True, 
                     maxy=maxy, fracTrain=fracTrain, logy=False, savfolder=savfolder, suffix='_fittedk_' )
    
    fit_linear_speed(results_WMP, results_OMP, idy='fitted_k',  idx=('log_ctrb_fb','fracVF'), logX=False, 
                     maxy=maxy, fracTrain=fracTrain, logy=False, savfolder=savfolder, suffix='_fittedk_fracVF_fb' )
    
    fit_linear_speed(results_WMP, results_OMP, idy='fitted_k',  idx=('log_ctrb_fb','hit_rate_pre'), logX=False, 
                     maxy=maxy, fracTrain=fracTrain, logy=False, savfolder=savfolder, suffix='_fittedk_hit_rate_' )
    
    fit_linear_speed(results_WMP, results_OMP, idy='fitted_k',  idx=('log_ctrb_fb','expVar_diff'), logX=False, 
                     maxy=maxy, fracTrain=fracTrain, logy=False, savfolder=savfolder, suffix='_fittedk_exp_Var_' )



    #### Regression of change in hit rate

    results_WMP['change_hit'] = (results_WMP['hit_rate_post'] - results_WMP['hit_rate_pre'])/(1- results_WMP['hit_rate_pre'])
    results_OMP['change_hit'] = (results_OMP['hit_rate_post'] - results_OMP['hit_rate_pre'])/(1- results_OMP['hit_rate_pre'])

    maxy=1.1
    miny=-.1

    print('change_hit versus ctrb_fb and ctrb_ff:')
    fit_linear_speed(results_WMP, results_OMP, idy='change_hit',  idx=('ctrb_fb','ctrb_ff'), logX=True, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=False, clampy=True, yfit_max=1, savfolder=savfolder, suffix='_changehit_ctrb_' )

    print('change_hit versus ctrb_fb and fracVF:')
    fit_linear_speed(results_WMP, results_OMP, idy='change_hit',  idx=('ctrb_fb','fracVF'), logX=True, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=False, clampy=True, yfit_max=1, savfolder=savfolder, suffix='_changehit_ctrb_fracVF_' )

    print('change_hit versus fracVF and expVar along decoder:')
    fit_linear_speed(results_WMP, results_OMP, idy='change_hit',  idx=('log_fracVF','expVar_decoder'), logX=False, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=False, clampy=True, yfit_max=1, savfolder=savfolder, suffix='_changehit_fracVF_evdecoder_' )

    print('change_hit versus ctrb_fb and fracVF_fb:')
    fit_linear_speed(results_WMP, results_OMP, idy='change_hit',  idx=('ctrb_fb','fracVF_fb'), logX=True, 
                     maxy=maxy, miny=miny, fracTrain=fracTrain, logy=False, clampy=True, yfit_max=1, savfolder=savfolder, suffix='_changehit_ctrb_fracVFfb_' )


    #####################################################
    # MISCELLANEOUS PLOTS
    #####################################################

    pp.figure()
    pre1=[jj[0] for jj in results_WMP['progress_asymm_pre'] if not np.isnan(jj)]
    pre2=[jj[0] for jj in results_OMP['progress_asymm_pre'] if not np.isnan(jj)]
    post1=[jj[0] for jj in results_WMP['progress_asymm_post'] if not np.isnan(jj)]
    post2=[jj[0] for jj in results_OMP['progress_asymm_post'] if not np.isnan(jj)]
    pp.violinplot( [pre1],[1], showmedians=True, quantiles=[0.05,0.95])
    if len(pre2)>0:
        pp.violinplot( [pre2],[2], showmedians=True, quantiles=[0.05,0.95])
    pp.violinplot( [post1],[4], showmedians=True, quantiles=[0.05,0.95])
    pp.violinplot( [post2],[5], showmedians=True, quantiles=[0.05,0.95])
    pp.ylabel('Progress asymmetry')
    pp.savefig(savfolder+'asymm_progress_violin.png')
    pp.close()

    pp.figure()
    pp.violinplot( [results_OMP['fracTotalVel'].T[0]],[2], showmedians=True, quantiles=[0.05,0.95])
    pp.violinplot( [results_WMP['fracTotalVel'].T[0]],[1], showmedians=True, quantiles=[0.05,0.95])
    pp.plot([0.5,2.5],[1,1],color='k')
    pp.ylabel('ratio of decoder projected variance')
    pp.xticks([1,2],['WMP','OMP'])
    #pp.xticks(np.arange(max(results_OMP['fracTotalVel'])))
    pp.savefig(savfolder+'fracTotalVel_violin.png')
    pp.close()

    pp.figure()
    pp.hist(results_WMP['dimVF'], color='r', alpha=0.5)
    pp.hist(results_OMP['dimVF'], color='b', alpha=0.5)
    pp.xlabel('rDim of vector field change')
    pp.savefig(savfolder+'dim_vf'+dic['save_suffix']+'.png')
    pp.close()

    #####################################################
    # COMPARE REGRESSIONS
    ##################################################### 




###############################################
