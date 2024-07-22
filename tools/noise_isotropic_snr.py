#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Perturb activity by isotropic gaussian noise and calculate SNR for different directions in activity space.
Main function is analyse_files() which takes a list of files and calculates SNR for different noise levels.
'''

import numpy as np
import scipy.linalg as ll
import tools.perturb_network_ as pn
import tools.analysis_helpers as ah

import matplotlib.pyplot as pp
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)
import glob, os


def get_proj_variance( X, W ):
    # W: N x K
    # sigma: N x N
    W2, _ = ll.qr(W)
    W2 = W2/ll.norm(W2, axis=0)
    W2=W2[:,:W.shape[1]]
    projX = X @ W2
    Xnew = projX @ W2.T # T x N # back in neural space
    v = np.mean(np.var(Xnew,axis=0))    # variance of individual neuron along time
    return v

def get_proj_norm( X, W ):
    # W: N x K
    # sigma: N x N
    W2, _ = ll.qr(W)
    W2 = W2/ll.norm(W2, axis=0)
    W2=W2[:,:W.shape[1]]
    projX = X @ W2
    Xnew = projX @ W2.T # T x N
    v = np.mean(ll.norm(Xnew,axis=1))   #neural activity norm (energy) -> but this depends on mean and shifting baseline so avoid.
    return v


def snr(file, noise_range=[0.03,0.05,0.1,0.25, 0.5, 1,3,10], nshuff=10, tOnset=100, tOffset=400, use_vel=True):
    """
    Calculate the signal-to-noise ratio (SNR) for a given file and noise range.

    Parameters:
    - file (str): The file path.
    - noise_range (list): A list of noise levels to test.
    - nshuff (int): The number of random directions to sample.
    - tOnset (int): The onset time.
    - tOffset (int): The offset time.
    - use_vel (bool): Whether to use velocity or not.

    Returns:
    - dict: A dictionary containing the SNR values for different variables.
    """

    mod = pn.postModel( file, {'jump_amp':0.0, 'bump_amp':0.0, 'delays':[100,101] })  # fix delays to ease mean subtraction
    mod.load_rnn()
    mod.model.sigma_n = 0.0

    p = mod.model.save_parameters()
    tdata = pn.gen_test_data( mod.params )
    dic = mod.test_model(tdata=tdata)
    meanX_0, _ , meanpos_0, _ =  analyse_X(dic, tOnset=tOnset, tOffset=tOffset)     #mean activity and beh trajectory for unperturbed trials

    nneu = dic['res']['activity1'].shape[2]
    wout = p['W_out_0']
    _, px_1 , wint = mod.get_intuitive_map(nPC=8, use_vel=use_vel )

    nnoise = len(noise_range)
    var_S_wout = np.zeros(nnoise)
    var_N_wout = np.zeros(nnoise)
    var_S_wint = np.zeros(nnoise)
    var_N_wint = np.zeros(nnoise)
    var_S_pc12 = np.zeros(nnoise)
    var_N_pc12 = np.zeros(nnoise)
    var_S_wrng = np.zeros(nnoise)
    var_N_wrng = np.zeros(nnoise)

    var_S_beh = np.zeros(nnoise)
    var_N_beh = np.zeros(nnoise)
    var_S_beh_int = np.zeros(nnoise)
    var_N_beh_int = np.zeros(nnoise)

    ctr=0
    for noise in noise_range:
        # run network with noise
        mod.model.sigma_n = noise
        dic = mod.test_model(tdata=tdata)   # use same stimulus for all noise levels

        #meanX, resX, meanpos, respos = analyse_X( dic )
        tr,tm,neu = dic['res']['activity1'][:,tOnset:tOffset,:].shape
        nout = dic['res']['output'].shape[2]
        resX = np.reshape(dic['res']['activity1'][:,tOnset:tOffset,:], (tr*tm,neu))-meanX_0     # activity noise from previous mean
        respos = np.reshape(dic['res']['output'][:,tOnset:tOffset,:], (tr*tm,nout))-meanpos_0   # beh noise from previous mean


        var_S_beh[ctr] = np.mean( np.var(meanpos_0,axis=0) )
        var_N_beh[ctr] = np.mean( np.var(respos,axis=0) )

        var_S_wout[ctr] =  get_proj_variance( meanX_0, wout )
        var_N_wout[ctr] =  get_proj_variance( resX, wout )

        var_S_wint[ctr] =  get_proj_variance( meanX_0, wint )
        var_N_wint[ctr] =  get_proj_variance( resX, wint )

        var_S_pc12[ctr] =  get_proj_variance( meanX_0, px_1.components_[:2,:].T )
        var_N_pc12[ctr] =  get_proj_variance( resX, px_1.components_[:2,:].T  )

        c1 = 0
        c2 = 0
        for ss in range(nshuff):
            wrng = np.random.rand(nneu,2)-0.5
            c1 += get_proj_variance( meanX_0, wrng )
            c2 += get_proj_variance( resX, wrng )
        var_S_wrng[ctr] = c1/nshuff
        var_N_wrng[ctr] = c2/nshuff


        
        ctr+=1

    p['W_out_0'] = wint
    mod.model.load_parameters(p)
    mod.model.sigma_n = 0.0
    dic = mod.test_model(tdata=tdata)
    meanX_0i, _ , meanpos_0i, _ =  analyse_X(dic, tOnset=tOnset, tOffset=tOffset)
    ctr=0
    for noise in noise_range:
        mod.model.sigma_n = noise
        dic = mod.test_model(tdata=tdata)
        #meanXi, resXi, meanposi, resposi = analyse_X( dic )
        nout = dic['res']['output'].shape[2]
        resposi = np.reshape(dic['res']['output'][:,tOnset:tOffset,:], (tr*tm,nout))-meanpos_0i
        var_S_beh_int[ctr] = np.mean(np.var(meanpos_0i,axis=0) )
        var_N_beh_int[ctr] = np.mean( np.var(resposi,axis=0) )
        ctr+=1


    res = {'var_S_wout':var_S_wout, 'var_N_wout':var_N_wout,'var_S_wint':var_S_wint, 'var_N_wint':var_N_wint,
            'var_S_pc12':var_S_pc12, 'var_N_pc12':var_N_pc12, 'var_S_wrng':var_S_wrng, 'var_N_wrng':var_N_wrng ,
            'var_S_beh':var_S_beh, 'var_N_beh':var_N_beh, 'var_S_beh_int':var_S_beh_int, 'var_N_beh_int':var_N_beh_int } 

    return res 

        


def analyse_X( dic, tOnset=100, tOffset=300 ):
    X = dic['res']['activity1'][:,tOnset:tOffset,:] # crop to movement period
    resX = np.zeros_like(X)
    mx = np.zeros_like(X)
    stim = dic['res']['stimulus']
    pos = dic['res']['output'][:,tOnset:tOffset,:] # crop to movement period
    respos = np.zeros_like(pos)
    mp = np.zeros_like(pos)
    

    # subtract condition average
    theta = np.arctan2( stim[:,-1,1], stim[:,-1,0] )
    alltheta = np.unique(theta)
    trials = [ [kk for kk in range(X.shape[0]) if theta[kk]==currtheta] for currtheta in alltheta]
    for jj in range(len(alltheta)):
        meanX = np.mean(X[trials[jj],:,:], axis=0)  # condition-averaged activity
        resX[trials[jj],:,:] = X[trials[jj],:,:] - meanX[np.newaxis,:,:] 
        mx[trials[jj],:,:] = meanX[np.newaxis,:,:] 
        meanpos = np.mean(pos[trials[jj],:,:], axis=0)
        respos[trials[jj],:,:] = pos[trials[jj],:,:] - meanpos[np.newaxis,:,:] 
        mp[trials[jj],:,:] = meanpos[np.newaxis,:,:] 

    meanX = mx
    meanpos = mp

    X = np.reshape( X, (X.shape[0]*X.shape[1],X.shape[2]))
    meanX = np.reshape( meanX, (meanX.shape[0]*meanX.shape[1], meanX.shape[2]))
    resX = np.reshape( resX, (resX.shape[0]*resX.shape[1],resX.shape[2]))

    pos = np.reshape( pos,  (pos.shape[0]*pos.shape[1],pos.shape[2]) )
    meanpos = np.reshape( meanpos, (meanpos.shape[0]*meanpos.shape[1], meanpos.shape[2]))
    respos = np.reshape( respos,  (respos.shape[0]*respos.shape[1],respos.shape[2]) )

    return meanX, resX, meanpos, respos



def analyse_files( files = None, savfolder='saved_plots/noise/',
                    noise_range=[0.03, 0.05, 0.1, 0.3, 1, 2, 5, 10],
                     fname = 'snr_allfiles_relu', nshuff=100, use_vel=True ):
    
    if files is None:
        files = glob.glob('use_models/relu_/*.npy')
    nfiles = len(files)
    nnoise = len(noise_range)

    snr_wout = np.zeros((nfiles,nnoise))
    snr_wint = np.zeros((nfiles,nnoise))
    snr_pc12 = np.zeros((nfiles,nnoise))
    snr_wrng = np.zeros((nfiles,nnoise))
    snr_beh = np.zeros((nfiles,nnoise))
    snr_beh_int = np.zeros((nfiles,nnoise))

    ratio_wout = np.zeros((nfiles,nnoise))
    ratio_wint = np.zeros((nfiles,nnoise))
    ratio_pc12 = np.zeros((nfiles,nnoise))
    ratio_wrng = np.zeros((nfiles,nnoise))
    ratio_beh = np.zeros((nfiles,nnoise))
    ratio_beh_int = np.zeros((nfiles,nnoise))
    
    

    for ff in range(nfiles):
        res = snr( files[ff], noise_range=noise_range, nshuff=nshuff, use_vel=use_vel )
        snr_wout[ff,:] = res['var_S_wout']/res['var_N_wout']
        snr_wint[ff,:] = res['var_S_wint']/res['var_N_wint']
        snr_pc12[ff,:] = res['var_S_pc12']/res['var_N_pc12']
        snr_wrng[ff,:] = res['var_S_wrng']/res['var_N_wrng']
        snr_beh[ff,:] = res['var_S_beh']/res['var_N_beh']
        snr_beh_int[ff,:] = res['var_S_beh_int']/res['var_N_beh_int']

        ratio_wout[ff,:] = 20*np.log10(np.sqrt(res['var_S_wout'])/np.sqrt(res['var_N_wout']))   # in dB
        ratio_pc12[ff,:] = 20*np.log10(np.sqrt(res['var_S_pc12'])/np.sqrt(res['var_N_pc12']))
        ratio_wint[ff,:] = 20*np.log10(np.sqrt(res['var_S_wint'])/np.sqrt(res['var_N_wint']))
        ratio_wrng[ff,:] = 20*np.log10(np.sqrt(res['var_S_wrng'])/np.sqrt(res['var_N_wrng']))
        ratio_beh[ff,:] = 20*np.log10(np.sqrt(res['var_S_beh'])/np.sqrt(res['var_N_beh']))
        ratio_beh_int[ff,:] = 20*np.log10(np.sqrt(res['var_S_beh_int'])/np.sqrt(res['var_N_beh_int']))
    
    if not os.path.exists(savfolder):
        os.makedirs(savfolder)
    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(snr_wout, axis=0) , yerr=np.std(snr_wout, axis=0),fmt='o-', color='b', label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(snr_wint, axis=0) , yerr=np.std(snr_wint, axis=0),fmt='o-',color='r', label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(snr_pc12, axis=0) , yerr=np.std(snr_pc12, axis=0),fmt='o-',color='g', label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(snr_wrng, axis=0) , yerr=np.std(snr_wrng, axis=0),fmt='o-',color='k', label='wrng')
    pp.legend()
    pp.savefig(savfolder+fname+'_ndim.png')
    #pp.savefig(savfolder+fname+'_ndim.svg')
    pp.close()



    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_beh, axis=0) ,  yerr=np.std(ratio_beh, axis=0) ,color='b', label='original')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_beh_int, axis=0) , yerr=np.std(ratio_beh_int, axis=0) ,color='r', label='intuitive')
    pp.legend()
    pp.savefig(savfolder+fname+'_beh_ndim.png')
    #pp.savefig(savfolder+fname+'_beh_ndim.svg')
    pp.close()


    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wout, axis=0) , yerr=np.std(ratio_wout, axis=0), fmt='o-',color='b', label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wint, axis=0) , yerr=np.std(ratio_wint, axis=0),fmt='o-',color='r', label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_pc12, axis=0) , yerr=np.std(ratio_pc12, axis=0),fmt='o-',color='g', label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wrng, axis=0) , yerr=np.std(ratio_wrng, axis=0),fmt='o-',color='m', label='wrng')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_beh, axis=0) , yerr=np.std(ratio_beh, axis=0),fmt='o-',color='y', label='beh')
    pp.legend()
    pp.plot([-1.5,1.1],[0,0],'k--')
    pp.xticks(np.arange(start=-1.5,stop=1.5,step=0.5))
    pp.xlabel('log noise sigma')
    pp.ylabel('SNR in dB')
    pp.ylim([-20,60])
    pp.savefig(savfolder+fname+'_ratio_ndim.png')
    #pp.savefig(savfolder+fname+'_ratio_ndim.svg')
    pp.close()




    