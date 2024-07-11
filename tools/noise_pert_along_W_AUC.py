import numpy as np
import scipy.linalg as ll
import tools.perturb_network_ as pn
import tools.analysis_helpers as ah
import tools.toolbox as tb

import matplotlib.pyplot as pp
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)

import glob, os

def get_proj_norm( X, W ):
    # W: N x K
    # X: T x Neu
    W2, _ = ll.qr(W)
    W2 = W2/ll.norm(W2, axis=0)
    W2=W2[:,:W.shape[1]]
    projX = X @ W2
    Xnew = projX @ W2.T # T x N
    v = np.mean(np.linalg.norm(Xnew,axis=1))
    return v


def get_proj_variance( X, W ):
    # W: N x K
    # X: T x Neu
    W2, _ = ll.qr(W)
    W2 = W2/ll.norm(W2, axis=0)
    W2=W2[:,:W.shape[1]]
    projX = X @ W2
    v = np.mean( np.var(projX, axis=0) )
    return v


def snr( file, noise_range=[1,3,5,10,20,30], nshuff=10 , jump_del=[150], tOnset=150, tOffset=300, use_vel=True, nTr=200 ):
    # jump_del=[150,160,180,200],
    mod = pn.postModel( file, {'jump_amp':0.0, 'bump_amp':0.0, 'delays':[100,101], 'jump_del': jump_del,'testing_perturb':nTr } )
    mod.load_rnn()
    p = mod.model.save_parameters()
    par=mod.params
    tdata = pn.gen_test_data( par )  # use same stimuli for testing
    ntest = par['testing_perturb']
    #tOnset = jump_del[0]+1
    
    dic = mod.test_model( tdata=tdata )
    meanX_0, _ , meanpos_0, _ =  analyse_X(dic)     # mean per trial (condition-specific mean)  #trial x time x neuron
    X = meanX_0[ntest:,tOnset:tOffset,:]                         # only the later period where perturbation happens
    X = np.reshape(X, (X.shape[0]*X.shape[1],X.shape[2]))
    pos = meanpos_0[ntest:,tOnset:tOffset,:]
    pos = np.reshape( pos, (pos.shape[0]*pos.shape[1], pos.shape[2]) )

    nneu = dic['res']['activity1'].shape[2]
    wout = p['W_out_0']
    _, px_1 , wint = mod.get_intuitive_map(nPC=8, use_vel=use_vel)
    pc12 = px_1.components_[:2,:].T

    nnoise = len(noise_range)
    var_S_wout = np.zeros(nnoise)
    var_N_wout = np.zeros(nnoise)
    var_S_wint = np.zeros(nnoise)
    var_N_wint = np.zeros(nnoise)
    var_S_pc12 = np.zeros(nnoise)
    var_N_pc12 = np.zeros(nnoise)
    var_S_wrng = np.zeros(nnoise)
    var_N_wrng = np.zeros(nnoise)


    var_S_wout_neu = np.zeros(nnoise)
    var_N_wout_neu = np.zeros(nnoise)
    var_S_wint_neu = np.zeros(nnoise)
    var_N_wint_neu = np.zeros(nnoise)
    var_S_pc12_neu = np.zeros(nnoise)
    var_N_pc12_neu = np.zeros(nnoise)
    var_S_wrng_neu = np.zeros(nnoise)
    var_N_wrng_neu = np.zeros(nnoise)


    var_S_wout_task = np.zeros(nnoise)
    var_N_wout_task = np.zeros(nnoise)
    var_S_wint_task = np.zeros(nnoise)
    var_N_wint_task = np.zeros(nnoise)
    var_S_pc12_task = np.zeros(nnoise)
    var_N_pc12_task = np.zeros(nnoise)
    var_S_wrng_task = np.zeros(nnoise)
    var_N_wrng_task = np.zeros(nnoise)

    hitrate_wout=np.zeros(nnoise)
    hitrate_pc12=np.zeros(nnoise)
    hitrate_wint=np.zeros(nnoise)
    hitrate_wrng=np.zeros(nnoise)

    ctr=0
    for noise in noise_range:
        
        var_S_wout[ctr], var_N_wout[ctr] , var_S_wout_neu[ctr], var_N_wout_neu[ctr], var_S_wout_task[ctr], var_N_wout_task[ctr], hitrate_wout[ctr] =   pert_X( mod=mod, tdata=tdata, W=wout, ntest=ntest, pos=pos, X=X, noise=noise , pc12=pc12, tOnset=tOnset, tOffset=tOffset )
        
        var_S_pc12[ctr], var_N_pc12[ctr] , var_S_pc12_neu[ctr], var_N_pc12_neu[ctr], var_S_pc12_task[ctr], var_N_pc12_task[ctr], hitrate_pc12[ctr]  =   pert_X( mod=mod, tdata=tdata, W=pc12, ntest=ntest, pos=pos, X=X, noise=noise , pc12=pc12, tOnset=tOnset, tOffset=tOffset)

        c1 = 0
        c2 = 0
        c3=0
        c4=0
        c5=0
        c6=0
        c7=0
        for ss in range(nshuff):
            wrng = np.random.rand(nneu,2)-0.5
            x1, x2, x3, x4, x5, x6, x7 = pert_X( mod=mod, tdata=tdata, W=wrng, ntest=ntest, pos=pos, X=X, noise=noise , pc12=pc12, tOnset=tOnset, tOffset=tOffset )
            c1 += x1
            c2 += x2
            c3 +=x3
            c4+= x4
            c5+=x5
            c6+=x6
            c7+=x7
        var_S_wrng[ctr] = c1/nshuff
        var_N_wrng[ctr] = c2/nshuff
        var_S_wrng_neu[ctr] = c3/nshuff
        var_N_wrng_neu[ctr] = c4/nshuff
        var_S_wrng_task[ctr] = c5/nshuff
        var_N_wrng_task[ctr] = c6/nshuff  
        hitrate_wrng[ctr] = c7/nshuff


        ctr+=1

    p['W_out_0'] = wint
    mod.model.load_parameters(p)
    mod.model.sigma_n = 0.0
    dic = mod.test_model(tdata=tdata)
    meanX_0i, _ , meanpos_0i, _ =  analyse_X(dic)
    Xi = meanX_0i[ntest:,tOnset:tOffset,:]
    Xi = np.reshape( Xi, (Xi.shape[0]*Xi.shape[1],Xi.shape[2]) )
    pos_i = meanpos_0i[ntest:,tOnset:tOffset,:]
    pos_i = np.reshape( pos_i, (pos_i.shape[0]*pos_i.shape[1], pos_i.shape[2]) )
    ctr=0
    for noise in noise_range:
        var_S_wint[ctr], var_N_wint[ctr], var_S_wint_neu[ctr], var_N_wint_neu[ctr], var_S_wint_task[ctr], var_N_wint_task[ctr], hitrate_wint[ctr]  =   pert_X( mod=mod, tdata=tdata, W=wint, ntest=ntest, pos=pos_i, X=Xi, noise=noise , pc12=pc12, tOnset=tOnset, tOffset=tOffset )
        ctr+=1


    res = {'var_S_wout':var_S_wout, 'var_N_wout':var_N_wout,'var_S_wint':var_S_wint, 'var_N_wint':var_N_wint,
            'var_S_pc12':var_S_pc12, 'var_N_pc12':var_N_pc12, 'var_S_wrng':var_S_wrng, 'var_N_wrng':var_N_wrng,
             'var_S_wout_neu':var_S_wout_neu, 'var_N_wout_neu':var_N_wout_neu,'var_S_wint_neu':var_S_wint_neu, 'var_N_wint_neu':var_N_wint_neu,
            'var_S_pc12_neu':var_S_pc12_neu, 'var_N_pc12_neu':var_N_pc12_neu, 'var_S_wrng_neu':var_S_wrng_neu, 'var_N_wrng_neu':var_N_wrng_neu,
             'var_S_wout_task':var_S_wout_task, 'var_N_wout_task':var_N_wout_task,'var_S_wint_task':var_S_wint_task, 'var_N_wint_task':var_N_wint_task,
            'var_S_pc12_task':var_S_pc12_task, 'var_N_pc12_task':var_N_pc12_task, 'var_S_wrng_task':var_S_wrng_task, 'var_N_wrng_task':var_N_wrng_task,
             'hitrate_wout':hitrate_wout, 'hitrate_pc12':hitrate_pc12, 'hitrate_wint':hitrate_wint, 'hitrate_wrng':hitrate_wrng } 

    return res 


def pert_X( mod, tdata, W, ntest, pos, X, noise , pc12, tOnset=150, tOffset=300):
    par=mod.params
    noisex = tb.gen_noisex( nSamples=par['testing_perturb'], dt=par['dt'], maxT=par['test_maxT'], useDelay=par['jump_del'], W=W, mag=noise)
    dic = mod.test_model(tdata=tdata, noisex=noisex)
    Xnew = dic['res']['activity1'][ntest:,tOnset:tOffset,:]
    Xint = np.cumsum( Xnew, axis=1)
    Xnew = np.reshape( Xnew, (Xnew.shape[0]*Xnew.shape[1], Xnew.shape[2]) )
    Xint = np.reshape( Xint, (Xint.shape[0]*Xint.shape[1], Xint.shape[2]) )
    resX = Xnew-X
    posnew = dic['res']['output'][ntest:,tOnset:tOffset,:]
    posnew = np.reshape( posnew, (posnew.shape[0]*posnew.shape[1],posnew.shape[2]) )
    respos = posnew - pos

    var_S_beh =   np.mean(np.trapz( np.abs(pos.T), dx=1)/pos.shape[0])
    var_N_beh = np.mean(np.trapz( np.abs(respos.T), dx=1)/respos.shape[0])
    var_S_neu = np.mean(np.trapz( np.abs(X.T), dx=1)/X.shape[0])
    var_N_neu = np.mean(np.trapz( np.abs(resX.T), dx=1)/resX.shape[0])
    projX = X@pc12
    projX2= Xnew@pc12
    err = projX2-projX
    var_S_task = np.mean(np.trapz( np.abs(projX.T), dx=1)/projX.shape[0])
    var_N_task = np.mean(np.trapz( np.abs(err.T), dx=1)/err.shape[0])

    use_trials=np.arange(ntest,dic['res']['activity1'].shape[0])
    perf = ah.get_performance( dic, use_trials=use_trials, thresh=0.1 )



    return var_S_beh, var_N_beh, var_S_neu, var_N_neu, var_S_task, var_N_task, perf['success']

def analyse_X( dic ):
    X = dic['res']['activity1']
    resX = np.zeros_like(X)
    stim = dic['res']['stimulus']
    pos = dic['res']['output']
    respos = np.zeros_like(pos)
    mp = np.zeros_like(pos)
    mx = np.zeros_like(X)

    # subtract condition average
    theta = np.arctan2( stim[:,-1,1], stim[:,-1,0] )
    alltheta = np.unique(theta)
    trials = [ [kk for kk in range(X.shape[0]) if theta[kk]==currtheta] for currtheta in alltheta]
    for jj in range(len(alltheta)):
        meanX = np.mean(X[trials[jj],:,:], axis=0)
        resX[trials[jj],:,:] = X[trials[jj],:,:] - meanX[np.newaxis,:,:] 
        mx[trials[jj],:,:] = meanX[np.newaxis,:,:] 
        meanpos = np.mean(pos[trials[jj],:,:], axis=0)
        respos[trials[jj],:,:] = pos[trials[jj],:,:] - meanpos[np.newaxis,:,:] 
        mp[trials[jj],:,:] = meanpos[np.newaxis,:,:] 

    meanX = mx
    meanpos = mp

    return meanX, resX, meanpos, respos



def analyse_files( files = None, savfolder='saved_plots/noise/', ext='.png',
                    noise_range=[1,5,10,20,30,50,100], tOffset=300,
                     fname = 'noisepert_AUC_relu', nshuff=10, use_vel=True ):
    

    if files is None:
        files = glob.glob('use_models/relu_/*.npy')
    nfiles = len(files)
    nnoise = len(noise_range)
    clr={'wout':'b', 'wint':'r', 'pc12':'g', 'wrng':'k'}

    snr_wout = np.zeros((nfiles,nnoise))
    snr_wint = np.zeros((nfiles,nnoise))
    snr_pc12 = np.zeros((nfiles,nnoise))
    snr_wrng = np.zeros((nfiles,nnoise))

    ratio_wout = np.zeros((nfiles,nnoise))
    ratio_wint = np.zeros((nfiles,nnoise))
    ratio_pc12 = np.zeros((nfiles,nnoise))
    ratio_wrng = np.zeros((nfiles,nnoise))

    ratio_wout_neu = np.zeros((nfiles,nnoise))
    ratio_wint_neu = np.zeros((nfiles,nnoise))
    ratio_pc12_neu = np.zeros((nfiles,nnoise))
    ratio_wrng_neu = np.zeros((nfiles,nnoise))

    ratio_wout_task = np.zeros((nfiles,nnoise))
    ratio_wint_task = np.zeros((nfiles,nnoise))
    ratio_pc12_task = np.zeros((nfiles,nnoise))
    ratio_wrng_task = np.zeros((nfiles,nnoise))

    hit_wout = np.zeros((nfiles,nnoise))
    hit_pc12 = np.zeros((nfiles,nnoise))
    hit_wint = np.zeros((nfiles,nnoise))
    hit_wrng = np.zeros((nfiles,nnoise))
    
    

    for ff in range(nfiles):
        res = snr( files[ff], noise_range=noise_range, nshuff=nshuff, use_vel=use_vel , tOffset=tOffset )
        snr_wout[ff,:] = res['var_S_wout']/res['var_N_wout']
        snr_wint[ff,:] = res['var_S_wint']/res['var_N_wint']
        snr_pc12[ff,:] = res['var_S_pc12']/res['var_N_pc12']
        snr_wrng[ff,:] = res['var_S_wrng']/res['var_N_wrng']

        hit_wout[ff,:] = res['hitrate_wout']
        hit_wint[ff,:] = res['hitrate_wint']
        hit_pc12[ff,:] = res['hitrate_pc12']
        hit_wrng[ff,:] = res['hitrate_wrng']


        ratio_wout[ff,:] = res['var_N_wout']/res['var_S_wout']#20*np.log10(np.sqrt(res['var_S_wout'])/np.sqrt(res['var_N_wout']))/20
        ratio_pc12[ff,:] = res['var_N_pc12']/res['var_S_pc12']#20*np.log10(np.sqrt(res['var_S_pc12'])/np.sqrt(res['var_N_pc12']))/20
        ratio_wint[ff,:] = res['var_N_wint']/res['var_S_wint']#20*np.log10(np.sqrt(res['var_S_wint'])/np.sqrt(res['var_N_wint']))/20
        ratio_wrng[ff,:] = res['var_N_wrng']/res['var_S_wrng']#20*np.log10(np.sqrt(res['var_S_wrng'])/np.sqrt(res['var_N_wrng']))/20

        ratio_wout_neu[ff,:] = res['var_N_wout_neu']/res['var_S_wout_neu']#20*np.log10(np.sqrt(res['var_S_wout_neu'])/np.sqrt(res['var_N_wout_neu']))/20
        ratio_pc12_neu[ff,:] = res['var_N_pc12_neu']/res['var_S_pc12_neu']#20*np.log10(np.sqrt(res['var_S_pc12_neu'])/np.sqrt(res['var_N_pc12_neu']))/20
        ratio_wint_neu[ff,:] = res['var_N_wint_neu']/res['var_S_wint_neu']#20*np.log10(np.sqrt(res['var_S_wint_neu'])/np.sqrt(res['var_N_wint_neu']))/20
        ratio_wrng_neu[ff,:] = res['var_N_wrng_neu']/res['var_S_wrng_neu']#20*np.log10(np.sqrt(res['var_S_wrng_neu'])/np.sqrt(res['var_N_wrng_neu']))/20

        ratio_wout_task[ff,:] = res['var_N_wout_task']/res['var_S_wout_task']#20*np.log10(np.sqrt(res['var_S_wout_task'])/np.sqrt(res['var_N_wout_task']))/20
        ratio_pc12_task[ff,:] = res['var_N_pc12_task']/res['var_S_pc12_task']#20*np.log10(np.sqrt(res['var_S_pc12_task'])/np.sqrt(res['var_N_pc12_task']))/20
        ratio_wint_task[ff,:] = res['var_N_wint_task']/res['var_S_wint_task']#20*np.log10(np.sqrt(res['var_S_wint_task'])/np.sqrt(res['var_N_wint_task']))/20
        ratio_wrng_task[ff,:] = res['var_N_wrng_task']/res['var_S_wrng_task']#20*np.log10(np.sqrt(res['var_S_wrng_task'])/np.sqrt(res['var_N_wrng_task']))/20
    
    neu_beh_wout = np.divide( ratio_wout , ratio_wout_neu)
    neu_beh_pc12 = np.divide( ratio_pc12, ratio_pc12_neu )
    neu_beh_wint = np.divide( ratio_wint, ratio_wint_neu )
    neu_beh_wrng = np.divide( ratio_wrng, ratio_wrng_neu )
    
    if not os.path.exists(savfolder):
        os.makedirs(savfolder)
    
    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(neu_beh_wout, axis=0) , yerr=np.std(neu_beh_wout, axis=0)/np.sqrt(nfiles-1),fmt='o-', color=clr['wout'], label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(neu_beh_wint, axis=0) , yerr=np.std(neu_beh_wint, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wint'], label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(neu_beh_pc12, axis=0) , yerr=np.std(neu_beh_pc12, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['pc12'], label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(neu_beh_wrng, axis=0) , yerr=np.std(neu_beh_wrng, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wrng'], label='wrng')
    pp.legend()    
    pp.ylabel('normalized impact on behavior')
    pp.xlabel('log noise sigma')
    pp.ylim([0,1.1])
    pp.savefig(savfolder+fname+'_neu_to_beh_t'+np.str_(tOffset)+ext)
    pp.close()


    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(snr_wout, axis=0) , yerr=np.std(snr_wout, axis=0)/np.sqrt(nfiles-1),fmt='o-', color=clr['wout'], label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(snr_wint, axis=0) , yerr=np.std(snr_wint, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wint'], label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(snr_pc12, axis=0) , yerr=np.std(snr_pc12, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['pc12'], label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(snr_wrng, axis=0) , yerr=np.std(snr_wrng, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wrng'], label='wrng')
    pp.legend()
    pp.ylabel('Signal-to-noise variance ratio')
    pp.xlabel('log noise sigma')
    pp.savefig(savfolder+fname+'_ndim_t'+np.str_(tOffset)+ext)
    pp.close()



    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wout, axis=0) , yerr=np.std(ratio_wout, axis=0)/np.sqrt(nfiles-1), fmt='o-',color=clr['wout'], label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wint, axis=0) , yerr=np.std(ratio_wint, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wint'], label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_pc12, axis=0) , yerr=np.std(ratio_pc12, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['pc12'], label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wrng, axis=0) , yerr=np.std(ratio_wrng, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wrng'], label='wrng')
    pp.legend()
    pp.xlabel('log noise sigma')
    pp.ylabel('AUC for pos')#'Log signal-to-noise')#pp.ylabel('SNR in dB')
    #pp.ylim([-1,70])
    pp.plot([0,2],[0,0],'k--')
    pp.savefig(savfolder+fname+'_ratio_beh_ndim_t'+np.str_(tOffset)+ext)
    pp.close()

    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wout_neu, axis=0) , yerr=np.std(ratio_wout_neu, axis=0)/np.sqrt(nfiles-1), fmt='o-',color=clr['wout'], label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wint_neu, axis=0) , yerr=np.std(ratio_wint_neu, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wint'], label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_pc12_neu, axis=0) , yerr=np.std(ratio_pc12_neu, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['pc12'], label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wrng_neu, axis=0) , yerr=np.std(ratio_wrng_neu, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wrng'], label='wrng')
    pp.legend()
    pp.xlabel('log noise sigma')
    pp.ylabel('AUC for r')#Log signal-to-noise for r')#pp.ylabel('SNR along full neural space in dB')
    #pp.ylim([-1,70])
    pp.plot([0,2],[0,0],'k--')
    pp.savefig(savfolder+fname+'_ratio_neu_ndim_t'+np.str_(tOffset)+ext)
    pp.close()

    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wout_task, axis=0) , yerr=np.std(ratio_wout_task, axis=0)/np.sqrt(nfiles-1), fmt='o-',color=clr['wout'], label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wint_task, axis=0) , yerr=np.std(ratio_wint_task, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wint'], label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_pc12_task, axis=0) , yerr=np.std(ratio_pc12_task, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['pc12'], label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(ratio_wrng_task, axis=0) , yerr=np.std(ratio_wrng_task, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wrng'], label='wrng')
    pp.legend()
    pp.xlabel('log noise sigma')
    pp.ylabel('AUC in task space')#Log signal-to-noise in task space')#pp.ylabel('SNR along task space in dB')
    #pp.ylim([-1,70])
    pp.plot([0,2],[0,0],'k--')
    pp.savefig(savfolder+fname+'_ratio_task_ndim_t'+np.str_(tOffset)+ext)
    pp.close()



    pp.figure()
    pp.errorbar(  np.log10(noise_range), np.mean(hit_wout, axis=0) , yerr=np.std(hit_wout, axis=0)/np.sqrt(nfiles-1), fmt='o-',color=clr['wout'], label='wout')
    pp.errorbar(  np.log10(noise_range), np.mean(hit_wint, axis=0) , yerr=np.std(hit_wint, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wint'], label='wint')
    pp.errorbar(  np.log10(noise_range), np.mean(hit_pc12, axis=0) , yerr=np.std(hit_pc12, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['pc12'], label='pc12')
    pp.errorbar(  np.log10(noise_range), np.mean(hit_wrng, axis=0) , yerr=np.std(hit_wrng, axis=0)/np.sqrt(nfiles-1),fmt='o-',color=clr['wrng'], label='wrng')
    pp.legend()
    pp.xlabel('log noise sigma')
    pp.ylabel('Hit rate')#Log signal-to-noise in task space')#pp.ylabel('SNR along task space in dB')
    #pp.ylim([-1,70])
    pp.plot([0,2],[0,0],'k--')
    pp.savefig(savfolder+fname+'_hitrate_t'+np.str_(tOffset)+ext)
    pp.close()





    