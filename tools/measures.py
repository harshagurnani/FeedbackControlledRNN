import numpy as np
import scipy.linalg as ll
from matplotlib import pyplot as pp
from scipy.ndimage import uniform_filter1d
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import torch


def compute_popov_test( Wrec, Wfb, Wout ):
    nneu = Wrec.shape[0]
    nfb = Wfb.shape[0]
    nout = Wout.shape[1]
    nA = nneu+nout

    A = Wrec-np.eye(nneu)

    A=A.T           #Post x Pre
    Wout = Wout.T   #Out x Neu
    Wfb = Wfb.T     #Neu x Fb

    FullCM = np.zeros((nA, nA+nfb))
    FullCM[:nneu,nneu:] = Wfb

    FullA = np.zeros((nA, nA))
    FullA[:nneu,:nneu] = A          # remaining columns are 0
    FullA[nneu:,:nneu] = Wout       # remaining columns are 0
    

    evalues = ll.eig(FullA, right=False, left=False)

    ranks = np.zeros((nA,1))

    for eid in range(nA):
        FullCM[:nA,:nA] = evalues[eid]*np.eye(nA)-FullA
        ranks[eid] = np.linalg.matrix_rank(FullCM)

    meanrank = np.sum(ranks>0)/nA           # if ==1 , FullA is controllable
    return meanrank

    

def compute_max_eval(  Wrec, Wfb, Wout ):
    nneu = Wrec.shape[0]
    nfb = Wfb.shape[0]          # should be equal to nout
    nout = Wout.shape[1]
    nA = nneu+nout

    A = Wrec-np.eye(nneu)

    A=A.T           #Post x Pre
    Wout = Wout.T   #Out x Neu
    Wfb = Wfb.T     #Neu x Fb

    FullA = np.zeros((nA, nA))
    FullA[:nneu,:nneu] = A          # 
    FullA[nneu:,:nneu] = Wout       # remaining columns are 0
    FullA[:nneu,nneu:] = Wfb        

    evalues = ll.eig(FullA, right=False, left=False)
    posE = evalues[np.real(evalues)>=-.05]
    isort = np.argsort(np.imag(evalues))
    largeE = evalues[np.abs(np.imag(evalues))>1.1*np.percentile(np.abs(np.imag(evalues)),q=80)]

    return evalues, posE, largeE

def exp_rate(t, a, b, c):
    return a * np.exp(-b *t) + c

def log_rate(t, a, m, k, t0):
    # eqn such that m is intercept at t=0, 
    return a / (1.0 + np.exp(-k * (t - t0))) + m

def fit_logit( hitrate,  p0 = (1.0, 0.0, 1/20.0, 50.0), bounds=([0.0,-0.1,1/5000.0,-300.0],[1.0,1.0,1.0,300.0]) ):  #bounds=([0.0,-0.1,1/2000.0,-150.0],[1.0,1.0,1.0,150.0]) 

    ntrials = hitrate.shape[0]
    tData = np.arange(ntrials)

    popt, pcov = curve_fit(log_rate, tData, hitrate, p0, bounds=bounds, maxfev=10000 )

    fitted_rate = log_rate(tData, popt[0],popt[1],popt[2],popt[3] )
    all_k = 1/popt[2]
    speed= popt[0]*popt[2]*1000    # a * k (~ deltaX/deltaT)
    RMSE = np.std(fitted_rate-hitrate)/np.std(hitrate)
    '''
    if np.random.random()<.05:
        pp.figure()
        pp.plot(hitrate)
        pp.plot(fitted_rate)
        pp.savefig('fittedrate_map_'+np.str_(np.random.randint(100))+'.png')
        pp.close()
    '''
    results = {'fitted_params': popt, 'fitted_k':popt[2], 'RMSE':RMSE, 'speed':speed, 'fitted_rate':fitted_rate }
    return results


def fit_exp( loss,  p0 = (0.5, 0.015, 0.05), bounds=([0.0, 0.001, 0],[5.0, 0.2, 3]) ):

    ntrials = loss.shape[0]
    tData = np.arange(ntrials)

    popt, pcov = curve_fit(exp_rate, tData, loss, p0, bounds=bounds, maxfev=10000 )

    fitted_rate = exp_rate(tData, popt[0],popt[1],popt[2] )

    speed= popt[0]*popt[1]*1000    # a * k (~ deltaX/deltaT)
    RMSE = np.std(fitted_rate-loss)/np.std(loss)
    
    '''
    np.random.seed()

    if np.random.random()<.02:
        print('plotting...')
        pp.figure()
        pp.plot(loss)
        pp.plot(fitted_rate)
        pp.savefig('fittedloss_map_'+np.str_(np.random.randint(100))+'.png')
        pp.close()
    '''
    results = {'fitted_params': popt, 'fitted_k':popt[1], 'RMSE':RMSE, 'speed':speed, 'fitted_rate':fitted_rate }
    return results

def compare_speed_eigval( all_files, idx=-1, avg_window=20 ):

    nfiles = len(all_files)
    speed = []
    fitted_k = []
    max_ev = []

    for ff in range(nfiles):
        train_res = all_files[ff]
        nmaps = len(train_res)
        Wrec = train_res[0][0]['posttrain']['params0']['W_rec_0']
        Wfb = train_res[0][0]['posttrain']['params0']['W_fbk_0']
        
        for map in range(nmaps):
            Wout = train_res[map][0]['wout'].copy()
            Wout

            nReps = len(train_res[map])
            hitrate = train_res[map][0]['posttrain']['lc'][idx]
            allh = np.zeros((hitrate.shape[0],nReps))
            for rep in range(nReps):
                allh[:,rep]= train_res[map][rep]['posttrain']['lc'][idx]
            hitrate = np.mean(allh, axis=1)
            res = fit_logit( uniform_filter1d(hitrate, size=avg_window) )
            speed.append( res['speed'])
            fitted_k.append( res['fitted_k'])
            

            evalues, posE, largeE = compute_max_eval(  Wrec, Wfb, Wout )
            max_ev.append(np.sum(np.real(posE)))

    return speed, fitted_k, max_ev



'''
ids = [8,90,14,56,86]
import numpy as np
import matplotlib.pyplot as pp
import matplo
from tools.measures import *
allres=[]
file='trained_wmp_trainFbk200_.npy'
folder = 'wmp/relu_/'
suffix='_new__PC8/'
for jj in ids:
    fname = folder+'Model_'+np.str_(jj)+'_movePC'+suffix+file
    ar = np.load(fname, allow_pickle=True).item()
    allres.append(ar['train_res'])
 
speed, fitted_k, max_ev = compare_speed_eigval(allres)


folder = 'omp/relu_/'
file='trained_omp_trainFbk200_.npy'
suffix='_new__PC8/'
allres=[]
for jj in ids:
    fname = folder+'Model_'+np.str_(jj)+'_movePC'+suffix+file
    ar = np.load(fname, allow_pickle=True).item()
    allres.append(ar['train_res'])

speed2, fitted_k2, max_ev2 = compare_speed_eigval(allres)

nmap=8

pp.close()
pp.scatter(max_ev,speed)
pp.scatter(max_ev2,speed2,c='r')
pp.savefig('speed.png')
pp.close()
pp.scatter(max_ev,fitted_k)
pp.scatter(max_ev2,fitted_k2,c='r')
pp.savefig('fittedk.png')

'''



def compute_fbk_alignment(  wmp_train, nVec = 8 ):

    Wrec0 = wmp_train['posttrain']['params0']['W_rec_0']
    Wrec1 = wmp_train['posttrain']['params1']['W_rec_0']
    Wfbk0 = wmp_train['posttrain']['params0']['W_in_0']
    Wfbk1 = wmp_train['posttrain']['params1']['W_in_0']
    wout = wmp_train['posttrain']['params1']['W_out_0']

    U0, s0, V0 = ll.svd(Wrec0, compute_uv=True)
    U1, s1, V1 = ll.svd(Wrec1, compute_uv=True)

    angles0 = ll.subspace_angles( Wfbk0.T, U0[:,:nVec] )
    angles1 = ll.subspace_angles( Wfbk1.T, U1[:,:nVec] ) 

    meanV0 = np.arccos(np.mean( np.cos(angles0) ))
    meanV1 = np.arccos(np.mean( np.cos(angles1) ))

    return angles0, angles1, meanV0, meanV1



def compare_speed_alignment( all_files, idx=-1, avg_window=20 ):

    nfiles = len(all_files)
    speed = []
    fitted_k = []
    angle_0 = []
    angle_1 = []
    delta_angle = []

    for ff in range(nfiles):
        train_res = all_files[ff]
        nmaps = len(train_res)
        
        for map in range(nmaps):

            nReps = len(train_res[map])
            hitrate = train_res[map][0]['posttrain']['lc'][idx]
            allh = np.zeros((hitrate.shape[0],nReps))
            for rep in range(nReps):
                allh[:,rep]= train_res[map][rep]['posttrain']['lc'][idx]
            hitrate = np.mean(allh, axis=1)
            res = fit_logit( uniform_filter1d(hitrate, size=avg_window) )
            speed.append( res['speed'])
            fitted_k.append( res['fitted_k'])
            

            angles0, angles1, meanV0, meanV1 = compute_fbk_alignment(  train_res[map][0] )
            #angle_0.append(meanV0)
            #angle_1.append(meanV1)
            angle_0.append(min(angles0))
            angle_1.append(min(angles1))

    delta_angle = np.array(angle_0) - np.array(angle_1)

    return speed, fitted_k, angle_0, angle_1, delta_angle





'''
ids = [8,90,14,56,86]
import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mpl
from tools.measures import *
allres=[]
file='trained_wmp_trainFbk200_.npy'
folder = 'wmp/relu_/'
suffix='_new__PC8/'
for jj in ids:
    fname = folder+'Model_'+np.str_(jj)+'_movePC'+suffix+file
    ar = np.load(fname, allow_pickle=True).item()
    allres.append(ar['train_res'])
 

speed, fitted_k, angle_0, angle_1, delta = compare_speed_alignment(allres)


folder = 'omp/relu_/'
file='trained_omp_trainFbk200_.npy'
suffix='_new__PC8/'
allres2=[]
for jj in ids:
    fname = folder+'Model_'+np.str_(jj)+'_movePC'+suffix+file
    ar = np.load(fname, allow_pickle=True).item()
    allres2.append(ar['train_res'])

speed2, fitted_k2, angle_02, angle_12, delta2 = compare_speed_alignment(allres2)

pp.close()
pp.scatter(np.rad2deg(angle_1),speed)
pp.scatter(np.rad2deg(angle_12),speed2,c='r')
#pp.ylim([200,510])
pp.savefig('speed.png')
pp.close()
pp.scatter(np.rad2deg(angle_1),fitted_k)
pp.scatter(np.rad2deg(angle_12),fitted_k2,c='r')
pp.savefig('fittedk.png')

'''



def compute_wt_change(  wmp_train, use_mod=None, usewt='W_in_0' ):

    #print(wmp_train['posttrain']['params1'].keys())
    if use_mod is None:
        Wfbk0 = wmp_train['posttrain']['params0'][usewt]
        Wfbk1 = wmp_train['posttrain']['params1'][usewt]
        #print(Wfbk1.shape)
    elif use_mod=='fbk_p':
        Wfbk0 = wmp_train['posttrain']['params0']['fbk_p'][usewt].cpu().T
        Wfbk1 = wmp_train['posttrain']['params1']['fbk_p'][usewt].cpu().T
        #print(Wfbk1.shape)
    elif use_mod=='ff_p':
        Wfbk0 = wmp_train['posttrain']['params0']['ff_p'][usewt].cpu().T
        Wfbk1 = wmp_train['posttrain']['params1']['ff_p'][usewt].cpu().T

    wout = wmp_train['posttrain']['params1']['W_out_0']


    meanFb0 = ll.norm(Wfbk0)
    meanFb1 = ll.norm(Wfbk1)
    deltaFb = ll.norm(Wfbk1-Wfbk0)

    return meanFb0, meanFb1, deltaFb

def compute_wt_angle(  wmp_train, usewt='W_in_0' ):
    ''' angle between columns of input wts before and after retraining'''
    Wfbk0 = wmp_train['posttrain']['params0'][usewt]
    Wfbk1 = wmp_train['posttrain']['params1'][usewt]

    nIP = np.zeros(Wfbk0.shape[1])
    for dim in range(Wfbk0.shape[1]):
        nIP[dim] = np.dot(Wfbk0[:,dim], Wfbk1[:,dim])

    return nIP


def compare_speed_wtchange( all_files, idx=-1, avg_window=20, usewt='W_fbk_0' ):

    nfiles = len(all_files)
    speed = []
    fitted_k = []
    meanFb0 = []
    meanFb1 = []
    deltaFb = []

    for ff in range(nfiles):
        train_res = all_files[ff]
        nmaps = len(train_res)
        
        for map in range(nmaps):

            nReps = len(train_res[map])
            hitrate = train_res[map][0]['posttrain']['lc'][idx]
            allh = np.zeros((hitrate.shape[0],nReps))
            for rep in range(nReps):
                allh[:,rep]= train_res[map][rep]['posttrain']['lc'][idx]
            hitrate = np.mean(allh, axis=1)
            res = fit_logit( uniform_filter1d(hitrate, size=avg_window) )
            speed.append( res['speed'])
            fitted_k.append( res['fitted_k'])
            

            fb0, fb1, delta = compute_wt_change(  train_res[map][0], usewt=usewt )
            #angle_0.append(meanV0)
            #angle_1.append(meanV1)
            meanFb0.append(fb0)
            meanFb1.append(fb1)
            deltaFb.append(delta)


    return speed, fitted_k, meanFb0, meanFb1, deltaFb



'''
ids = [8,90,56,86]
import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mpl
from tools.measures import *
allres=[]
file='trained_wmp_trainFbk200_.npy'
folder = 'wmp/relu_/'
suffix='_new__PC8/'
for jj in ids:
    fname = folder+'Model_'+np.str_(jj)+'_movePC'+suffix+file
    ar = np.load(fname, allow_pickle=True).item()
    allres.append(ar['train_res'])
 

speed, fitted_k, meanFb0, meanFb1, deltaFb = compare_speed_wtchange(allres)


folder = 'omp/relu_/'
file='trained_omp_trainFbk200_.npy'
suffix='_new__PC8/'
allres2=[]
for jj in ids:
    fname = folder+'Model_'+np.str_(jj)+'_movePC'+suffix+file
    ar = np.load(fname, allow_pickle=True).item()
    allres2.append(ar['train_res'])

speed2, fitted_k2, meanFb02, meanFb12, deltaFb2= compare_speed_wtchange(allres2)

pp.close()
pp.scatter(meanFb1,speed)
pp.scatter(meanFb12,speed2,c='r')
#pp.ylim([100,510])
pp.savefig('speed.png')
pp.close()
pp.scatter(meanFb1,fitted_k)
pp.scatter(meanFb12,fitted_k2,c='r')
pp.savefig('fittedk.png')

s1 = np.array(speed)
s2=np.array(speed2)
s = np.hstack((s1,s2))

k1 = np.array(fitted_k)
k2=np.array(fitted_k2)
k = np.hstack((k1,k2))

f1= np.array(meanFb1)
f2= np.array(meanFb12)
f = np.hstack((f1,f2))

np.corrcoef(f,s)
np.corrcoef(k,s)
'''


'''
def get_vectorfld_change( res, params0, params1 ):

    # reshape:
    usetm = np.arange(200,500)

    rt = res['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    stim = res['stimulus'][:,usetm,:]
    stim = np.reshape( stim, (stim.shape[0]*stim.shape[1],stim.shape[2]) )
    pos = res['output'][:,usetm,:]
    pos = np.reshape( pos, (pos.shape[0]*pos.shape[1],pos.shape[2]) )
    tgt = res['end_target'][:,usetm,:]
    tgt = np.reshape( tgt, (tgt.shape[0]*tgt.shape[1],tgt.shape[2]) )

    zpost_ff = stim @ params1['W_in_0']
    zpost_fb = (pos-tgt) @ params1['W_fbk_0']

    zpre_ff = stim @ params0['W_in_0']
    zpre_fb = (pos-tgt) @ params0['W_fbk_0']

    nNeu = params1['W_rec_0'].shape[0]
    zpost_rec = rt @ (params1['W_rec_0']-np.eye(nNeu))
    zpre_rec = rt @ (params0['W_rec_0']-np.eye(nNeu))

    zpost = zpost_fb  + zpost_ff + zpost_rec
    zpre = zpre_fb + zpre_ff + zpre_rec

    delZ = (zpost-zpre)
    ev = ll.eig(delZ.T@delZ, right=False,left=False)
    ev = abs(ev)
    dim = np.power(np.sum(ev),2)/np.sum(np.power(ev,2*np.ones(ev.shape)))

    fracZ = np.divide( ll.norm(delZ,axis=1), ll.norm(zpre,axis=1) )
    fracZ_ff = np.divide( ll.norm((zpost_ff-zpre_ff),axis=1), ll.norm(zpre,axis=1) )
    fracZ_fb = np.divide( ll.norm((zpost_fb-zpre_fb),axis=1), ll.norm(zpre,axis=1) )
    fracZ_rec = np.divide( ll.norm((zpost_rec-zpre_rec),axis=1), ll.norm(zpre,axis=1) )

    velpre = zpre @ params1['W_out_0']
    velpost = zpost @ params1['W_out_0']

    fracvel = np.divide( ll.norm(velpost-velpre,axis=1), ll.norm(velpre,axis=1) )
    

    fracZ = np.nanmean(fracZ)
    fracZ_ff = np.nanmean(fracZ_ff)
    fracZ_fb = np.nanmean(fracZ_fb)
    fracZ_rec = np.nanmean(fracZ_rec)
    fracvel = np.nanmean(fracvel)

    return fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim

'''

def get_vectorfld_prcptron( model, res, params0, params1, nonlinearity='relu', alpha=0.2 ):

    # reshape:
    usetm = np.arange(200,500)

    rt = res['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    stim = res['stimulus'][:,usetm,:]
    stim = np.reshape( stim, (stim.shape[0]*stim.shape[1],stim.shape[2]) )
    pos = res['output'][:,usetm,:]
    pos = np.reshape( pos, (pos.shape[0]*pos.shape[1],pos.shape[2]) )
    tgt = res['end_target'][:,usetm,:]
    tgt = np.reshape( tgt, (tgt.shape[0]*tgt.shape[1],tgt.shape[2]) )
    erri = (pos-tgt) 

    err_tensor = torch.Tensor(erri)
    rate_tensor = torch.Tensor(rt)
    
    zpost_ff = stim @ params1['W_in_0']
    wnew = model.save_parameters()
    wnew['fbk_p'] = params1['fbk_p']
    model.load_parameters(wnew)
    currfb = model.fb_mod(err_tensor, rate_tensor).detach().numpy()
    zpost_fb = currfb @ params1['W_fbk_0']

    zpre_ff = stim @ params0['W_in_0']
    wnew['fbk_p'] = params0['fbk_p']
    model.load_parameters(wnew)
    currfb2 = model.fb_mod(err_tensor, rate_tensor).detach().numpy()
    zpre_fb = currfb2 @ params0['W_fbk_0']

    nNeu = params1['W_rec_0'].shape[0]
    zpost_rec = rt @ (params1['W_rec_0']-np.eye(nNeu))
    zpre_rec = rt @ (params0['W_rec_0']-np.eye(nNeu))

    zpost = zpost_fb  + zpost_ff + zpost_rec
    zpre = zpre_fb + zpre_ff + zpre_rec

    delZ = (zpost-zpre)

    xpost = rt + alpha* (zpost + params1['bias_n'])#(1-alpha)*rt + alpha* (zpost + params1['bias_n'])
    xpre = rt + alpha * (zpre + params0['bias_n'])#(1-alpha)*rt + alpha * (zpre + params0['bias_n'])
    if nonlinearity=='relu':
        xpost[xpost<0]=0
        xpre[xpre<0]=0

    
    delX = (xpost-xpre)
    ev = ll.eig(delX.T@delX, right=False,left=False)
    ev = abs(ev)
    dim = np.power(np.sum(ev),2)/np.sum(np.power(ev,2*np.ones(ev.shape)))
    
    fracZ = np.divide( ll.norm(delZ,axis=1), ll.norm(zpre,axis=1) )
    fracZ_ff = np.divide( ll.norm((zpost_ff-zpre_ff),axis=1), ll.norm(zpre,axis=1) )
    fracZ_fb = np.divide( ll.norm((zpost_fb-zpre_fb),axis=1), ll.norm(zpre,axis=1) )
    fracZ_rec = np.divide( ll.norm((zpost_rec-zpre_rec),axis=1), ll.norm(zpre,axis=1) )

    velpre = zpre @ params1['W_out_0']
    velpost = zpost @ params1['W_out_0']

    fracvel = np.divide( ll.norm(velpost-velpre,axis=1), ll.norm(velpre,axis=1) )
    

    fracZ = np.nanmean(fracZ)
    fracZ_ff = np.nanmean(fracZ_ff)
    fracZ_fb = np.nanmean(fracZ_fb)
    fracZ_rec = np.nanmean(fracZ_rec)
    fracvel = np.nanmean(fracvel)

    return fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim


def get_vectorfld_in_U( res, params0, params1, U , nPC=8):

    # reshape:
    usetm = np.arange(200,500)
    U = U.T

    rt = res['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    stim = res['stimulus'][:,usetm,:]
    stim = np.reshape( stim, (stim.shape[0]*stim.shape[1],stim.shape[2]) )
    pos = res['output'][:,usetm,:]
    pos = np.reshape( pos, (pos.shape[0]*pos.shape[1],pos.shape[2]) )
    tgt = res['end_target'][:,usetm,:]
    tgt = np.reshape( tgt, (tgt.shape[0]*tgt.shape[1],tgt.shape[2]) )

    zpost_ff = stim @ params1['W_in_0']
    zpost_fb = (pos-tgt) @ params1['W_fbk_0']

    zpre_ff = stim @ params0['W_in_0']
    zpre_fb = (pos-tgt) @ params0['W_fbk_0']

    nNeu = params1['W_rec_0'].shape[0]
    zpost_rec = rt @ (params1['W_rec_0']-np.eye(nNeu))
    zpre_rec = rt @ (params0['W_rec_0']-np.eye(nNeu))

    zpost = zpost_fb  + zpost_ff + zpost_rec
    zpre = zpre_fb + zpre_ff + zpre_rec
    zpre_inU = zpre @ U[:,:nPC] @ U[:,:nPC].T
    zpost_inU = zpost @ U[:,:nPC] @ U[:,:nPC].T


    delZ = (zpost-zpre)
    delZ_inU = delZ @ U[:,:nPC] @ U[:,:nPC].T
    ev = ll.eig(delZ.T@delZ, right=False,left=False)
    ev = abs(ev)
    dim = np.power(np.sum(ev),2)/np.sum(np.power(ev,2*np.ones(ev.shape)))


    delZ_ff_inU = (zpost_ff-zpre_ff)@ U[:,:nPC] @ U[:,:nPC].T
    delZ_fb_inU = (zpost_fb-zpre_fb)@ U[:,:nPC] @ U[:,:nPC].T
    delZ_rec_inU = (zpost_rec-zpre_rec)@ U[:,:nPC] @ U[:,:nPC].T
    
    # fraction of vector field change in U versus total vector field change
    fracZ = np.divide( ll.norm(delZ_inU,axis=1), ll.norm(zpre_inU,axis=1) )
    fracZ_ff = np.divide( ll.norm(delZ_ff_inU,axis=1), ll.norm(zpre_inU,axis=1) )
    fracZ_fb = np.divide( ll.norm(delZ_fb_inU,axis=1), ll.norm(zpre_inU,axis=1) )
    fracZ_rec = np.divide( ll.norm(delZ_rec_inU,axis=1), ll.norm(zpre_inU,axis=1) )

    velpre = zpre_inU @ params1['W_out_0']
    velpost = zpost_inU @ params1['W_out_0']

    fracvel = np.divide( ll.norm(velpost-velpre,axis=1), ll.norm(velpre,axis=1) )
    

    fracZ = np.nanmean(fracZ)
    fracZ_ff = np.nanmean(fracZ_ff)
    fracZ_fb = np.nanmean(fracZ_fb)
    fracZ_rec = np.nanmean(fracZ_rec)
    fracvel = np.nanmean(fracvel)

    return fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim



def get_velocity_change( res_post, res_orig, res_pre, params0, params1 ):

    # reshape:
    usetm = np.arange(200,500)

    rt = res_orig['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    vel_orig = rt @ params1['W_out_0']
    vel_orig = get_variance_in_U( res_orig['activity1'][:,usetm,:], params1['W_out_0'])
    #var_orig = np.trace(rt.T @ rt)

    rt = res_post['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    vel_post = rt @ params1['W_out_0']
    vel_post = get_variance_in_U( res_post['activity1'][:,usetm,:], params1['W_out_0'])
    #var_post = np.trace(rt.T @ rt)

    rt = res_pre['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    vel_pre = rt @ params1['W_out_0']
    vel_pre = get_variance_in_U( res_pre['activity1'][:,usetm,:], params1['W_out_0'])
    
    #velVar = np.sum(np.var(vel_post, axis=0))/np.sum(np.var(vel_orig, axis=0))
    velVar = vel_post/vel_orig
    #fracVar_orig = vel_orig/var_orig
    #fracVar_post = vel_post/var_post


    return velVar, vel_orig, vel_post, vel_pre






def get_vectorfld_change( res, params0, params1 , nonlinearity='relu', alpha=0.2):

    # reshape:
    usetm = np.arange(200,500)

    rt = res['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    stim = res['stimulus'][:,usetm,:]
    stim = np.reshape( stim, (stim.shape[0]*stim.shape[1],stim.shape[2]) )
    pos = res['output'][:,usetm,:]
    pos = np.reshape( pos, (pos.shape[0]*pos.shape[1],pos.shape[2]) )
    tgt = res['end_target'][:,usetm,:]
    tgt = np.reshape( tgt, (tgt.shape[0]*tgt.shape[1],tgt.shape[2]) )

    zpost_ff = stim @ params1['W_in_0']
    zpost_fb = (pos-tgt) @ params1['W_fbk_0']

    zpre_ff = stim @ params0['W_in_0']
    zpre_fb = (pos-tgt) @ params0['W_fbk_0']

    nNeu = params1['W_rec_0'].shape[0]
    zpost_rec = rt @ (params1['W_rec_0']-np.eye(nNeu))
    zpre_rec = rt @ (params0['W_rec_0']-np.eye(nNeu))

    zpost = zpost_fb  + zpost_ff + zpost_rec
    zpre = zpre_fb + zpre_ff + zpre_rec

    xpost = (1-alpha)*rt + alpha* (zpost + params1['bias_n'])
    xpre = (1-alpha)*rt + alpha * (zpre + params0['bias_n'])
    if nonlinearity=='relu':
        xpost[xpost<0]=0
        xpre[xpre<0]=0

    delZ = (zpost-zpre)
    delX = (xpost-xpre)
    ev = ll.eig(delX.T@delX/delX.shape[0], right=False,left=False)
    ev = abs(ev)
    dim = np.power(np.sum(ev),2)/np.sum(np.power(ev,2*np.ones(ev.shape)))

    fracZ = np.divide( ll.norm(delZ,axis=1), ll.norm(zpre,axis=1) )
    fracZ_ff = np.divide( ll.norm((zpost_ff-zpre_ff),axis=1), ll.norm(zpre,axis=1) )
    fracZ_fb = np.divide( ll.norm((zpost_fb-zpre_fb),axis=1), ll.norm(zpre,axis=1) )
    fracZ_rec = np.divide( ll.norm((zpost_rec-zpre_rec),axis=1), ll.norm(zpre,axis=1) )

    velpre = zpre @ params1['W_out_0']
    velpost = zpost @ params1['W_out_0']

    fracvel = np.divide( ll.norm(velpost-velpre,axis=1), ll.norm(velpre,axis=1) )
    

    fracZ = np.nanmean(fracZ)
    fracZ_ff = np.nanmean(fracZ_ff)
    fracZ_fb = np.nanmean(fracZ_fb)
    fracZ_rec = np.nanmean(fracZ_rec)
    fracvel = np.nanmean(fracvel)

    return fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim










def get_vectorfld_change_all( res, params0, params1 , nonlinearity='relu', alpha=0.2):

    # reshape:
    usetm = np.arange(200,500)
    rt = res['activity1'][:,usetm,:]
    rt = np.reshape( rt, (rt.shape[0]*rt.shape[1],rt.shape[2]) )
    stim = res['stimulus'][:,usetm,:]
    stim = np.reshape( stim, (stim.shape[0]*stim.shape[1],stim.shape[2]) )
    pos = res['output'][:,usetm,:]
    pos = np.reshape( pos, (pos.shape[0]*pos.shape[1],pos.shape[2]) )
    tgt = res['end_target'][:,usetm,:]
    tgt = np.reshape( tgt, (tgt.shape[0]*tgt.shape[1],tgt.shape[2]) )


    
    # weight-driven currents
    zpost_ff = stim @ params1['W_in_0']
    zpost_fb = (pos-tgt) @ params1['W_fbk_0']

    zpre_ff = stim @ params0['W_in_0']
    zpre_fb = (pos-tgt) @ params0['W_fbk_0']

    nNeu = params1['W_rec_0'].shape[0]
    zpost_rec = rt @ (params1['W_rec_0']-np.eye(nNeu))
    zpre_rec = rt @ (params0['W_rec_0']-np.eye(nNeu))

    zpost = zpost_fb  + zpost_ff + zpost_rec
    zpre = zpre_fb + zpre_ff + zpre_rec

    # wt-driven activity change
    xpost = (1-alpha)*rt + alpha* (zpost + params1['bias_n'])
    xpre = (1-alpha)*rt + alpha * (zpre + params0['bias_n'])
    if nonlinearity=='relu':
        xpost[xpost<0]=0
        xpre[xpre<0]=0

    # dimensionality of activity change
    delX = (xpost-xpre)
    ev = ll.eig(delX.T@delX/delX.shape[0], right=False,left=False)
    ev = abs(ev)
    dim = np.power(np.sum(ev),2)/np.sum(np.power(ev,2*np.ones(ev.shape)))

    # wt-driven current change (vector field change)
    delZ = (zpost-zpre)
    fracZ = np.divide( ll.norm(delZ,axis=1), ll.norm(zpre,axis=1) )
    fracZ_ff = np.divide( ll.norm((zpost_ff-zpre_ff),axis=1), ll.norm(zpre,axis=1) )
    fracZ_fb = np.divide( ll.norm((zpost_fb-zpre_fb),axis=1), ll.norm(zpre,axis=1) )
    fracZ_rec = np.divide( ll.norm((zpost_rec-zpre_rec),axis=1), ll.norm(zpre,axis=1) )

    velpre = zpre @ params1['W_out_0']
    velpost = zpost @ params1['W_out_0']
    fracvel = np.divide( ll.norm(velpost-velpre,axis=1), ll.norm(velpre,axis=1) )
    
    # average across timepoints
    fracZ = np.nanmean(fracZ)
    fracZ_ff = np.nanmean(fracZ_ff)
    fracZ_fb = np.nanmean(fracZ_fb)
    fracZ_rec = np.nanmean(fracZ_rec)
    fracvel = np.nanmean(fracvel)

    results = {'fracZ':fracZ, 'fracZ_ff':fracZ_ff, 'fracZ_fb':fracZ_fb, 'fracvel':fracvel, 'fracZ_rec':fracZ_rec, 'dim':dim}


    # propatage once
    rt_1_post = (1-alpha)*rt + alpha* ( rt @ params1['W_rec_0'] + stim @ params1['W_in_0'] + (pos-tgt) @ params1['W_fbk_0'] + params1['bias_n'] )
    rt_1_pre = (1-alpha)*rt + alpha* ( rt @ params0['W_rec_0'] + stim @ params0['W_in_0'] + (pos-tgt) @ params0['W_fbk_0'] + params0['bias_n'] )
    if nonlinearity=='relu':
        rt_1_post[rt_1_post<0] = 0
        rt_1_pre[rt_1_pre<0] = 0


    # trajectory-driven changes
    y_1 = rt_1_post - rt_1_pre
    delerror_1 = rt_1_post @ params1['W_out_0'] - rt_1_pre @ params0['W_out_0']
    
    y_rec = y_1 @ params1['W_rec_0']
    y_fb = delerror_1 @ params1['W_fbk_0']
    y_tot = y_rec + y_fb
    fracY_fb = np.divide( ll.norm(y_fb,axis=1), ll.norm(zpre,axis=1) )  #ll.norm(y_fb,axis=1)#
    fracY_rec = np.divide( ll.norm(y_rec,axis=1), ll.norm(zpre,axis=1) ) #ll.norm(y_rec,axis=1)#
    angle_y = np.zeros((y_tot.shape[1],1))
    for tt in range(len(angle_y)):
        angle_y[tt] = np.rad2deg(np.arccos(np.dot( y_rec[tt]/ll.norm(y_rec[tt]), y_fb[tt]/ll.norm(y_fb[tt]))))
    

    fracY_fb = np.nanmean(fracY_fb)
    fracY_rec = np.nanmean(fracY_rec)
    angle_y = np.nanmedian(angle_y)
    print('angle between state-driven rec and fbk dep VF change: '+np.str_(angle_y))
    y_tot = np.nanmean( ll.norm(y_tot, axis=1))

    results.update({'y_tot':y_tot, 'fracY_fb':fracY_fb, 'fracY_rec':fracY_rec, 'angle_y':angle_y })


    return fracZ, fracZ_ff, fracZ_fb, fracvel, fracZ_rec, dim, y_tot, fracY_fb, fracY_rec, angle_y


def get_variance_in_U( X, U, npc=8 ):
    tr, tm, neu = X.shape
    X = np.reshape(X, (tr*tm,neu))
    X = X-np.mean(X,axis=0)       
    sigma = X.T @ X             # covariance matrix

    uv = U[:,:npc].T            # orthonormalized dimensions
    ev_within = np.trace( uv @ sigma @ uv.T )
    ev_total = np.trace(sigma)

    return ev_within/ev_total       # fractional variance



def get_change_IM( Xpre, Xpost, U, wpert, npc=8 ):
    tr, tm, neu = Xpre.shape
    Xpre = np.reshape(Xpre, (tr*tm,neu))
    Xpre = Xpre-np.mean(Xpre,axis=0)

    tr, tm, neu = Xpost.shape
    Xpost = np.reshape(Xpost, (tr*tm,neu))
    Xpost = Xpost-np.mean(Xpost,axis=0)

    sigma_pre = Xpre.T @ Xpre               # covariance matrix
    sigma_post = Xpost.T @ Xpost

    uv = U[:,:npc].T
    ev_within_pre = np.diag( uv @ sigma_pre @ uv.T )    # ev along different pcs
    ev_total_pre = np.trace(sigma_pre)

    ev_within_post = np.diag( uv @ sigma_post @ uv.T )
    ev_total_post = np.trace(sigma_post)

    u1 = np.sqrt(ev_within_pre/ev_total_pre)            # variance-along-dim vector
    u1 = u1/ll.norm(u1)
    u2 = np.sqrt(ev_within_post/ev_total_post)
    u2 = u2/ll.norm(u2)

    px1 = PCA( n_components=npc)
    px2 = PCA( n_components=npc)
    px1.fit(Xpre)               # new manifolds
    px2.fit(Xpost)
    angles = np.rad2deg( ll.subspace_angles(px1.components_.T, px2.components_.T ))


    influence_pre = np.zeros((npc,wpert.shape[1]))
    influence_post = np.zeros((npc,wpert.shape[1]))
    for jj in range(npc):
        for kk in range(wpert.shape[1]):
            influence_pre[jj,kk] = u1[jj]* uv[jj,:]@wpert[:,kk]         #impact of each pc dim on each wpert dim
            influence_post[jj,kk] = u2[jj]* uv[jj,:]@wpert[:,kk] 
    influence_pre = influence_pre/ll.norm(influence_pre,axis=0)
    influence_post = influence_post/ll.norm(influence_post, axis=0)

    inf_overlap = np.trace(influence_pre.T @ influence_post)/wpert.shape[1]

    return np.dot(u1, u2), angles, inf_overlap


def get_decoder_overlap( wout0, wout1 ):
    wout0 = wout0/ll.norm(wout0, axis=0)
    wout1 = wout1/ll.norm(wout1, axis=0)
    ov = np.trace( wout0.T@wout1 )/wout1.shape[1]
    return ov
