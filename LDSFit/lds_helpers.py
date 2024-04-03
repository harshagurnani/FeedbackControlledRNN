import torch
import numpy as np
import scipy.linalg as ll
import matplotlib.pyplot as pp

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
sys.path.insert(0, parentdir) 

import tools.perturb_network_ as pnet

import LDSFit.lds as lds

import torch.nn as nn
import torch.optim as optim
import LDSFit.fit_model as fm


def generate_data( file, usetm=None, trialAvg=True, device='cpu' ):
    '''load model, run the model and reorganize into training data'''

    if trialAvg:
        delays=[100,101]
    else:
        delays=[100,200]

    rnn = pnet.postModel(file, {'jump_amp':0.01, 'delays':delays, 'device':device})
    rnn.load_rnn()

    dic = rnn.test_model()      # numpy
    X = dic['res']['activity1'] # batch, time, neu
    p = dic['res']['output']    # batch, time, xy
    tgt = dic['res']['end_target']
    error = p-tgt
    stim = dic['res']['stimulus'] # batch, time, input

    # reshape into time x batch x n
    xt = np.swapaxes(X,0,1)
    ut = np.concatenate( (stim, error), axis=2 )
    ut = np.swapaxes(ut,0,1) 

    # Restrict time?
    if usetm is None:
        usetm = np.arange( start=100,stop=600)
    xt = xt[usetm,:,:]
    ut = ut[usetm,:,:]
    xt.shape

    if trialAvg:
        theta = np.arctan2( ut[-1,:,1], ut[-1,:,0])
        nstim = len(np.unique(theta))
        tm,batch,nx = xt.shape
        ni = ut.shape[2]
        all_xt = np.zeros( (tm,nstim,nx))
        all_ut = np.zeros( (tm,nstim,ni))
        ctr=0
        for angle in np.unique(theta):
            ids = (theta==angle)
            all_xt[:,ctr,:] = np.mean(xt[:,ids,:], axis=1)
            all_ut[:,ctr,:] = np.mean(ut[:,ids,:], axis=1)
            ctr+=1

    else:
        all_xt=xt
        all_ut=ut

    all_xt = torch.Tensor( all_xt).to(device=device)
    all_ut = torch.Tensor( all_ut ).to(device=device)

    return all_ut, all_xt



def make_model( ut, nz=3, nx=10, ni=5, A_sigma=0.5, C_sigma=0.1, B_sigma=0.5, x0_sigma=.2, batch=10, nonlinearity=None, 
               A0=None, B0=None, C0=None, x0=None, d0=None, device='cpu', doplot=False):
    maxx = 50
    norms = 100
    while norms>maxx:
        model = lds.LNDS( n_states=nz, n_obs=nx, n_input=ni, A_sigma=A_sigma , C_sigma=C_sigma, B_sigma=B_sigma, x0_sigma=x0_sigma, batch=batch, nonlinearity=nonlinearity,
                         A0=A0, B0=B0, C0=C0, x0=x0, d0=d0, device=device)
        zg,xg=model(ut, dt=0.1)
        norms = np.mean(ll.norm(xg.cpu().detach().numpy(), axis=2))
        print(norms)
    if doplot:
        pp.figure()
        pp.plot(xg[:,1,:20].cpu().detach().numpy())
        pp.close()
    
    return model


def generate_train( all_ut, all_xt, epoch=100 , device='cpu'):
    tm, bsample, nx = all_xt.shape
    ni = all_ut.shape[2]

    us = torch.zeros( size=(epoch,tm,bsample,ni), device=device)
    xs = torch.zeros( size=(epoch,tm,bsample,nx), device=device)
    for ee in range(epoch):
        us[ee] = all_ut[:,:bsample,:]
        xs[ee] = all_xt[:,:bsample,:]

    return us, xs



def training_schedule( model, criterion, all_ut, all_xt,
                      epoch1=400, lr1=0.01, epoch2=600, lr2=0.005, epoch3=0, lr3=0.002,
                      alp0=1e-5, gam0=1e-5 , bet0=1e-5, device='cpu' ):
    loss=[]

    # ROUND 1:
    if epoch1>0:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr1)
        us, xs = generate_train(all_ut, all_xt, epoch=epoch1, device=device)
        lc = fm.fit( model, opt, criterion, epoch1, us, xs, alp0=alp0, gam0=gam0 , bet0=bet0, device=device)
        loss.append(lc)

    if epoch2>0:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr2)
        us, xs = generate_train(all_ut, all_xt, epoch=epoch2, device=device)
        lc = fm.fit( model, opt, criterion, epoch2, us, xs, alp0=alp0, gam0=gam0 , bet0=bet0, device=device)
        loss.append(lc)
    
    if epoch3>0:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr3)
        us, xs = generate_train(all_ut, all_xt, epoch=epoch3, device=device)
        lc = fm.fit( model, opt, criterion, epoch3, us, xs, alp0=alp0, gam0=gam0 , bet0=bet0, device=device)
        loss.append(lc)

    return loss


def analyse_file_CV_2( file, train_frac=0.9, nz=6, maxe1a=30 , maxe2a=30, maxe1b=30, maxe2b=30, model=None, C_sigma=0.1, d_sigma=0.1, device='cpu', nonlinearity='relu') :
    
    all_ut, all_xt = generate_data( file, device=device )

    tm, batch, nx = all_xt.shape
    np.random.seed()
    ptrain = np.random.random(nx)
    trainx = all_xt[:,:,ptrain<=train_frac]
    testx = all_xt[:,:,ptrain>train_frac]

    trainid = ptrain<=train_frac
    testid = ptrain>train_frac
    print(np.where(testid)[0])

    # Make model with nx=trainx
    ni=all_ut.shape[2]
    model = make_model( all_ut, nz=nz, ni=ni, nx=trainx.shape[2], nonlinearity=nonlinearity , batch=all_ut.shape[1], device=device)

    # train original model
    criterion = torch.nn.MSELoss(reduction='none')
    loss = training_schedule( model, criterion, all_ut, trainx, epoch1=maxe1a, lr1=0.01, epoch2=maxe2a, lr2=0.005, device=device)

    # create full model:
    C = torch.randn(size=(nz,nx), device=device) * C_sigma
    C[:,np.where(trainid)[0]] = model.C.detach()

    d=torch.randn(size=(1,nx), device=device) *d_sigma
    d[:, trainid] = model.d.detach()

    ### traintrials
    ntr = all_ut.shape[1]
    traintrials = np.random.choice( ntr, np.int_(np.round(0.8*ntr)), replace=False )
    testtrials = [jj for jj in range(ntr) if jj not in traintrials]
    print('testrials= '+np.str_(testtrials))
    use_ut = all_ut[:,traintrials,:]
    use_xt = all_xt[:,traintrials,:]
    x0=model.x0.detach()
    use_x0 = x0[:,traintrials,:]

    #model_full = make_model( all_ut, nz=nz, ni=ni, nx=nx, nonlinearity=nonlinearity , batch=all_ut.shape[1], C0=C, d0=d, A0=model.A.detach(), B0=model.B.detach(), x0=model.x0.detach(), device=device)
    model_full = make_model( use_ut, nz=nz, ni=ni, nx=nx, nonlinearity=nonlinearity , batch=use_ut.shape[1], C0=C, d0=d, A0=model.A.detach(), B0=model.B.detach(), x0=use_x0, device=device)
    model_full.A.requires_grad = False
    model_full.B.requires_grad = False
    model_full.x0.requires_grad = False

    # retrain on C, x0, d:
    #loss_full = training_schedule( model_full, criterion, all_ut, all_xt, epoch1=maxe1b, lr1=0.01, epoch2=maxe2b, lr2=0.005, device=device)
    loss_full = training_schedule( model_full, criterion, use_ut, use_xt, epoch1=maxe1b, lr1=0.01, epoch2=maxe2b, lr2=0.005, device=device)

    z_orig, x_orig = model(all_ut, dt=0.1)
    x_pred = torch.relu(z_orig @ model_full.C + model_full.d)
    x_test_pred = x_pred[:,:,testid].cpu().detach().numpy()
    x_test_orig = all_xt[:,:,testid].cpu().numpy()
    error = x_test_orig[:,testtrials,:]-x_test_pred[:,testtrials,:]
    tm,batch,neu = error.shape
    error = np.reshape( np.swapaxes(error,0,1), [tm*batch, neu])
    den = np.reshape( np.swapaxes(x_test_orig[:,testtrials,:],0,1), [tm*batch, neu])
    
    print(error.shape)
    cev = 1 - np.sum(np.var(error, axis=0))/np.sum(np.var(den,axis=0))

    ## Post-training:
    model_full2 = make_model( all_ut, nz=nz, ni=ni, nx=nx, nonlinearity=nonlinearity , batch=all_ut.shape[1], 
                             C0=model_full.C.detach(), d0=model_full.d.detach(), A0=model.A.detach(), B0=model.B.detach(), x0=x0, device=device)
    zg,xg=model_full2(all_ut, dt=0.1)
    xg_test = xg[:,:,testid].cpu().detach().numpy()
    tm,batch,neu = xg_test.shape
    xg_test = np.reshape( np.swapaxes(xg_test,0,1), [tm*batch, sum(testid)])
    xt_test = np.reshape( np.swapaxes(testx.cpu().numpy(),0,1), [tm*batch, sum(testid)])

    pp.figure(figsize=(14,4))
    pp.plot(xg_test, color='k')
    pp.plot(xt_test, color='r')
    pp.savefig('testneu_activity.png')
    pp.close()

    return loss_full, model_full, all_ut, all_xt, cev