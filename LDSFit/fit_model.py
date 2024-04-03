import torch
import torch.nn as nn
import numpy as np

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
sys.path.insert(0, parentdir) 

def loss( model, input, xt, x0=None, criterion=None, device='cpu' ):
    if criterion is None:
        criterion = torch.nn.MSELoss(reduction='none')

    xgen = model( input, x0, device=device )
    loss = criterion( xgen, xt )

    return loss

def fit( modelt, optimizert, criteriont, nepoch, input, ytrue, x0=None, device='cpu', dt=0.1,
        loss_t=20, alp0=1e-2, gam0=1e-3 , bet0=1e-3, clipv=0.5, printt=10 ):
    
    torch.autograd.set_detect_anomaly(True)

    lc = []
    for epoch in range(nepoch):
        train_running_loss = 0.0
        modelt.train()

        # one training step
        optimizert.zero_grad()
        zt, yobs = modelt( input[epoch], x0=x0, dt=dt )
        
        nTm = yobs.shape[0]


        loss_maxt = 800 # <----- remove this later
        if loss_maxt is None:
            loss_maxt=nTm
        else:
            loss_maxt=min(loss_maxt, nTm)  # make sure not longer than sim series
        
        # calculate loss pnly between loss_t and loss_maxt
        loss = criteriont(yobs[loss_t:loss_maxt], ytrue[epoch,loss_t:loss_maxt] )      

        loss_train = loss.mean() 

        # add regularization
        # term 1: parameters
        regx = alp0*modelt.A.norm(2)
        regin = alp0*modelt.B.norm(2)
        regobs = gam0*modelt.C.norm(2)
        
        # term 2: rates
        reg0act = 0.0#bet0*zt.pow(2).mean()

        loss = loss_train+regin+regx+regobs#+reg0act
        loss.backward()

        #####################################

        torch.nn.utils.clip_grad_norm_(modelt.parameters(), clipv)
        
        optimizert.step()
        
        modelt.eval()
        lc.append(loss.cpu().detach().numpy()) 
        if ( epoch % printt )==0:
            print('epoch: '+np.str_(epoch)+', loss: '+np.str_(lc[-1]))

    return lc



