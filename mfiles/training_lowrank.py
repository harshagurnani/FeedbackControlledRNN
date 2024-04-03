#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Training function for the 2D reaching task for low-rank networks

    
Harsha Gurnani, 2024
'''
# #################  TRAINING FUNCTION ##################### 

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import scipy.linalg as ll


def train_LR(modelt, optimizert, criteriont, tt, targett, stimt, trajt=None, perturb=None, dt=1, 
          alp0=1e-3, bet0=1.9*1e-3, gam0=1e-4,                              # regularisation parameters
          alp1=1e-3, bet1=1.9*1e-3, gam1=1e-4, 
          clipv=0.2, loss_mask=None, loss_t=20, loss_maxt=800,
          dist_thresh = 0.15, hit_period=200, print_t=1 ):
    '''
    model, optimizer, criterion - as standard 
    tt = number of training epochs (each with batch update)
    Inputs to model:
    (can replace later)
        targett = target tensors (tt x time x batch x n_output)
        stim_t = stim tensor (tt x time x batch x n_inputs)
        trajt (optional) = desired trajectory tensor (tt x time x batch x n_inputs) -> used to compute loss if not None
        perturb (optional) = external perturbation added to tensor (tt x time x batch x n_outputs)
    dt = model timestep (in ms/ arbitrary units?)
    '''

    device = modelt.device
    torch.autograd.set_detect_anomaly(True)

    lc = []
    dw_fbk_0 = torch.ones(modelt.W_fbk.shape).to(device=device)   # initialize
    dw_inp_0 = torch.ones(modelt.W_in.shape).to(device=device)   # initialize
    all_dw = torch.empty((0,modelt.W_rec.shape[1])).to(device=device)

    if modelt.m_vectors.requires_grad and (not modelt.W_fbk.requires_grad):
        dw_rec_0 = torch.ones(modelt.W_rec.shape).to(device=device)   # initialize
    elif modelt.fb_has_hidden:
        if modelt.fb_mod.output.weight.requires_grad:
            dw_fbk_0 = torch.ones(modelt.fb_mod.output.weight.shape).to(device=device)   # initialize
            all_dw = torch.empty((0,modelt.fb_mod.output.weight.shape[1])).to(device=device)
    
        

    for epoch in range(tt):
        toprint = OrderedDict()
        train_running_loss = 0.0        
        modelt.train()
        if 'fb_mod' in modelt._modules.keys():
            modelt._modules['fb_mod'].train()

        # one training step
        optimizert.zero_grad()      # empty gradients
        # run one batch - (gradients will be accumulated)
        if perturb is not None:
            output,r0 = modelt(dt, stimt[epoch], targett[epoch], perturb[epoch])
        else:
            output,r0 = modelt(dt, stimt[epoch], targett[epoch])
        
        # only use certain time points for loss - to replace later with mask
        nTm = output.shape[0]
        if loss_maxt is None:
            loss_maxt=nTm
        else:
            loss_maxt=min(loss_maxt, nTm)  # make sure not longer than sim series
        
        # calculate loss only between loss_t and loss_maxt - can add mask later 
        if trajt is None:
            loss = criteriont(output[loss_t:loss_maxt], targett[epoch,loss_t:loss_maxt])        # here target is same as trajectory (e.g. for continuous tracking?) 
        else:
            loss = criteriont(output[loss_t:loss_maxt], trajt[epoch,loss_t:loss_maxt])          # use trajectory explicity 
                

        dist_from_target = output[hit_period:] -  targett[epoch,hit_period:]
        # make sure not collecting gradients here:
        dist_from_target = dist_from_target.cpu().detach().numpy() # to cpu
        dist_from_target = ll.norm(dist_from_target, axis=2)
        hittarget = np.sum((dist_from_target < dist_thresh ), axis=0) # if dist less at any time
        hittarget = np.mean(hittarget > 0)                            # what frac of batches had hits
        toprint['Average hit rate'] = hittarget
        
        

        if loss_mask is not None:
            loss = loss_mask @ loss   # if you want to not weigh all errors equally
        loss_train = loss.mean() 
        toprint['Loss'] = loss_train

        # add regularization
        # term 1: parameters
        regin = alp0*modelt.W_in.norm(2)        # weight L2 regularization - inputs (ff and fbk)
        regfbk = alp0*modelt.W_fbk.norm(2)
        regrec = gam0*modelt.m_vectors.norm(2) + gam0*modelt.n_vectors.norm(2)     # weight L2 regularization for rec
        toprint['R_l0inp'] = regin
        toprint['R_l0rec'] = regrec
        toprint['R_l0fbk'] = regfbk

        # term 2: rates
        reg0act = bet0*r0.pow(2).mean()         # activity regularization
        toprint['R_l0rate'] = reg0act

        loss = loss_train+regin+regrec+regfbk+reg0act

        #  term 3: submodule parameters?
        if modelt.ff_has_hidden:               #'Perceptron' in modelt.ff_type:
            reg_ff_mod = alp1 * modelt.ff_mod.output.weight.norm(2)
            toprint['ff_mod'] = reg_ff_mod
            loss = loss + reg_ff_mod
        
        if modelt.fb_has_hidden:               #'Perceptron' in modelt.fb_type:
            reg_fb_mod = alp1 * modelt.fb_mod.output.weight.norm(2)
            toprint['fb_mod'] = reg_fb_mod
            loss = loss + reg_fb_mod

            
        
        #####################################
        # compute gradients 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelt.parameters(), clipv)
        grads = [ param.grad.detach().flatten()  for param in modelt.parameters()  if param.grad is not None ]
        gradnorm = torch.cat(grads).norm()
        toprint['gradnorm'] = gradnorm

        if modelt.W_fbk.requires_grad:
            dw_fbk = modelt.W_fbk.grad.detach()
            corr = 0
            for id in range(modelt.W_fbk.shape[0]):
                corr+= torch.dot( dw_fbk[id,:],dw_fbk_0[id,:])/(torch.norm(dw_fbk_0[id,:])*torch.norm(dw_fbk[id,:]))
            corr = corr/modelt.W_fbk.shape[0]
            dw_fbk_0 = dw_fbk.detach().clone()
            toprint['corr_grad_fbk'] = corr

            all_dw = torch.concatenate((all_dw,dw_fbk),dim=0)
            cov = (all_dw.T @ all_dw).to(device='cpu')
            cov=cov.numpy()
            ev,_ = np.linalg.eig(cov )
            dim = np.sum(ev)**2/np.sum(ev**2)
            toprint['dim_dw_fbk'] = dim
            corr = corr.item()
        
        elif modelt.fb_has_hidden:
            if modelt.fb_mod.output.weight.requires_grad:
                dw_fbk = modelt.fb_mod.output.weight.grad.detach()
                dw_fbk = dw_fbk
                corr = 0
                for id in range(modelt.fb_mod.output.weight.shape[0]):
                    corr+= torch.dot( dw_fbk[id,:],dw_fbk_0[id,:])/(torch.norm(dw_fbk_0[id,:])*torch.norm(dw_fbk[id,:]))
                corr = corr/modelt.fb_mod.output.weight.shape[0]
                dw_fbk_0 = dw_fbk.detach().clone()
                toprint['corr_grad_fbk'] = corr

                all_dw = torch.concatenate((all_dw,dw_fbk),dim=0)
                cov = (all_dw.T @ all_dw).to(device='cpu')
                cov=cov.numpy()
                ev,_ = np.linalg.eig(cov )
                dim = np.sum(ev)**2/np.sum(ev**2)
                toprint['dim_dw_fbk'] = dim
                corr = corr.item()
        else:
            corr=1
            dim = 0

        if modelt.W_in.requires_grad:
            dw_inp = modelt.W_in.grad.detach()
            corr_in = 0
            for id in range(modelt.W_in.shape[0]):
                corr_in+= torch.dot( dw_inp[id,:],dw_inp_0[id,:])/(torch.norm(dw_inp_0[id,:])*torch.norm(dw_inp[id,:]))
            corr_in = corr_in/modelt.W_in.shape[0] #corr_in/modelt.W_fbk.shape[0] <---- change inp scaling before to 2/3
            dw_inp_0 = dw_inp.detach().clone()
            toprint['corr_grad_inp'] = corr_in
            corr_in = corr_in.item()
        else:
            corr_in=1


        #  -------> optimizer updates parameters
        optimizert.step()

        # save loss terms 
        if modelt.ff_has_hidden:              # 'Perceptron' in modelt.ff_type:
            reg_ff_mod =  reg_ff_mod.detach().item()
        else:
            reg_ff_mod = 0
        if modelt.fb_has_hidden:               #'Perceptron' in modelt.fb_type:
            reg_fb_mod =  reg_fb_mod.detach().item()
        else:
            reg_fb_mod = 0         
        train_running_loss = [loss_train.detach().item(),regin.detach().item(),
                            regrec.detach().item(), regfbk.detach().item(),  reg0act.detach().item(),
                            reg_ff_mod, reg_fb_mod, dim,
                            corr_in, corr, gradnorm.detach().item(), 
                            hittarget ]
        
        
        lc.append(train_running_loss)   
        
        # set grad off (unless train called again)
        modelt.eval()
        if (epoch % print_t)==0:        # print loss terms 
            print(('Epoch=%d | '%(epoch)) +" | ".join("%s=%.4f"%(k, v) for k, v in toprint.items()))
        

    return lc
    






