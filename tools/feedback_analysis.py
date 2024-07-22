#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pp
import scipy.linalg as ll
import glob

import torch

import tools.perturb_network_ as pn

from scipy.stats import circmean, circstd
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)


def get_v_results_v2(mod, x_0, u_t, e_t, Wout, k=None, dt=1, nonlin=torch.relu, transientT=100):
    '''
    Calculates the velocity change and error transformation of a feedback-controlled RNN.

    Parameters:
        mod (object): The feedback-controlled RNN model.
        x_0 (Tensor): The initial state of the RNN. Shape: 1 x batch x neurons.
        u_t (Tensor): The input tensor. Shape: time x batch x nInputs.
        e_t (Tensor): The error tensor. Shape: time x batch x nInputs.
        Wout (Tensor): The output weight matrix.
        k (int, optional): The number of time steps to consider. Defaults to None.
        dt (int, optional): The time step size. Defaults to 1.
        nonlin (function, optional): The nonlinearity function. Defaults to torch.relu.
        transientT (int, optional): The number of transient time steps. Defaults to 100.

    Returns:
        Tuple: A tuple containing the following tensors:
            - delv_t (Tensor): The velocity change tensor. Shape: time x batch x neurons.
            - rout (Tensor): The norm of change in activity tensor for the output direction. Shape: time x batch.
            - rnull (Tensor): The norm of change in activity tensor for the output null space. Shape: time x batch.
    
    '''
    if k is None:
        k=u_t.shape[0]
    
    if e_t is None:
        e_t = torch.zeros( (u_t.shape[0], u_t.shape[1], 2) )
    
    if len(x_0.shape)<3:
        x_0 = x_0.reshape( 1,-1, x_0.shape[-1] )
    
    r_t_base = torch.zeros( (u_t.shape[0],u_t.shape[1], x_0.shape[2]) )
    r_t_corr = torch.zeros_like( r_t_base )


    ### first go to fixed point
    x_t = x_0
    r_0 = nonlin( x_t )
    for tm in range( transientT ):
        x_t_base = x_t + mod.alpha * ( - x_t + mod.bias_n 
                                        + r_0 @ mod.W_rec 
                                        + u_t[0, :,:] @ mod.W_in 
                                      )
        x_t = x_t_base
        r_0 = nonlin( x_t )


    ### then look at how error is transformed into velocity change
    x_0 = x_t.detach().clone()
    r_0 = nonlin( x_t )

    for tm in range( u_t.shape[0] ):
        x_t_base = x_t + mod.alpha * ( - x_t + mod.bias_n 
                                        + r_0 @ mod.W_rec 
                                        + u_t[tm, :,:] @ mod.W_in 
                                      )
        x_t = x_t_base
        r_0 = nonlin( x_t )
        r_t_base[tm,:,:] = r_0
    
    x_t = x_0.clone()
    r_0 = nonlin( x_t )

    for tm in range( u_t.shape[0] ):
        x_t_corr = x_t + mod.alpha * ( - x_t + mod.bias_n 
                                        + r_0 @ mod.W_rec 
                                        + u_t[tm, :,:] @ mod.W_in 
                                        + e_t[tm, :,:] @ mod.W_fbk )
        x_t = x_t_corr
        r_0 = nonlin( x_t )
        r_t_corr[tm,:,:] = r_0
    

    delv_t = r_t_corr @ Wout - r_t_base @ Wout

    delr_t = r_t_corr - r_t_base
    
    q,r = ll.qr( Wout.numpy() )
    w1 = q[:,:2]
    w2 = q[:,2:]
    w1 = torch.Tensor( w1/ll.norm(w1,axis=0) )
    w2 = torch.Tensor( w2/ll.norm(w2,axis=0) )

    rnull = torch.zeros_like( delr_t)
    rout = torch.zeros_like( delr_t)
    for tm in range( u_t.shape[0] ):
        rout[tm] = torch.transpose( w1 @ w1.T @ delr_t[tm].T, 1,0 )
        rnull[tm] = torch.transpose( w2 @ w2.T @ delr_t[tm].T, 0,1 )

    rout = torch.norm(rout, dim=2)
    rnull = torch.norm(rnull, dim=2)

    return delv_t, rout, rnull




def analyse_all_files( files=None, suffix='', plot_type='vpeak', nt=21, ntgt=13 , scale=1 ):
    '''
    Main file to get dynamics of error correction in feedback-controlled RNNs.

    Send scale = 1 or scale=[0.1,0.3,0.5,0.8,1.0]
    '''

    ntm = 100
    nb=1000
    nin=3
    nout=2
    nneu=100
    transientT=100 

    
    if files is None:
        files = glob.glob('use_models/relu_/*tau5*.npy')

    cm = mpl.colormaps['hot']
    cm3 = mpl.colormaps['viridis']
    cm2 = mpl.colormaps['inferno']

    if isinstance( scale, float ) or isinstance( scale, int ):
        all_av = np.zeros( (len(files),ntm, nt, ntgt ) )
        ff=0
        for file in files:

            net = pn.postModel( file, {'jump_amp':0.0})
            net.load_rnn()
            mod = net.model

            u_t = torch.zeros( (ntm, nb, nin) )
            u_t[:,:,2]=.5
            e_t = torch.zeros( (ntm, nb, nout) )
            x_0 = torch.randn( (1,nb,nneu))

            test_angles = np.linspace( -np.pi, np.pi, nt )#21
            targets = np.linspace(-np.pi, np.pi, ntgt)#13

            av = np.zeros( (ntm, nt, ntgt) )
            vnorm = np.zeros( (ntm, nt, ntgt) )

            p = mod.save_parameters()
            Wout = torch.Tensor( p['W_out_0'])

            ctr=-1
            e_t = torch.zeros( (ntm, nb, nout) )
            for tgt in targets:
                ctr+=1
                u_t[:,:,0]=np.cos(tgt)
                u_t[:,:,1]=np.sin(tgt)
                for aa in range( len(test_angles) ):
                    angle = test_angles[aa]
                    e_t[0:2,:,0] = np.cos(angle)*scale
                    e_t[0:2,:,1] = np.sin(angle)*scale
                    dv, rn, ro = get_v_results_v2( mod, x_0, u_t, e_t, torch.Tensor(Wout) , transientT=transientT )
                    av[:,aa,ctr] = circmean( torch.atan2(dv[:,:,1],dv[:,:,0]).detach().numpy(), axis=1 )   #torch.mean( torch.atan2(dv[:,:,1],dv[:,:,0]),axis=1 ).detach().numpy()
                    vnorm[:,aa,ctr] = np.mean( ll.norm( dv.detach().numpy(), axis=2), axis=1 )        
            
            all_av[ff]  = av

            
            # first 20 timesteps
            if plot_type=='t20':
                for tt in range(0,20):
                    for jj in range(ntgt):
                        pp.scatter( test_angles, np.mod( 4*np.pi + av[tt,:,jj]  , 2*np.pi), color=cm(tt/20), s=10, alpha=0.2) #- test_angles
                pp.ylim([-.1,2*np.pi+.1])
                pp.xlabel('angular error')
                pp.ylabel('angular delta velocity')
                pp.savefig('delv_'+suffix+'.png')

            elif plot_type=='vpeak':
                for jj in range(ntgt):
                    for aa in range(nt):
                        ix = np.argmax( vnorm[:,aa,jj] )
                        pp.scatter( test_angles[aa], np.mod( 4*np.pi + av[ix,aa,jj] , 2*np.pi), color=cm3(jj/ntgt), alpha=0.3 )
                pp.ylim([-.1,2*np.pi+.1])
                pp.xlabel('angular error')
                pp.ylabel('angular delta velocity (tpeak)')
                pp.savefig('delv_vpeak_'+suffix+'.png')
            
            ff+=1 

        nf=len(files)
        if plot_type=='error':
            error = np.mod(4*np.pi+ all_av - np.tile(test_angles.reshape(1,1,-1,1), (nf,ntm,1,ntgt)) ,2*np.pi)  ## is this close to pi? [file,tm,error,target]
            error = np.transpose(error,[2,3,0,1])
            error = np.reshape( error, (-1,nf,ntm) )
            mean_error = circmean( circmean( error, axis=0 ), axis=0 ) # average across targets then across files
            std_error = circstd( error.reshape(-1,ntm), axis=0 ) #std across all files and targets
            pp.errorbar( x=np.arange(ntm), 
                y= mean_error,
                yerr= std_error, 
                color='r', lw=3, alpha=0.5)
            pp.yticks(np.linspace(np.pi/4,3*np.pi/2,11))
            pp.xlabel('time')
            pp.ylabel('angular velocity - target')
            pp.savefig('delv_error_'+suffix+'.png')


        
    if isinstance(scale, list):
        all_vnorm= np.zeros( (len(files),ntm, nt, ntgt ) )
        for sc in scale:
            ff=0
            for file in files:

                net = pn.postModel( file, {'jump_amp':0.0})
                net.load_rnn()
                mod = net.model
                u_t = torch.zeros( (ntm, nb, nin) )
                u_t[:,:,2]=.5
                e_t = torch.zeros( (ntm, nb, nout) )
                x_0 = torch.randn( (1,nb,nneu))

                test_angles = np.linspace( -np.pi, np.pi, nt )#21
                targets = np.linspace(-np.pi, np.pi, ntgt)#13

                av = np.zeros( (ntm, nt, ntgt) )
                vnorm = np.zeros( (ntm, nt, ntgt) )

                p = mod.save_parameters()
                Wout = torch.Tensor( p['W_out_0'])

                
                ctr=-1
                e_t = torch.zeros( (ntm, nb, nout) )
                for tgt in targets:
                    ctr+=1
                    u_t[:,:,0]=np.cos(tgt)
                    u_t[:,:,1]=np.sin(tgt)
                    for aa in range( len(test_angles) ):
                        angle = test_angles[aa]
                        e_t[0:2,:,0] = np.cos(angle)*sc
                        e_t[0:2,:,1] = np.sin(angle)*sc

                        dv, _, _ = get_v_results_v2( mod, x_0, u_t, e_t, torch.Tensor(Wout) , transientT=transientT )
                        av[:,aa,ctr] = circmean( torch.atan2(dv[:,:,1],dv[:,:,0]).detach().numpy(), axis=1 )   #torch.mean( torch.atan2(dv[:,:,1],dv[:,:,0]),axis=1 ).detach().numpy()
                        vnorm[:,aa,ctr] = np.mean( ll.norm( dv.detach().numpy(), axis=2), axis=1 )        
                
                all_vnorm[ff] = vnorm
                ff+=1

            y = np.mean(np.mean( np.mean(all_vnorm,axis=3),axis=2),axis=0)
            yerr=np.median( np.std( np.mean(all_vnorm,axis=2),axis=2),axis=0)
            pp.errorbar( x=np.arange(ntm), y=y, yerr = yerr, color=cm2(sc), lw=sc)

        pp.xlabel('time')
        pp.ylabel('velocity norm')
        for sc in scale:  
            pp.plot([0],[0],c=cm2(sc), label='s='+np.str_(sc))        
        pp.legend()
        pp.savefig('vnorm_allfiles.png')

    pp.close()



    





