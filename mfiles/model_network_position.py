#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network consists of input (feedforward and feedback) modules, an rnn and a readout - position decoder

model parameters:
    - recurrent weights
    - input weights
    - feedback weights
    - nonlinearity
    - biases
    - readout
    - (optional)feedforward and feedback modules

    
Harsha Gurnani, 2023
"""

import torch
import torch.nn as nn
import copy
from mfiles.model_network import *
import mfiles.model_controllers as mc
from collections import OrderedDict
from math import sqrt
import numpy as np



class RNN_w_TargetError_Pos(RNN_w_TargetError):
    def __init__(self, n_inputs, n_outputs, n_neurons, tau, dtype,                  # architecture parameters
                 dt=1, device='cpu',  fbk_state=False,                              # 
                 W_in_0=None, W_rec_0=None, W_fbk_0=None,                           # weight initialisations
                 W_in_mask=None, W_fbk_mask=None,                                   # non-trainable mask to use for sparse training
                 add_bias_n=False, bias_n=None,  sigma_n=0.0,                       # add bias terms to rnn units?
                 readout = None, W_out_0=None, add_bias=False, bias_0 = None,       # readout - either another module or linear
                 learn_inp=True, learn_rec=True, learn_fbk=True,                    # train which parameters? (for ff/fb modules - set within module)
                 nonlinearity='relu', input_nonlinearity=None, decode_p=False,      # neuronal properties
                 ff_mod=None, fb_mod=None, **kwargs ):                              # feedforward and feedback modules - if none, set up as linear
        " Initialise network architecture and set initial weights"

        super(RNN_w_TargetError_Pos, self).__init__( n_inputs=n_inputs, n_outputs=n_outputs, n_neurons=n_neurons, tau=tau, dtype=dtype, device=device )

        # initialisation parameters
        init_params = {'hidden_init':0.5, 'hidden_init_scale':0.1, 'error_init':0.5, 'error_init_scale':0.1,
                       'sig_inp_0':0.5, 'sig_fbk_0':0.5, 'sig_rec_0':1.05, 'sig_out_0':0.5}
        for key,val in kwargs.items():
            init_params.update({key:val})

        
        # Setting up custom network rather than using RNN module (recurrent layer) - to separate inputs easily
        self.init_params = init_params      # for initialising states 
        self.n_neurons = n_neurons          # Number of neurons
        self.n_inputs = n_inputs            # Dimension of feedforward stim vector = 3
        self.n_outputs = n_outputs          # Dimension of readout      = 2
        self.tau = tau                      # Neuronal decay timescale
        self.alpha = dt/self.tau            # dt/tau -> scaling of update size for discretized dynamics ~0.1 or 0.2 -> updated at run time
        self.dtype = dtype                  # data type (depends on cpu or gpu)
        self.device = device                # device holding model (can transfer model to it later)
        self.sigma_n = sigma_n
        
        self.fbk_state = fbk_state      # send rnn state in feedback controller (bool)
        self.add_bias = add_bias
        self.add_bias_n = add_bias_n
        if W_in_mask is not None:
            self.W_in_mask = torch.Tensor(W_in_mask).to(device=device)
        else:
            self.W_in_mask = None
        if W_fbk_mask is not None:
            self.W_fbk_mask = torch.Tensor(W_fbk_mask).to(device=device)
        else:
            self.W_fbk_mask = None
        
        self.input_nonlinearity = self.set_nonlinearity( input_nonlinearity )
        self.nonlinearity = self.set_nonlinearity( nonlinearity )
        if fb_mod is None:
            self.n_fbinputs = n_outputs     # Dimension of feedback input (= n_outputs if output error)
        else:
            self.n_fbinputs = fb_mod.n_output
        if ff_mod is None:
            self.n_ffinputs = self.n_inputs
        else:
            self.n_ffinputs = ff_mod.n_output
            


        self.init_weights( learn_inp=learn_inp, learn_rec=learn_rec, learn_fbk=learn_fbk, W_in_0=W_in_0, W_rec_0=W_rec_0, W_fbk_0=W_fbk_0, 
                           add_bias_n=add_bias_n, bias_n=bias_n )

        # set up input and output modules
        self.setup_inputs( ff_mod, fb_mod)
        if readout is None:
            self.readout = setup_linear_mod( self.n_neurons, self.n_outputs, add_bias=add_bias, bias=bias_0, dtype=self.dtype , device=self.device )
            self.readout.weight *= self.init_params['sig_out_0']
            if W_out_0 is not None:
                self.readout.weight.copy_( torch.Tensor(W_out_0)) # copy readout weights supplied
        else:
            self.readout = readout

    
    
    # ------------------- forward pass ------------------
    def f_step(self,xin,x1,r1,err,noisex):
        "Network dynamics: Update activity in single step"
        '''Args:    Are of 1 x batch_size x N   shape, where N is relevant dimension
            - xin = feedforward stim (n_inputs)
            - x1 = network state (n_neurons)
            - r1 = nonlinear network state (n_neurons)
            - err = readout (n_outputs)
        '''
        
        # calculate input currents:
        inp = self.ff_mod( xin )            # uses feedforward stim
        
        if self.fbk_state:
            fbk = self.fb_mod( err, r1 )        # uses error (and optionally, network state)
        else:
            fbk = self.fb_mod( err )

        # update network activity
        x1 = x1 + self.alpha * ( - x1  + self.bias_n        # bias current
                                + r1 @ self.W_rec           # recurrent current
                                + inp @ self.W_in           # ff input current
                                + fbk @ self.W_fbk          # fbk current
                                + torch.randn(size = x1.shape).to(device=self.device)* self.sigma_n  # random noise
                                + noisex        )           # injected noise
        
        # apply nonlinearity
        r1 = self.nonlinearity(x1)

        return x1,r1
 

    def forward(self, dt, stim, target, perturb=None, noisex=None ):
        " entire forward pass of network for all tsteps (in a trial - seq dependence) and batches (parallel) "
        '''
        Args:
            - dt (FLOAT:   in milliseconds )
            - stim    (TENSOR:  time x batch x n_inputs)
            - target  (TENSOR:  time x batch x n_outputs)
            - perturb (TENSOR:  optional external perturbation, same size as target)
        '''

        self.alpha = dt/self.tau
        self.batch_size = stim.size(1)
        tsteps = stim.size(0)
        
        hiddeni, erri = self.init_hidden()
        x1 = hiddeni
        r1 = self.nonlinearity(x1)
        
        currpos = torch.zeros(self.batch_size, self.n_outputs).to(dtype=self.dtype, device=self.device)             #.to(self.device)     # current Cursor state
        
        if perturb is None:         # cursor perturbation
            perturb=torch.zeros( tsteps, self.batch_size, self.n_outputs).to(dtype=self.dtype, device=self.device)  #.to(self.device)
            #print('not perturbing output')
        if noisex is None:          # activity perturbation
            noisex=torch.zeros( tsteps, self.batch_size, self.n_neurons).to(dtype=self.dtype, device=self.device)
            #print('not injecting noise in rnn')

        # All tsteps TENSOR
        hidden0 = torch.zeros( tsteps, self.batch_size, self.n_neurons).to(dtype=self.dtype, device=self.device)   #.to(self.device)
        cursor0 = torch.zeros( tsteps, self.batch_size, self.n_outputs).to(dtype=self.dtype, device=self.device)   #.to(self.device)
        for tm in range( tsteps ):

            x1,r1 = self.f_step(stim[tm],x1,r1,erri,noisex[tm])                    # run batch in parallel
            

            #### ONLY UPDATE THIS!
            hidden0[tm] = r1
            new_position = self.readout(hidden0[tm])                        # position readout in cm
            
            # rnn output updates current position plus any perturbation
            cursor0[tm] = new_position + perturb[tm]   # in cm, (dt is in ms)
            currpos = cursor0[tm]
            erri = currpos - target[tm]
            
            
        return cursor0, hidden0
    
