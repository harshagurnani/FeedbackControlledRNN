#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network consists of input (feedforward and feedback) modules, an rnn and a readout

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
import mfiles.model_controllers as mc
import mfiles.model_effectors as me
from collections import OrderedDict
from math import sqrt
import numpy as np



class RNN_w_TargetError(nn.Module):
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

        super(RNN_w_TargetError, self).__init__()

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
        self.sigma_n = sigma_n              # rnn activity noise sigma
        
        self.fbk_state = fbk_state      # send rnn state in feedback controller (bool)
        self.add_bias = add_bias        # bias to readout (keep false)
        self.add_bias_n = add_bias_n    # bias current to neurons (generally true)

        # mask gradients for input weights:
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

        # set up effector 
        self.effector = me.twoD_reach_velocity  # not being used currently



    ## -------------- initialisations ------------------
    def init_weights( self, learn_inp=True, learn_rec=True, learn_fbk=True, W_in_0=None, W_rec_0=None, W_fbk_0=None, add_bias_n=False, bias_n=None ):
        ''' 
        Weights are either sampled from normal distribution so that each input is O(1) ... OR
        if weight matrices are given - set accordingly. Make sure weights are Tensors!
        '''

        # Define network parameters: (trainable parameters -> set requires_grad )
        self.W_in = nn.Parameter( torch.Tensor(self.n_inputs, self.n_neurons).to(dtype=self.dtype, device=self.device) )#.to(device=self.device) )    #input weights -> trained
        self.W_in.requires_grad = learn_inp
        self.W_rec = nn.Parameter( torch.Tensor(self.n_neurons, self.n_neurons).to(dtype=self.dtype, device=self.device) )#.to(device=self.device) ) # recurrent weights -> trained
        self.W_rec.requires_grad = learn_rec
        self.W_fbk = nn.Parameter( torch.Tensor(self.n_fbinputs, self.n_neurons).to(dtype=self.dtype, device=self.device) ) #.to(device=self.device) ) # error feedback weights 
        self.W_fbk.requires_grad = learn_fbk                                      # Learn if true
        
        self.bias_n = nn.Parameter(torch.Tensor(1,self.n_neurons).to(dtype=self.dtype, device=self.device) ) #.to(device=self.device))                # bias term
        if self.add_bias_n:
            self.bias_n.requires_grad = learn_rec
        else:
            self.bias_n.requires_grad = False    

        # Initialise parameters:
        with torch.no_grad():
            if W_in_0 is None:
                print('sigma in = '+np.str_(self.init_params['sig_inp_0']))
                self.W_in.normal_(std=self.init_params['sig_inp_0']/sqrt(self.n_inputs))
                #if self.W_in_mask is not None:
                #    self.W_in.mul_( self.W_in_mask )
            else:
                self.W_in.copy_( W_in_0 )
            if self.W_in_mask is not None:
                self.W_in.mul_( self.W_in_mask )
            
            
            if W_rec_0 is None:
                self.W_rec.normal_(std=self.init_params['sig_rec_0']/sqrt(self.n_neurons))
            else:
                self.W_rec.copy_( W_rec_0 )
            
            if W_fbk_0 is None:
                self.W_fbk.normal_(std=self.init_params['sig_fbk_0']/sqrt(self.n_outputs)) 
                #if self.W_fbk_mask is not None:
                #    self.W_fbk.mul_( self.W_fbk_mask )
            else:
                self.W_fbk.copy_( W_fbk_0 )
            if self.W_fbk_mask is not None:
                self.W_fbk.mul_( self.W_fbk_mask )

            if add_bias_n:
                if bias_n is None:
                    self.bias_n.uniform_(-0.3,0.3)
                else:
                    self.bias_n.copy_( bias_n )
            else:
                self.bias_n.fill_(0)

        # Mask wt gradients for inputs?
        if self.W_fbk_mask is not None and learn_fbk:
            self.W_fbk.register_hook((lambda grad: grad.mul_(self.W_fbk_mask)))
        if self.W_in_mask is not None and learn_inp:
            self.W_in.register_hook((lambda grad: grad.mul_(self.W_in_mask)))
            


    def set_nonlinearity(self, nonlinearity ):
        if nonlinearity=='tanh':
            return nn.functional.tanh
        elif nonlinearity=='relu':
            return nn.functional.relu
        elif nonlinearity=='logsigmoid': 
            return nn.functional.logsigmoid
        elif nonlinearity is None:
            return noNonlin
        

    def setup_inputs(self, ff_mod, fb_mod):
        # set up feedforward and feedback modules - allow to swap later by changing these externally
        if ff_mod is None:
            self.ff_mod = noNonlin
            self.ff_type = 'none'
            self.ff_has_hidden = False
        else:
            self.ff_mod = ff_mod        #.to(device=self.device)
            self.ff_type = ff_mod.inp_type
            self.ff_has_hidden = ff_mod.has_hidden
        if fb_mod is None:
            self.fb_mod = noNonlin
            self.fb_type = 'none'
            self.fbk_state = False
            self.fb_has_hidden = False
            print('No feedback module detected ... network state will not be used for feedback')
        else:
            self.fb_mod = fb_mod        #.to(device=self.device)
            self.fb_type = fb_mod.inp_type
            self.fb_has_hidden = fb_mod.has_hidden



    def init_hidden(self):
        " Method to generate random numbers for initial state "
        hidden0 = ((torch.rand(self.batch_size, self.n_neurons)-self.init_params['hidden_init'])*self.init_params['hidden_init_scale']).to(dtype=self.dtype, device=self.device)  #.to(device=self.device)
        error0 = ((torch.rand(self.batch_size, self.n_outputs)-self.init_params['error_init'])*self.init_params['error_init_scale']).to(dtype=self.dtype, device=self.device)     #.to(device=self.device)
        return hidden0, error0
    

    # ------------------- forward pass ------------------
    def f_step(self,xin,x1,r1,err):
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
                                + torch.randn(size = x1.shape).to(device=self.device)* self.sigma_n ) # noise
        
        # apply nonlinearity
        r1 = self.nonlinearity(x1)

        return x1,r1
 

    def forward(self, dt, stim, target, perturb=None):
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
        
        if perturb is None:
            perturb=torch.zeros( tsteps, self.batch_size, self.n_outputs).to(dtype=self.dtype, device=self.device)  #.to(self.device)
            #print('not perturbing output')

        # All tsteps TENSOR
        hidden0 = torch.zeros( tsteps, self.batch_size, self.n_neurons).to(dtype=self.dtype, device=self.device)   #.to(self.device)
        cursor0 = torch.zeros( tsteps, self.batch_size, self.n_outputs).to(dtype=self.dtype, device=self.device)   #.to(self.device)
        for tm in range( tsteps ):

            x1,r1 = self.f_step(stim[tm],x1,r1,erri)                    # run batch in parallel
            
            hidden0[tm] = r1
            velocity = self.readout(hidden0[tm])                        # velocity readout in cm/s
            
            # rnn output updates current position plus any perturbation
            cursor0[tm] = velocity*(dt/1000) + currpos  + perturb[tm]   # in cm, (dt is in ms)      #self.effector( self, r1, currpos, perturb[tm], dt)
            currpos = cursor0[tm]
            erri = currpos - target[tm]
            
            
        return cursor0, hidden0
    

    # ---------------------- commonly accessed parameters ------------------- 
    def load_parameters(self, params):
        state_dict = self.state_dict()

        # Update network weights from arguments
        # input dictionary is on cpu -> send to device
        state_dict['W_in'] = torch.Tensor(params['W_in_0']).to(dtype=self.dtype, device=self.device)  
        state_dict['W_rec'] = torch.Tensor(params['W_rec_0']).to(dtype=self.dtype, device=self.device) 
        state_dict['readout.weight'] = torch.Tensor(params['W_out_0'].T).to(dtype=self.dtype, device=self.device) 
        if self.add_bias:
            state_dict['readout.bias'] = torch.Tensor(params['bias_0']).to(dtype=self.dtype, device=self.device) 
        state_dict['W_fbk'] = torch.Tensor(params['W_fbk_0']).to(dtype=self.dtype, device=self.device) 
        if self.add_bias_n:
            state_dict['bias_n'] = torch.Tensor(params['bias_n']).to(dtype=self.dtype, device=self.device) 

        # load for children
        if 'fbk_p' in params.keys():
            if params['fbk_p'] is not None:
                self.fb_mod.load_parameters( params['fbk_p'] )
        if 'ff_p' in params.keys():
            if params['ff_p'] is not None:
                self.ff_mod.load_parameters( params['ff_p'] )
        
        ## Load --------------------
        self.load_state_dict(state_dict, strict=True)

        ## Update masks ------------
        if 'W_in_mask' in params.keys():
            if params['W_in_mask'] is not None:
                self.W_in_mask = torch.Tensor( params['W_in_mask'] ).to(device=self.device)
            else:
                self.W_in_mask = None
            
        if 'W_fbk_mask' in params.keys():
            if params['W_fbk_mask'] is not None:
                self.W_fbk_mask = torch.Tensor( params['W_fbk_mask'] ).to(device=self.device)
            else:
                self.W_fbk_mask = None

    def save_parameters(self):
        win = self.W_in.cpu().detach().numpy().copy()
        wrec = self.W_rec.cpu().detach().numpy().copy()
        wout = self.readout.weight.cpu().detach().numpy().copy()
        wout = wout.T       # torch.nn.linear module stores weights as n_output x n_input tensor
        #print(wout.shape)
        wfbk = self.W_fbk.cpu().detach().numpy().copy()
        if self.add_bias:
            bias_0 = self.readout.bias.cpu().detach().numpy().copy()
        else:
            bias_0 = None
        if self.add_bias_n:
            bias_n = self.bias_n.cpu().detach().numpy().copy()
        else:
            bias_n = None
        
        if self.fb_type != 'none': 
            fbk_p = copy.deepcopy(self.fb_mod.state_dict())
        else:
            fbk_p=None
        if self.ff_type != 'none':
            ff_p = copy.deepcopy(self.ff_mod.state_dict())
        else:
            ff_p = None


        ## save masks ----------------------
        win_mask = None
        wfb_mask = None
        if self.W_in_mask is not None:
            win_mask = self.W_in_mask.cpu().detach().numpy().copy()
        if self.W_fbk_mask is not None:
            wfb_mask = self.W_fbk_mask.cpu().detach().numpy().copy()


        dic = {'W_in_0':win,'W_rec_0':wrec, 'W_out_0':wout, 'W_fbk_0':wfbk, 
               'bias_0':bias_0, 'bias_n':bias_n, 'fbk_p':fbk_p, 'ff_p': ff_p,
               'W_in_mask':win_mask, 'W_fbk_mask':wfb_mask }
        
        return dic



    



######## HELPERS - NOT METHODS ##########################


def setup_linear_mod( input_dim, output_dim, add_bias=False, bias=None, weights=None, train=False, dtype=torch.FloatTensor, device='cpu' ):
        '''
        Create a linear module with given parameters.
        Used for readout module
        '''
        #print(output_dim)
        layer = nn.Linear( in_features=input_dim, out_features=output_dim, bias=add_bias, device=device, dtype=dtype)  #, device=device , dtype=dtype, device=device
        if weights is not None:
            layer.weight.copy_(weights)         # need to ensure is tensor and on device
        layer.weight.requires_grad = train

        if add_bias:
            layer.bias.requires_grad = train
            if bias is None:
                layer.bias.normal_(std=1/sqrt(output_dim))
            else:
                layer.bias.copy_( bias )

        return layer




def noNonlin( x ):
    ''' No nonlinearity - identity transfer function '''
    return x


