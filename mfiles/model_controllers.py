#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for feedforward or feedback input transformations:
At the moment, classes are specified explicitly as 1 or 2 layer feedforward networks - instead of using sequential class for MLP

The 'forward' function is rewritten in each case - this is the fn called by default, 
when a call is made to the module. 
Eg:
ff_mod = Input_Linear_Nonlinear( n_inp=3, n_out=100 )
y = ff_mod(x) is equivalent to : y = ff_mod.forward(x)

Harsha Gurnani, 2023
"""

import torch
import torch.nn as nn
from math import sqrt


# --------------------------------------------------------------------------------------------#
#                       SUPERCLASS FOR ALL CONTROLLER MODULES
# --------------------------------------------------------------------------------------------#

class Controller( nn.Module ):
    def __init__(self, n_inp, n_out, device='cpu', dtype=torch.FloatTensor, inp_type='none' ):
        super(Controller, self).__init__()
        ''' Essential attributes for all controllers '''

        self.device = device            # device to load module on
        self.dtype = dtype              # tensor type
        self.n_input = n_inp            # input dimension
        self.n_output = n_out           # output dimension
        self.inp_type = inp_type        # controller type

        # only defaults - proper intialisation elsewhere
        self.has_hidden = False         # has a hidden layer?     
        self.output = None              # output module or function - the last transformation?

    def forward(self, input, input2=None ):
        # dummy 
        if self.output is None:
            outv = input
        else:
            outv = self.output( input )
        return outv
    
    def load_parameters( self, params ):
        state_dict = self.state_dict()
        for key,val in params.items():
            state_dict[key] = val
        self.load_state_dict(state_dict, strict=True)

    def save_parameters( self ):
        return self.state_dict()
    

# --------------------------------------------------------------------------------------------#








# ------------------------------------------------------------#
# Type 1: simple linear transformation + optional point nonlinearity (eg. for clamping )
class Input_Linear_NonLinear( Controller ):
    ''' CONTAINS: 
            A single Linear output layer that takes in [input].
        Optionally a nonlinearity can be applied to the linear module's output.
    '''
    def __init__(self, n_inp, n_out, device='cpu', dtype=torch.FloatTensor, 
                 weight_out=None, add_bias=False, bias=None, train=False, nonlinearity=None, 
                 inp_type='linear' ):           # should only need to modify this for special cases
        super().__init__(   n_inp=n_inp, n_out=n_out, device=device,  dtype=dtype, inp_type=inp_type )

        self.has_hidden = False
        self.output = nn.Linear( n_inp, n_out, bias=add_bias, dtype=dtype, device=self.device) # device=device  # automatically generates weight (and bias) as Parameters
        self.train_out = train
        self.nonlinearity = set_nonlinearity( nonlinearity=nonlinearity )
        self.add_bias = add_bias
        # Initial values of parameters:
        if weight_out is not None:                      # else, initialised with std = 1/sqrt(n_in)
            self.output.weight.copy_(weight_out)        # need to ensure is tensor and on device
        self.output.weight.requires_grad = self.train_out        
        if bias is not None:
            self.output.bias.copy_(bias)
            self.output.bias.requires_grad = self.train_out
        

    def forward(self, input, input2=None ):
        # dummy input2 not used
        outv = self.nonlinearity(self.output(input))
        return outv
    

    def load_parameters( self, params ):
        state_dict = self.state_dict()
        state_dict['output.weight'] = torch.Tensor(params['weight_out'].T).to(dtype=self.dtype, device=self.device) 
        if self.add_bias:
            state_dict['output.bias'] = torch.Tensor(params['bias']).to(dtype=self.dtype, device=self.device) 
        self.load_state_dict(state_dict, strict=True)

    def save_parameters(self):
        wout = self.output.weight.cpu().detach().numpy().copy()
        wout = wout.T 
        if self.add_bias:
            bias_0 = self.output.bias.cpu().detach().numpy().copy()
        else:
            bias_0 = None
        
        dic = { 'weight_out':wout, 'bias':bias_0 }
        return dic
    


# ------------------------------------------------------------#
# Type 2: simple linear transformation that includes stim input and a transformed input2 (usually rnn state)
# ------------------------------------------------------------#

class Input_Linear_NonLinear_2Inp( Input_Linear_NonLinear ):
    ''' CONTAINS: 
            Initialised with a Linear output layer (that takes in [input, F(input2)]). F is identity function by default.
        - Optionally has another Linear (or custom) layer that transforms input2 (returns F(input2) ) before sending to output layer
    '''

    def __init__(self, n_inp, n_out, n_neu=None, n_mf=None, inp_type='linear_2Inp',
                    sample_type=None, sample_neu=None, train_neu=None, train=None, # additional parameters to subsample input2 
                    **kwargs ):                     # any other belonging to linear controller class
        # Pass essential parameters for dimensions, sampler and training - everything else can come however:
       
        # the lin_nonlin layer should be taking mfs+stim as stacked input -> dim = n_inp + n_mf
        if n_mf is None:
            n_mf = n_neu                # remember to pass n_neu otherwise error
        n_totinp = n_inp + n_mf         # input dim = stim + sampled rnn
        super().__init__(   n_inp=n_totinp, n_out=n_out, inp_type=inp_type, train=train,  **kwargs  )
        

        # add extra attributes
        self.n_neu = n_neu
        self.n_mf = n_mf
        self.sample_type = sample_type        
        self.train_neu = train_neu      # train neu sampler?    note: for the output layer train_out set to 'train' 

        # optional submodule:
        if self.sample_type is None:
            self.sample_neu = noNonlin       
        elif self.sample_type == 'linear':
            # basic linear module - no bias
            self.sample_neu = nn.Linear( in_features=self.n_neu, out_features=self.n_mf, bias=False, dtype=self.dtype, device=self.device )   #, device=device        
            self.sample_neu.weight.requires_grad = train_neu
        elif self.sample_type == 'lin_nonlin':
            # linear + nonlinearity
            self.sample_neu = Input_Linear_NonLinear( n_inp=self.n_neu, n_out=self.n_mf, train=train_neu, dtype=self.dtype, device=self.device)    #, device=device 
        elif self.sample_type == 'custom':
            # send your own
            self.sample_neu = sample_neu
            # because of this - no new load parameters function -> you can define your own if needed


    def forward(self, input, input2 ):
        # just rewrite forward pass to use input2
        subr = self.sample_neu(input2)                    # subsample input 2 (= rnn state)
        input = torch.concat( (input, subr), dim=-1 )     # stack inputs 
        
        outv = self.nonlinearity(self.output(input))
        return outv
    
    def load_parameters( self, params ):
        #------- REMOVE LATER----------------------
        state_dict = {} #self.state_dict()
        state_dict['output.weight'] = torch.Tensor(params['output.weight']).to(dtype=self.dtype, device=self.device) 
        if self.add_bias:
            state_dict['output.bias'] = torch.Tensor(params['output.bias']).to(dtype=self.dtype, device=self.device) 
        state_dict['sample_neu.weight'] = torch.Tensor(params['sample_neu.weight']).to(dtype=self.dtype, device=self.device) 
        if self.add_bias:
            state_dict['sample_neu.bias'] = torch.Tensor(params['sample_neu.bias']).to(dtype=self.dtype, device=self.device) 
        self.load_state_dict(state_dict, strict=True)





# --------------------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------------------#







# ------------------------------------------------------------#
# Type 3: 2-layer perceptron with possible nonlinearity:
# ------------------------------------------------------------#
class Perceptron( Controller ):
    ''' CONTAINS: 
            A hidden layer : Linear Module + nonlinearity
            An output layer: Linear Module + nonlinearity
        Useful to model an expansion-compression/ compression-expansion/ cerebellar-like architecture
        All main attributes are assigned rather than to submodules (for now...)
    '''
    def __init__(self, n_inp, n_hid, n_out,  inp_type='Perceptron', device='cpu', dtype=torch.FloatTensor, 
                 weight_hid=None, weight_out=None,
                 add_bias=False, bias=None, add_bias_hid=None, bias_hid=None, add_bias_out=None, bias_out=None,
                 train_hid=False, train_out=False,
                 nonlinearity_hid=None, nonlinearity_out=None ):
        super().__init__(   n_inp=n_inp, n_out=n_out, device=device,  dtype=dtype, inp_type=inp_type )

        if add_bias_hid is None:
            add_bias_hid = add_bias
        if add_bias_out is None:
            add_bias_out = add_bias

        self.n_hidden = n_hid
        self.has_hidden = True
        self.add_bias = add_bias
        self.add_bias_hid = add_bias_hid
        self.add_bias_out = add_bias_out

        self.hidden = nn.Linear( self.n_input, self.n_hidden, bias=self.add_bias_hid, dtype=dtype, device=self.device)      #  # hidden layer     #, device=device 
        self.train_hid = train_hid
        self.output = nn.Linear( self.n_hidden, self.n_output, bias=self.add_bias_out, dtype=dtype, device=self.device)     # # output layer      #, device=device 
        self.train_out = train_out

        # initialised with std = 1/sqrt(n_in)
        # HIDDEN LAYER
        if weight_hid is not None:
            self.hidden.weight.copy_(weight_hid)        # need to ensure is tensor and on device
        self.hidden.weight.requires_grad = train_hid

        if self.add_bias_hid and (bias_hid is not None):                           # bias randomly assigned - or update externally
            self.hidden.bias.copy_(bias_hid)
            self.hidden.bias.requires_grad = train_hid

        # OUTPUT LAYER
        if weight_out is not None:
            self.output.weight.copy_(weight_out)        # need to ensure is tensor and on device
        self.output.weight.requires_grad = train_out
        
        if self.add_bias_out and (bias_out is not None):               # bias randomly assigned - or update externally
            self.output.bias.copy_(bias_out)
            self.output.bias.requires_grad = train_out

        # set unit nonlinearities (optional)
        self.nonlinearity_hid = set_nonlinearity( nonlinearity=nonlinearity_hid )
        self.nonlinearity_out = set_nonlinearity( nonlinearity=nonlinearity_out )
        

    def forward(self, input, input2=None ):
        #Only stim input is used, n_neu/input2 is redundant - only to maintain uniformity of inputs
        hid = self.nonlinearity_hid( self.hidden(input) )
        outv = self.nonlinearity_out(self.output(hid))
        return outv


    def load_parameters( self, params ):
        state_dict = self.state_dict()
        # HIDDEN LAYER      -- need to transpose weights as linear by default assumes right mult
        state_dict['hidden.weight'] = torch.Tensor(params['weight_hid'].T).to(dtype=self.dtype, device=self.device) 
        if self.add_bias_hid:
            state_dict['hidden.bias'] = torch.Tensor(params['bias_hid']).to(dtype=self.dtype, device=self.device) 
        # OUTPUT LAYER
        state_dict['output.weight'] = torch.Tensor(params['weight_out'].T).to(dtype=self.dtype, device=self.device) 
        if self.add_bias_out:
            state_dict['output.bias'] = torch.Tensor(params['bias_out']).to(dtype=self.dtype, device=self.device) 
        self.load_state_dict(state_dict, strict=True)


    def save_parameters(self):
        whid = self.hidden.weight.cpu().detach().numpy().copy()
        whid = whid.T
        wout = self.output.weight.cpu().detach().numpy().copy()
        wout = wout.T 
        if self.add_bias_hid:
            bias_hid = self.hidden.bias.cpu().detach().numpy().copy()
        else:
            bias_hid = None
        if self.add_bias_out:
            bias_out = self.output.bias.cpu().detach().numpy().copy()
        else:
            bias_out = None

        
        dic = {  'weight_hid':whid, 'bias_hid':bias_hid ,  'weight_out':wout, 'bias_out':bias_out }
        return dic
    



# ------------------------------------------------------------#
# Type 4: Extension of 2-layer perceptron that uses the second input:
# ------------------------------------------------------------#
class Perceptron_2Inp( Perceptron ):
    ''' CONTAINS: 
            A hidden layer : Linear Module + nonlinearity
            An output layer: Linear Module + nonlinearity
            - Sampling Layer (optional): Transforms input2 (returns F(input2) ) before sending to hidden layer
    '''
    def __init__(self, n_inp, n_hid, n_out, inp_type = 'Perceptron_2Inp', device='cpu', dtype=torch.FloatTensor,
                 n_neu=None, n_mf = None, sample_type=None, sample_neu=None, # additional parameters to subsample input2             
                 train_hid=False, train_out=False, train_neu=None,  
                 **kwargs ):
        
        if (n_mf is None) or (sample_type is None):     # needs n_mf == n_neu 
            n_mf = n_neu
        n_totinp = n_inp + n_mf         # input dim = stim + sampled rnn

        # initialise all from parent
        super().__init__( n_inp=n_totinp, n_hid=n_hid, n_out=n_out, device=device, dtype=dtype, inp_type=inp_type, \
                         train_hid=train_hid, train_out=train_out, **kwargs  )
        
        # add extra attributes
        self.n_neu = n_neu
        self.n_mf = n_mf
        self.sample_type = sample_type
        self.train_neu = train_neu

        # add submodule:
        if self.sample_type is None:
            self.sample_neu = noNonlin          # needs n_mf == n_neu 
        elif self.sample_type == 'linear':
            # basic linear module - no bias
            self.sample_neu = nn.Linear( in_features=self.n_neu, out_features=self.n_mf, bias=False, dtype=self.dtype, device=self.device )   #, device=device        
            self.sample_neu.weight.requires_grad = train_neu
        elif self.sample_type == 'lin_nonlin':
            # linear + nonlinearity
            self.sample_neu = Input_Linear_NonLinear( n_inp=self.n_neu, n_out=self.n_mf, train=train_neu, dtype=self.dtype, device=self.device)    #, device=device 
        elif self.sample_type == 'custom':
            # send your own
            self.sample_neu = sample_neu

        # because of this - no new load parameters function -> you can define your own if needed
            
        
    # just rewrite forward pass
    def forward(self, input, input2 ):
        
        subr = self.sample_neu(input2)                    # subsample input 2 (= rnn state)
        input = torch.concat( (input, subr), dim=-1 )      # stack inputs 
        
        hid = self.nonlinearity_hid( self.hidden(input) )
        outv = self.nonlinearity_out(self.output(hid) )
        
        return outv
    

    def load_parameters( self, params ):
        state_dict = self.state_dict()
        for key in params.keys():
            state_dict[key] = torch.Tensor(params[key]).to(dtype=self.dtype, device=self.device) 
        
        self.load_state_dict(state_dict, strict=True)

    def save_parameters(self):
        state_dict = self.state_dict()
        dic = {}
        for key in state_dict.items():
            dic.update({key: state_dict[key].cpu().detach().numpy().copy()})

        return dic



########################

def noNonlin( x ):
    return x


def set_nonlinearity(nonlinearity):
        if nonlinearity=='tanh':
            return nn.functional.tanh
        elif nonlinearity=='relu':
            return nn.functional.relu
        elif nonlinearity=='logsigmoid': 
            return nn.functional.logsigmoid
        elif nonlinearity is None:
            return noNonlin

