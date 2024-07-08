#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Container to do basic simulations :
Methods to simulate an RNN with linear readout plus a feedback that depends on your model selected
The model is trained to produce planar hand position trajectories.
Targets are perturbed at a random delay to force the network to use feedback input
We use pytorch and the Adam optimizer to train the model.

The model returns some statistics about the trained model.

#Removing the other models for now


Harsha Gurnani, 2023
Revised Mar 2024
"""

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tools.toolbox as tb
from tools.init_params import *

import plotting.plot_helpers as phelp
from matplotlib import pyplot as pp

import mfiles.model_network as mnet
import mfiles.model_network_position as mpos
import mfiles.model_lowrank as mlow
import mfiles.model_controllers as mcon
import mfiles.training as mtrain
import mfiles.training_lowrank as mtrain2

import scipy.linalg as ll



# ----------------------------------------------- #
#    BASIC NETWORK OR EXPERIMENT CLASS            #
###################################################
''' construct, train and test a model '''

class mainModel:
    def __init__(self, model_path, init_p ):
        '''Initialize object for creating, training, perturbing models'''
        self.model_path = model_path
        if init_p is None:
            init_p = get_params()
            init_p = get_ml_params( init_p )
        else:
            init_p = get_params(init_params=init_p)
            init_p = get_ml_params( init_params=init_p )
        
        self.params = init_p
        self.params0 = {}       # pre-training parameters
        self.params1 = {}       # post-training parameters
        self.model = None
        self.criterion = None
        self.optimizer = None


    #################    CLASS METHODS

    def construct_network(self):
        torch.manual_seed(self.params['rand_seed'])
        np.random.seed(self.params['rand_seed'])
        dtype = self.params['dtype']
        device = torch.device(self.params['device'])  

        # -------- instantiate network

        # create input modules:
                
        fbk_mod = get_fbk_model( self.params, module='fb' )
        ff_mod = get_fbk_model( self.params , module='ff' )

        if self.params['use_sigma']:
            if self.params['model_type'] == 'Target':
                # ----- Feedback is in the form of vector to target
                if self.params['decode_p']:
                    self.model = mpos.RNN_w_TargetError_Pos(n_inputs=self.params['input_dim'], n_outputs=self.params['output_dim'], n_neurons=self.params['n1'], tau=self.params['tau'], dtype=self.params['dtype'], device=self.params['device'],
                                        nonlinearity=self.params['nonlinearity'], add_bias=self.params['add_bias'],bias_0=self.params['bias_0'], add_bias_n=self.params['add_bias_n'], bias_n=self.params['bias_n'], sigma_n=self.params['sigma_n'],
                                        W_in_0 = self.params['W_in_0'], W_rec_0=self.params['W_rec_0'], W_fbk_0 = self.params['W_fbk_0'], W_out_0=self.params['W_out_0'], 
                                        W_in_mask=self.params['W_in_mask'], W_fbk_mask=self.params['W_fbk_mask'], 
                                        learn_inp=self.params['train_inp'], learn_rec=self.params['train_rec'], learn_fbk=self.params['train_fbk'],
                                        fbk_state=self.params['fbk_state'], ff_mod=ff_mod, fb_mod=fbk_mod,
                                        sig_inp_0=self.params['sig_inp_0'], sig_fbk_0=self.params['sig_fbk_0'], sig_rec_0=self.params['sig_rec_0'], sig_out_0=self.params['sig_out_0'] )                    
                else:
                    self.model = mnet.RNN_w_TargetError(n_inputs=self.params['input_dim'], n_outputs=self.params['output_dim'], n_neurons=self.params['n1'], tau=self.params['tau'], dtype=self.params['dtype'], device=self.params['device'],
                                        nonlinearity=self.params['nonlinearity'], add_bias=self.params['add_bias'],bias_0=self.params['bias_0'], add_bias_n=self.params['add_bias_n'], bias_n=self.params['bias_n'], sigma_n=self.params['sigma_n'],
                                        W_in_0 = self.params['W_in_0'], W_rec_0=self.params['W_rec_0'], W_fbk_0 = self.params['W_fbk_0'], W_out_0=self.params['W_out_0'], 
                                        W_in_mask=self.params['W_in_mask'], W_fbk_mask=self.params['W_fbk_mask'], 
                                        learn_inp=self.params['train_inp'], learn_rec=self.params['train_rec'], learn_fbk=self.params['train_fbk'],
                                        fbk_state=self.params['fbk_state'], ff_mod=ff_mod, fb_mod=fbk_mod,
                                        sig_inp_0=self.params['sig_inp_0'], sig_fbk_0=self.params['sig_fbk_0'], sig_rec_0=self.params['sig_rec_0'], sig_out_0=self.params['sig_out_0'] )
            elif self.params['model_type'] == 'LowRank':
                self.model = mlow.RNN_LowRank(n_inputs=self.params['input_dim'], n_outputs=self.params['output_dim'], n_neurons=self.params['n1'], tau=self.params['tau'], dtype=self.params['dtype'], rank = self.params['rank'], device=self.params['device'],
                                        nonlinearity=self.params['nonlinearity'], add_bias=self.params['add_bias'],bias_0=self.params['bias_0'], add_bias_n=self.params['add_bias_n'], bias_n=self.params['bias_n'], sigma_n=self.params['sigma_n'],
                                        W_in_0 = self.params['W_in_0'], W_rec_0=self.params['W_rec_0'], W_fbk_0 = self.params['W_fbk_0'], W_out_0=self.params['W_out_0'], 
                                        W_in_mask=self.params['W_in_mask'], W_fbk_mask=self.params['W_fbk_mask'], 
                                        learn_inp=self.params['train_inp'], learn_rec=self.params['train_rec'], learn_fbk=self.params['train_fbk'],
                                        fbk_state=self.params['fbk_state'], ff_mod=ff_mod, fb_mod=fbk_mod,
                                        sig_inp_0=self.params['sig_inp_0'], sig_fbk_0=self.params['sig_fbk_0'], sig_rec_0=self.params['sig_rec_0'], sig_out_0=self.params['sig_out_0'] )
        else:
            if self.params['model_type'] == 'Target':
                if self.params['decode_p']:
                    self.model = mpos.RNN_w_TargetError_Pos(n_inputs=self.params['input_dim'], n_outputs=self.params['output_dim'], n_neurons=self.params['n1'], tau=self.params['tau'], dtype=self.params['dtype'], device=self.params['device'], 
                                        nonlinearity=self.params['nonlinearity'], add_bias=self.params['add_bias'],bias_0=self.params['bias_0'], add_bias_n=self.params['add_bias_n'], bias_n=self.params['bias_n'], sigma_n=self.params['sigma_n'],
                                        W_in_0 = self.params['W_in_0'], W_rec_0=self.params['W_rec_0'], W_fbk_0 = self.params['W_fbk_0'], W_out_0=self.params['W_out_0'], \
                                        W_in_mask=self.params['W_in_mask'], W_fbk_mask=self.params['W_fbk_mask'], 
                                        learn_inp=self.params['train_inp'], learn_rec=self.params['train_rec'], learn_fbk=self.params['train_fbk'],
                                        fbk_state=self.params['fbk_state'], ff_mod=ff_mod, fb_mod=fbk_mod )                    
                else:
                    self.model = mnet.RNN_w_TargetError(n_inputs=self.params['input_dim'], n_outputs=self.params['output_dim'], n_neurons=self.params['n1'], tau=self.params['tau'], dtype=self.params['dtype'], device=self.params['device'], 
                                        nonlinearity=self.params['nonlinearity'], add_bias=self.params['add_bias'],bias_0=self.params['bias_0'], add_bias_n=self.params['add_bias_n'], bias_n=self.params['bias_n'], sigma_n=self.params['sigma_n'],
                                        W_in_0 = self.params['W_in_0'], W_rec_0=self.params['W_rec_0'], W_fbk_0 = self.params['W_fbk_0'], W_out_0=self.params['W_out_0'], \
                                        W_in_mask=self.params['W_in_mask'], W_fbk_mask=self.params['W_fbk_mask'], 
                                        learn_inp=self.params['train_inp'], learn_rec=self.params['train_rec'], learn_fbk=self.params['train_fbk'],
                                        fbk_state=self.params['fbk_state'], ff_mod=ff_mod, fb_mod=fbk_mod )
            elif self.params['model_type'] == 'LowRank':
                self.model = mlow.RNN_LowRank(n_inputs=self.params['input_dim'], n_outputs=self.params['output_dim'], n_neurons=self.params['n1'], tau=self.params['tau'], dtype=self.params['dtype'], rank=self.params['rank'], device=self.params['device'], 
                                        nonlinearity=self.params['nonlinearity'], add_bias=self.params['add_bias'],bias_0=self.params['bias_0'], add_bias_n=self.params['add_bias_n'], bias_n=self.params['bias_n'], sigma_n=self.params['sigma_n'],
                                        W_in_0 = self.params['W_in_0'], W_rec_0=self.params['W_rec_0'], W_fbk_0 = self.params['W_fbk_0'], W_out_0=self.params['W_out_0'], \
                                        W_in_mask=self.params['W_in_mask'], W_fbk_mask=self.params['W_fbk_mask'], 
                                        learn_inp=self.params['train_inp'], learn_rec=self.params['train_rec'], learn_fbk=self.params['train_fbk'],
                                        fbk_state=self.params['fbk_state'], ff_mod=ff_mod, fb_mod=fbk_mod )
                    
        # update pre-train parameters
        self.params0 = self.model.save_parameters()
        # Put model on device:
        self.model.to(device=device)
    

    
    ##################
    
    def test_model(self, dic=None, tdata=None, noisex=None):
        '''Test output of current model'''
        p = self.params
        if tdata is None:
            tdata = gen_test_data( self.params )
        
        # put inputs on device:    
        stim = torch.Tensor(tdata['stimulus'].transpose(1,0,2)).to(dtype=p['dtype'], device=self.params['device'])
        target = torch.Tensor(tdata['target_output'].transpose(1,0,2)).to(dtype=p['dtype'], device=self.params['device'])       # this is only for training rnn toproduce certain trajectory - not input to model
        hold = torch.Tensor(tdata['hold_target'].transpose(1,0,2)).to(dtype=p['dtype'], device=self.params['device'])
        test_bump = torch.Tensor(tdata['test_bump'].transpose(1,0,2)).to(dtype=p['dtype'], device=self.params['device'])
        if noisex is not None:
            noise_rnn = torch.Tensor( noisex.transpose(1,0,2)).to(dtype=p['dtype'], device=self.params['device'])
        else:
            noise_rnn=None

        if p['model_type']=='Target':
            testout,testl1 = self.model(p['dt'], stim, hold, test_bump, noisex=noise_rnn)
        elif p['model_type']=='LowRank':
            testout,testl1 = self.model(p['dt'], stim, hold, test_bump)
        
        output = testout.cpu().detach().numpy().transpose(1,0,2)
        activity1 = testl1.cpu().detach().numpy().transpose(1,0,2)

        res = {'target':tdata['target_output'], 'stimulus':tdata['stimulus'], 'delays': tdata['del_used'], \
                'test_bump':tdata['test_bump'], 'end_target':tdata['hold_target'], 'noisex':noisex,
                'activity1':activity1,'output':output, 'sim_params':p}  
        
        if dic is not None:
            dic.update({'res': res})
        else:
            dic = {'res':res}
        
        return dic
    


    ##################

    def train_model( self, training_data=None, update=False, **kwargs):
        p = self.params
        for key, val in kwargs.items():
            p.update({key:val})

        if training_data is None:
            training_data = gen_training_data( params=p )

        # should already be tensors
        for inp in {'stim', 'target', 'hold', 'bump'}:          #training_data.keys():
            training_data[inp] = training_data[inp].to(dtype=self.params['dtype'], device=self.params['device'])        

        # define loss function
        self.criterion = self.set_criterion()               # will add params later

        # create optimizer    
        self.optimizer = self.set_optim()                   # will add params later

        # save pre-training model params
        #params0 = self.model.save_parameters()
        self.params0 = self.model.save_parameters()
        if p['loss_mask'] is not None:
            p['loss_mask'] = torch.Tensor(p['loss_mask']).to(dtype=self.params['dtype'], device=self.params['device'])

        # TRAIN
        if p['model_type'] == 'Target':
            print('model target')
            lc = mtrain.train(modelt=self.model, optimizert=self.optimizer, criteriont=self.criterion, tt=p['training_epochs'], 
                    trajt=training_data['target'], stimt=training_data['stim'], targett=training_data['hold'], perturb=training_data['bump'],    # should be on same device as model
                    alp0=p['alpha1'], bet0=p['beta1'], gam0=p['gamma1'], alp1=p['alpha2'], bet1=p['beta2'], gam1=p['gamma2'], 
                    clipv=p['clipgrad'], dt=p['dt'],  dist_thresh=p['dist_thresh'], hit_period=p['hit_period'],
                    loss_mask=p['loss_mask'], loss_t=p['loss_t'], loss_maxt=p['loss_maxt'], print_t=p['print_t'] )
        
        elif p['model_type'] == 'LowRank':
            print('model target')
            lc = mtrain2.train_LR(modelt=self.model, optimizert=self.optimizer, criteriont=self.criterion, tt=p['training_epochs'], 
                    trajt=training_data['target'], stimt=training_data['stim'], targett=training_data['hold'], perturb=training_data['bump'],    # should be on same device as model
                    alp0=p['alpha1'], bet0=p['beta1'], gam0=p['gamma1'], alp1=p['alpha2'], bet1=p['beta2'], gam1=p['gamma2'], 
                    clipv=p['clipgrad'], dt=p['dt'],  dist_thresh=p['dist_thresh'], hit_period=p['hit_period'],
                    loss_mask=p['loss_mask'], loss_t=p['loss_t'], loss_maxt=p['loss_maxt'], print_t=p['print_t'] )


        # save post-training model params
        self.params1 = self.model.save_parameters()
        dic = {'params0': self.params0, 'params1':self.params1, 'simparams':p, 'lc':np.array(lc) }

        return dic
    
    def train_model_WP( self, training_data=None, update=False, **kwargs):
        ''' Train via weight perturbation '''
        p = self.params
        for key, val in kwargs.items():
            p.update({key:val})

        if training_data is None:
            training_data = gen_training_data( params=p )

        # should already be tensors
        for inp in {'stim', 'target', 'hold', 'bump'}:          #training_data.keys():
            training_data[inp] = training_data[inp].to(dtype=self.params['dtype'], device=self.params['device'])        

        # define loss function
        self.criterion = self.set_criterion()               # will add params later


        # save pre-training model params
        #params0 = self.model.save_parameters()
        self.params0 = self.model.save_parameters()
        if p['loss_mask'] is not None:
            p['loss_mask'] = torch.Tensor(p['loss_mask']).to(dtype=self.params['dtype'], device=self.params['device'])

        # TRAIN

        if p['model_type'] == 'Target':
            print('model target')
            lc = mtrain.train_WP(modelt=self.model, lossfn = mtrain.model_loss, criteriont=self.criterion, tt=p['training_epochs'], 
                    trajt=training_data['target'], stimt=training_data['stim'], targett=training_data['hold'], perturb=training_data['bump'],    # should be on same device as model
                    eta=p['wp.eta'],  out_sigma=p['wp.out_sigma'], hid_sigma=p['wp.hid_sigma'], inp_sigma=p['wp.inp_sigma'], fbk_sigma=p['wp.fbk_sigma'], #rec_sigma=1e-10,
                    train_out=p['fb']['train_out'], train_hid=p['fb']['train_hid'], train_inp=p['train_inp'], train_fbk=p['train_fbk'], #train_rec=p['train_rec'],
                    alp0=p['alpha1'], bet0=p['beta1'], gam0=p['gamma1'], alp1=p['alpha2'], bet1=p['beta2'], gam1=p['gamma2'], 
                    clipv=p['clipgrad'], dt=p['dt'],  dist_thresh=p['dist_thresh'], hit_period=p['hit_period'],
                    loss_mask=p['loss_mask'], loss_t=p['loss_t'], loss_maxt=p['loss_maxt'], print_t=p['print_t'] )

        # save post-training model params
        self.params1 = self.model.save_parameters()
        dic = {'params0': self.params0, 'params1':self.params1, 'simparams':p, 'lc':np.array(lc) }

        return dic
    

    ##################
    # create optimizer    --- allow flexibility later
    def set_optim( self ):
        p = self.params
        return optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=p['lr'])
    
    ##################
    # define loss function   --- allow flexibility later
    def set_criterion( self ):
        p = self.params
        return nn.MSELoss(reduction='none')


    ##################
    def save_results( self, dic, savname=None ):
        if savname is None:
            savname = self.model_path
        np.save(savname + self.params['suffix1'],     dic)



    def plot_model_results( self, dic, savname=None, plot_idx=None ):
        if savname is None:
            savname = self.model_path

        res=dic['res']    
        ntrials, tsteps, n1 = res['output'].shape

        if plot_idx is None:
            plot_idx = np.random.choice( range(ntrials), 20, replace=False )        # subset of plotted trials

        # Plot readout and error colored by go-cue delay
        if self.params['plot_readout']:
            pname = 'test_readout'+self.params['suffix1']+'.png'
            phelp.plot_readout_trace( output=res['output'], stimulus=res['stimulus'], target=res['target'], savfolder=savname, savname=pname, plot_idx=plot_idx )

        # Plot 2D trajectory colored by reach direction
        if self.params['plot_2D']:
            pname = 'test_trajectory'+self.params['suffix1']+'.png'
            phelp.plot_2D_traj( output=res['output'], stimulus=res['stimulus'], savfolder=savname, savname=pname, plot_idx=plot_idx )



    ##################



# ----------------------------------------------- #
#                    HELPERS                      #
###################################################



# -------------- setup sub - modules

def get_fbk_model( params, module='fb' ):
    '''
    Return one of the controller modules,
    module specifies which params to read ('ff' or 'fb')
    '''

    # overwrite relevant dim
    if module=='ff':
        params[module]['n_input'] = params['input_dim']
    elif module=='fb':
        params[module]['n_input'] = params['output_dim']
        params[module]['n_neu'] = params['n1']

    if params[module]['type'] == 'none':
        return None                     # no additional module
    
    elif params[module]['type'] == ' linear':
        mod = mcon.Input_Linear_NonLinear( n_inp=params[module]['n_input'], n_out=params[module]['n_output'], weight_out=params[module]['weight_out'],
                                        add_bias=params[module]['add_bias'], bias=params[module]['bias'], dtype=params['dtype'], device=params['device'],
                                        train=params[module]['train_out'], nonlinearity=params[module]['nonlin_out'])  #, has_hidden=params[module]['has_hidden']
        mod.to(params['device'])

    elif params[module]['type'] == 'linear_2Inp':
        mod = mcon.Input_Linear_NonLinear_2Inp( n_inp=params[module]['n_input'], n_out=params[module]['n_output'], n_neu = params['n1'], n_mf = params[module]['n_mf'], 
                                                sample_type=params[module]['sample_type'], sample_neu=params[module]['sample_neu'], train_neu=params[module]['train_neu'], train=params[module]['train_out'], 
                                                weight_out=params[module]['weight_out'], add_bias=params[module]['add_bias'], bias=params[module]['bias'], dtype=params['dtype'], device=params['device'],
                                                nonlinearity=params[module]['nonlin_out']) #, has_hidden=params[module]['has_hidden']
        mod.to(params['device'])

    
    elif params[module]['type'] == 'Perceptron':
        mod = mcon.Perceptron( n_inp=params[module]['n_input'], n_hid=params[module]['n_hidden'], n_out=params[module]['n_output'], 
                                weight_hid=params[module]['weight_hid'], weight_out=params[module]['weight_out'],
                                add_bias=params[module]['add_bias'], bias=params[module]['bias'], dtype=params['dtype'], device=params['device'],
                                train_hid=params[module]['train_hid'], nonlinearity_hid=params[module]['nonlin_hid'],
                                train_out=params[module]['train_out'], nonlinearity_out=params[module]['nonlin_out'])  #, has_hidden=params[module]['has_hidden']
        mod.to(params['device'])

    elif params[module]['type'] == 'Perceptron_2Inp':
        mod = mcon.Perceptron_2Inp( n_inp=params[module]['n_input'], n_hid=params[module]['n_hidden'], n_out=params[module]['n_output'], 
                                weight_hid=params[module]['weight_hid'], weight_out=params[module]['weight_out'],
                                add_bias=params[module]['add_bias'], bias=params[module]['bias'], dtype=params['dtype'], device=params['device'],
                                train_hid=params[module]['train_hid'], nonlinearity_hid=params[module]['nonlin_hid'],
                                train_out=params[module]['train_out'], nonlinearity_out=params[module]['nonlin_out'],
                                n_neu = params['n1'], n_mf = params[module]['n_mf'], train_neu=params[module]['train_neu'],
                                sample_type=params[module]['sample_type'], sample_neu=params[module]['sample_neu'])  #, has_hidden=params[module]['has_hidden'] 
        mod.to(params['device'])
    
    return mod








def gen_training_data( params ):
    '''
    Generate training data for a 2D target reaching task
    '''
    p=params
    st, to, du, et = tb.gen_data_2( p['ntrials'], dt=p['dt'], maxT=p['maxT'], \
                                    fixedDelay=False, delays=p['delays'], vel=p['velocity'])

    # convert stimulus and target to pytorch form
    stim = torch.zeros(p['training_epochs'], p['tsteps'], p['batch_size'], p['input_dim']).type(p['dtype'])
    target = torch.zeros(p['training_epochs'], p['tsteps'], p['batch_size'], p['output_dim']).type(p['dtype'])
    hold = torch.zeros(p['training_epochs'], p['tsteps'], p['batch_size'], p['output_dim']).type(p['dtype'])
    delays = torch.zeros( (p['training_epochs'], p['batch_size']) )


    for j in range(p['training_epochs']):
        idx = np.random.choice(range(p['ntrials']),p['batch_size'],replace=False)
        stimtmp = np.zeros((p['batch_size'],p['tsteps'],p['input_dim']))
        stimtmp = st[idx,:,:]
        stim[j] = torch.Tensor(stimtmp.transpose(1,0,2)).type(p['dtype'])
        target[j] = torch.Tensor(to[idx].transpose(1,0,2)).type(p['dtype'])
        hold[j] = torch.Tensor(et[idx].transpose(1,0,2)).type(p['dtype'])
        delays[j,:] = torch.Tensor(du[idx]).type(p['dtype'])


    # Generate perturbation data
    add_vel_bump = torch.zeros(p['training_epochs'],p['tsteps'],p['batch_size'],p['output_dim']).type(p['dtype'])
    id1=p['training_epochs']-p['perturb_epochs']
    if p['train_perturb']:
        for j in range(p['perturb_epochs']):
            for k in range(p['batch_size']):
                tmp = tb.perturb_bump(dt=p['dt'],T=p['maxT'],bump_delay=np.random.randint(low=50,high=p['maxT']), amp=p['bump_amp'])
                add_vel_bump[id1+j,:,k,:]=torch.Tensor(tmp).type(p['dtype'])

    tdata = {'stim': stim, 'target':target, 'hold':hold, 'bump':add_vel_bump, 'delays':delays }

    return tdata




def gen_test_data( params ):
    '''
    Generate input data for a 2D reaching task
    '''
    p = params
    tsteps = np.int_(p['test_maxT']/ p['dt'])
    if p['test_type']=='discrete':
        n_jump = len(p['jump_del'])
        testing_trials= p['testing_perturb']*(n_jump+1)
        

        tstim, tout, tdel, thold = tb.gen_data_discrete( p['testing_perturb'], useTheta=p['test_theta'], \
                            dt=p['dt'], maxT=p['test_maxT'], fixedDelay=False, delays=p['delays'], add_hold=True , vel=p['velocity'])

        stimulus = np.zeros([testing_trials,tsteps,p['input_dim']])
        target_output = np.zeros([testing_trials,tsteps,p['output_dim']])
        del_used = np.zeros(testing_trials)
        new_bump = np.zeros([testing_trials, tsteps, 2])
        hold_target = np.zeros([testing_trials,tsteps,p['output_dim']])

        for pid in range(n_jump+1):
            tr = (pid)*p['testing_perturb']
            tre= tr+p['testing_perturb']
            stimulus[tr:tre,:,:] = tstim
            target_output[tr:tre,:,:] = tout
            hold_target[tr:tre,:,:] = thold
            del_used[tr:tre] = tdel
            if pid>0:
                jid = np.int_(p['jump_del'][pid-1]/p['dt'])
                for jj in range(p['testing_perturb']):
                    trial=tr+jj
                    dirn = np.random.choice([0,1])
                    new_bump[trial,jid,dirn] = np.random.choice( [-p['jump_amp'], p['jump_amp']]) # single large perturbation 

    elif p['test_type']=='random':
        testing_trials = p['testing_trials']
        stimulus, target_output, del_used, hold_target = tb.gen_data_2( testing_trials, dt=p['dt'], maxT=p['test_maxT'], \
                                                    fixedDelay=False, delays=p['delays'], vel=p['velocity'] )

        new_bump = np.zeros(shape=(testing_trials,tsteps,2))
        for jj in range(p['testing_perturb']):
            trial=testing_trials-p['testing_perturb']+jj
            new_bump[trial,:,:] = torch.Tensor(tb.perturb_bump(dt=p['dt'],T=p['test_maxT'],\
                                    bump_delay=np.random.randint(low=50,high=p['test_maxT']-100), amp=p['bump_amp'])).type(p['dtype'])            
    
    elif p['test_type'] == 'no_perturb':
        print('no perturbation')
        testing_trials = p['testing_trials']
        stimulus, target_output, del_used, hold_target = tb.gen_data_2( testing_trials, dt=p['dt'], maxT=p['test_maxT'], \
                                                    fixedDelay=False, delays=p['delays'] , vel=p['velocity'])
        new_bump = np.zeros(shape=(testing_trials,tsteps,2))






    tdata = {'stimulus': stimulus, 'target_output': target_output, 'hold_target':hold_target, 'test_bump':new_bump, 'del_used':del_used}
    return tdata





# ----------------------------------------------- #
###################################################






