#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


"""
More methods to do perturbations and load pre-existing models


Harsha Gurnani, 2024
"""



import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import tools.toolbox as tb
from tools.init_params import *
from tools.run_network_ import *
import tools.analysis_helpers as ah
import tools.wout_perturbations as wpert

import plotting.plot_helpers as phelp
from matplotlib import pyplot as pp

'''
import mfiles.model_network as mnet
import mfiles.model_lowrank as ml
import mfiles.model_controllers as mcon
import mfiles.training as mtrain
'''

import scipy.linalg as ll



# ----------------------------------------------- #
#    EXTENDED CLASS FOR MODEL MANIPULATIONS       #
###################################################


class postModel( mainModel ):
    def __init__(self, model_path, adapt_p ):
        super().__init__( model_path, adapt_p )
        #self.adapt_params = self.params.copy()      # further manipulations to this
        postModel.intuitive = {}                     # fit an intuitive decoder - won't be automatically filled - useful keys: 'wout', 'PCRes', 'LinModel', 'nPC', 'params'


    def load_rnn(self, ch_params=None ):
        '''Create RNN module based on trained model'''
        a = np.load( self.model_path, allow_pickle=True).item()
        self.simparams = a['simparams']                 # save params during model construction
        self.trained_model = a['params1'].copy()        # save a copy of pretrained network - DONT TOUCH!
        self.model_params = a['params1']                # (trainable/fixed) Parameters of the torch modules in the network (see load/save methods)

        # reload specific structural parameters (not training parameters)
        p = self.simparams
        update_params = {'rand_seed', 'model_type', 'n1', 'tau', 'input_dim', 'output_dim', 'velocity', 'fb', 'ff', 
                        'nonlinearity', 'nonlinearity2',  'add_bias', 'bias_0', 'add_bias_n', 'bias_n',
                        'sample_rnn',  'input_transform', 'fbk_state', 'decode_p', 
                        'use_sigma', 'sig_inp_0', 'sig_fbk_0', 'sig_rec_0', 'sig_out_0',
                            'W_in_0', 'W_rec_0', 'W_fbk_0', 'W_out_0' }
        for key in update_params:
            self.params.update( {key: self.simparams[key] })
        if ch_params is not None:
            for key in ch_params:
                self.params.update( {key: ch_params[key] })
        if 'sigma_n' in self.simparams.keys():
            self.params.update( {'sigma_n': self.simparams['sigma_n'] })
        if 'rank' in self.simparams.keys():
            self.params.update( {'rank': self.simparams['rank'] })

        # instantiate network
        self.construct_network()
        self.model.load_parameters( self.model_params )     # loads parameters from trained model (also has keys for submodule)

    def restore_model( self ):
        '''Reload original trained model'''
        self.model_params = self.trained_model.copy()
        self.model.load_parameters( self.model_params )

    def reset_training_params( self, params=None ):
        if params is None:
            params=self.params
        self.model.W_in.requires_grad = params['train_inp']
        self.model.W_fbk.requires_grad = params['train_fbk']
        if params['model_type']=='Target':
            self.model.W_rec.requires_grad = params['train_rec']
        elif params['model_type']=='LowRank':
            self.model.m_vectors.requires_grad = params['train_rec']
            self.model.n_vectors.requires_grad = params['train_rec']
            
        
        # set for fbk module: (not for ff yet)
        if self.model.fb_type != 'none':
            if ('linear' in self.model.fb_mod.inp_type) or ('Perceptron' in self.model.fb_mod.inp_type):
                self.model.fb_mod.train_out = params['train_out']
                self.model.fb_mod.output.weight.requires_grad = params['train_out']
                print('setting train out ... ')
                if self.model.fb_mod.add_bias_out:
                    self.model.fb_mod.output.bias.requires_grad = params['train_out']
            
                if self.model.fb_mod.has_hidden:
                    self.model.fb_mod.train_hid = params['train_hid']
                    self.model.fb_mod.hidden.weight.requires_grad = params['train_hid']
                    if self.model.fb_mod.add_bias_hid:
                        self.model.fb_mod.hidden.bias.requires_grad = params['train_hid']

                if '2Inp' in self.model.fb_mod.inp_type:
                    self.model.fb_mod.train_neu = params['train_neu']
                    if self.model.fb_mod.sample_type is not None:
                        self.model.fb_mod.sample_neu.weight.requires_grad = params['train_neu']


    def perturb_wmp( self ):
        return None
    
    def perturb_omp( self ):
        return None
    

    def get_intuitive_map( self, nPC=10, ch_params={} ):
        ''' fit a new readout weight matrix by fitting a linear regression between observed velocity
        and activity projected in top 'nC' principal components
        - currently only supports linear readouts without bias (as intercept of fit is not separated between neuron mean and readout bias)
        '''

        p = get_perturb_params(self.params)
        for key,val in ch_params.items():
            p['intuitive.'+key] = val
        
        # generate data:
        p['jump_amp'] = 1e-3
        p['bump_amp'] = 1e-3
        tdata = gen_test_data( params=p )
        dic = self.test_model( tdata=tdata )

        reg, px_1 = ah.intuitive_decoder(  dic['res'], nPC=nPC, dt=p['dt'], scaleCoef=p['intuitive.scaleCoef'], nTest=p['intuitive.nTest'], noise_x=p['intuitive.noiseX'],
                                        scaleX=p['intuitive.scaleX'], fitPC=p['intuitive.fitPC'], add_bias=p['intuitive.add_bias'] , use_tm=p['intuitive.use_tm'] )
        
        w_fit = reg.coef_
        U = px_1.components_

        U_full = w_fit @ U
        wout = U_full.T

        return reg, px_1 , wout
    


    def generate_wmps( self, intuitive=None, wout=None, LinModel=None, PCRes=None, nPerms=3, ch_params={} ):

        # check if using pre-saved map
        if intuitive is None:
            intuitive = self.intuitive      
        if LinModel is None:
            LinModel = intuitive['LinModel']
        if PCRes is None:
            PCRes = intuitive['PCRes']
        if wout is None:
            wout = intuitive['wout']
        
        p = get_perturb_params( self.params )
        p['wmp.nPerms'] = nPerms
        for key,val in ch_params.items():
            p['wmp.'+key] = val

        p['wmp.nDim'] = PCRes.n_components_
        p['wmp.nList_p'] = min(p['wmp.nList_p'], math.factorial(p['wmp.nDim']))

        list_perturb, ranks = wpert.get_random_permutations( p['wmp.nDim'], p['wmp.nList_p'] )

        check_a = wpert.select_wmp_wout( p['wmp.nDim'], LinModel.coef_, PCRes.components_, wout, theta_range=p['wmp.theta_wout'], p=list_perturb, maps=ranks  )

        use_maps = ranks[check_a]       # only those within angle range
        cid = min( len(use_maps), p['wmp.nList_sub'] )

        # generate data:
        p['jump_amp'] = 1e-3
        p['bump_amp'] = 1e-3
        tdata = gen_test_data( params=p )
        dic = self.test_model( tdata=tdata )
        X = dic['res']['activity1']
        stim = dic['res']['stimulus']
        speed_check, angle_check_mean, angle_check_med, ratio = wpert.select_wmp_openloopv( X=X, wout0=wout, maps=use_maps[:cid], lin_coef=LinModel.coef_, U=PCRes.components_, use_p=True, p=list_perturb, ratio_lim=p['wmp.ratio_lim'], stim=stim, limA=[20,80]  )


        return check_a, speed_check, angle_check_mean, angle_check_med, list_perturb, use_maps, ratio
    

    def run_wmp( self, intuitive=None, LinModel=None, PCRes=None, map=None, list_perm=None, ch_params={} ):
        
        if intuitive is None:
            intuitive = self.intuitive
        if LinModel is None:
            LinModel = intuitive['LinModel']
        if PCRes is None:
            PCRes = intuitive['PCRes']

        p = get_perturb_params( self.params )
        p['wmp.nDim'] = PCRes.n_components_
        for key,val in ch_params.items():
            p['wmp.'+key] = val
        

        if map is None:
            map = 1
        if list_perm is None:
            list_perm, _ = wpert.get_random_permutations( p['wmp.nDim'],None)

        idx = wpert.get_perm_order( list_perm, map )
        w_new = LinModel.coef_[:,idx]
        Wout_full = w_new @ PCRes.components_


        wout = Wout_full.T
        self.restore_model()
        newp = self.model.save_parameters()
        newp['W_out_0'] = wout
        self.model.load_parameters(newp)
        dic = self.test_model()

        return dic, wout
    


    def post_filter_wmp( self, intuitive=None, LinModel=None, PCRes=None, use_maps=None, list_perm=None, ch_params={}, plot_num=10, savpath=None, get_perf=False, maptype='wmp', neuron_gps=None, use_vratio=False ):
        if intuitive is None:
            intuitive = self.intuitive
        if LinModel is None:
            LinModel = intuitive['LinModel']
        if PCRes is None:
            PCRes = intuitive['PCRes']
        if (neuron_gps is None) and (maptype=='omp') :
            neuron_gps = self.omp['neuron_groups']
        wout = intuitive['wout']

        nDim=PCRes.n_components_
        lin_coef=LinModel.coef_
        U=PCRes.components_

        if savpath is None:
            savpath=self.model_path

        if maptype=='rmp':
            mpnm = 'wmp'
        else:
            mpnm = maptype
        p = get_perturb_params( self.params )
        p[mpnm+'.nDim'] = nDim
        for key,val in ch_params.items():
            p[mpnm+'.'+key] = val

        if use_maps is None:
            use_maps = 1
        if list_perm is None:
            list_perm, _ = wpert.get_random_permutations( p[mpnm+'.nDim'],None)

        nmaps = len(use_maps)
        nplot = min(plot_num, nmaps)
        #plot_idx = np.random.choice(nmaps,nplot,replace=False)
        plot_done=0
        all_v = np.zeros(nmaps)
        if get_perf:
            perfrate = np.zeros((nmaps,4))
        for map in range(nmaps):
            self.restore_model()
            newp = self.model.save_parameters()
            #newp['W_out_0'] = wout
            #self.model.load_parameters(newp)
            dic = self.test_model()
            meanv_ori = wpert.mean_speeds(  X=dic['res']['output'], stim=dic['res']['stimulus'] )


            
            if maptype=='wmp':
                p_order = wpert.get_perm_order( list_perm, use_maps[map] )  #idx
                _, Wout_full, _ = wpert.get_wmp( nDim, lin_coef, U, Wout_orig=wout.T, idx=p_order )
            elif maptype=='omp':
                #print('Using OMP generator')
                p_order = wpert.get_perm_order( list_perm, use_maps[map] )  #idx
                _, Wout_full, _ = wpert.get_omp( allgps=neuron_gps, n_groups0=nDim, lin_coef=lin_coef, U=U, Wout_orig=wout.T, gp_order=p_order )
            elif maptype=='rmp':
                #print('Using OMP generator')
                Wout_full = use_maps[map].T
            #w_new = LinModel.coef_[:,idx]
            #Wout_full = w_new @ PCRes.components_
            newp['W_out_0'] = Wout_full.T
            self.model.load_parameters(newp)
            dic = self.test_model()
            #print(ll.norm(Wout_full.T-wout))
            
            # get closed loop velocity of perturbed readout
            meanv = wpert.mean_speeds(  X=dic['res']['output'], stim=dic['res']['stimulus'] )
            #meanv = 1
            all_v[map] = np.nanmedian(meanv)
            
            if use_vratio:
                all_v[map] = all_v[map]/np.nanmedian(meanv_ori)     # make it a ratio instead of absolute speed
            print(all_v[map])
            keepv = (all_v[map]>p[mpnm+'.closedloopv'][0]) and (all_v[map]<p[mpnm+'.closedloopv'][1])
            doperf=True
            if get_perf:
                perfres = ah.get_performance( dic, nTest=200, thresh=0.15)
                perfrate[map,:] = np.array([perfres['min_rdist'], perfres['success'], perfres['traj_error'], np.median(perfres['acq_time'])])
                doperf = (perfres['success']>0.2) and (perfres['success']<0.8)
            
            if keepv and (np.random.random()>0.1) and (plot_done<nplot) and (doperf) :
                print('Plotting with closedloop v= '+np.str_(all_v[map]))
                self.plot_model_results(dic, savname=savpath+maptype+'_'+np.str_(map)+'_PC8_') # HAS MODEL SUFFIX
                plot_done +=1 
            
            
        

        if get_perf:
            return use_maps, all_v, perfrate
        else:
            return use_maps, all_v
        

    def generate_omps( self, intuitive=None, wout=None, LinModel=None, PCRes=None, nPerms=3, ch_params={} ):

        # check if using pre-saved map
        if intuitive is None:
            intuitive = self.intuitive      
        if LinModel is None:
            LinModel = intuitive['LinModel']
        if PCRes is None:
            PCRes = intuitive['PCRes']
        if wout is None:
            wout = intuitive['wout']
        
        p = get_perturb_params( self.params )
        p['omp.nPerms'] = nPerms
        for key,val in ch_params.items():
            p['omp.'+key] = val

        p['omp.nDim'] = PCRes.n_components_
        p['omp.nList_p'] = min(p['omp.nList_p'], math.factorial(p['omp.nDim']))

        list_perturb, ranks = wpert.get_random_permutations( p['omp.nDim'], p['omp.nList_p'] )

        # generate data:
        p['jump_amp'] = 1e-3
        p['bump_amp'] = 1e-3
        tdata = gen_test_data( params=p )
        dic = self.test_model( tdata=tdata )
        X = dic['res']['activity1']
        stim = dic['res']['stimulus']
        
        X1 = np.reshape( X, (X.shape[0]*X.shape[1],X.shape[2]))
        stdR = np.std( X1, axis=0 )

        check_a, allgps = wpert.select_omp_wout( p['omp.nDim'], LinModel.coef_, PCRes.components_, wout, stdR, theta_range=p['omp.theta_wout'], p=list_perturb, maps=ranks  )
        self.omp = {'neuron_groups': allgps}    # need to generate OMPs later using group structure

        use_maps = ranks[check_a]       # only those within angle range
        cid = min( len(use_maps), p['omp.nList_sub'] )

        speed_check, angle_check_mean, angle_check_med, ratio = wpert.select_map_openloopv( X=X, wout0=wout, maps=use_maps[:cid], lin_coef=LinModel.coef_, U=PCRes.components_, use_p=True, p=list_perturb, ratio_lim=p['omp.ratio_lim'], stim=stim, limA=[20,80] , mapgen='omp', neuron_gps=allgps )


        return check_a, speed_check, angle_check_mean, angle_check_med, list_perturb, use_maps, ratio
    

    def generate_random_decoders( self, intuitive=None, wout=None, LinModel=None, PCRes=None, nPerms=3, ch_params={} ):

        # check if using pre-saved map
        if intuitive is None:
            intuitive = self.intuitive
        if PCRes is None:
            PCRes = intuitive['PCRes']
        if LinModel is None:
            LinModel = intuitive['LinModel']
        if wout is None:
            wout = intuitive['wout']
        
        p = get_perturb_params( self.params )
        p['wmp.nPerms'] = nPerms
        for key,val in ch_params.items():
            p['wmp.'+key] = val

        p['wmp.nDim'] = PCRes.n_components_
        #p['wmp.nList_p'] = min(p['wmp.nList_p'], math.factorial(p['wmp.nDim']))

        list_perturb, ranks = wpert.get_random_permutations( p['wmp.nDim'], min(p['wmp.nList_p'], math.factorial(p['wmp.nDim'])) )
        print(p['wmp.nList_p'])

        # generate data:
        p['jump_amp'] = 1e-3
        p['bump_amp'] = 1e-3
        tdata = gen_test_data( params=p )
        dic = self.test_model( tdata=tdata )
        X = dic['res']['activity1']
        stim = dic['res']['stimulus']

        check_a, new_maps = wpert.select_rmp_wout( p['wmp.nDim'], LinModel.coef_, PCRes.components_, wout, theta_range=p['wmp.theta_wout'], p=list_perturb, maps=np.arange(p['wmp.nList_p'])  )

        use_maps = [new_maps[jj] for jj in range(len(check_a)) if check_a[jj]]       # only those within angle range
        cid = min( len(use_maps), p['wmp.nList_sub'] )
        use_maps = [use_maps[jj] for jj in range(cid)]

        speed_check, angle_check_mean, angle_check_med, ratio = wpert.select_map_openloopv( X=X, wout0=wout, maps=use_maps, lin_coef=LinModel.coef_, U=PCRes.components_, use_p=True, p=list_perturb, ratio_lim=p['wmp.ratio_lim'], stim=stim, limA=[20,80] , mapgen='rmp' )


        return check_a, speed_check, angle_check_mean, angle_check_med, list_perturb, use_maps, ratio
    









