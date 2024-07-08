#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Default parameters for model generation and management.
Can consider YAML.

Harsha Gurnani, 2023
'''

import numpy as np
import torch

# ----------------------------------------------- #
#           DEFAULT PARAMETERS                    #
###################################################


def get_params( init_params=None, rand_seed=None ):
    if rand_seed is None:
        rand_seed= np.random.randint(300)
    

    params={}
    params['dtype'] = torch.float32#torch.FloatTensor
    params['device'] = "cpu"            # 'mps' for mac m1 processors

    # Basic parameters:
    params['rand_seed'] = rand_seed
    params['n1'] = 256                  # neurons in RNN
    params['tau'] = 10                  # time constant of network dx/dt
    params['nonlinearity']='relu'       # rnn nonlinearity
    params['nonlinearity2'] = 'relu'    # input nonlinearity -  not really used anymore
    params['output_dim'] = 2            # cursor x,y velocity
    params['input_dim'] = 3             # target x, y + hold

    params['rank'] = 4                  # only for low-rank networks
    
    # ---- will call controller afterwards----
    params['fb'] = None                 # params for feedback module            
    params['ff'] = None                 # params for feedforward inp module
    # ----------------------------------------

    # run params (if evaluating model)
    params['bump_amp'] = 0.02   # perturbation amplitudes
    params['jump_amp'] = 0.1 
    params['dt'] = 1            # ms
    params['maxT'] = 500        # ms (simulation length)
    params['tsteps'] = np.int_(params['maxT']/params['dt'])
    params['delays'] = [100,200]  # ms, delay to go cue sampled from this range
    params['ntrials'] = 2000      # to generate data to sample for training
    params['velocity'] = 0.050    # sets timescale for target trajectory

    # Model type
    params['model_type'] = 'Target'     # or 'LowRank' #WILL UPDATE OPTIONS LATER ---- Or 'Error', 'Implicit' (error to arget, error to trajectory or pure state feedback)
    params['add_bias'] = False          # bias (intercept) for readout
    params['add_bias_n'] = False        # bias current for neurons
    params['sigma_n'] = 0.0             # activity noise sigma
    # optional:
    params['input_transform'] = False   # transform input (and feedback signals) by submodules?
    params['fbk_state'] = False         # use rnn state in feedback controller?
    params['sample_rnn'] = False        # subsample rnn for computing feedback?
    params['decode_p'] = False          # readout is position rather than velocity?

    # Weight initialisations
    params['use_sigma'] = False         # set scale of network weights at initialisation
    params['sig_inp_0'] = 0.3           
    params['sig_fbk_0'] = 0.3
    params['sig_rec_0'] = 1
    params['sig_out_0'] = 1             # remember, total cursor velocity depends on network activity and output weights -> scale will determine O(x)
    

    # initialised parameters:
    params['W_in_0'] = None             # can be numpy matrices
    params['W_rec_0'] = None
    params['W_fbk_0'] = None
    params['W_out_0'] = None
    params['bias_0'] = None
    params['bias_n'] = None
    params['W_in_mask'] = None
    params['W_fbk_mask'] = None


    # for a main run?
    params['train_model'] = True
    params['train_perturb'] = True
    params['test_model'] = True
    params['save_data'] =True

    # Test parameters
    params['testing_trials']=100  # total number of test trials
    params['testing_perturb']=50  # how many of them with perturbation
    params['test_maxT'] = 600 #ms
    params['test_type'] = 'discrete' # 'random'         
    params['jump_del'] = [150, 300, 500] #ms            # total no. of trials = (1+jump_del)*testing_perturb
    params['test_theta'] = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    # Plot parameters:
    params['plot_data'] = False        # plot neural activity traces
    params['plot_readout']=True        # plot network output ("velocity")
    params['plot_2D']=True             # plot trajectory
    if params['train_model']:
        params['suffix1']='_with_training_seed_'+str(rand_seed)
    else:
        params['suffix1']='_no_training_seed_'+str(rand_seed)

    #Overwrite if needed
    if init_params is not None:
        for x in init_params.keys():
            params[x] = init_params[x]

    # update children dicts:
    params['fb'] = get_controller_params( params['fb'], nneu=params['n1'] )
    params['ff'] = get_controller_params( params['ff'] )

    # update state sampling:
    if params['fb']['sample_type'] is not None:         # modify this if new modules don't need network state
        params['fbk_state'] = True

    # reset for sanity
    params['tsteps'] = np.int_(params['maxT']/params['dt'])


    return params




# --------------------------------------------#
# more params for training
# --------------------------------------------#
def get_ml_params( init_params=None ):

    params={}
    
    params['batch_size'] = 32           # for each update epoch
    params['training_epochs'] = 100     # total number 
    #  Last X trials to add perturbation to position during training - only if train_perturb is True
    params['perturb_epochs'] = 50       # should be less than training_epochs.
    params['lr'] = 2e-4                 # learning rate

    params['training_epochs_2'] = 120
    params['perturb_epochs_2'] = 50

    # Gradient computation for rnn parameters 
    params['train_inp'] = True          # train rnn's input weights (W_in_0)
    params['train_fbk'] = True          # train rnn's recurrent weights and neuronal bias (W_rec_0)
    params['train_rec'] = True          # train rnn's feedback weights (W_fbk_0)

    # Regularisation parameters
    params['alpha1'] = 3e-4 # reg inp & out
    params['alpha2'] = 3e-4 # reg ff & fb
    params['gamma1'] = 3e-4 # reg rec 1
    params['gamma2'] = 3e-4 # reg rec 2
    params['beta1'] = 0.01 # reg rate 1  # from 0.8 , or 0.01?
    params['beta2'] = 0.01 # reg rate 2  # from 0.8 , or 0.01?

    # Parameters for Weight Perturbation:
    params['wp.eta'] = 0.001#  # 0.005 for relu med ## 0.001 for slow relu# 0.001 for percp2 and relu slow ##### 0.01 for relu
    params['wp.out_sigma'] = 0.01#0.05
    params['wp.hid_sigma'] = 0.01#0.05
    params['wp.inp_sigma'] = 0.01#0.05  0.05 for all, 0.01 for small
    params['wp.fbk_sigma'] = 0.01#0.05

    # Training performance
    params['clipgrad'] = 0.2        # clip gradient norm
    params['loss_mask'] = None
    params['loss_t'] = 20           # min time for trajectory error
    params['loss_maxt'] = 800       # max time for trajectory error
    params['dist_thresh'] = 0.15    # neighbourhood around target to determine hit
    params['hit_period'] = 200      # beginning of timepoint where target reaching is considered a hit
    params['print_t'] = 10          # print loss every x steps


    #Overwrite if needed  +  Add custom parameters
    if init_params is not None:
        for x in init_params.keys():
            params[x] = init_params[x]

    return params


# --------------------------------------------#
# params for controller submodules
# --------------------------------------------#
def get_controller_params( init_p, nneu=1 ):
    params={}
    params['type'] = 'none'             # module type - none(empty), 'linear','Perceptron','Perceptron_2Inp'
    
    # for all modules  - good to keep at least the output naming across new modules
    params['n_input'] = 3
    params['n_output'] = 2
    params['n_hidden'] = 3
    params['weight'] = None             # initialisation of weight in Linear/Nonlinear module
    params['weight_hid'] = None         # initialisation of hidden layer weights
    params['weight_out'] = None         # initialisation of output layer weights
    
    # bias terms
    params['add_bias'] = False
    params['bias'] = None
    params['add_bias_hid'] = False
    params['bias_hid'] = None
    params['add_bias_out'] = False
    params['bias_out'] = None

    # for output layer of controller
    params['nonlin_out'] = None         # nonlinearity
    params['train_out'] = False         # train weights
    
    # for perceptron
    params['nonlin_hid'] = None         # hidden layer nonlinearity
    params['train_hid'] = False

    # for perceptron_2inp               
    params['n_mf'] = nneu               # output of rnn sampling module ( no. of "mossy fibres")
    params['train_neu'] = False         # train rnn sampler
    params['sample_type'] = None
    params['sample_neu'] = None         # sampling MODULE or FUNCTION

    #params['has_hidden'] = False
    #params['alp1'] = 0.001


    #Overwrite if needed  +  Add custom parameters
    if init_p is not None:
        for x in init_p.keys():
            params[x] = init_p[x]

    return params




# --------------------------------------------#
# params for perturbation experiments
# --------------------------------------------#
def get_perturb_params( init_p=None ):
    params={}

    params['intuitive.noiseX'] = 0.01
    params['intuitive.scaleX'] = False
    params['intuitive.nPC'] = 10
    params['intuitive.fitPC'] = True
    params['intuitive.nTest'] = 200
    params['intuitive.testfrac'] = 0.10
    params['intuitive.add_bias'] = False
    params['intuitive.use_vel'] = True
    params['intuitive.scaleCoef'] = True
    params['intuitive.use_tm'] = None

    params['wmp.nPerms'] = 1
    params['wmp.nReps'] = 1
    params['wmp.nList_p'] = np.int_(1e5)
    params['wmp.nList_sub'] = 100
    params['wmp.theta_wout'] = [20,75]      # angle with intuitive decoder: in degrees
    params['wmp.ratio_lim'] = [0.3, 3]      # ratio of open loop velocities wrt intuitive decoder
    params['wmp.velAngle'] = [20,80]        # angular diff in vel per target (acceptable range)

    params['wmp.closedloopv'] = [1,5]       # not a ratio

    params['omp.nPerms'] = 1
    params['omp.nReps'] = 1
    params['omp.nList_p'] = np.int_(1e5)
    params['omp.nList_sub'] = 100
    params['omp.theta_wout'] = [20,85]      # angle with intuitive decoder: in degrees
    params['omp.ratio_lim'] = [0.3, 3]      # ratio of open loop velocities wrt intuitive decoder
    params['omp.velAngle'] = [20,80]        # angular diff in vel per target (acceptable range)

    params['omp.closedloopv'] = [0.3,5]     

    if init_p is not None:
        for x in init_p.keys():
            params[x] = init_p[x]

    #print( ' setting WMP to use as many PCs as intuitive decoder ... ')
    params['omp.nPC'] = params['intuitive.nPC']

    return params

