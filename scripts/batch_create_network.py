'''
Batch Script: Create a set of models with different parameters
Fixed initialisation scales.
Network architectures:
- Simple network (in use_models/linear_/). Modeltype = 'none'
- Simple network with relu nonlinearity at rnn (in use_models/relu_). Modeltype = 'none'
- Relu RNN with 1-layer feedback (uses r and err, has relu nonlinearity) (in use_models/layer1_relu_). Modeltype = 'layer1'
- Relu RNN with 2-layer feedback (uses r and err, has hidden layer that is not trained) (in use_models/perceptron_relu_). Modeltype = 'layer2'


Example calls:
$ python batch_create_network.py -F 'relu_/' -nf 5 -te 400 -pe 250
$ python batch_create_network.py -F 'relu_sparse_wt_/' -nf 5 -te 400 -pe 250 -wt 0.3
$ python batch_create_network.py -F 'relu_sparse_neu_/' -nf 5 -te 400 -pe 250 -neu 0.3
$ python batch_create_network.py -F 'relu_layer1_/' -nf 5 -te 400 -pe 250 -lr 0.0002 -mtype 'layer1' -nout 8
$ python batch_create_network.py -F 'relu_layer1_wsample_/' -nf 5 -te 400 -pe 250 -lr 0.0002 -mtype 'layer1' -nout 8 -nmf 10
$ python batch_create_network.py -F 'relu_perceptron_wsample_/' -nf 5 -te 400 -pe 250 -lr 0.0002 -mtype 'layer2' -nout 8 -nhid 50 -nmf 10


HG. Aug 2023
'''


import numpy as np
import math
import os
import sys
import inspect
import argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import torch
import tools.run_network_ as rnet



# ------------------ HELPERS ------------------


def init_mask( size, neu_col=True, wt_frac=1.0, neu_frac=1.0, eps=1e-8 , seed=0):

    mask = np.ones( size )
    np.random.seed(seed)
    if wt_frac<1:
        set_zero = np.random.random(size=size)
        mask[set_zero>wt_frac]= eps # set to small epsilon
    if neu_frac<1:
        if not neu_col:
            mask =mask.T
        nneu = mask.shape[1]
        set_neu = np.random.choice( nneu, np.int_(nneu*(1-neu_frac)) , replace=False)
        mask[:,set_neu] = eps # set to small epsilon
        if not neu_col:
            mask=mask.T

    return mask


# -----------------------------------------------

# construct for network
def gen_masks( n1, init_p, wt_frac=1.0, neu_frac=1.0, n_inp_currents=3, n_fbk_currents=2, seed=0 ):
    # generate masks
    W_in_mask = None
    W_fbk_mask = None
    if wt_frac*neu_frac<1.0:     
        W_in_mask = init_mask( (n_inp_currents,n1), neu_col=True, wt_frac=wt_frac, neu_frac=neu_frac , seed=seed-1)
        W_fbk_mask = init_mask( (n_fbk_currents,n1), neu_col=True, wt_frac=wt_frac, neu_frac=neu_frac, seed=seed+1) # just to vary seed for ff versus fb input
        init_p.update({ 'W_in_mask':W_in_mask, 'W_fbk_mask':W_fbk_mask})
    
    return init_p


def gen_architecture( modeltype, init_p, nmf=0, n_output = 15, n_hidden = 50 ):

    # sampling rnn parameters:
    # n_output = no. of feedback currents
    nonlin_out = 'relu'         # output nonlinearity
    train_out = True
    train_neu = False           # train sampler weights?
    sample_type = 'linear'
    # n_hidden  = hidden layer units in 2-layer (Perceptron) fb modules
    nonlin_hid = 'relu'         # hidden layer nonlinearity
    train_hid = False
    if nmf>0:
        n_mf = nmf              # sample rnn features - as second input to fbk module

    train_rec = True
    train_inp = True 

    if modeltype == 'none':
        train_fbk = True           # train final fbk weights -
        fbk_p = None
        n_fbk_currents = 2

    elif modeltype == 'layer1':
        train_fbk = False           # - or use random projection?
        if n_mf>0:
            mname = 'linear_2Inp'
        else:
            mname = 'linear'
        n_fbk_currents = n_output
        fbk_p = {'type': mname,
            'n_output':n_output, 'nonlin_out':nonlin_out, 'train_out':train_out,   # Output layer ("Transformation/Expansion recoding")
            'n_mf': n_mf, 'train_neu':train_neu, 'sample_type':sample_type}        # RNN sampling layer    ("mfs")
    
        
    elif modeltype == 'layer2': 
        train_fbk = False           # - or use random projection?
        if n_mf>0:
            mname = 'Perceptron_2Inp'
        else:
            mname = 'Perceptron'
        n_fbk_currents = n_output
        fbk_p = {'type': mname,
            'n_output':n_output, 'nonlin_out':nonlin_out, 'train_out':train_out,    # Output layer ("pkj")
            'n_hidden': n_hidden, 'nonlin_hid': nonlin_hid, 'train_hid':train_hid,  # Hidden layer ("grcs")
            'n_mf': n_mf, 'train_neu':train_neu, 'sample_type':sample_type}         # RNN sampling layer    ("mfs")
    


    init_p.update({'train_fbk':train_fbk, 'train_rec':train_rec, 'train_inp':train_inp ,
                    'fb':fbk_p})
    return init_p, n_fbk_currents






# --------------------------------------------------------------------#
#######################################################################
#                    Create a batch of models
#######################################################################
# --------------------------------------------------------------------#

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Models')
    parser.add_argument('-pf', '--prefix', type=str, default='', 
                        help='Prefix for trained model files')
    parser.add_argument('-F', '--folder', type=str, default='relu_sparse_wt_/', 
                        help='Folder to save trained models')
    parser.add_argument('-nf', '--nfiles', type=int, default=3, help='Number of models to generate') 
    parser.add_argument('-n1', '--n1', type=int, default=100, help='Number of RNN units') 
    parser.add_argument('-tau', '--tau', type=int, default=5, help='Tau (decay timescale) of RNN units') 
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Learning rate for ADAM') 
    parser.add_argument('-maxT', '--maxT', type=int, default=650, help='Trial Sim time during training') 
    parser.add_argument('-tMt', '--test_maxT', type=int, default=800, help='Trial Sim time to test model') 
    parser.add_argument('-te', '--training_epochs', type=int, default=10, help='Total no. of training epochs') 
    parser.add_argument('-pe', '--perturb_epochs', type=int, default=10, help='Training epochs with perturbations')
    parser.add_argument('-mtype', '--modeltype', type=str, default='none', 
                        help='Type of feedback module architectures')
    parser.add_argument('-wt', '--wt_frac', type=float, default=1.0, help='Fraction of non-zero input weights at initialisation')
    parser.add_argument('-neu', '--neu_frac', type=float, default=1.0, help='Fraction of neurons with non-zero input wts at initialisation') 
    parser.add_argument('-nonlin', '--nonlinearity', type=str, default='relu', help='Nonlinearity at RNN units') 
    parser.add_argument('-nmf', '--nmf', type=int, default=0, help='Number of outputs of RNN sampling. 0 if not sampling RNN state') 
    parser.add_argument('-nout', '--n_out', type=int, default=15, help='Number of output currents for different architectures (ignored for simple feedback)') 
    parser.add_argument('-nhid', '--n_hid', type=int, default=50, help='Number of hidden units in feedback module - only used for 2layer/Perceptron module') 
    parser.add_argument('-idx', '--index', type=int, default=1, help='CUDA device index') 
    parser.add_argument('-recsig','--sig_rec_0', type=float, default=1, help='Sigma of recurrent weight distribution')
    parser.add_argument('-sign','--sigma_n', type=float, default=0.0, help='activity noise sigma')
    parser.add_argument('-del0','--delay0', type=int, default=100, help='start of go cue distribution')
    parser.add_argument('-del1','--delay1', type=int, default=200, help='end of go cue distribution')
    
    args = parser.parse_args()
    


    # Global parameters ------------------------------------------------
    
    n1=args.n1              # no. of rnn units
    tau = args.tau
    nonlinearity = args.nonlinearity    #'relu'
    add_bias_n = True       # bias current in rnn units

    n_inp_currents = 3
    #n_fbk_currents = 2     # returned by architecture setup - depends on modeltype
    
    wt_frac =   args.wt_frac        # setup sparse wts
    neu_frac =   args.neu_frac
    use_sigma = True
    sig_rec_0 = args.sig_rec_0           # rnn wt sigma
    sig_inp_0 = 0.5 * math.sqrt(1/wt_frac) * math.sqrt(1/neu_frac) 
    sig_fbk_0 = 0.5 * math.sqrt(1/wt_frac) * math.sqrt(1/neu_frac) 
    sig_out_0 = 5


    velocity = 0.05
    
    training_epochs = args.training_epochs
    perturb_epochs = args.perturb_epochs
    lr = args.lr
    maxT = args.maxT
    test_maxT = args.test_maxT

    nseeds = args.nfiles #<------------------------- No. of model files

    # Feedback module parameters
    modeltype = args.modeltype
    nmf = args.nmf
    n_out = args.n_out
    n_hid = args.n_hid

    # Startup operations ----------------------------------------------

    # set up folders:
    if not os.path.exists('use_models/'):
            os.mkdir('use_models/')

    # open cuda connection
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.index)
    else:
        device = torch.device('cpu')

    prefix = args.prefix+'tau'+np.str_(tau)+'_'
    savfolder = 'use_models/' +args.folder

    if not os.path.exists(savfolder):
            os.mkdir(savfolder)

    # Initialize RNN params ----------------------------------------------

    init_p =  { 'n1':n1,  'tau':tau, 'training_epochs':training_epochs, 'perturb_epochs':perturb_epochs, 'lr':lr, 'use_sigma':use_sigma,
                'sig_rec_0':sig_rec_0, 'sig_inp_0':sig_inp_0, 'sig_fbk_0':sig_fbk_0, 'sig_out_0':sig_out_0 , 'velocity':velocity,
                'maxT':maxT, 'test_maxT':test_maxT, 'add_bias_n':add_bias_n, 'device':device,
                'nonlinearity':nonlinearity, 'sigma_n':args.sigma_n, 'delays':[args.delay0, args.delay1] }
    


    # Construct and train networks ----------------------------------

    for jj in range(nseeds):
        
        print(f'file '+np.str_(jj+1)+' of '+np.str_(nseeds)+' of type "' + modeltype +'" ... ')
        seed = np.random.choice(100)

        init_p, n_fbk_currents = gen_architecture( modeltype, init_p, nmf=nmf, n_output=n_out, n_hidden=n_hid)          # generate feedback modules
        init_p = gen_masks( n1=n1, init_p=init_p, n_fbk_currents=n_fbk_currents, wt_frac=wt_frac, neu_frac=neu_frac, seed=seed)
        
        p =rnet.get_params( init_p, rand_seed=seed )            # fill remaining fields needed with defaults
        network = rnet.mainModel( savfolder+prefix, p )
        network.construct_network()

        print(network.model)        # confirm architecture

        dic = network.train_model()
        network.save_results( dic, savname=savfolder+prefix )

        dic = network.test_model(dic)
        network.plot_model_results( dic )
