import numpy as np
import os
import sys
import argparse
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
if '/scripts' in currentdir[-10:]:
    parentdir = os.path.dirname(currentdir)
    parentdir=parentdir+'/'
    sys.path.insert(0, parentdir) 
else:
    parentdir=currentdir



#import for_glados.train_model_wmp as tmw
import tools.train_model_perturbation as tmw
import tools.train_model_WP as tmwp

import torch



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train on WMPs that pass all criteria, change Rec wts')
    
    parser.add_argument('-F', '--folder', type=str, default='relu_/', 
                        help='Folder name in use_models containing model files')
    parser.add_argument('-file', '--fn', type=str, default='wmp_tested_movepc_PC8.npy', 
                        help='Filename of WMP sweep results')
    parser.add_argument('-n1', '--n1', type=int, default=100, help='Number of RNN units, used for normalisation') 
    parser.add_argument('-sf', '--suffix', type=str, default='_movePC_PC8', 
                        help='Suffix for saved results')
    parser.add_argument('-sf2', '--suffix2', type=str, default='', 
                        help='Suffix for current results')
    parser.add_argument('-map', '--maptype', type=str, default='wmp', 
                        help='Perturbation to readout - maptype= wmp or omp')
    parser.add_argument('-twp', '--train_wp', type=int, default=0, 
                        help='Train using weight perturbation instead of Adam? False by default')

    
    parser.add_argument('-seed', '--train_seed', type=int, default=1010203, help='Seed for WMP selection') 
    parser.add_argument('-sign', '--sigma_n', type=float, default=0.0, help='Activity noise sigma') 
    parser.add_argument('-nf', '--nfiles', type=int, default=5, help='Max number of model files to use') 
    parser.add_argument('-idx', '--index', type=int, default=1, help='CUDA device index') 
    parser.add_argument('-ntr', '--ntrain', type=int, default=10, help='Max number of WMPs to train on') 
    parser.add_argument('-nr', '--nreps', type=int, default=1, help='Number of training repetitions for each WMP') 
    parser.add_argument('-cv0', '--cvrange0', type=float, default=1, help='Min value of avg closed loop velocity') 
    parser.add_argument('-cv1', '--cvrange1', type=float, default=5, help='Max value of avg closed loop velocity') 
    parser.add_argument('-sr0', '--success_range0', type=float, default=0.2, help='Min success rate for training') 
    parser.add_argument('-sr1', '--success_range1', type=float, default=0.7, help='Max success rate for training') 
    

    parser.add_argument('-inp', '--traininp', type=int, default=1, help='Train feedforward input weights?') 
    parser.add_argument('-fbk', '--trainfbk', type=int, default=1, help='Train feedback input weights?') 
    parser.add_argument('-rec', '--trainrec', type=int, default=0, help='Train recurrent weights?') 
    parser.add_argument('-fbout', '--trainfbout', type=int, default=0, help='Train output layer of a feedback module (eg for perceptron like modules)?') 
    
    parser.add_argument('-pc', '--nPC_fit', type=int, default=8, help='Number of PCs previously used for dim reduction for inuitive decoder')#metavar='pc',     
    parser.add_argument('-maxT', '--maxT', type=int, default=800, help='Trial Sim time during training') 
    parser.add_argument('-tMt', '--test_maxT', type=int, default=1500, help='Trial Sim time to test model') 
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Learning rate for ADAM') 
    parser.add_argument('-te', '--training_epochs', type=int, default=10, help='Total no. of raining epochs') 
    parser.add_argument('-pe', '--perturb_epochs', type=int, default=10, help='Training epochs with perturbations') 
    parser.add_argument('-bsz', '--batch_size', type=int, default=1, help='Batch size for training')

    parser.add_argument('-alp1', '--alpha1', type=float, default=5e-2, help='Regularisation weight for input weights (will be divided by n1)') 
    parser.add_argument('-gam1', '--gamma1', type=float, default=5e-2, help='Regularisation weight for input weights (will be divided by n1)') 
    parser.add_argument('-bet1', '--beta1', type=float, default=0.01, help='Regularisation weight for activity (will NOT  be divided by n1)') 
    parser.add_argument('-alp2', '--alpha2', type=float, default=5e-2, help='Regularisation weight for input module output weights (will be divided by n1)') 

    # --- Note:  for anything else call train wmp with your custom args
    
    

    args = parser.parse_args()

    print(args.maptype)
    if args.maptype=='wmp':
        pfolder='wmp/'
    elif args.maptype=='omp':  
        pfolder='omp/'
    elif args.maptype=='rmp':
        pfolder='rmp/'
    fname = parentdir+pfolder+args.folder+args.fn

    # load results
    allw = np.load( fname, allow_pickle=True)
    nf = len(allw)


    # set device:
    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.index)
    else:
        device = torch.device('cpu')


    
    trainrec = (args.trainrec>0)
    trainfbout = (args.trainfbout>0)
    traininp = (args.traininp>0)
    trainfbk = (args.trainfbk>0)

    n1 = args.n1
    pert_params = {'nPC':args.nPC_fit, 'ntrain':args.ntrain, 'nReps':args.nreps, 
                  'cv_range':[args.cvrange0,args.cvrange1], 'success_range':[args.success_range0,args.success_range1]}

    adapt_p = { 'maxT':args.maxT, 'tsteps':args.maxT, 'test_maxT':args.test_maxT, 'lr': args.lr, 'batch_size':args.batch_size, 
               'training_epochs':args.training_epochs, 'perturb_epochs':args.perturb_epochs,
                'alpha1':args.alpha1/n1, 'alpha2':args.alpha2/n1, 'gamma1':args.gamma1/n1, 'gamma2':args.gamma1/n1,  'beta1':args.beta1,'beta2':args.beta1, 
                'train_inp' : traininp, 'train_rec': False, 'train_fbk':trainfbk, 'train_inp_pre': False, 'train_fbk_pre': False, 
                'train_out':False,  'train_neu': False,  'train_hid':False , 'train_rec_mf':False,
                'thresh': 0.15, 'loss_mask':None, 'loss_t':300, 'jump_amp': 0.01, 'hit_period': 200, 
                'sigma_n':args.sigma_n }
    
    if trainrec:
        print('Training recurrent instead of input weights ... ')
        adapt_p.update( {'train_inp' : False, 'train_rec': True, 'train_fbk':False, 'train_inp_pre': False, 'train_fbk_pre': False, 'train_rec_mf':False} )
    if trainfbout:
        print('Training output layer of feedback module instead of input weights ... ')
        adapt_p.update( {'train_inp' : traininp, 'train_rec': False, 'train_fbk':False, 'train_inp_pre': False, 'train_fbk_pre': False, 'train_rec_mf':False, 
                         'train_out':True, 'train_neu': False, 'train_hid':False } )

    print(args)
    #print(adapt_p)
    print(len(allw))

    # do all files:  
      
    resname = 'trained_'+args.maptype
    nf = min(nf,args.nfiles )
    for ff in range(nf):
        rdic = allw[ff]
        if args.train_wp:
            tmwp.train_wmp(rdic, pfolder=pfolder, maptype=args.maptype, folder = args.folder, suffix=args.suffix,
                          resname=resname, suffix_save=args.suffix2, device=device, 
                          pert_params=pert_params, adaptparams=adapt_p, index=args.index, train_seed=args.train_seed  )
        else:
            tmw.train_wmp(rdic, pfolder=pfolder, maptype=args.maptype, folder = args.folder, suffix=args.suffix,
                          resname=resname, suffix_save=args.suffix2, device=device, 
                          pert_params=pert_params, adaptparams=adapt_p, index=args.index, train_seed=args.train_seed  )
        
    


