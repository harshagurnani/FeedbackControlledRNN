''' 
generate networks with different parameters 
- wout scale, wrnn scale, wfb scale, fbout scale
- architecture
- wt sparsity, neu sparsity, 
- perturbation size, (bump_amp) 
- #neurons
- beta?
compute:
- angle between pcs and wout (as a fn of #pcs)
- between fbk and wout
- between right eigenvectors and wout
- fit inutiive:
- between wout and wintuit
'''

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


import torch
import tools.run_network_ as rnet
import tools.perturb_network_ as pnet
import tools.wout_angles as wang
import tools.analysis_helpers as ah
import scripts.batch_create_network as bcs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Models and assess structure')
    parser.add_argument('-F', '--folder', type=str, default='relu_sparse_/', 
                        help='Folder to save results')
    parser.add_argument('-nlin', '--nonlin', type=str, default='relu', 
                        help='Nonlinearity for RNN units, give "None" for linear networks')
    parser.add_argument('-tau', '--tau', type=int, default=5, help='Tau (decay timescale) of RNN units') 
    parser.add_argument('-spr', '--setsparse', type=str, default='None', 
                        help='Where to add sparsity? "None", "wt" or "neu"')
    parser.add_argument('-inpsig', '--sig_inp_0', type=float, default=0.5, help='Input sigma. If -1, set to feedback sigma in sweep') 
    
    parser.add_argument('-nr', '--nreps', type=int, default=3, help='Number of repeats for each parameter set') 
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='Learning rate for ADAM') 
    parser.add_argument('-maxT', '--maxT', type=int, default=650, help='Trial Sim time during training') 
    parser.add_argument('-tMt', '--test_maxT', type=int, default=800, help='Trial Sim time to test model') 
    parser.add_argument('-te', '--training_epochs', type=int, default=10, help='Total no. of training epochs') 
    parser.add_argument('-pe', '--perturb_epochs', type=int, default=10, help='Training epochs with perturbations')
    parser.add_argument('-bump', '--bump_amp', type=int, default=0.05, help='Perturbation amplitude for training') 
    parser.add_argument('-jump', '--jump_amp', type=int, default=0.1, help='Perturbation amplitude for testing') 
    
    parser.add_argument('-mtype', '--modeltype', type=str, default='none', 
                        help='Type of feedback module architectures')
    parser.add_argument('-nmf', '--nmf', type=int, default=0, help='Number of outputs of RNN sampling. 0 if not sampling RNN state') 
    parser.add_argument('-nout', '--n_out', type=int, default=15, help='Number of output currents for different architectures (ignored for simple feedback)') 
    parser.add_argument('-nhid', '--n_hid', type=int, default=50, help='Number of hidden units in feedback module - only used for 2layer/Perceptron module') 
    
    parser.add_argument('-pc', '--nPC_fit', type=int, default=8, help='Number of PCs to use for dim reduction for inuitive decoder')#metavar='pc',     
    parser.add_argument('-thr', '--thresh', type=int, default=0.1, help='Maximum distance from target to consider hit') 
    parser.add_argument('-idx', '--index', type=int, default=1, help='CUDA device index') 
    parser.add_argument('-sign', '--sigma_n', type=float, default=0.0, help='Activity noise sigma')
    
    
    
    args = parser.parse_args()

    # Global parameters ------------------------------------------------
    mfolder = 'oblique_dynamics/'
    if not os.path.exists( mfolder ):
        os.mkdir(mfolder)

    folder = args.folder    #'relu_sparse_/'#'linear_gpu/'#'simple_gpu/'
    nonlin = args.nonlin    #'relu'
    if nonlin=='None':
        nonlin=None
    
    savfolder = mfolder + folder
    if not os.path.exists( savfolder ):
        os.mkdir(savfolder)


    print(savfolder)




    # ---------------------

    # parameter sweep:
    wout_scale = [ 1, 2, 5]    # [2,5]                  # readout scale
    n_wout = len(wout_scale)

    #velocity_scale = [0.05]
    #n_vel = len(velocity_scale)

    rnn_scale = [0.3, 0.5,1,1.25]   # [0.5,1]             # rnn initial g (1.5 is chaotic; greater g harder to train)
    n_wrec = len(rnn_scale)

    n_neurons = [100]                   # no. of rnn units
    n_netn1 = len(n_neurons)

    fbk_sig = [0.5]#[0.0, 0.05, 0.1, 0.5]  # [0.5, 1.0]                # input wt scale
    n_fbksig = len(fbk_sig)


    bump_amp = [0.05]#[0.0, 0.05]
    n_bump_amp = len(bump_amp)


    setsparse = args.setsparse
    frac_wt = [1.0]
    frac_neu = [1.0]
    sparse_frac = [1.0]
    if setsparse=='wt':
        frac_wt = [0.3, 0.6, 1.0]       # fbk (or ff input) wt sparsity - what fraction of all wts nonzero
        sparse_frac = frac_wt
    elif setsparse=='neu':
        frac_neu = [0.3, 0.6, 1.0]      # fbk (or ff input) wt sparsity - what fraction of neurons receive feedback (or stim)
        sparse_frac = frac_neu
    n_sparse = len(sparse_frac)
    

    nReps = args.nreps                  # repeats for same parameter

    # ---------------------
    
    # open cuda connection

    if torch.cuda.is_available():
        device = torch.device('cuda', index=args.index)     # use a different gpu
    else:
        device = torch.device('cpu')


    # Run parameters
    n_inp_currents = 3
    n_fbk_currents = 2      # depends on model type
    sig_inp_0 = args.sig_inp_0  # input wt sigma

    # Feedback module parameters
    modeltype = args.modeltype
    nmf = args.nmf
    n_out = args.n_out
    n_hid = args.n_hid

    te = args.training_epochs
    pe = args.perturb_epochs
    lr = args.lr
    maxT = args.maxT
    test_maxT = args.test_maxT
    #bump_amp = 0.02
    velocity_scale = 0.05

    # - no fb/ff modules unless model type specifies

    init_p =  {'training_epochs':te, 'perturb_epochs':pe, 'maxT':maxT, 'test_maxT':test_maxT, 
               'jump_amp':args.jump_amp, 'sigma_n': args.sigma_n, 
                'device':device, 'tau':args.tau, 'use_sigma':True, 'add_bias_n':True,
                'nonlinearity':nonlin, 'velocity':velocity_scale }
    
    #
    # 'sig_inp_0':0.5, 'sig_fbk_0':0.5,  
    
    # intuitive parameters:
    nPC = args.nPC_fit           #8
    thresh = args.thresh    #0.1

    # angles:
    nPC_angle=30

    all_angles = []
    save_ctr = 5   # save every n files
    ctr = 0
    use_tm = np.arange( start=200,stop=400 )

    print('nfiles = '+np.str_(n_wout*n_wrec*n_netn1*n_fbksig*n_sparse*nReps*n_bump_amp))
    for wo in range(n_wout):
        sig_out_0 = wout_scale[wo]

        for wfb in range(n_fbksig):
            sig_fbk_0 = fbk_sig[wfb]
            if sig_inp_0<-1:
                sig_inp_0=sig_fbk_0

            for rr in range(n_wrec):
                sig_rec_0 = rnn_scale[rr]

                for nn in range(n_netn1):
                    n1 = n_neurons[nn]

                    for sp in range(n_sparse):
                        fracwts = sparse_frac[sp]

                        for rep in range(nReps):
                            seed = np.random.choice(100)

                            for bmp in range(n_bump_amp):
                                amp = bump_amp[bmp]

                                # set params
                                init_p.update({'seed':seed, 'sig_out_0':sig_out_0, 'sig_rec_0':sig_rec_0, 'n1':n1, 'sig_fbk_0':sig_fbk_0, 'sig_inp_0':sig_inp_0, 'bump_amp':amp })
                                init_p, n_fbk_currents = bcs.gen_architecture( modeltype, init_p, nmf=nmf, n_output=n_out, n_hidden=n_hid)          # generate feedback modules

                                # generate masks --------------------------------
                                W_in_mask = None
                                W_fbk_mask = None
                                wt_frac = 1
                                neu_frac = 1
                                if setsparse=='wt':
                                    wt_frac=fracwts
                                elif setsparse=='neu':
                                    neu_frac=fracwts
                                if fracwts<1:     
                                    init_p = bcs.gen_masks( n1=n1, init_p=init_p, n_fbk_currents=n_fbk_currents, wt_frac=wt_frac, neu_frac=neu_frac, seed=seed)
                                    #W_in_mask = bcs.init_mask( (n_inp_currents,n1), neu_col=True, wt_frac=wt_frac, neu_frac=neu_frac , seed=seed-1)
                                    #W_fbk_mask = bcs.init_mask( (n_fbk_currents,n1), neu_col=True, wt_frac=wt_frac, neu_frac=neu_frac, seed=seed+1) # just to vary seed for ff versus fb input
                                else:
                                    init_p.update({ 'W_in_mask':W_in_mask, 'W_fbk_mask':W_fbk_mask})
                                # ------------------------------------------------

                                
                                suffix = '_wout_'+np.str_(np.int_(sig_out_0*1000))\
                                        + '_rec_'+np.str_(np.int_(sig_rec_0*1000))\
                                        + '_n1_'+np.str_(np.int_(n1))\
                                        + '_wtfrac'+np.str_(np.int_(wt_frac*100))\
                                        + '_neufrac'+np.str_(np.int_(neu_frac*100))\
                                        + '_bump_'+np.str_(np.int_(amp*1000))\
                                        + '_seed_'+np.str_(seed)\
                                        + '_rep_'+np.str_(rep)
                                
                                print( suffix )
                                
                                init_p.update({'suffix1': suffix })

                                # ------------------------------------------------
                                if (sig_fbk_0==0) | (fracwts==0):
                                    init_p.update({'train_fbk':False})      # don't train non-existent weights
                                # construct and train network:
                                p =rnet.get_params( init_p, rand_seed=seed )
                                network = pnet.postModel( savfolder, p )
                                network.construct_network()
                                print('sigma_n = '+np.str_(network.model.sigma_n))
                                dic = network.train_model()
                                
                                # look at angles and performance in trained model ----------
                                dic = network.test_model(dic)       # has parameters and data
                                # do something with this:
                                trained_ang = wang.study_trained_dynamics( dic['params1'], dic['res'], nPC=nPC_angle, starttm=200, stoptm=450 )
                                res_trained = ah.get_performance( dic, thresh=thresh)
                                wres_trained = wang.get_all_wchange(dic['params0'], dic['params1'])
                                gencorr_trained = wang.get_gen_correlation(  dic['params1'], dic['res'] )
                                gencorr_var = wang.get_gen_corr_ratios(  dic['params1'], dic['res'], nPC=60, nexclude=20  )
                                trained_delW_ang = wang.study_angle_delW( dic['params1'], dic['params0'], nPC=nPC_angle )


                                # get intuitive map: ---------------------------------------
                                reg, px1, wout_in = network.get_intuitive_map( nPC=nPC, ch_params={'use_tm':use_tm} )
                                intuitive ={ 'PCRes':px1, 'wout':wout_in }
                                # do something with this:
                                intuitive_ang = wang.study_intuitive_dynamics( dic['params1'], dic['res'], intuitive )

                                # performance of intuitive map -----------------------------
                                newp = network.model.save_parameters()
                                newp['W_out_0'] = wout_in
                                #mask = newp['W_in_mask']
                                #print(np.sum(mask>1e-5))                    # check sparsity
                                #print(np.sum(np.sum(mask, axis=0)>1e-5))    # check sparsity
                                network.model.load_parameters( newp )
                                dic2 = network.test_model()
                                res_intuit = ah.get_performance( dic2, thresh=thresh)

                                currdic = {'trained_angles':trained_ang, 'intuitive_angles':intuitive_ang, 'simparams':network.params, 'velocity':velocity_scale,
                                        'sig_out_0':sig_out_0, 'sig_rec_0':sig_rec_0, 'n1':n1, 'seed':seed, 'frac_wt':wt_frac, 'frac_neu':neu_frac, 'sig_fbk_0':sig_fbk_0,
                                        'perf_trained':res_trained, 'perf_intuit':res_intuit, 'wres_trained':wres_trained,
                                        'nPC_angle':nPC_angle, 'thresh':thresh , 'gencorr_trained':gencorr_trained, 'gencorr_var':gencorr_var,
                                        'trained_delW_angles':trained_delW_ang}
                                
                                all_angles.append( currdic )
                                ctr+=1
                                if (ctr % save_ctr)==0:
                                    np.save(savfolder+'wout_angles', all_angles )


    # save results
    np.save(savfolder+'wout_angles', all_angles )

                    
    # do some plotting:



