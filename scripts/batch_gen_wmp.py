import numpy as np
import os
import sys
import inspect
import argparse
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir=parentdir+'/'
sys.path.insert(0, parentdir) 


import torch
import tools.run_network_ as rnet
import tools.perturb_network_ as pnet
import glob
import scipy.linalg as ll




# parameters:
def sweep_wmp(loadname='use_models/relu_', savfolder='wmp/relu_', n1files=None, maxfiles=10, suffix = '_PC7', save_pn = True, index=1, 
              fit_maxT = 1000,  test_maxT=1500, noiseX=0.1, nTest=200, nPC_fit=7,  sigma_n=0.0, 
              nList_sub=5000, ratio_lim=[0.5,2], velAngle=[30,75], tm_range=[200,500] , device=None ):
    
    print('...Starting WMP Sweep for models in '+loadname+' ...')
    # all model files
    if n1files is None:
        n1files = glob.glob( parentdir+loadname+'*training*.npy')
    #print( parentdir+loadname+'*training*.npy')
    nf = len(n1files)
    nf = min(nf,maxfiles)
    n1files = n1files[0:nf]

    if not os.path.exists(savfolder):
        os.mkdir(savfolder)


    use_tm = np.arange(start=tm_range[0], stop=tm_range[1])
    intuitive_p ={'noiseX':noiseX, 'nTest':nTest, 'nPC':nPC_fit, 'use_tm':use_tm }      # fit intuitive decoder - params
    wmp_p = {'nList_sub':nList_sub, 'ratio_lim':ratio_lim, 'velAngle':velAngle}         # how many maps to sample? criteria for wmp

    # open cuda connection
    if torch.cuda.is_available():
        device = torch.device('cuda', index=index)
    else:
        device = torch.device('cpu')

    # saving:
    results = []
    allparams = {'device':device, 'wmp_p':wmp_p , 'intuitive_p':intuitive_p, 'use_tm':use_tm, 'test_maxT':test_maxT}

    # search for wmp:
    for ff in range(nf):
        print(f'file '+np.str_(ff+1)+' of '+np.str_(nf)+' ... ')
        file = n1files[ff]
        print(file)
        newExp = pnet.postModel( model_path=file, adapt_p=None )
        newExp.params['device'] = device
        rdic = {'file':file, 'savfolder':savfolder, 'intuitive':{} }        
        newExp.load_rnn()           # load model
        # update sigma_n
        newExp.model.sigma_n = sigma_n
        # model details:
        seed = newExp.simparams['rand_seed']
        subf = savfolder+ 'Model_'+np.str_(seed)+'_movePC'+suffix+'/' # added use_tm to be during movepc
        if not os.path.exists( subf ):
            os.mkdir(subf)
        rdic.update({'rand_seed':seed, 'subf':subf})
        
        # fit intuitive:
        newExp.params['test_maxT']= fit_maxT
        rc, px1, wout = newExp.get_intuitive_map(nPC=intuitive_p['nPC'], ch_params=intuitive_p)
        newp = newExp.model.save_parameters()
        ang_int_wout = ll.subspace_angles( wout, newp['W_out_0'] )      # angle between intuitive decoder and w_out_0
        #print( np.rad2deg(ang_int_wout) )

        # load intuititive decoder
        newp['W_out_0'] = wout
        newExp.model.load_parameters(newp)
        newExp.intuitive.update( {'LinModel':rc, 'PCRes':px1, 'wout':wout} )
        newExp.params['jump_amp'] = 0.05
        newExp.params['test_maxT']= test_maxT
        dic = newExp.test_model()
        newExp.plot_model_results(dic, savname=subf+'fit_')
        rdic.update({'intuitive' :newExp.intuitive.copy() })


        # collect wmps:
        ca, speed, angle, medangle, pperm, mapids, ratio = newExp.generate_wmps( ch_params=wmp_p )
        z = (medangle * angle ) * speed # all criteria pass or not
        cmaps = mapids[:len(z)]
        usemap= cmaps[z]
        rdic.update({'wout_angle':ca, 'mean_velangle':angle, 'median_velangle':medangle, 'speed':speed, 
                    'perm':pperm, 'rank_id_0':mapids, 'sratio':ratio, 'npass':sum(z), 'use_wmp_id':usemap, 
                    'ang_int_wout':ang_int_wout })


        good_maps, allspeed, perfrate = newExp.post_filter_wmp( use_maps=usemap, list_perm=pperm, plot_num=10, savpath=subf, get_perf=True, maptype='wmp', use_vratio=False )
        rdic.update({'good_maps':good_maps, 'closedv':allspeed, 'perfrate':perfrate})
        newExp.wmp_maps = {'good_maps':good_maps, 'closedv':allspeed, 'perfrate':perfrate}
        print(len(good_maps))
        if save_pn:
            np.save(subf+'WMP_Maps', {'newExp':newExp, 'intuitive':rdic['intuitive']})

        results.append( rdic )

    np.save( savfolder+'wmp_tested_movepc'+suffix, results)        # movepc for fit

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sweep across WMPs and determine their properties and goodness')
    
    parser.add_argument('-F', '--folder', type=str, default='relu_/', 
                        help='Folder name in use_models containing model files')
    parser.add_argument('-sf', '--suffix', type=str, default='', 
                        help='Suffix for saving results')
    parser.add_argument('-sign', '--sigma_n', type=float, default=0.0, help='Activity noise sigma') 
    parser.add_argument('-nf', '--nfiles', type=int, default=5, help='Max number of model files to use') 
    parser.add_argument('-idx', '--index', type=int, default=1, help='CUDA device index') 
    parser.add_argument('-pc', '--nPC_fit', type=int, default=7, help='Number of PCs to use for dim reduction for inuitive decoder')#metavar='pc',     
    parser.add_argument('-tF', '--fit_maxT', type=int, default=1000, help='Sim time to calculate PCs') 
    parser.add_argument('-tM', '--test_maxT', type=int, default=1500, help='Sim time to test model') 
    parser.add_argument('-nlist', '--nList_sub', type=int, default=5000, help='Max number of random WMPs to check') 
    parser.add_argument('-ratio0', '--ratio_lim0', type=float, default=0.5, help='Min ratio between open loop velocities') 
    parser.add_argument('-ratio1', '--ratio_lim1', type=float, default=2.0, help='Max ratio between open loop velocities') 
    parser.add_argument('-vel0', '--velRange0', type=float, default=30, help='Min angular difference between open loop velocities (deg)') 
    parser.add_argument('-vel1', '--velRange1', type=float, default=75, help='Max angular difference between open loop velocities (deg)') 
    parser.add_argument('-tm0', '--usetm0', type=int, default=200, help='Min time to include for getting top PCs') 
    parser.add_argument('-tm1', '--usetm1', type=int, default=500, help='Max time to include for getting top PCs') 
    

    args = parser.parse_args()
    
    loadname='use_models/'+args.folder
    savfolder='wmp/'+args.folder
    suffix=args.suffix+'_PC'+np.str_(args.nPC_fit)
    ratio_lim = [args.ratio_lim0, args.ratio_lim1]
    velAngle =  [args.velRange0, args.velRange1]
    tm_range = [args.usetm0, args.usetm1]

    print(args)

    sweep_wmp(loadname=loadname, savfolder=savfolder, maxfiles=args.nfiles, fit_maxT=args.fit_maxT, test_maxT=args.test_maxT , index = args.index,
               nList_sub=args.nList_sub,ratio_lim=ratio_lim, velAngle=velAngle, tm_range=tm_range, suffix=suffix, nPC_fit=args.nPC_fit, sigma_n=args.sigma_n )