
# ---------------------------------------------- ##
# ---------------------------------------------- ##
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'

def get_folders( opt ):

    if opt=='relu_test_':
        path=parentdir
        subf = 'relu_/'         # Main top-level architecture (eg in use_models)
        save_suffix ='_relu_test_inp200_'
        # wt change for feedback:
        use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
        use_mod = None### 'fbk_p'#None# 
        percp=False
        
        ###### Will search for all files in wmp['folder']+'/Model_**'+wmp['suffix']+wmp['file'][jj]
        wmp={}                  # FOR WMPs
        wmp['folder'] = path+'wmp/'+subf
        wmp['file']   = ['trained_wmp_trainInp200.npy']         # filenames
        wmp['save_suffix']=save_suffix                          # plot suffix
        wmp['suffix']='_PC8'                                   # loading directory suffix
        wmp['savfolder'] = wmp['folder']+'plots_test_/'           # save directory
        npro=True

        ###### Will search for all files in omp['folder']+'/Model_**'+omp['suffix']+omp['file'][jj]
        omp={}                  # FOR OMPs
        omp['folder'] = path+'omp/'+subf
        omp['file']=  ['trained_omp_trainInp200.npy']           # filenames
        omp['save_suffix']=save_suffix
        omp['suffix']='_PC8'
        omp['savfolder'] = omp['folder']+'plots_test_/'
        npro=True

        # FOR COMBINED DATA
        joint={}                
        joint['savfolder'] = parentdir+'saved_plots/pert/relu_test_/'
        joint['save_suffix']=save_suffix+'combined_'
        clrs = ['r','b']        # WMP and OMP color respectively

        rmp={}

    dic = {'wmp':wmp, 'omp':omp, 'joint':joint, 'clrs':clrs, 'rmp':rmp, 
            'path':path, 'subf':subf, 'save_suffix':save_suffix, 
            'use_fbwt':use_fbwt, 'use_mod':use_mod, 'percp':percp }
            
    return dic



'''
path='/data/users/hg84/FeedbackLearning/Data/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='rec_All_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_trainRec200_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_rec2_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']= ['trained_omp_trainRec200_.npy']# ['trained_omp_trainRec200_.npy']#['trained_omp_trainFbk200_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_rec2_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_rec_All_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_recAll_'
clrs = ['r','b']        # WMP and OMP color respectively

'''

# ---------------------------------------------- ##
# ---------------------------------------------- ##

'''
subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fb_All_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_trainFbk200_noise_.npy',  'trained_wmp_trainFbk200_noise_run2_.npy'] #['trained_wmp_trainRec200_.npy']# ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy',  'trained_wmp_trainFbk_run2_.npy']  #['trained_wmp_trainFbk200_.npy']#, 'trained_wmp_trainFbk_run2_.npy']        # all possible files to combine   #'trained_wmp_trainFbk_rep_.npy'['trained_wmp_trainFbOut200_.npy']#  'trained_wmp_trainFbk_run2_.npy',['trained_wmp_trainFbOut200_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fb_noise_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']= ['trained_omp_trainFbk200_noise_run2_.npy']# ['trained_omp_trainRec200_.npy']#['trained_omp_trainFbk200_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fb_noise_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fb_All_noise_model1_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_model1_'
clrs = ['r','b']        # WMP and OMP color respectively

### -------------------------------------------- ###
### -------------------------------------------- ###

subf = 'relu_noisy_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fb_All_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmptrain_Fbk200_.npy']# ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy',  'trained_wmp_trainFbk_run2_.npy']  #['trained_wmp_trainFbk200_.npy']#, 'trained_wmp_trainFbk_run2_.npy']        # all possible files to combine   #'trained_wmp_trainFbk_rep_.npy'['trained_wmp_trainFbOut200_.npy']#  'trained_wmp_trainFbk_run2_.npy',['trained_wmp_trainFbOut200_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fb_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']= ['trained_omptrain_Fbk200_.npy']#['trained_omp_trainFbk200_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fb_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fb_All_2_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_'
clrs = ['r','b']        # WMP and OMP color respectively
'''

### -------------------------------------------- ###
### -------------------------------------------- ###
'''
path='/data/users/hg84/FeedbackLearning/Data/'

subf = 'percp2_wide_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbOut_All_'
# wt change for feedback:
use_fbwt= 'output.weight'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = 'fbk_p'### 'fbk_p'#None# 
percp=True

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmptrain_FbOut200_.npy', 'trained_wmptrain_FbOut200_run2_.npy'] #['trained_wmptrain_FbOut200_run3_.npy', 'trained_wmptrain_FbOut200_run4_.npy']# ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy']#,  'trained_wmp_trainFbk_run2_.npy']  #['trained_wmp_trainFbk200_.npy']#, 'trained_wmp_trainFbk_run2_.npy']        # all possible files to combine   #'trained_wmp_trainFbk_rep_.npy'['trained_wmp_trainFbOut200_.npy']#  'trained_wmp_trainFbk_run2_.npy',['trained_wmp_trainFbOut200_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbOut_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbOut200_.npy', 'trained_omp_trainFbOut200_run2_.npy']#['trained_omp_FbOut200_run2_.npy', 'trained_omp_FbOut200_run4_.npy']#['trained_omp_trainFb_.npy']#['trained_omp_trainFb_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbOut_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fbOut_All_2_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_'
clrs = ['r','b']        # WMP and OMP color respectively


### -------------------------------------------- ###
### -------------------------------------------- ###
### -------------------------------------------- ###
### -------------------------------------------- ###
'''

'''
path='/home/hg84/Documents/Github/FeedbackLearning/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbk_wp_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_trainFbk500_WP_run_med_.npy', 'trained_wmp_trainFbk500_WP_run_med_run2_.npy']#['trained_wmp_trainFbk500_WP_run_slow_run2_.npy']#['trained_wmp_trainFbk500_WP_run_slow_.npy']#['trained_wmp_trainFbk200_WP_run2_.npy', 'trained_wmp_trainFbk200_WP_run3_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_wpmed_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbk500_WP_run_med_.npy', 'trained_omp_trainFbk500_WP_run_med_run2_.npy']#['trained_omp_trainFbk500_WP_run_slow_.npy']#['trained_omp_trainFbk200_WP_run2_.npy', 'trained_omp_trainFbk200_WP_run3_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_wpmed_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_WP_med_logspeed_2_/' 
joint['save_suffix']=save_suffix+'combined_wp_'
clrs = ['r','b']        # WMP and OMP color respectively
'''


'''
path='/home/hg84/Documents/Github/FeedbackLearning/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbk_wp_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_trainFbk500_WP_run_slow_run2_.npy', 'trained_wmp_trainFbk500_WP_run_slow_run3_.npy']#['trained_wmp_trainFbk500_WP_run_slow_.npy']#['trained_wmp_trainFbk200_WP_run2_.npy', 'trained_wmp_trainFbk200_WP_run3_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_wpslow_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbk500_WP_run_slow_.npy',  'trained_omp_trainFbk500_WP_run_slow_run3_.npy']#['trained_omp_trainFbk200_WP_run2_.npy', 'trained_omp_trainFbk200_WP_run3_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_wpslow_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_WP_slow_logspeed_2_/' 
joint['save_suffix']=save_suffix+'combined_wp_'
clrs = ['r','b']        # WMP and OMP color respectively
'''




'''
path='/home/hg84/Documents/Github/FeedbackLearning/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbk_wp_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_trainFbk500_WP_run_slow_long1_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_wplong_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbk500_WP_run_slow_long1_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_wplong_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_WP_slow_long_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_wp_'
clrs = ['r','b']        # WMP and OMP color respectively




####################


path='/home/hg84/Documents/Github/FeedbackLearning/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbk_wp_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_trainFbk500_WP_run_slow_small_.npy', 'trained_wmp_trainFbk500_WP_run_slow_small_run2_.npy', 'trained_wmp_trainFbk500_WP_run_slow_small_run3_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_wpsmall_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbk500_WP_run_slow_small_.npy',  'trained_omp_trainFbk500_WP_run_slow_small_run2_.npy', 'trained_omp_trainFbk500_WP_run_slow_small_run3_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_wpsmall_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_WP_slow_small_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_wp_'
clrs = ['r','b']        # WMP and OMP color respectively
'''


####################

'''
path='/data/users/hg84/FeedbackLearning/Data/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbk_bsz_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = [ 'trained_wmp_bsz32_r2_.npy', 'trained_wmp_bsz32_r3_.npy', 'trained_wmp_bsz32_r4_.npy', 'trained_wmp_bsz32_r5_.npy']#'trained_wmp_bsz32_.npy',
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_bsz_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  [ 'trained_omp_bsz32_r2_.npy', 'trained_omp_bsz32_r3_.npy' , 'trained_omp_bsz32_r4_.npy', 'trained_omp_bsz32_r5_.npy'  ]#'trained_omp_bsz32_.npy',
omp['save_suffix']=save_suffix  
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_bsz_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_bsz_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_bsz_'
clrs = ['r','b']        # WMP and OMP color respectively
'''


### -------------------------------------------- ###
### -------------------------------------------- ###


'''
subf = 'percp2_expansion_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbOut_WP_'
# wt change for feedback:
use_fbwt= 'output.weight'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = 'fbk_p'### 'fbk_p'#None# 
percp=True

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_trainFbOut500_WP_.npy', 'trained_wmp_trainFbOut500_WP_run2_.npy', 'trained_wmp_trainFbOut500_WP_run3_.npy',  'trained_wmp_trainFbOut500_WP_run4_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbOut_WP_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbOut500_WP_.npy', 'trained_omp_trainFbOut500_WP_run2_.npy', 'trained_omp_trainFbOut500_WP_run3_.npy', 'trained_omp_trainFbOut500_WP_run4_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbOut_WP_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fbOut_logspeed_WP_2_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_WP_'
clrs = ['r','b']        # WMP and OMP color respectively
'''

# ---------------------------------------------- ##
# ---------------------------------------------- ##
### -------------------------------------------- ###
'''

path='/home/hg84/Documents/Github/FeedbackLearning/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbk_dim_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_testdim_.npy', 'trained_wmp_testdim_run2_.npy', 'trained_wmp_testdim_run3_.npy', 'trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy',  'trained_wmp_trainFbk_run2_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_ls_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_testdim_.npy', 'trained_omp_testdim_run2_.npy', 'trained_omp_testdim_run3_.npy', 'trained_omp_trainFbk200_.npy', 'trained_omp_trainFbk200_run3_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_ls_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_Largeset_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_'
clrs = ['r','b']        # WMP and OMP color respectively

'''
### -------------------------------------------- ###
### -------------------------------------------- ###
'''
path='/home/hg84/Documents/Github/FeedbackLearning/'


subf = 'percp2_expansion_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbOut_WP_'
# wt change for feedback:
use_fbwt= 'output.weight'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = 'fbk_p'### 'fbk_p'#None# 
percp=True

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_trainFbk500_WP_run_slow_small_.npy', 'trained_wmp_trainFbk500_WP_run_slow_small_run2_.npy', 'trained_wmp_trainFbk500_WP_run_slow_small_run3_.npy', 'trained_wmp_trainFbk500_WP_run_slow_small_run4_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbOut_WP_small_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbk500_WP_run_slow_small_.npy', 'trained_omp_trainFbk500_WP_run_slow_small_run2_.npy', 'trained_omp_trainFbk500_WP_run_slow_small_run3_.npy', 'trained_omp_trainFbk500_WP_run_slow_small_run4_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbOut_WP_small_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fbOut_logspeed_WP_slow_small_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_WP_'
clrs = ['r','b']        # WMP and OMP color respectively
'''



'''
path='/data/users/hg84/FeedbackLearning/Data/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbk_dim_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_testdim_.npy', 'trained_wmp_testdim_run2_.npy', 'trained_wmp_testdim_run3_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_dim2_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_testdim_.npy', 'trained_omp_testdim_run2_.npy', 'trained_omp_testdim_run3_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_dim2_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_dim_logspeed_2_/' 
joint['save_suffix']=save_suffix+'combined_'
clrs = ['r','b']        # WMP and OMP color respectively
'''

'''
subf = 'percp2_expansion_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbOut_All_'
# wt change for feedback:
use_fbwt= 'output.weight'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = 'fbk_p'### 'fbk_p'#None# 
percp=True

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmptrain_FbOut200_run3_.npy',  'trained_wmptrain_FbOut200_run4_.npy', 'trained_wmp_trainFbOut_run5_.npy'] #'trained_wmp_trainFbOut200_.npy',  #['trained_wmptrain_FbOut200_run3_.npy', 'trained_wmptrain_FbOut200_run4_.npy']# ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy']#,  'trained_wmp_trainFbk_run2_.npy']  #['trained_wmp_trainFbk200_.npy']#, 'trained_wmp_trainFbk_run2_.npy']        # all possible files to combine   #'trained_wmp_trainFbk_rep_.npy'['trained_wmp_trainFbOut200_.npy']#  'trained_wmp_trainFbk_run2_.npy',['trained_wmp_trainFbOut200_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbOut_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_FbOut200_run2_.npy', 'trained_omp_FbOut200_run4_.npy', 'trained_omp_trainFbOut_run5_.npy']# 'trained_omp_trainFbOut200_.npy', #['trained_omp_FbOut200_run2_.npy', 'trained_omp_FbOut200_run4_.npy']#['trained_omp_trainFb_.npy']#['trained_omp_trainFb_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbOut_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fbOut_logspeed_2_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_'
clrs = ['r','b']        # WMP and OMP color respectively
'''

### -------------------------------------------- ###
### -------------------------------------------- ###

'''
path='/data/users/hg84/FeedbackLearning/Data/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='rec_slower_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_rec200_slower_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_recs_slower_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']= ['trained_omp_rec200_slower_.npy']# ['trained_omp_trainRec200_.npy']#['trained_omp_trainFbk200_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_rec_slower_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_rec_slower_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_rec_slower_'
clrs = ['r','b']        # WMP and OMP color respectively


### -------------------------------------------- ###


path='/home/hg84/Documents/Github/FeedbackLearning/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fbk_slow_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_fbk200_slow_.npy', 'trained_wmp_fbk200_slow_run2_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbk_slow_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']= ['trained_omp_fbk200_slow_.npy', 'trained_omp_fbk200_slow_run2_.npy']# 
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbk_slow_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fbk_slow_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_fbk_slow_'
clrs = ['r','b']        # WMP and OMP color respectively



### -------------------------------------------- ###


path='/home/hg84/Documents/Github/FeedbackLearning/'

subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fbk_slow_long_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_fbk200_slow_long_.npy', 'trained_wmp_fbk200_slow_long_run2_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbk_slow_long_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']= ['trained_omp_fbk200_slow_long_.npy', 'trained_omp_fbk200_slow_long_run2_.npy']# 
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbk_slow_long_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fbk_slow_long_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_fbk_slow_long_'
clrs = ['r','b']        # WMP and OMP color respectively
'''


### -------------------------------------------- ###
### -------------------------------------------- ###


'''
# ---------------------------------------------- ##

path='/home/hg84/Documents/Github/FeedbackLearning/'
subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fb_only_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_trainFbkOnly_.npy',  'trained_wmp_trainFbkOnly_run2_.npy',  'trained_wmp_trainFbkOnly_run3_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbonly_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbkOnly_.npy',  'trained_omp_trainFbkOnly_run2_.npy',  'trained_omp_trainFbkOnly_run3_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbonly_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_FbkOnly_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_fbonly_'
clrs = ['r','b']        # WMP and OMP color respectively


# ---------------------------------------------- ##
# ---------------------------------------------- ##
# ---------------------------------------------- ##





path='/home/hg84/Documents/Github/FeedbackLearning/'
subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='inp_only_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_trainInpOnly_.npy',  'trained_wmp_trainInpOnly_run2_.npy',  'trained_wmp_trainInpOnly_run3_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_inponly_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainInpOnly_.npy',  'trained_omp_trainInpOnly_run2_.npy',  'trained_omp_trainInpOnly_run3_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_inponly_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_InpOnly_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_inponly_'
clrs = ['r','b']        # WMP and OMP color respectively


# ---------------------------------------------- ##
# ---------------------------------------------- ##

path='/home/hg84/Documents/Github/FeedbackLearning/'
subf = 'relu_rank4_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fb_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run2_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fb_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbk200_.npy',  'trained_omp_trainFbk200_run2_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fb_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_fball_'
clrs = ['r','b']        # WMP and OMP color respectively

# ---------------------------------------------- ##
# ---------------------------------------------- ##

path='/home/hg84/Documents/Github/FeedbackLearning/'
subf = 'relu_rank2_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fb_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run2_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fb_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbk200_.npy',  'trained_omp_trainFbk200_run2_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fb_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_fball_'
clrs = ['r','b']        # WMP and OMP color respectively

# ---------------------------------------------- ##
# ---------------------------------------------- ##

'''