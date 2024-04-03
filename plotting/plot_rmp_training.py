import numpy as np
import os, glob
import matplotlib.pyplot as pp
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)

import tools.pert_analyses as pa
import plotting.plot_pert_training as pmt
from sklearn.linear_model import LinearRegression

import pandas as pd
import seaborn as sns


#########################################################

### REGRESSION FUNCTIONS ---------------------------------

def fit_linear_speed( *args, savfolder='', idy='log_speed', id1='ctrb_fb', id2='fracVF', suffix='', maxy=40, fracTrain=0.8 , logy=True, clampy=False, clrs=['r','b','k'], cmaps=['Reds','Blues', 'Greys'] ):

    all_dic = args
    ndic = len(args)
    #print(ndic)
    #print(all_dic[0].keys())

    # concatenate data
    y = all_dic[0][idy]
    x1 = np.log10(all_dic[0][id1])
    x2 = np.log10(all_dic[0][id2])
    for jj in range(1, ndic):
        y = np.vstack((y, all_dic[jj][idy] ))
        x1 = np.vstack((x1, np.log10(all_dic[jj][id1])))
        x2 = np.vstack((x2, np.log10(all_dic[jj][id2])))
    y = np.reshape(y, (len(y)))
    x1 = np.reshape(x1, (len(x1)))
    x2 = np.reshape(x2, (len(x2)))

    X = np.vstack((x1,x2)).T
    nX = len(X)


    # Run regression
    trainid = np.random.choice(np.arange(nX), np.int_(np.floor(fracTrain*nX)), replace=False)
    testid = np.arange(nX)
    keepid = [xx not in trainid for xx in testid]
    testid = testid[keepid]
    trainX = X[trainid,:]
    trainY = y[trainid]
    testX = X[testid,:]
    testY = y[testid]

    reg = LinearRegression().fit(trainX,trainY)
    print( reg.score(testX, testY))
    print(reg.coef_)
    ypred = reg.predict(testX)
    if clampy:
        ymax = max(y)
        ypred[ypred>ymax]=ymax
    print( 'explained var = '+np.str_(1 - np.mean(np.power(testY-ypred,2*np.ones(testY.shape)))/np.var(testY)))

    f1 = pp.figure()
    if logy:
        pp.plot([0,10**maxy],[0,10**maxy],color='k')
    else:
        miny = -0.3
        pp.plot([miny,maxy],[miny,maxy],color='k')

    for jj in range(ndic):
        nx1 = len(np.log10(all_dic[jj][id1]))
        Xpart = np.concatenate(( np.reshape(np.log10(all_dic[jj][id1]), (nx1,1)),  np.reshape(np.log10(all_dic[jj][id2]), (nx1,1)) ), axis=1)
        ypart = all_dic[jj][idy]
        ypart = np.reshape(ypart, (len(ypart),1))
        zpart = reg.predict(Xpart)
        zpart = np.reshape(zpart, (len(zpart),1))
        if clampy:
            zpart[zpart>ymax]=ymax

        data1 = pd.DataFrame( np.concatenate((ypart,zpart), axis=1) , columns=['true','pred'])
        if logy:
            data1=10**data1

        sns.kdeplot( data1, x='true', y='pred', cmap=cmaps[jj], fill=True, alpha=0.4)

        if logy:
            pp.scatter(10**ypart, 10**zpart,c=clrs[jj], alpha=0.2)
            pp.xlim([0,10**maxy])
            pp.ylim([0,10**maxy])
        else:
            pp.scatter(ypart, zpart, c=clrs[jj], alpha=0.2)
            #pp.xlim([miny,maxy])
            #pp.ylim([miny,maxy])
        
    pp.xlabel('true')
    pp.ylabel('predicted')
    pp.savefig(savfolder+'regression_fit'+suffix+'.svg')
    pp.close(f1)





def fit_linear_speed_noexp2( *args, savfolder='', idy='log_speed', id1='ctrb_fb', id2='fracVF', suffix='' , maxy=40, fracTrain=0.8, logy=True, clampy=False, ymax=1, clrs=['r','b','k'], cmaps=['Reds','Blues', 'Greys'] ):
    all_dic = args
    ndic = len(args)

    # concatenate data
    y = all_dic[0][idy]
    x1 = np.log10(all_dic[0][id1])
    x2 = all_dic[0][id2]
    for jj in range(1, ndic):
        y = np.vstack((y, all_dic[jj][idy] ))
        x1 = np.vstack((x1, np.log10(all_dic[jj][id1])))
        x2 = np.vstack((x2, all_dic[jj][id2]))              # not log
    y = np.reshape(y, (len(y)))
    x1 = np.reshape(x1, (len(x1)))
    x2 = np.reshape(x2, (len(x2)))

    X = np.vstack((x1,x2)).T
    nX = len(X)


    # Run regression
    trainid = np.random.choice(np.arange(nX), np.int_(np.floor(fracTrain*nX)), replace=False)
    testid = np.arange(nX)
    keepid = [xx not in trainid for xx in testid]
    testid = testid[keepid]
    trainX = X[trainid,:]
    trainY = y[trainid]
    testX = X[testid,:]
    testY = y[testid]

    reg = LinearRegression().fit(trainX,trainY)
    print( reg.score(testX, testY))
    print(reg.coef_)
    ypred = reg.predict(testX)
    if clampy:
        ymax = max(y)
        ypred[ypred>ymax]=ymax
    print( 'explained var = '+np.str_(1 - np.mean(np.power(testY-ypred,2*np.ones(testY.shape)))/np.var(testY)))


    f1 = pp.figure()
    if logy:
        pp.plot([0,10**maxy],[0,10**maxy],color='k')
    else:
        miny = -0.3
        pp.plot([miny,maxy],[miny,maxy],color='k')

    for jj in range(ndic):
        nx1 = len(np.log10(all_dic[jj][id1]))
        Xpart = np.concatenate(( np.reshape(np.log10(all_dic[jj][id1]), (nx1,1)),  np.reshape(all_dic[jj][id2], (nx1,1)) ), axis=1)
        ypart = all_dic[jj][idy]
        ypart = np.reshape(ypart, (len(ypart),1))
        zpart = reg.predict(Xpart)
        zpart = np.reshape(zpart, (len(zpart),1))
        if clampy:
            zpart[zpart>ymax]=ymax

        data1 = pd.DataFrame( np.concatenate((ypart,zpart), axis=1) , columns=['true','pred'])
        if logy:
            data1=10**data1

        sns.kdeplot( data1, x='true', y='pred', cmap=cmaps[jj], fill=True, alpha=0.4)

        if logy:
            pp.scatter(10**ypart, 10**zpart,c=clrs[jj], alpha=0.2)
            pp.xlim([0,10**maxy])
            pp.ylim([0,10**maxy])
        else:
            pp.scatter(ypart, zpart, c=clrs[jj], alpha=0.2)
            #pp.xlim([miny,maxy])
            #pp.ylim([miny,maxy])
        
    pp.xlabel('true')
    pp.ylabel('predicted')
    pp.savefig(savfolder+'regression_fit'+suffix+'.svg')
    pp.close(f1)

#########################################################
    




path='/home/hg84/Documents/Github/FeedbackLearning/'
# Set up folders
### -------------------------------------------- ###


subf = 'percp2_expansion_/'         # Main top-level architecture (eg in use_models)
save_suffix ='_fbOut_All_'
# wt change for feedback:
use_fbwt= 'output.weight'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = 'fbk_p'### 'fbk_p'#None# 
percp=True

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   = ['trained_wmp_trainFbOut200_.npy', 'trained_wmptrain_FbOut200_run3_.npy',  'trained_wmptrain_FbOut200_run4_.npy'] #['trained_wmptrain_FbOut200_run3_.npy', 'trained_wmptrain_FbOut200_run4_.npy']# ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy']#,  'trained_wmp_trainFbk_run2_.npy']  #['trained_wmp_trainFbk200_.npy']#, 'trained_wmp_trainFbk_run2_.npy']        # all possible files to combine   #'trained_wmp_trainFbk_rep_.npy'['trained_wmp_trainFbOut200_.npy']#  'trained_wmp_trainFbk_run2_.npy',['trained_wmp_trainFbOut200_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fbOut_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']=  ['trained_omp_trainFbOut200_.npy', 'trained_omp_FbOut200_run2_.npy', 'trained_omp_FbOut200_run4_.npy']#['trained_omp_FbOut200_run2_.npy', 'trained_omp_FbOut200_run4_.npy']#['trained_omp_trainFb_.npy']#['trained_omp_trainFb_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fbOut_/'
npro=True

joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fbOut_logspeed_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_'
clrs = ['r','b']        # WMP and OMP color respectively


# ---------------------------------------------- ##
# ---------------------------------------------- ##
# ---------------------------------------------- ##
# ---------------------------------------------- ##
# ---------------------------------------------- ##

path='/home/hg84/Documents/Github/FeedbackLearning/'
subf = 'relu_/'         # Main top-level architecture (eg in use_models)
save_suffix ='fb_All_'
# wt change for feedback:
use_fbwt= 'W_fbk_0'##'W_fbk_0'#'output.weight' #'W_fbk_0'#
use_mod = None### 'fbk_p'#None# 
percp=False

wmp={}                  # FOR WMPs
wmp['folder'] = path+'wmp/'+subf
wmp['file']   =  ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy',  'trained_wmp_trainFbk_run2_.npy'] #['trained_wmp_trainRec200_.npy']# ['trained_wmp_trainFbk200_.npy',  'trained_wmp_trainFbk200_run3_.npy',  'trained_wmp_trainFbk_run2_.npy']  #['trained_wmp_trainFbk200_.npy']#, 'trained_wmp_trainFbk_run2_.npy']        # all possible files to combine   #'trained_wmp_trainFbk_rep_.npy'['trained_wmp_trainFbOut200_.npy']#  'trained_wmp_trainFbk_run2_.npy',['trained_wmp_trainFbOut200_.npy']
wmp['save_suffix']=save_suffix                          # plot suffix
wmp['suffix']='_PC8/'                                   # loading directory suffix
wmp['savfolder'] = wmp['folder']+'plots_fb_new_/'               # save directory
npro=True

omp={}                  # FOR OMPs
omp['folder'] = path+'omp/'+subf
omp['file']= ['trained_omp_trainFbk200_.npy', 'trained_omp_trainFbk200_run3_.npy']# ['trained_omp_trainRec200_.npy']#['trained_omp_trainFbk200_.npy', 'trained_omp_trainFbk200_run3_.npy']  #['trained_omp_trainFb_.npy']#['trained_omp_trainFbk200_.npy']
omp['save_suffix']=save_suffix
omp['suffix']='_PC8/'
omp['savfolder'] = omp['folder']+'plots_fb_new_/'
npro=True

rmp={}                  # FOR RMPs
rmp['folder'] = path+'rmp/'+subf
rmp['file']   =  ['trained_rmp_test.npy', 'trained_rmp_run2_.npy', 'trained_rmp_run3_.npy', 'trained_rmp_run4_.npy']
rmp['save_suffix']=save_suffix                          # plot suffix
rmp['suffix']='_PC8/'                                   # loading directory suffix
rmp['savfolder'] = rmp['folder']+'plots_fb_new_/'               # save directory
npro=True


joint={}                # FOR COMBINED DATA
joint['savfolder'] = omp['folder']+'plots_combined_fb_All_logspeed_rmp_/' 
joint['save_suffix']=save_suffix+'combined_fbAll_'
clrs = ['r','b', 'k']        # WMP and OMP color respectively


# ---------------------------------------------- ##
# ---------------------------------------------- ##







# ------- WMP ------------- #
dic = wmp
print(dic['savfolder'])


allfolders = glob.glob(dic['folder']+'Model_*_PC8')     # all models
res = []
# collect each training result
for ff in range(len(allfolders)):
    model = allfolders[ff]+'/'
    for file in dic['file']:
        fname = model+file
        if os.path.isfile(fname):
            res.append( pa.plot_training_results(file=file,folder=model, savfolder=model, hit_thresh=0.1, new_progress =npro, use_mod=use_mod, use_fbwt=use_fbwt, percp=percp) )
# combine all results
allres = res[0].copy()
for key in res[0].keys():
    for ff in range(len(res)-1):
        allres[key] = np.vstack((allres[key], res[ff+1][key]))

if not os.path.isdir(dic['savfolder']):
    os.makedirs(dic['savfolder'])

pmt.plot_all(allres, savfolder=dic['savfolder'], suffix=dic['save_suffix'])

# ------- OMP ------------- #

dic = omp
print(dic['savfolder'])
allfolders = glob.glob(dic['folder']+'Model_*_PC8')
res = []
for ff in range(len(allfolders)):
    model = allfolders[ff]+'/'
    for file in dic['file']:
        fname = model+file
        if os.path.isfile(fname):
            res.append( pa.plot_training_results(file=file,folder=model, savfolder=model, hit_thresh=0.1, new_progress =npro, use_mod=use_mod, use_fbwt=use_fbwt, percp=percp ) )


allres2 = res[0].copy()
for key in res[0].keys():
    for ff in range(len(res)-1):
        allres2[key] = np.vstack((allres2[key], res[ff+1][key]))


if not os.path.isdir(dic['savfolder']):
    os.makedirs(dic['savfolder'])

pmt.plot_all(allres2, savfolder=dic['savfolder'], suffix=dic['save_suffix'])



# ------- RMP ------------- #
dic = rmp
print(dic['savfolder'])


allfolders = glob.glob(dic['folder']+'Model_*_PC8')     # all models
res = []
# collect each training result
for ff in range(len(allfolders)):
    model = allfolders[ff]+'/'
    for file in dic['file']:
        fname = model+file
        if os.path.isfile(fname):
            res.append( pa.plot_training_results(file=file,folder=model, savfolder=model, hit_thresh=0.1, new_progress =npro, use_mod=use_mod, use_fbwt=use_fbwt, percp=percp) )
# combine all results
allres3 = res[0].copy()
for key in res[0].keys():
    for ff in range(len(res)-1):
        allres3[key] = np.vstack((allres3[key], res[ff+1][key]))

if not os.path.isdir(dic['savfolder']):
    os.makedirs(dic['savfolder'])

pmt.plot_all(allres3, savfolder=dic['savfolder'], suffix=dic['save_suffix'])


##### JOINT

dic = joint
if not os.path.isdir(dic['savfolder']):
    os.makedirs(dic['savfolder'])

savfolder = dic['savfolder']
print(savfolder)

svdic = {'allres':allres, 'allres2':allres2, 'allres3':allres3, 'dic':dic }
np.save(savfolder+'analyses_'+save_suffix, svdic)

pmt.plot_progress_2(allres, allres2, allres3,       savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_activity_norm_2(allres, allres2, allres3,  savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_asymmetry_2(allres, allres2, allres3,      savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_expvar_2(allres, allres2, allres3,         savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)#
pmt.plot_expvar_hist(allres, allres2, allres3,         savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)#
pmt.plot_hitrate_change_2(allres, allres2, allres3, savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_hitrate_input(allres, allres2, allres3,    savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_speed_input(allres, allres2, allres3,      savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_learning_speed_2(allres, allres2, allres3,  savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_speed_controls(allres,allres2, allres3,     savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_grads(allres,allres2, allres3,     savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)




# ---------------------
alpha = np.vstack((allres['obsv'], allres2['obsv'], allres3['obsv']))
allff = np.vstack((allres['ctrb_ff'], allres2['ctrb_ff'], allres3['ctrb_ff']))
allfb = np.vstack((allres['ctrb_fb'], allres2['ctrb_fb'], allres3['ctrb_fb']))
speed = np.vstack((allres['speed'], allres2['speed'], allres3['speed']))


useid =(speed >0) & (np.log10(allff)<6) & (np.log10(allfb)<6)
print('Observability vs speed:')
print(np.corrcoef(np.log10(alpha[useid].T),speed[useid].T))

print('FF controllability vs speed:')
print(np.corrcoef(np.log10(allff[useid].T),speed[useid].T))

print('FB controllability vs speed:')
print(np.corrcoef(np.log10(allfb[useid].T),speed[useid].T))

# ---------------------


pmt.plot_speed_vectorfld(allres,allres2, allres3,     savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_loss_speed_controls(allres,allres2, allres3,    savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)

pmt.plot_var_dyn(allres,allres2, allres3,    savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_success_dyn(allres,allres2, allres3,    savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
#pnt.plot_hitrate_change_2(allres, allres2,  savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)
pmt.plot_ctrb_U(allres,allres2,  allres3,   savfolder=savfolder, suffix=dic['save_suffix'], clrs=clrs)




#### -------------------------------------------------

allres['expVar_pre'][np.isnan(allres['expVar_pre'])]=1
allres2['expVar_pre'][np.isnan(allres2['expVar_pre'])]=1
allres3['expVar_pre'][np.isnan(allres3['expVar_pre'])]=1
allres2['expVar_post'][np.isnan(allres2['expVar_post'])]=1
allres['expVar_post'][np.isnan(allres['expVar_post'])]=1
allres3['expVar_post'][np.isnan(allres3['expVar_post'])]=1

allres['expVar_diff'] = allres['expVar_post']  - allres['expVar_pre']
allres2['expVar_diff'] = allres2['expVar_post']  - allres2['expVar_pre']
allres3['expVar_diff'] = allres3['expVar_post']  - allres3['expVar_pre']

allres['log_speed'] = np.log10(allres['speed'])
allres2['log_speed'] = np.log10(allres2['speed'])
allres3['log_speed'] = np.log10(allres3['speed'])



#print(allres['fracVF_fb'])

fracTrain=0.8
maxy=np.log10(75)
np.random.seed()

print('speed versus ctrb_fb and fracVF:')
fit_linear_speed(allres, allres2, allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='fracVF', maxy=maxy, fracTrain=fracTrain )
print('speed versus ctrb_ff and fracVF:')
fit_linear_speed(allres, allres2, allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_ff', id2='fracVF', maxy=maxy, fracTrain=fracTrain , suffix='_ctrb_ff')

allres['fracVF_fb_inv'] = 1/allres['fracVF_fb']
allres2['fracVF_fb_inv'] = 1/allres2['fracVF_fb']
allres3['fracVF_fb_inv'] = 1/allres3['fracVF_fb']
allres['fracVF_inv'] = 1/allres['fracVF']
allres2['fracVF_inv'] = 1/allres2['fracVF']
allres3['fracVF_inv'] = 1/allres3['fracVF']
print('speed versus ctrb_fb and 1/fracVF:')
fit_linear_speed_noexp2(allres, allres2,allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='fracVF_inv', maxy=maxy, fracTrain=fracTrain, suffix='_fracVF_inv' )
print('speed versus ctrb_fb and 1/fracVF_fb:')
fit_linear_speed_noexp2(allres, allres2,allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='fracVF_fb_inv', suffix='_fracVF_fb_inv', maxy=maxy, fracTrain=fracTrain )



allres['fracVF_fb_100'] = 100 * allres['fracVF_fb']
allres2['fracVF_fb_100'] = 100 * allres2['fracVF_fb']
allres3['fracVF_fb_100'] = 100 * allres3['fracVF_fb']
print('speed versus ctrb_fb and fracVF_fb:')
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='fracVF_fb', suffix='_fracVF_fb', maxy=maxy, fracTrain=fracTrain )
#fit_linear_speed_noexp2(allres, allres2,savfolder, idy='log_speed',  id1='fracVF_fb', id2='ctrb_fb', suffix='_ctrb_fb', maxy=maxy, fracTrain=fracTrain )

print('speed versus ctrb_fb and hit rate:')
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='hit_rate_pre', suffix='_hit_rate_' , maxy=maxy, fracTrain=fracTrain)
#print('speed versus ctrb_fb and expVar_dff:')
#fit_linear_speed_noexp2(allres, allres2,allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='expVar_diff', suffix='_exp_Var_', maxy=maxy , fracTrain=fracTrain)


##############################

pp.figure()
pp.hist(allres['fracTotalVel'], color='r', alpha=0.3)
pp.hist(allres2['fracTotalVel'], color='b', alpha=0.3)
pp.hist(allres3['fracTotalVel'], color='k', alpha=0.3)
m1 = np.median(allres['fracTotalVel'])
m2 = np.median(allres2['fracTotalVel'])
m3 = np.median(allres3['fracTotalVel'])
pp.plot([m1,m1],[0,65], 'r')
pp.plot([m2,m2],[0,65], 'b')
pp.plot([m3,m3],[0,65], 'k')
pp.xlabel('ratio of decoder projected variance')
pp.xticks(np.arange(max(allres2['fracTotalVel'])))
pp.savefig(savfolder+'fracTotalVel.svg')
pp.close()



pp.figure()

pp.violinplot( [allres2['fracTotalVel'].T[0]],[2], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [allres['fracTotalVel'].T[0]],[1], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [allres3['fracTotalVel'].T[0]],[3], showmedians=True, quantiles=[0.05,0.95])

#pp.hist(allres['fracTotalVel'], color='r', alpha=0.5)
#pp.hist(allres2['fracTotalVel'], color='b', alpha=0.5)
#m1 = np.median(allres['fracTotalVel'])
#m2 = np.median(allres2['fracTotalVel'])
#pp.plot([m1,m1],[0,65], 'r')
#pp.plot([m2,m2],[0,65], 'b')
pp.plot([0.5,3.5],[1,1],color='k')
pp.ylabel('ratio of decoder projected variance')
#pp.xticks(np.arange(max(allres2['fracTotalVel'])))
pp.savefig(savfolder+'fracTotalVel_violin.svg')
pp.close()


pp.figure()
dvar = allres['fracTotalVel']*allres['expVar_decoder']/allres['expVar_post']
dvar2 = allres2['fracTotalVel']*allres2['expVar_decoder']/allres2['expVar_post']
dvar3 = allres3['fracTotalVel']*allres3['expVar_decoder']/allres3['expVar_post']
pp.violinplot( [dvar2.T[0]],[2], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [dvar.T[0]],[1], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [dvar3.T[0]],[3], showmedians=True, quantiles=[0.05,0.95])
pp.ylabel('decoder projected variance (fraction)')
pp.ylim([0,1])
#pp.xticks(np.arange(max(allres2['fracTotalVel'])))
pp.savefig(savfolder+'fracTotalVel_fractionU.svg')
pp.close()

pp.figure()

pp.violinplot( [allres2['fracVar_post'].T[0]],[2], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [allres['fracVar_post'].T[0]],[1], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [allres3['fracVar_post'].T[0]],[3], showmedians=True, quantiles=[0.05,0.95])
pp.ylabel('decoder projected variance (fraction)')
pp.ylim([0,1])
#pp.xticks(np.arange(max(allres2['fracTotalVel'])))
pp.savefig(savfolder+'fracTotalVel_fractionX.svg')
pp.close()



pp.figure()
pp.violinplot( [allres2['fracVar_pre'].T[0]],[2], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [allres['fracVar_pre'].T[0]],[1], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [allres3['fracVar_pre'].T[0]],[3], showmedians=True, quantiles=[0.05,0.95])
pp.ylabel('decoder projected variance (fraction)')
#pp.xticks(np.arange(max(allres2['fracTotalVel'])))
pp.ylim([0,1])
pp.savefig(savfolder+'fracTotalVel_fractionX_pre.svg')
pp.close()



pp.figure()
pp.scatter(allres['Fbk_overlap'], allres['Out_overlap'], color='r', alpha=0.5)
pp.scatter(allres2['Fbk_overlap'], allres2['Out_overlap'], color='b', alpha=0.5)
pp.scatter(allres3['Fbk_overlap'], allres3['Out_overlap'], color='k', alpha=0.5)
pp.xlabel('Fbk overlap w RSV')
pp.ylabel('Wout overlap w LSV')
pp.savefig(savfolder+'overlap.svg')
pp.close()

#print(allres['Fbk_overlap'])



#############################################



print('speed versus ctrb_fb and out overlap:')
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='Out_overlap', suffix='_overlap_', maxy=maxy , fracTrain=fracTrain)
print('speed versus ctrb_fb and fbk overlap:')
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='log_speed',  id1='ctrb_fb', id2='Fbk_overlap', suffix='_fbk_overlap_', maxy=maxy , fracTrain=fracTrain)


maxy2=0.2
fit_linear_speed(allres, allres2, allres3, savfolder=savfolder, idy='fitted_k',  id1='ctrb_fb', id2='fracVF', maxy=maxy2, suffix='_k', fracTrain=fracTrain )
fit_linear_speed_noexp2(allres, allres2,allres3, savfolder=savfolder, idy='fitted_k',  id1='ctrb_fb', id2='fracVF_fb', suffix='_k_fracVF_fb', maxy=maxy2, fracTrain=fracTrain )
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='fitted_k',  id1='ctrb_fb', id2='hit_rate_pre', suffix='_k_hit_rate_' , maxy=maxy2, fracTrain=fracTrain)
#fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='fitted_k',  id1='ctrb_fb', id2='expVar_diff', suffix='_k_exp_Var_', maxy=maxy2 , fracTrain=fracTrain)
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='fitted_k',  id1='ctrb_fb', id2='Out_overlap', suffix='_k_overlap_', maxy=maxy2 , fracTrain=fracTrain)
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='fitted_k',  id1='ctrb_fb', id2='Fbk_overlap', suffix='_k_fbk_overlap_', maxy=maxy2 , fracTrain=fracTrain)





pp.figure()
pp.hist(allres['dimVF'], color='r', alpha=0.3)
pp.hist(allres2['dimVF'], color='b', alpha=0.3)
pp.hist(allres3['dimVF'], color='k', alpha=0.3)
pp.xlabel('rDim of vector field change')
pp.savefig(savfolder+'dim_vf'+save_suffix+'.svg')
pp.close()

#############################################


allres['change_hit'] = (allres['hit_rate_post'] - allres['hit_rate_pre'])/(1- allres['hit_rate_pre'])
allres2['change_hit'] = (allres2['hit_rate_post'] - allres2['hit_rate_pre'])/(1- allres2['hit_rate_pre'])
allres3['change_hit'] = (allres3['hit_rate_post'] - allres3['hit_rate_pre'])/(1- allres3['hit_rate_pre'])

maxy3=1
print('change_hit versus ctrb_fb and ctrb_ff:')
fit_linear_speed(allres, allres2, allres3, savfolder=savfolder, idy='change_hit',  id1='ctrb_fb', id2='ctrb_ff', maxy=maxy3, fracTrain=fracTrain, suffix='_changehit_ctrb_' , logy=False, clampy=True)
print('change_hit versus ctrb_fb and fracVF:')
fit_linear_speed(allres, allres2, allres3, savfolder=savfolder, idy='change_hit',  id1='ctrb_fb', id2='fracVF', maxy=maxy3, fracTrain=fracTrain, suffix='_changehit_ctrb_fracVF_' , logy=False , clampy=True)
print('change_hit versus fracVF and expVar along decoder:')
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='change_hit',  id1='fracVF', id2='expVar_decoder', maxy=maxy3, fracTrain=fracTrain, suffix='_changehit_fracVF_evdecoder_' , logy=False , clampy=True)
print('change_hit versus fracVF and angle with IM:')
fit_linear_speed_noexp2(allres, allres2, allres3, savfolder=savfolder, idy='change_hit',  id1='fracVF', id2='theta_intuit_wperturb', maxy=maxy3, fracTrain=fracTrain, suffix='_changehit_fracVF_evim_' , logy=False , clampy=True)
print('change_hit versus ctrb_fb and fracVF_fb:')
fit_linear_speed(allres, allres2, allres3, savfolder=savfolder, idy='change_hit',  id1='ctrb_fb', id2='fracVF_fb', maxy=maxy3, fracTrain=fracTrain, suffix='_changehit_ctrb_fracVFfb_' , logy=False , clampy=True)


pp.figure()
pre1=[jj[0] for jj in allres['progress_asymm_pre'] if not np.isnan(jj)]
pre2=[jj[0] for jj in allres2['progress_asymm_pre'] if not np.isnan(jj)]
pre3=[jj[0] for jj in allres3['progress_asymm_pre'] if not np.isnan(jj)]
post1=[jj[0] for jj in allres['progress_asymm_post'] if not np.isnan(jj)]
post2=[jj[0] for jj in allres2['progress_asymm_post'] if not np.isnan(jj)]
post3=[jj[0] for jj in allres3['progress_asymm_post'] if not np.isnan(jj)]
pp.violinplot( [pre1],[1], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [pre2],[2], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [pre3],[3], showmedians=True, quantiles=[0.05,0.95])

pp.violinplot( [post1],[5], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [post2],[6], showmedians=True, quantiles=[0.05,0.95])
pp.violinplot( [post3],[7], showmedians=True, quantiles=[0.05,0.95])
pp.ylabel('Progress asymmetry')
pp.savefig(savfolder+'asymm_progress_violin.svg')
pp.close()