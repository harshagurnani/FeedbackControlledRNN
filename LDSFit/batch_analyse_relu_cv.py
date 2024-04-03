import torch
import numpy as np
import scipy.linalg as ll
import matplotlib.pyplot as pp
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
sys.path.insert(0, parentdir) 

from LDSFit.lds_helpers import *


allz=[4, 6, 8, 10, 12]
#allz=[10]

train_frac=0.92
maxe1a=300 
maxe2a=400
maxe1b=100
maxe2b=200

iters=10
files=['use_models/relu_/tau5__with_training_seed_90.npy', 
       'use_models/relu_/tau5__with_training_seed_14.npy',
       'use_models/relu_/tau5__with_training_seed_56.npy']

np.random.seed()
torch.random.seed()

if torch.cuda.is_available():
       device = torch.device('cuda', index=1)
else:
       device = torch.device('cpu')

all_loss = []
loss_summary = []
all_cev = []
currentz = []
all_ev = []

device='cpu'
for nz in allz:
    for ff in files:
        for jj in range(iters):
            loss, model_full, all_ut, all_xt, cev = analyse_file_CV_2( file=ff, train_frac=train_frac, nz=nz, maxe1a=maxe1a , maxe2a=maxe2a, maxe1b=maxe1b, maxe2b=maxe2b, device=device)
            all_loss.append( loss )
            all_cev.append( cev )
            loss_summary.append(loss[1][-1])  
            e,v  = ll.eig(model_full.A.cpu().detach().numpy())
            all_ev.append(e)
            currentz.append(nz)

            dic = {'files':files, 'cev':all_cev, 'ev':all_ev, 'lc':all_loss, 'loss_final':loss_summary, 'nz':currentz, 'train_frac':train_frac,
                     'maxe1a':maxe1a, 'maxe2a':maxe2a, 'maxe1b':maxe1b, 'maxe2b':maxe2b }
            np.save( 'relu_iters_ldsAll_cv2', dic)



print(all_cev)


import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)

ids = [ jj  for jj in range(len(all_cev)) if (all_cev[jj]>0.9 and currentz[jj]==nz) ]
fig = pp.figure(figsize=(6,6))
ax = fig.add_subplot(111)
for jj in ids:
    ax.scatter(np.real(all_ev[jj]), np.imag(all_ev[jj]), color='m',alpha=0.5)

ax.plot([0,0],[-1.3,1.3],'k--')
cc=pp.Circle((-1,0), 1, color='b', fill=False)
ax.add_patch(cc)
pp.savefig('ev_relu_iters_lds12_concat_cv2.png')
pp.close()

'''
a = np.load(file,allow_pickle=True).item()
all_cev=a['cev']
all_ev=a['ev']
currentz=a['nz']

'''


'''
allf=['/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds4_cv.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds4_cv2.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds6_cv.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds6_cv2.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds8_cv.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds8_cv2.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds10_cv.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds10_cv2.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds12_cv.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds12_cv2.npy',
       '/home/hg84/Documents/Github/FeedbackLearning/LDSFit/relu_iters_lds14_cv.npy']

       
cev_summary = np.zeros(0,)
nz_summary = np.zeros(0,)

for file in allf:
       a = np.load(file,allow_pickle=True).item()
       all_cev=a['cev']
       all_ev=a['ev']
       currentz=a['nz']
       cev_summary = np.concatenate((cev_summary, all_cev), axis=0)
       nz_summary = np.concatenate((nz_summary, currentz), axis=0)

       
import seaborn as sns
import pandas as pd
df = pd.DataFrame({'Group': nz_summary, 'Data': cev_summary})


pp.figure()
sns.violinplot(x='Group', y='Data', data=df, density_norm='width')
pp.xlabel('nZ')
pp.ylabel('cev')
pp.savefig('LDSfit_summary_cev.png')
pp.close()

# Calculate median and percentile values for each group
summary_df = df.groupby('Group')['Data'].describe(percentiles=[0.05, 0.5, 0.95])

# Extract relevant statistics
medians = summary_df['50%'].values
q5 = summary_df['5%'].values
q95 = summary_df['95%'].values
groups = summary_df.index.values

pp.figure()
pp.bar(groups, q95 - q5, bottom=q5, color='blue', alpha=0.3, label='90% Interval')
pp.hlines(medians, xmin=groups - 0.4, xmax=groups + 0.4, color='red', linewidth=2, label='Median')
pp.xlabel('nZ')
pp.ylabel('cev')
pp.xticks(np.unique(nz_summary))
pp.savefig('LDSfit_summary_cev2.png')
pp.close()
'''