#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import scipy.linalg as ll
import matplotlib.pyplot as pp
import glob
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
sys.path.insert(0, parentdir) 
plotdir = parentdir+'saved_plots/LDSFit/'

from LDSFit.lds_helpers import *


allz=[4, 6, 8, 10, 12]
noPrep = False
Fbk = True

train_frac=0.92
maxe1a=300
maxe2a=400
maxe1b=100
maxe2b=200

iters=40
files=None
if files is None:
     files=glob.glob('use_models/relu_/*.npy')

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
            loss, model_full, all_ut, all_xt, cev = analyse_file_CV_2( file=ff, noPrep=noPrep, fbk=Fbk, train_frac=train_frac, nz=nz, maxe1a=maxe1a , maxe2a=maxe2a, maxe1b=maxe1b, maxe2b=maxe2b, device=device)
            all_loss.append( loss )
            all_cev.append( cev )
            loss_summary.append(loss[1][-1])  
            e,v  = ll.eig(model_full.A.cpu().detach().numpy())
            all_ev.append(e)
            currentz.append(nz)

            dic = {'files':files, 'cev':all_cev, 'ev':all_ev, 'lc':all_loss, 'loss_final':loss_summary, 'nz':currentz, 'train_frac':train_frac,
                     'maxe1a':maxe1a, 'maxe2a':maxe2a, 'maxe1b':maxe1b, 'maxe2b':maxe2b }
            np.save( 'relu_iters_'+np.str_(iters)+'_ldsAll_cv2', dic)



print(all_cev)

all_cev = np.array(all_cev)
currentz = np.array(currentz)


###############################################################################
# PLOTTING RESULTS
##############################################################################


import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)
import seaborn as sns
import pandas as pd

nz=8
min_cev=0.7


########################################
# Plot for example nz=8
########################################



ids = [ jj  for jj in range(len(all_cev)) if (all_cev[jj]>min_cev and currentz[jj]==nz) ]
fig = pp.figure(figsize=(6,6))
ax = fig.add_subplot(111)
for jj in ids:
    ax.scatter(np.real(all_ev[jj]), np.imag(all_ev[jj]), color='m',alpha=0.5)

ax.plot([0,0],[-1.3,1.3],'k--')
cc=pp.Circle((-1,0), 1, color='b', fill=False)
ax.add_patch(cc)
pp.savefig('ev_relu_noPrep_iters_lds12_concat_cv2.png')
pp.close()


########################################
# Plot summary
########################################

df = pd.DataFrame({'Group': currentz, 'Data': all_cev})

pp.figure()
sns.violinplot(x='Group', y='Data', data=df, density_norm='width')
pp.xlabel('nZ')
pp.ylabel('cev')
pp.savefig(plotdir+'LDSfit_summary_cev.png')
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
pp.xticks(np.unique(currentz))
pp.savefig(plotdir+'LDSfit_summary_cev2.png')
pp.close()

########################################
# Plot Rotational EV
########################################

idx=np.intersect1d( np.where(all_cev>min_cev)[0], np.where(currentz==nz)[0] )

rot_ev_all = []

for ii in idx:
       ee = all_ev[ii]
       imaginary_part = np.imag(ee)
       real_part = np.real(ee)
       nonzero_imaginary_indices = np.nonzero(imaginary_part)[0]
       if len(nonzero_imaginary_indices) > 0:
              slowest_eigenvalue_index = np.argmax(real_part[imaginary_part>0])
              slowest_eigenvalue = ee[nonzero_imaginary_indices[slowest_eigenvalue_index]]
              rot_ev_all.append(slowest_eigenvalue)

rot_ev_all = np.array(rot_ev_all)
pp.figure()
pp.hist(np.real(rot_ev_all), bins=np.linspace(-1, 1, 20), color='r', alpha=0.3, label='Real')
pp.hist(np.imag(rot_ev_all), bins=np.linspace(-1, 1, 20), color='b', alpha=0.3, label='Imaginary')
pp.savefig(plotdir+'LDSfit_summary_rot_ev.png')
pp.close()

pp.figure()
# Convert eigenvlues to decay timescale and rotational frequency
sampling_time = 0.1
decay_timescale = -1 / np.real(rot_ev_all)
rotational_frequency = np.abs(np.imag(rot_ev_all) / (2 * np.pi * sampling_time))

# Plot decay timescale
pp.hist(decay_timescale, bins=np.linspace(0, 10, 20), color='b', alpha=0.3, label='Decay Timescale')
pp.xlabel('Decay Timescale')
pp.ylabel('Frequency')
pp.savefig(plotdir+'LDSfit_summary_decay_timescale.png')
pp.close()

# Plot rotational frequency
pp.hist(rotational_frequency, bins=np.linspace(-5, 5, 20), color='g', alpha=0.3, label='Rotational Frequency')
pp.xlabel('Rotational Frequency')
pp.ylabel('Frequency')
pp.savefig(plotdir+'LDSfit_summary_rotational_frequency.png')
pp.close()