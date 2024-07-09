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
plotdir = parentdir+'saved_plots/'

from LDSFit.lds_helpers import *


nz= 8
noPrep = False
Fbk = True

train_frac=0.92
maxe1a=300 
maxe2a=400
maxe1b=100
maxe2b=200

iters=1
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
        np.save( 'relu_iters_ldsAll_cv2', dic)

print(all_cev)


import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)

ids = [ jj  for jj in range(len(all_cev)) if (all_cev[jj]>0.8 and currentz[jj]==nz) ]
fig = pp.figure(figsize=(6,6))
ax = fig.add_subplot(111)
for jj in ids:
    ax.scatter(np.real(all_ev[jj]), np.imag(all_ev[jj]), color='m',alpha=0.5)

ax.plot([0,0],[-1.3,1.3],'k--')
cc=pp.Circle((-1,0), 1, color='b', fill=False)
ax.add_patch(cc)
pp.savefig(plotdir+'ev_relu_iters_lds'+np.str_(nz)+'_concat_cv2.png')
pp.close()