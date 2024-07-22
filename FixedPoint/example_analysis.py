#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys, os
homedir = str(Path(__file__).resolve().parent.parent)
os.chdir(homedir)
sys.path.append(homedir)

######################################################

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as pp
import scipy.linalg as ll

from FixedPoint.model_fp import RNN_simple
from FixedPoint.analyzer_joint import FixedPoint_Joint as FixedPoint


import tools.perturb_network_ as pn

from sklearn.decomposition import PCA
import scipy.stats as stats
import matplotlib as mpl

cmap='viridis'
cm = mpl.colormaps[cmap]

######################################################
# GLOBAL SETTINGS
######################################################
sv= homedir + '/use_models/relu_/'
fn='tau5__with_training_seed_6.npy'
plotdir =  homedir + '/saved_plots/FPS/'
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

dtype=torch.FloatTensor
if torch.cuda.is_available():
        device = torch.device('cuda', index=0)
else:
    device = torch.device('cpu')

# load model data
oldm= np.load(sv+fn,allow_pickle=True).item()

pmodel = pn.postModel( sv+fn, {'jump_amp':0.0})
pmodel.load_rnn()
dic = pmodel.test_model()

n_in=3
n_out = 2
n_fb=n_out
n_hid=oldm['params1']['W_rec_0'].shape[0]

# recreate network in open loop
model = RNN_simple(n_in=n_in,n_out=2,n_hid=n_hid, device=device).to(device)
p = oldm['params1'].copy()
model.load_parameters(p)

# create fixed point analyzer object
analyzer2 = FixedPoint(model=model, device=device, lr_decay_epoch=5000, speed_tor=5e-06, lambda0=0.1)#, speed_tor=1e-07, gamma=0.1)
N_pts = 1

####################################################
# EXAMPLE FIXED POINT
####################################################


# example position
theta = 45/180* np.pi # in radians

# preparatory period (go-input is off)
input_signal = np.zeros((N_pts,n_in))
input_signal[:,0] = np.cos(theta)
input_signal[:,1] = np.sin(theta)
input_ff = torch.Tensor( input_signal ).type(dtype)

# Hold target is zero
input_signal_tgt = np.zeros((N_pts,n_fb))
input_fb = torch.Tensor( input_signal_tgt ).type(dtype)

hiddenx = ((torch.rand(1, n_hid)-0.5)*0.1).type(dtype)
hiddenp = ((torch.rand(1, n_out)-0.5)*0.1).type(dtype)

# get fixed point
fixed_point1, fixed_point1_pos, _ = analyzer2.find_fixed_point(hiddenx.to(device), hiddenp.to(device), input_ff.to(device), input_fb.to(device) ,view=True)

# check one timestep forward
new_fb_input = fixed_point1_pos - input_fb
v1, z1,r1 = model(input_ff.to(device), new_fb_input.to(device), fixed_point1)
v2, z2,r2 = model(input_ff.to(device), new_fb_input.to(device), z1)

# for stability analysis, we get the entire jacobian matrix and then we can use eigendecomposition to see if there are any positive real eigenvalues
jaci = analyzer2.calc_input_jacobian( fixed_point1, fixed_point1_pos, input_ff[0,:], input_fb[0,:] )
jac = analyzer2.calc_hiddenx_jacobian( fixed_point1, fixed_point1_pos, input_ff[0,:], input_fb[0,:] )
[evs,v]=ll.eig(jac)
maxev=max(np.real(evs))


W_out = oldm['params1']['W_out_0']
# angles with readout directions
#angle_w_cue = np.rad2deg( ll.subspace_angles(jaci[:,:2], W_out) )
#angle_w_hold = np.rad2deg( ll.subspace_angles(jaci[:,2:3], W_out) )
#angle_w_fb = np.rad2deg( ll.subspace_angles(jaci[:,3:], W_out) )


# go period (go-input is ON)
GO = 0.5
input_signal[:,2] = GO
input_ff = torch.Tensor( input_signal ).type(dtype)
input_signal_tgt[:,0] = np.cos(theta)
input_signal_tgt[:,1] = np.sin(theta)
input_fb = torch.Tensor( input_signal_tgt ).type(dtype)

fixed_point2, fixed_point2_pos, _ = analyzer2.find_fixed_point(hiddenx.to(device), hiddenp.to(device), input_ff.to(device), input_fb.to(device) ,view=True)


######################################################
# ALL FIXED POINTS FOR PREPARATORY PERIOD
######################################################


data=dic['res']['activity1']
data = np.reshape(data, (data.shape[0]*data.shape[1],n_hid))
nT=data.shape[0]

n_reps = 3
all_theta = np.linspace(start=0,stop=2*np.pi,num=37)
all_theta = all_theta[:-1]
ntheta = len(all_theta)
all_fps = torch.zeros((n_reps, ntheta, n_hid))
all_fps_pos = torch.zeros((n_reps, ntheta, n_out))
ev_fps = np.zeros((n_reps, ntheta))
is_stable = np.zeros((n_reps, ntheta))
X = np.zeros((n_reps, ntheta,n_hid))

### Preparatory period
result_ok = False
for jj in range(ntheta):
    theta = all_theta[jj]
    input_signal = np.zeros((1,n_in))
    input_signal[:,0] = np.cos(theta)
    input_signal[:,1] = np.sin(theta)
    input_ff = torch.Tensor( input_signal ).type(dtype)
    input_tgt = np.zeros((1,n_out))
    #input_fb[:,0] = np.cos(theta)
    #input_fb[:,1] = np.sin(theta)
    input_tgt = torch.Tensor( input_tgt ).type(dtype)
    for ctr in range(n_reps):
        result_ok = False
        print('Theta = ' + np.str_(np.int_(theta/np.pi*180)) + ', ctr = '+ np.str_(ctr) )
        while not result_ok:
            # different  initialisations - sample from trajectory
            idx = np.random.choice(nT)
            hiddenx = torch.Tensor( data[idx:idx+1,:]).type(dtype)
            initp = ((torch.rand(1, n_out)-0.5)*0.1).type(dtype)
            #hiddenx = ((torch.rand(1, n_hid)-0.5)*0.5*ctr).type(dtype)
            fp, fp_pos, result_ok = analyzer2.find_fixed_point(hiddenx, initp, input_ff, input_tgt, view=True)
            all_fps[ctr,jj,:] = fp
            all_fps_pos[ctr,jj,:] = fp_pos
            jac = analyzer2.calc_hiddenx_jacobian( fp, fp_pos, input_ff, input_tgt)
            [evs,v]=ll.eig(jac)
            ev_fps[ctr,jj]=max(np.real(evs))
            is_stable[ctr,jj] = (ev_fps[ctr,jj]<0)
            fb_input = fp_pos - input_tgt
            _,z,_ = model(input_ff, fb_input, fp )
            X[ctr, jj,:] = z.detach().numpy()




Y = np.reshape(X, (n_reps*ntheta, n_hid))
Y = Y + (-0.5+np.random.random((Y.shape)))*.01 # just jitter a bit so that variance is not zero

nPC=4
px = PCA(n_components=nPC)

new_fps = model.nonlinearity(all_fps).detach().numpy()
new_fps = np.reshape( new_fps, (n_reps*ntheta,n_hid))
Zz = px.fit_transform(new_fps)

#Z1=Zz[ev_fps<0,:]
#Z2=Zz[ev_fps>0,:]

Zz = np.reshape( Zz, (n_reps, ntheta, 4) )

fig=pp.figure()
ax=fig.add_subplot(projection='3d')
for jj in range(n_reps):
    ax.scatter(Zz[jj,:,0], Zz[jj,:,1], Zz[jj,:,2], color=cm(all_theta/(2*np.pi)), alpha=0.5)

ax.view_init(80, 30)
pp.xlabel('PC1')
pp.ylabel('PC2')
pp.savefig(plotdir+'fps_prep.png')
ax.view_init(10, 30)
pp.savefig(plotdir+'fps_prep_view2.png')
pp.close(fig)

# --------

# angles with prep FP subsace
#prep_sub = px.components_

#angle_w_cue = np.rad2deg( ll.subspace_angles(jaci[:,:2], prep_sub.T) )
#angle_w_hold = np.rad2deg( ll.subspace_angles(jaci[:,2:3], prep_sub.T) )
#angle_w_fb = np.rad2deg( ll.subspace_angles(jaci[:,3:], prep_sub.T) )




######################################################
# ALL FIXED POINTS FOR MOVEMENT PERIOD
######################################################


data=dic['res']['activity1']
data = np.reshape(data, (data.shape[0]*data.shape[1],n_hid))
nT=data.shape[0]

n_reps = 3
all_theta = np.linspace(start=0,stop=2*np.pi,num=37)
all_theta = all_theta[:-1]
ntheta = len(all_theta)
all_fps = torch.zeros((n_reps, ntheta, n_hid))
all_fps_pos = torch.zeros((n_reps, ntheta, n_out))
ev_fps = np.zeros((n_reps, ntheta))
is_stable = np.zeros((n_reps, ntheta))
X = np.zeros((n_reps, ntheta,n_hid))

### Movement period
result_ok = False
for jj in range(ntheta):
    theta = all_theta[jj]
    input_signal = np.zeros((1,n_in))
    input_signal[:,0] = np.cos(theta)
    input_signal[:,1] = np.sin(theta)
    input_signal[:,2] = 0.5
    input_ff = torch.Tensor( input_signal ).type(dtype)
    input_tgt = np.zeros((1,n_out))
    input_tgt[:,0] = np.cos(theta)
    input_tgt[:,1] = np.sin(theta)
    input_tgt = torch.Tensor( input_tgt ).type(dtype)
    for ctr in range(n_reps):
        result_ok = False
        print('Theta = ' + np.str_(np.int_(theta/np.pi*180)) + ', ctr = '+ np.str_(ctr) )
        while not result_ok:
            # different  initialisations - sample from trajectory
            idx = np.random.choice(nT)
            hiddenx = torch.Tensor( data[idx:idx+1,:]).type(dtype)
            initp = ((torch.rand(1, n_out)-0.5)*0.1).type(dtype)
            #hiddenx = ((torch.rand(1, n_hid)-0.5)*0.5*ctr).type(dtype)
            fp, fp_pos, result_ok = analyzer2.find_fixed_point(hiddenx, initp, input_ff, input_tgt, view=True)
            all_fps[ctr,jj,:] = fp
            all_fps_pos[ctr,jj,:] = fp_pos
            jac = analyzer2.calc_hiddenx_jacobian( fp, fp_pos, input_ff, input_tgt)
            [evs,v]=ll.eig(jac)
            ev_fps[ctr,jj]=max(np.real(evs))
            is_stable[ctr,jj] = (ev_fps[ctr,jj]<0)
            fb_input = fp_pos - input_tgt
            _,z,_ = model(input_ff, fb_input, fp )
            X[ctr, jj,:] = z.detach().numpy()




Y = np.reshape(X, (n_reps*ntheta, n_hid))
Y = Y + (-0.5+np.random.random((Y.shape)))*.01 # just jitter a bit so that variance is not zero

nPC=4
px = PCA(n_components=nPC)

new_fps = model.nonlinearity(all_fps).detach().numpy()
new_fps = np.reshape( new_fps, (n_reps*ntheta,n_hid))
Zz = px.fit_transform(new_fps)

#Z1=Zz[ev_fps<0,:]
#Z2=Zz[ev_fps>0,:]

Zz = np.reshape( Zz, (n_reps, ntheta, 4) )

fig=pp.figure()
ax=fig.add_subplot(projection='3d')
for jj in range(n_reps):
    ax.scatter(Zz[jj,:,0], Zz[jj,:,1], Zz[jj,:,2], color=cm(all_theta/(2*np.pi)), alpha=0.5)

ax.view_init(80, 30)
pp.xlabel('PC1')
pp.ylabel('PC2')
pp.savefig(plotdir+'fps_move.png')
ax.view_init(10, 30)
pp.savefig(plotdir+'fps_move_view2.png')
pp.close(fig)

fp_pos = all_fps_pos.detach().numpy()
f2=pp.figure()
for jj in range(n_reps):
    pp.scatter( all_theta[1:], np.mod( np.arctan2(fp_pos[jj,1:,1],fp_pos[jj,1:,0]), 2*np.pi), alpha=0.5, c=cm(all_theta[1:]/(2*np.pi)) )

pp.ylim([0,2*np.pi])
pp.xlabel('Theta')
pp.ylabel('FPS angle (rad)')
pp.savefig(plotdir+'fps_angle_move.png')
pp.close(f2)



# ----------
####################################################
# Vary Hold input
####################################################

hold_inp = [0,0.1,0.2,0.3,0.4,0.5]
n_go = len(hold_inp)
n_reps = n_go

all_theta = np.linspace(start=0,stop=2*np.pi,num=37)
all_theta = all_theta[:-1]
ntheta = len(all_theta)

all_fps_H = torch.zeros((n_reps, ntheta, n_hid))
all_fps_pos_H = torch.zeros((n_reps, ntheta, n_out))
ev_fps_H = np.zeros((n_reps, ntheta))
is_stable_H = np.zeros((n_reps, ntheta))
X_H = np.zeros((n_reps, ntheta,n_hid))

result_ok = False
for jj in range(ntheta):
    theta = all_theta[jj]
    input_signal = np.zeros((1,n_in))
    input_signal[:,0] = np.cos(theta)
    input_signal[:,1] = np.sin(theta) 
    for ctr in range(n_reps):
        input_signal[:,2] = hold_inp[ctr] #0.5# hold
        input_ff = torch.Tensor( input_signal ).type(dtype)
        input_tgt = np.zeros((1,n_out))
        input_tgt[:,0] = np.cos(theta)*hold_inp[ctr]*2
        input_tgt[:,1] = np.sin(theta)*hold_inp[ctr]*2
        input_tgt = torch.Tensor( input_tgt ).type(dtype)   
        result_ok = False
        print('Theta = ' + np.str_(np.int_(theta/np.pi*180)) + ', ctr = '+ np.str_(ctr) )
        while not result_ok:
            # different  initialisations - sample from trajectory
            idx = np.random.choice(nT)
            hiddenx = torch.Tensor( data[idx:idx+1,:]).type(dtype)
            fp, fp_pos, result_ok = analyzer2.find_fixed_point(hiddenx, initp, input_ff, input_tgt, view=True)
            all_fps_H[ctr,jj,:] = fp
            all_fps_pos_H[ctr,jj,:] = fp_pos
            jac = analyzer2.calc_hiddenx_jacobian( fp, fp_pos, input_ff, input_tgt )
            [evs,v]=ll.eig(jac)
            ev_fps_H[ctr,jj]=max(np.real(evs))
            is_stable_H[ctr,jj] = (ev_fps_H[ctr,jj]<0)
            _, z, r = model(input_ff, fb_input, fp )
            X_H[ctr, jj,:] = z.detach().numpy()


new_fps_H = model.nonlinearity(all_fps_H).detach().numpy()
new_fps_H = np.reshape( new_fps_H, (n_reps*ntheta,n_hid))
Zz_H = px.transform(new_fps_H)
Zz_H = np.reshape( Zz_H, (n_reps, ntheta, nPC) )

fig=pp.figure()
ax=fig.add_subplot(projection='3d')
for jj in range(n_reps):
    ax.scatter(Zz_H[jj,:,0], Zz_H[jj,:,1], Zz_H[jj,:,3], color=cm(all_theta/(2*np.pi)), s=jj*10+10, alpha=0.5)

pp.xlabel('PC1')
pp.ylabel('PC2')
ax.view_init(80, 30)
pp.savefig(plotdir+'fps_hold_varied.png')
ax.view_init(10, 30)
pp.savefig(plotdir+'fps_hold_varied_view2.png')
pp.close(fig)