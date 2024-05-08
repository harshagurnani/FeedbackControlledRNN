import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'
sys.path.insert(0, parentdir) 

import matplotlib.pyplot as pp
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)

import scipy.linalg as ll
from sklearn.decomposition import PCA
import tools.perturb_network_ as pnet


def prep_mov_angle( res, moveOnset=200, moveOffset=350, prepOnset=100, nPC=8 ):
    X  = res['activity1']
    delay = res['delays']

    nTr, nTm, nNeu = X.shape

    tr=0
    movTm = np.int_(np.arange( start=moveOnset, stop=moveOffset))
    prepTm = np.int_(np.arange( start=prepOnset, stop=delay[tr]))
    moveX =  X[tr, movTm,:] 
    prepX =  X[tr, prepTm,:] 
    nMov = len(movTm)
    nPrep = len(prepTm)
    ids = np.random.choice( nTm, size=nMov+nPrep, replace=False )
    shuff_moveX = X[tr,ids[:nMov],:]
    shuff_prepX = X[tr,ids[nMov:],:]


    for kk in range(nTr-1):
        tr = kk+1
        prepTm = np.int_(np.arange( start=prepOnset, stop=delay[tr]))
        x1 = X[tr,movTm,:]
        x2 = X[tr,prepTm,:]
        moveX=np.concatenate( (moveX, x1), axis=0 )
        prepX=np.concatenate( (prepX, x2), axis=0 )
        nMov = len(movTm)
        nPrep = len(prepTm)

        # shuffle control
        ids = np.random.choice( nTm, size=nMov+nPrep, replace=False )
        x3 = X[tr,ids[:nMov],:]
        x4 = X[tr,ids[nMov:],:]
        shuff_moveX=np.concatenate( (shuff_moveX, x3), axis=0 )
        shuff_prepX=np.concatenate( (shuff_prepX, x4), axis=0 )




    px_move = PCA( n_components=nPC )
    px_move.fit( moveX )

    px_prep = PCA( n_components=nPC )
    px_prep.fit(prepX)

    px_shuff_move = PCA( n_components=nPC )
    px_shuff_move.fit( shuff_moveX )

    px_shuff_prep = PCA( n_components=nPC )
    px_shuff_prep.fit( shuff_prepX )

    nAngles = np.zeros((nPC,2))
    shuffAngles = np.zeros((nPC,2))
    for jj in range(nPC-1):
        pc = jj+1
        xa = ll.subspace_angles( px_move.components_[:pc+1,:].T, px_prep.components_[:pc+1,:].T )
        nAngles[pc,:] = np.array( [min(xa[:pc]), max(xa[:pc])] )

        xa = ll.subspace_angles( px_shuff_move.components_[:pc+1,:].T, px_shuff_prep.components_[:pc+1,:].T )
        shuffAngles[pc,:] = np.array( [min(xa[:pc]), max(xa[:pc])] )

    return nAngles, shuffAngles



def analyse_file( filename ):
    mod = pnet.postModel( filename, {} )
    mod.load_rnn()
    mod.params['jump_amp'] = 0.01
    mod.params['bump_amp'] = 0.01
    #mod.params['delays'] = [150,151]
    dic = mod.test_model()

    angles, shuff_angles = prep_mov_angle( dic['res'], moveOnset=220, moveOffset=350, prepOnset=80, nPC=8 )

    return angles, shuff_angles

'''
X=X-np.mean(X,axis=0)
U = np.concatenate( (px_move.components_[:4,:], px_prep.components_[:4,:] ), axis=0)
Q, R = np.linalg.qr( U.T )
projX = X @ Q

fig = pp.figure()
ax = fig.add_subplot(projection='3d')
for jj in range(20):
    ps = np.int_(delay[jj])
    ax.plot( projX[jj,prepOnset:ps,1], projX[jj,prepOnset:ps,5],projX[jj,prepOnset:ps,4], color='b')
    ax.plot( projX[jj,ps:moveOffset,1], projX[jj,ps:moveOffset,5],projX[jj,ps:moveOffset,4],  color='r')

pp.xlabel('move2')
pp.ylabel('prep1')
ax.view_init(20,300)
pp.savefig('prepx_3d.png')
pp.close()


U_p =  np.concatenate((wout.T, px_prep.components_ ), axis=0 )
Q_p, _ = np.linalg.qr( U_p.T )

U_m=  np.concatenate((wout.T, px_move.components_ ), axis=0 )
Q_m, _ = np.linalg.qr( U_m.T )

proj_p = X @ Q_p
proj_m = X @ Q_m


fig = pp.figure()
ax = fig.add_subplot(projection='3d')
for jj in range(20):
    ps = np.int_(delay[jj])
    ax.plot( proj_p[jj,prepOnset:ps,3], proj_p[jj,prepOnset:ps,2],proj_p[jj,prepOnset:ps,0], color='b')
    ax.plot( proj_p[jj,ps:moveOffset,3], proj_p[jj,ps:moveOffset,2],proj_p[jj,ps:moveOffset,0],  color='r')

pp.xlabel('prep2')
pp.ylabel('prep1')
#pp.zlabel('prep1')

ax.view_init(20,300)
pp.savefig('prepx_3d.png')
pp.close()

fig2=pp.figure()
s1 = fig2.add_subplot(212)
s2 = fig2.add_subplot(211)
for jj in range(20):
    ps = np.int_(delay[jj])
    s1.plot( projX[jj,prepOnset:moveOffset,0],  color='b')
    s2.plot( projX[jj,prepOnset:moveOffset,4],  color='r')
    #s1.plot( projX[jj,prepOnset:moveOffset,1],  color='g')
    #s2.plot( projX[jj,prepOnset:moveOffset,4],  color='y')

fig2.savefig('projX_q1_tanh.png')
'''
'''
fig2=pp.figure()
s1 = fig2.add_subplot(212)
s2 = fig2.add_subplot(211)
for jj in range(10):
    ps = np.int_(delay[jj])
    s1.plot( ll.norm(projPrep[jj,prepOnset:moveOffset,:4],axis=1),  color='b')
    s2.plot( ll.norm(projMove[jj,prepOnset:moveOffset,:4],axis=1),  color='r')

fig2.savefig('projX_prep_move_tanh.png') 


from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

stim = res['stimulus']
cm = pp.get_cmap('viridis')
fig2=pp.figure(figsize=[8,10])
s1 = fig2.add_subplot(212)
s2 = fig2.add_subplot(211)
for jj in range(50):
   ps = np.int_(delay[jj])
   theta = np.arctan2(stim[jj,-1,1],stim[jj,-1,0])/(2*np.pi)+0.5
   s1.plot( projX[jj,ps:moveOffset,2],projX[jj,ps:moveOffset,0] , color=cm(theta))
   s2.plot( projX[jj,prepOnset:ps,4],projX[jj,prepOnset:ps,5],  color=cm(theta))
   s1.scatter( projX[jj,ps,2],projX[jj,ps,0] , color='k')
   s2.scatter( projX[jj,prepOnset,4],projX[jj,prepOnset,5],  color='k')
   t1= Affine2D().rotate_deg(np.rad2deg(np.arctan2(projX[jj,moveOffset,2],projX[jj,moveOffset,0])))
   t2= Affine2D().rotate_deg(np.rad2deg(np.arctan2(projX[jj,ps,4],projX[jj,ps,5])))
   s1.scatter( projX[jj,moveOffset,2],projX[jj,moveOffset,0] , 130,color=cm(theta), marker=MarkerStyle("^", "full", t1))
   s2.scatter( projX[jj,ps,4],projX[jj,ps,5],  130,color=cm(theta),  marker=MarkerStyle("^", "full", t2))

fig2.savefig('projX_q12_2D_tanh4.png')



fig2=pp.figure()
s1 = fig2.add_subplot(212)
s2 = fig2.add_subplot(211)
for jj in range(20):
   ps = np.int_(delay[jj])
   s1.plot( proj_m[jj,prepOnset:moveOffset,2] , color='b')
   s2.plot( proj_p[jj,prepOnset:moveOffset,2],  color='r')

fig2.savefig('proj_wout_q2.png')


fig2=pp.figure()
s1 = fig2.add_subplot(212)
s2 = fig2.add_subplot(211)
for jj in range(20):
   ps = np.int_(delay[jj])
   s1.plot( proj_m[jj,prepOnset:moveOffset,0] , color='b')
   s2.plot( proj_p[jj,prepOnset:moveOffset,0],  color='r')

fig2.savefig('proj_wout.png')


fig2=pp.figure()
for jj in range(20):
   ps = np.int_(delay[jj])
   theta = np.arctan2(stim[jj,-1,1],stim[jj,-1,0])/(2*np.pi)+0.5
   pp.plot( proj_m[jj,ps:moveOffset,0] ,proj_p[jj,ps:moveOffset,1], color=cm(theta))
   #s2.plot( proj_p[jj,ps:moveOffset,1],  color='r')

fig2.savefig('proj_wout_12.png')



fig = pp.figure()
ax = fig.add_subplot(projection='3d')
yy, xx = np.meshgrid(np.arange(-3,4,0.5),np.arange(-6,6,1))
z = 0*xx
ax.plot_surface(xx, yy, z, alpha=0.2)
for jj in range(50):
    ps = np.int_(delay[jj])
    theta = np.arctan2(stim[jj,-1,1],stim[jj,-1,0])/(2*np.pi)+0.5
    #ax.plot( projX[jj,prepOnset:ps,1], projX[jj,prepOnset:ps,2],projX[jj,prepOnset:ps,40], color=cm(theta))
    ax.scatter(projX[jj,ps,1], projX[jj,ps,2],projX[jj,ps,0], color='k')
    ax.plot( projX[jj,ps:moveOffset,1], projX[jj,ps:moveOffset,2],projX[jj,ps:moveOffset,0],  color=cm(theta))

pp.xlabel('move2')
pp.ylabel('move3')
#pp.zlabel('prep1')

ax.view_init(190,0)
pp.savefig('projX_3d_plane.png',dpi=200)
pp.close()




stim = res['stimulus']
cm = pp.get_cmap('viridis')
fig2=pp.figure()
s1 = fig2.add_subplot(111)
for jj in range(50):
   ps = np.int_(delay[jj])
   theta = np.arctan2(stim[jj,-1,1],stim[jj,-1,0])/(np.pi)
   s1.plot( projX[jj,ps:moveOffset,0],projX[jj,ps:moveOffset,1] , color=cm(theta))
   s1.scatter( projX[jj,ps,0],projX[jj,ps,1] , color='k')
   t1= Affine2D().rotate_deg(-np.rad2deg(np.arctan2(projX[jj,moveOffset,0],projX[jj,moveOffset,1])))
   s1.scatter( projX[jj,moveOffset,0],projX[jj,moveOffset,1] , 130,color=cm(theta), marker=MarkerStyle("^", "full", t1))

pp.savefig('projX_dic_pre_wpert.png',dpi=200)
'''


