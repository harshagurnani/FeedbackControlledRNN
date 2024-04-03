import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import os

def plot_2D_traj( output, stimulus=None, target=None, cmap='viridis', plot_idx=None, tm0 = 30, \
                    savfolder = 'results/dumpplot/', savname='2D_traj.png' ):
    """ Plot RNN readout in 2D phase space """

    
    if plot_idx is None:
        nTrials = output.shape[0]
        plot_idx = np.random.choice( range(nTrials), 20, replace=False )

    
    cm = mpl.colormaps[cmap]
    plt.figure()
    for idx in plot_idx:
        if stimulus is not None:
            theta =  np.arctan2(stimulus[idx,-1,1], stimulus[idx,-1,0])
            cidx = .5 + 0.5*(theta/np.pi)
            plt.plot(output[idx,tm0:,0],output[idx,tm0:,1], color=cm(cidx), lw=4)
            plt.scatter( np.cos(theta), np.sin(theta), marker='^', color=cm(cidx), edgecolors='k' )
            plt.scatter( output[idx,-1,0], output[idx,-1,1], marker='o', color=cm(cidx), edgecolors='r' )
        else:
            plt.scatter(output[idx,tm0:,0],output[idx,tm0:,1], color='k')
    plt.xlim((-1.8,1.8))
    plt.ylim((-1.8,1.8))
    plt.savefig( savfolder+ savname )
    plt.close()



def plot_readout_trace( output, stimulus=None, target=None, delays = [100,200],\
                        cmap='viridis', plot_idx=None,  tm0 = 30, \
                        savfolder = 'results/dumpplot/', savname='output.png' ):
    """ Plot RNN readout as time-varying components """


    
    tm = np.arange(output.shape[1])
    if plot_idx is None:
        nTrials = output.shape[0]
        plot_idx = np.random.choice( range(nTrials), 20, replace=False )

    cm = mpl.colormaps[cmap]
    plt.figure()
    for idx in plot_idx:
        if stimulus is not None:
            tms = np.where(stimulus[idx,:,2]>0)
            tms=tms[0][0]
            tms  = (delays[1]-tms)/(delays[1]-delays[0])
            if tms<0:
                tms=0
            elif tms>1:
                tms=1
            plt.plot(tm[tm0:], output[idx,tm0:,:], color=cm(tms))
        else:
            plt.plot(tm[tm0:], output[idx,tm0:,:], color='k')

        if target is not None:
            plt.plot(tm[tm0:], output[idx,tm0:,:]-target[idx,tm0:,:], 'r--')
    
    plt.savefig( savfolder+ savname )
    plt.close()



def plot_traces( activity, plot_neu=None, plot_idx=None, tm0 = 30, \
                savfolder = 'results/dumpplot/', savname='rnn_activity.png' ):
    """ Plot RNN activity as time-varying components """


    nTr, nTm, nNeu = activity.shape
    
    if plot_idx is None:
        plot_idx = np.random.choice( range(nTr), 30, replace=False )

    if plot_neu is None:
        plot_neu = np.random.choice( range(nNeu), 30, replace=False )


    use_tm = np.arange(tm0,min(1000,nTm))
    Y = activity[np.ix_(plot_idx,use_tm,plot_neu)]
    Y = np.reshape(Y,(Y.shape[0]*Y.shape[1], Y.shape[2]))
    plt.figure()
    plt.plot( use_tm, Y )    
    plt.savefig( savfolder+ savname )
    plt.close()


