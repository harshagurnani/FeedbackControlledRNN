#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Regression tools for fitting linear models to learning outcomes.
'''
import numpy as np
import matplotlib.pyplot as pp
import matplotlib as mpl
font = {'size'   : 20, 'sans-serif':'Arial'}
mpl.rc('font', **font)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sns

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)+'/'


def fit_linear_speed(*args,  idy='log_speed', idx=('ctrb_fb', 'fracVF'), logX=False,
                     nfit=40, fracTrain=0.8,  logy=True, clampy=False , yfit_max=1, # fitting extras
                     suffix='', savfolder='', miny=1,maxy=120, ext='.png', 
                     doplot=True, clrs =['r','b','k'] , cmaps=['Reds','Blues', 'Greys'], eps=1e-4 ):                # plotting
    """
    Fits a linear regression model to the given data and returns the trained model along with training and testing scores.

    Parameters:
    *args : dict
        Multiple dictionaries containing each data set. E,g for WMP, OMP etc
    idy : str, optional
        The key of the dependent variable in the dictionaries. Default is 'log_speed'.
    idx : tuple, optional
        The keys of the independent variables in the dictionaries. Default is ('ctrb_fb', 'fracVF').
    logX : bool, optional
        Whether to take the logarithm of the independent variables. Default is False.
    nfit : int, optional
        The number of iterations to fit the model. Default is 40.
    fracTrain : float, optional
        The fraction of data to use for training. Default is 0.8.
    logy : bool, optional
        Whether the dependent variable is previously transformed by log10. Default is True.
    clampy : bool, optional
        Whether to clamp the predicted values of the dependent variable. Default is False.
    yfit_max : int, optional
        The maximum value for the predicted dependent variable. Default is 1.
    suffix : str, optional
        A suffix to add to the saved file name. Default is an empty string.
    savfolder : str, optional
        The folder path to save the output file. Default is an empty string.
    miny : int, optional
        The minimum value for the y-axis in the plot. Default is 1.
    maxy : int, optional
        The maximum value for the y-axis in the plot. Default is 120.
    ext : str, optional
        The file extension for the saved plot. Default is '.png'.
    doplot: bool, optional
        Whether to plot regression results. Default is True.
    clrs : list, optional
        The colors to use for plotting. Default is ['r','b','k'].
    cmaps : list, optional
        The color maps to use for plotting. Default is ['Reds','Blues', 'Greys'].

    Returns:
    reg : LinearRegression
        The trained linear regression model.
    train_score : list
        The training scores for each iteration.
    test_score : list
        The testing scores for each iteration.
    test_score_clamp : list
        The testing scores with clamped predicted values for each iteration.
    """

    all_dic = args
    ndic = len(args)

    # Combine data across datasets:
    y = all_dic[0][idy]
    X_var = [np.empty((0,1)) for jj in range(len(idx))] # initialize as list first, each element of X_var will be one predictor stacked across datasets
    
    for jj in range(len(idx)):
        X_var[jj] = all_dic[0][idx[jj]]
    
    for jj in range(1, ndic):
        y = np.vstack((y, all_dic[jj][idy] ))
        for kk in range(len(idx)):
            X_var[kk] = np.vstack((X_var[kk], all_dic[jj][idx[kk]]))    # each x_var is stacked across datasets, 

    nx0 = [np.nan for jj in range(ndic)]
    
    # concatenate all X variables for each dataset:
    X_dataset = [np.empty((0,1)) for jj in range(ndic)]
    for jj in range(ndic):
        nx0[jj] = len(all_dic[jj][idx[0]])
        X_dataset[jj] = np.reshape(all_dic[jj][idx[0]], (nx0[jj],1))
    
    for jj in range(len(idx)-1):
        cid=jj+1
        for kk in range(ndic):
            X_dataset[kk] = np.concatenate(( X_dataset[kk],  np.reshape(all_dic[kk][idx[cid]], (nx0[kk],1)) ), axis=1) # each dic is one dataset, add columns for each predictor

    # make full predictor matrix
    X = X_dataset[0]
    for kk in range(ndic-1):
        X = np.vstack((X,X_dataset[kk+1]))

    if logX:
        X[X<eps]=eps
        X = np.log10(X)
        for kk in range(ndic):
            X_dataset[kk][X_dataset[kk]<eps]=eps
            X_dataset[kk] = np.log10(X_dataset[kk])
    nX = np.sum(np.array(nx0))

    # Split data
    train_score = list()
    test_score = list()
    test_score_clamp = list()
    best_reg = None
    best_score = -1000

    # Fit over iterations
    for kk in range(nfit):
        trainX, testX, trainY, testY = train_test_split( X,y, train_size=fracTrain, shuffle=True)
        reg = LinearRegression().fit(trainX,trainY)
        train_score.append( reg.score(trainX,trainY) )
        tsc = reg.score( testX, testY)
        test_score.append( tsc )
        ypred = reg.predict(testX)
        if clampy:
            ypred[ypred>yfit_max]=yfit_max
            tsc =  1 - np.mean(np.power(testY-ypred,2*np.ones(testY.shape)))/np.var(testY)
            test_score_clamp.append( tsc )
        if tsc>best_score:
            best_reg = reg
            best_score = tsc
    
    
    
    f1 = pp.figure()
    f1 = pp.figure()
    ax = f1.add_subplot(111)
    if logy:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot([10**miny,10**maxy],[10**miny,10**maxy],color='k')
    else:
        pp.plot([miny,maxy],[miny,maxy],color='k')

    for kk in range(ndic):
        ypred = best_reg.predict(X_dataset[kk])   
        ypred = np.reshape( ypred, (nx0[kk],1))
        ytrue = np.reshape(all_dic[kk][idy], (nx0[kk],1)) 
        if clampy:
            ypred[ypred>yfit_max]=yfit_max
        data = pd.DataFrame( np.concatenate((ytrue,ypred), axis=1) , columns=['true','pred'])
        if logy:
            data=10**data
            ypred = 10**ypred
            ytrue=10**ytrue
        sns.kdeplot( data, x='true', y='pred', cmap=cmaps[kk], fill=True, alpha=0.4)
        pp.scatter(ytrue, ypred,c=clrs[kk], alpha=0.2)
    
    pp.xlabel('true')
    pp.ylabel('predicted')
    pp.savefig(savfolder+'regression_fit'+suffix+ext)
    pp.close(f1)
    print(best_score)

    return best_reg, train_score, test_score, test_score_clamp