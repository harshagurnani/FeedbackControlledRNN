import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 



import matplotlib.pyplot as pp
import matplotlib as mpl
import numpy as np
import os
import sys
import inspect
import matplotlib as mpl
font = {'size'   : 10, 'sans-serif':'Arial'}
mpl.rc('font', **font)

def plot_oblique_results( topfolder = 'relu_test_/', file='wout_angles.npy', plotPC=8, ext='.png', min_hit=0.3 ):

    #### Plot angles as fn of wout scale ?
    savfolder = parentdir+'saved_plots/oblique_dynamics/'+topfolder
    file  = savfolder+'wout_angles.npy'
    allAng = np.load( file, allow_pickle=True )

    nfiles = len( allAng )
    tmp_dic = allAng[0]
    nPC_angle = tmp_dic['nPC_angle']

    Out_Fbk = np.zeros((nfiles,2))
    Out_PCs = np.zeros((nfiles,2))
    Curve_Out_PCs = np.zeros((nfiles,nPC_angle))
    Fbk_PCs = np.zeros((nfiles,2))

    Out_Int = np.zeros((nfiles,2))
    Curve_ID_RecA_RSV = np.zeros((nfiles,nPC_angle))
    Curve_ID_RecA_LSV = np.zeros((nfiles,nPC_angle))

    plotPC=8
    Out_RecA = np.zeros((nfiles,2))
    Out_RecA_L = np.zeros((nfiles,2))
    Curve_Out_RecA_L = np.zeros((nfiles,nPC_angle))
    Curve_Out_RecA = np.zeros((nfiles,nPC_angle))
    Out_delRecA_R = np.zeros((nfiles,2))
    Out_delRecA_L = np.zeros((nfiles,2))

    Fbk_RecA_R = np.zeros((nfiles,2))
    Fbk_RecA_L = np.zeros((nfiles,2))
    Curve_Fbk_RecA = np.zeros((nfiles,nPC_angle))
    Curve_Fbk_RecA_RSV = np.zeros((nfiles,nPC_angle))
    Fbk_delRecA_R = np.zeros((nfiles,2))
    Fbk_delRecA_L = np.zeros((nfiles,2))

    HitRate_T_I = np.zeros((nfiles,2))
    MinDist_T_I = np.zeros((nfiles,2))

    Wout = np.zeros(nfiles)
    Wrec = np.zeros(nfiles)
    Wfbk = np.zeros(nfiles)
    SparseWt = np.zeros(nfiles)
    SparseNeu = np.zeros(nfiles)
    n_neu = np.zeros(nfiles)

    rec_change = np.zeros((nfiles,2))
    fbk_change = np.zeros((nfiles,2))

    gencorr = np.zeros(nfiles)
    bumpamp = np.zeros(nfiles)
    henrici = np.zeros(nfiles)
    ctrb_ff = np.zeros(nfiles)
    ctrb_fb = np.zeros(nfiles)
    # read results:
    for ff in range( nfiles ):

        # ---------- connectivity parameters ----------------------------------------------------------
        Wout[ff] = allAng[ff]['sig_out_0']
        Wrec[ff] = allAng[ff]['sig_rec_0']
        n_neu[ff] = allAng[ff]['n1']
        if 'sig_fbk_0' in allAng[ff].keys():
            Wfbk[ff] = allAng[ff]['sig_fbk_0']
        if 'frac_wt' in allAng[ff].keys():
            SparseWt[ff] = allAng[ff]['frac_wt']
        if 'frac_neu' in allAng[ff].keys():
            SparseNeu[ff] = allAng[ff]['frac_neu']
        bumpamp[ff] = allAng[ff]['simparams']['bump_amp']

        # ---------  ANGLE OF  WOUT with weights and intrinsic dynamics ---------------------------------
        Out_Fbk[ff,:] = allAng[ff]['trained_angles']['Out_Fbk']                 # max and min angle
        Out_PCs[ff,:] = allAng[ff]['trained_angles']['Out_PCs'][plotPC,:]       # final PC
        Out_RecA[ff,:] = allAng[ff]['trained_angles']['Out_RightSV'][plotPC,:]  # with recurrent sensitive dirns
        Out_RecA_L[ff,:] = allAng[ff]['trained_angles']['Out_LeftSV'][plotPC,:] # with recurrent amplified dirns
        Out_Int[ff,:] = allAng[ff]['intuitive_angles']['Out_Intuitive']         # with intuitive decoder
        Out_delRecA_R[ff,:] = allAng[ff]['trained_delW_angles']['Out_RightSV'][plotPC,:]  # with deltaW - recurrent sensitive dirns
        Out_delRecA_L[ff,:] = allAng[ff]['trained_delW_angles']['Out_LeftSV'][plotPC,:] # with deltaW - recurrent amplified dirns

        # curve of successive vectors
        Curve_Out_PCs[ff,:] = np.nanmin( allAng[ff]['trained_angles']['Out_PCs'], axis=1 )      # with top pcs (min principle angle)
        Curve_Out_RecA[ff,:] = np.nanmin( allAng[ff]['trained_angles']['Out_RightSV'], axis=1 ) # with right SVs (sensitive dirns) (min principle angle)
        Curve_Out_RecA_L[ff,:] = np.nanmin( allAng[ff]['trained_angles']['Out_LeftSV'], axis=1 ) # with left SVs (amplified dirns) (min principle angle)

        # ---------  ANGLE OF  WFBK with intrinsic dynamics ---------------------------------
        Fbk_RecA_R[ff,0] = allAng[ff]['trained_angles']['Fbk_RightSV'][plotPC,0]  # with recurrent sensitive dirns
        Fbk_RecA_L[ff,0] = allAng[ff]['trained_angles']['Fbk_LeftSV'][plotPC,0] # with recurrent amplified dirns
        Fbk_RecA_R[ff,1] = allAng[ff]['trained_angles']['Fbk_RightSV'][plotPC,-1]  # with recurrent sensitive dirns
        Fbk_RecA_L[ff,1] = allAng[ff]['trained_angles']['Fbk_LeftSV'][plotPC,-1] # with recurrent amplified dirns
        Curve_Fbk_RecA[ff,:] = np.nanmin( allAng[ff]['trained_angles']['Fbk_LeftSV'], axis=1 )   
        Curve_Fbk_RecA_RSV[ff,:] = np.nanmin( allAng[ff]['trained_angles']['Fbk_RightSV'], axis=1 )   
        Fbk_delRecA_R[ff,0] = allAng[ff]['trained_delW_angles']['Fbk_RightSV'][plotPC,0]  # with deltaW - recurrent sensitive dirns
        Fbk_delRecA_L[ff,0] = allAng[ff]['trained_delW_angles']['Fbk_LeftSV'][plotPC,0] # with deltaW- recurrent amplified dirns
        Fbk_delRecA_R[ff,1] = allAng[ff]['trained_delW_angles']['Fbk_RightSV'][plotPC,-1]  # with deltaW - recurrent sensitive dirns
        Fbk_delRecA_L[ff,1] = allAng[ff]['trained_delW_angles']['Fbk_LeftSV'][plotPC,-1] # with deltaW - recurrent amplified dirns


        # ---------  ANGLE OF  PCs with intrinsic dynamics ---------------------------------
        Curve_ID_RecA_RSV[ff,:] = np.nanmin( allAng[ff]['trained_angles']['ID_PCs'], axis=1 )           
        Curve_ID_RecA_LSV[ff,:] = np.nanmin( allAng[ff]['trained_angles']['ID_PCs_LeftSV'], axis=1 )    

        # --------- PERFORMANCE -------------------------------------------------------------
        HitRate_T_I[ff,0] = allAng[ff]['perf_trained']['success']
        HitRate_T_I[ff,1] = allAng[ff]['perf_intuit']['success']
        MinDist_T_I[ff,0] = allAng[ff]['perf_trained']['min_rdist']
        MinDist_T_I[ff,1] = allAng[ff]['perf_intuit']['min_rdist']

        fbk_change[ff,:] = np.array([allAng[ff]['wres_trained']['W_fbk_0']['neunorm_1'], allAng[ff]['wres_trained']['W_fbk_0']['delW_neu'] ])
        rec_change[ff,:] = np.array([allAng[ff]['wres_trained']['W_rec_0']['neunorm_1'], allAng[ff]['wres_trained']['W_rec_0']['delW_neu'] ])

        gencorr[ff] = allAng[ff]['gencorr_trained']
        #print(allAng[ff]['gencorr_var'])
        try:
            henrici[ff] = allAng[ff]['trained_rec_HI']
            ctrb_ff[ff] = allAng[ff]['control_wout']['ctrb_ff']
            ctrb_fb[ff] = allAng[ff]['control_wout']['ctrb_fb']
        except:
            print('old file has no control analyses')


    
    ## --------------------------------------------------------------------------------------------------- ##
    #### ------------------------- Plotting -----------------------------------------------------------------
    ## --------------------------------------------------------------------------------------------------- ##

    if np.sum(SparseNeu)>0:
        pass
    elif np.sum(SparseWt)>0:
        pass

    usef = np.arange(nfiles)

    #usef = usef[bumpamp<0.05]

    ## --------------------------------------------------------------------------------------------------- ##

    # angle between readout and feedback weights (feedback doesnt directly drive along readout)
    plot_principal_angles( Wout[usef], np.rad2deg(Out_Fbk[usef]), savfolder, xlabel='Output wt scale', ylabel='angle of readout with feedback wts', fname='out_fbk_', ext=ext)

    # angle between readout and top (plotPC=8) pcs
    plot_principal_angles( Wout[usef], np.rad2deg(Out_PCs[usef]), savfolder, xlabel='Output wt scale', ylabel='Angle w '+np.str_(plotPC)+'-PC space', fname='out_pcs_', ext=ext)

    # angle between readout and top X pcs
    plot_curve( np.rad2deg(Curve_Out_PCs[usef,:]), savfolder=savfolder, xval=None, xlabel='#  activity PCs', ylabel='angle with wout', fname='out_pccurve_', ext=ext)

    # angle between readout and right and left singular vectors of Wrec.T
    plot_principal_angles( Wout[usef], np.rad2deg(Out_RecA[usef,:]), savfolder, xlabel='Output wt scale', ylabel='Angle w '+np.str_(plotPC)+'-SVs (sensitive) of WRec', fname='out_rec_', ext=ext)
    plot_principal_angles( Wout[usef], np.rad2deg(Out_delRecA_R[usef,:]), savfolder, xlabel='Output wt scale', ylabel='Angle w '+np.str_(plotPC)+'-SVs (sensitive) of delta-WRec', fname='out_delrec_', ext=ext)
    plot_principal_angles( Wout[usef], np.rad2deg(Out_RecA_L[usef,:]), savfolder, xlabel='Output wt scale', ylabel='Angle w '+np.str_(plotPC)+'-left SVs (amplified) of WRec', fname='out_rec_left_', ext=ext)
    plot_principal_angles( Wout[usef], np.rad2deg(Out_delRecA_L[usef,:]), savfolder, xlabel='Output wt scale', ylabel='Angle w '+np.str_(plotPC)+'-left SVs (amplified) of delta-WRec', fname='out_delrec_left_', ext=ext)

    # angle between readout and cumulative right and left singular vectors of Wrec.T
    plot_curve( np.rad2deg(Curve_Out_RecA[usef,:]), savfolder=savfolder, xlabel='# Right singular vectors', ylabel='angle with wout', fname='out_reccurve_right_', ext=ext)

    plot_curve( np.rad2deg(Curve_Out_RecA_L[usef,:]), savfolder=savfolder, xlabel='# Left singular vectors', ylabel='angle with wout', fname='out_reccurve_left_', ext=ext)

    # angle between feedback weights and left and right singular vectors of wrec.T
    plot_curve( np.rad2deg(Curve_Fbk_RecA[usef,:]), savfolder=savfolder, xlabel='# Left singular vectors', ylabel='angle with wfbk', fname='fbk_reccurve_left_', ext=ext)

    plot_curve( np.rad2deg(Curve_Fbk_RecA_RSV[usef,:]), savfolder=savfolder, xlabel='# Right singular vectors', ylabel='angle with wfbk', fname='fbk_reccurve_right_', ext=ext)


    # Performance - hit rate and distance to target
    f6 = pp.figure()
    x1=(np.random.random(Wout[usef].shape)-.5)*0.1       # jitter points
    x2=(np.random.random(Wout[usef].shape)-.5)*0.1
    pp.scatter( Wout[usef]+x1, HitRate_T_I[usef,0], label='trained')
    pp.scatter( Wout[usef]+x2, HitRate_T_I[usef,1], label='intuitive')
    pp.xlabel('Output wt scale')
    pp.ylabel('Hit rate')
    pp.legend()
    pp.savefig(savfolder+'hit_rate_'+ext)
    pp.close(f6)



    f7 = pp.figure()
    pp.scatter( Wout[usef]+x1, MinDist_T_I[usef,0], label='trained')
    pp.scatter( Wout[usef]+x2, MinDist_T_I[usef,1], label='intuitive')
    pp.xlabel('Output wt scale')
    pp.ylabel('Min distance to target')
    pp.legend()
    pp.savefig(savfolder+'min_dist_'+ext)
    pp.close(f7)


    #--------- Weight change
    plot_scatter( Wout[usef], [rec_change[usef,1], fbk_change[usef,1]], ['rec','fbk'],
                savfolder=savfolder, xlabel='Output wt scale', ylabel='Change in wts (per neuron)', fname='out_wtdel_', ext=ext )

    plot_scatter( Wout[usef], [rec_change[usef,0], fbk_change[usef,0]], ['rec','fbk'],
                savfolder=savfolder, xlabel='Output wt scale', ylabel='Final wt norm (per neuron)', fname='out_wtfinal_', ext=ext )

    ### generalised correlation
    plot_scatter( Wout[usef], [gencorr[usef]], ['trained'],
                savfolder=savfolder, xlabel='Output wt scale', ylabel='Generalized correlation with activity', fname='out_gencorr_', ext=ext )

    ### angle between readout and intuitive decoder
    yval_list = [np.rad2deg(Out_Int[usef,0]), np.rad2deg(Out_Int[usef,1])]
    plot_scatter( Wout[usef], yval_list, ['maxA','minA'],
                savfolder=savfolder, xlabel='Output wt scale', ylabel='Angle w Intutitive decoder', fname='out_intD_', ext=ext )

    yval_list = [np.rad2deg(Out_Int[usef,0]), np.rad2deg(Out_Int[usef,1])]
    plot_scatter( Wrec[usef], yval_list, ['maxA','minA'],
                savfolder=savfolder, xlabel='Recurrent wt scale', ylabel='Angle of wout w Intutitive decoder', fname='rec_intD_', ext=ext )

    ### angle between readout and top pcs
    yval_list = [np.rad2deg(Out_PCs[usef,0]), np.rad2deg(Out_PCs[usef,1])]
    plot_scatter( Wrec[usef], yval_list, ['maxA','minA'],
                savfolder=savfolder, xlabel='Recurrent wt scale', ylabel='Angle of wout w '+np.str_(plotPC)+'-PC space', fname='rec_pcs_', ext=ext )


    #### ---- 2d plots ----------------------------------------------------------------------------------

    #goodtrain = (HitRate_T_I[:,0]>0.3)
    goodtrain = (HitRate_T_I[usef,0]>min_hit)            # plot all or only successfully trained models???
    usef = usef[goodtrain>0]

    # ------------ plot angle between readout and intuitive decoder 
    #----------------------------------------------------------------
    # # vs wout and wrec scale
    fname=savfolder+'out_rec_intD'+ext
    z1=np.rad2deg(Out_Int[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax,do_int=True, include_z=goodtrain )

    # vs wout and wfbk scale
    fname=savfolder+'out_fbk_intD'+ext
    z1=np.rad2deg(Out_Int[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wfbk[usef], xname='Wt Out scale', yname='Fbk wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax,do_int=True, include_z=goodtrain )

    # vs wrec and wfbk scale
    fname=savfolder+'rec_fbk_intD'+ext
    z1=np.rad2deg(Out_Int[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wrec[usef], x2=Wfbk[usef], xname='Wt Rec scale', yname='Fbk wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax,do_int=True, include_z=goodtrain )


    # ------------ plot angle between readout  and top pcs
    #----------------------------------------------------------------
    # vs wout and wrec scale
    fname=savfolder+'out_rec_pcs'+ext
    z1=np.rad2deg(Out_PCs[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # vs wfbk and perturbation scale
    fname=savfolder+'fbk_amp_pcs'+ext
    z1=np.rad2deg(Out_PCs[usef,0])
    vmin = min(0, min(z1))
    vmax=max(0.5, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=bumpamp[usef], xname='Wt Fbk scale', yname='Perturbation amp', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )



    # ------------ plot angle between readout  and wrec_left/wrec_right
    #----------------------------------------------------------------
    # left sv - vs wout and wrec scale
    fname=savfolder+'out_rec_lsv_out'+ext
    z1=np.rad2deg(Out_RecA_L[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # right sv - vs wout and wrec scale
    fname=savfolder+'out_rec_rsv_out'+ext
    z1=np.rad2deg(Out_RecA[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    fname=savfolder+'out_rec_dellsv_out'+ext
    z1=np.rad2deg(Out_delRecA_L[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # right sv - vs wout and wrec scale
    fname=savfolder+'out_rec_delrsv_out'+ext
    z1=np.rad2deg(Out_delRecA_R[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # ------------ plot angle between feedback and wrec_left/wrec_right
    #----------------------------------------------------------------
    # left sv - vs wout and wrec scale
    fname=savfolder+'out_rec_lsv_fbk'+ext
    z1=np.rad2deg(Fbk_RecA_L[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # right sv - vs wout and wrec scale
    fname=savfolder+'out_rec_rsv_fbk'+ext
    z1=np.rad2deg(Fbk_RecA_R[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )


    # left sv - vs wfbk and wrec scale
    fname=savfolder+'fbk_rec_lsv_fbk'+ext
    z1=np.rad2deg(Fbk_RecA_L[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=Wrec[usef], xname='Wt Fbk scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # right sv - vs wfbk and wrec scale
    fname=savfolder+'fbk_rec_rsv_fbk'+ext
    z1=np.rad2deg(Fbk_RecA_R[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=Wrec[usef], xname='Wt Fbk scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    fname=savfolder+'fbk_rec_dellsv_fbk'+ext
    z1=np.rad2deg(Fbk_delRecA_L[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=Wrec[usef], xname='Wt Fbk scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # right sv - vs wfbk and wrec scale
    fname=savfolder+'fbk_rec_delrsv_fbk'+ext
    z1=np.rad2deg(Fbk_delRecA_R[usef,0])
    vmin = min(20, min(z1))
    vmax=max(75, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=Wrec[usef], xname='Wt Fbk scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )

    # ------------ plot generalised correlation along wout
    #----------------------------------------------------------------
    # vs wout and wrec scale
    fname=savfolder+'out_rec_gencorr'+ext
    z1=gencorr[usef]
    vmin = min(0, min(z1))
    vmax=max(0.5, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wt Out scale', yname='Rec wt scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )

    # vs wfbk and perturbation scale
    fname=savfolder+'fbk_amp_gencorr'+ext
    z1=gencorr[usef]
    vmin = min(0, min(z1))
    vmax=max(0.5, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=bumpamp[usef], xname='Wt Fbk scale', yname='Perturbation amp', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )


    ## ------------ plot versus sparsity: angle with intuitive decoder and top pcs
    #----------------------------------------------------------------
    nwtfrac  = len(np.unique(SparseWt))
    nneufrac  = len(np.unique(SparseNeu))

    if nwtfrac>1:
        fname=savfolder+'out_sparsewt_intD'+ext
        z1=np.rad2deg(Out_Int[usef,1])
        vmin = min(20, min(z1))
        vmax=max(75, max(z1))
        plot_heatmap( x1=Wout[usef], x2=SparseWt[usef], xname='Wt Out scale', yname='Input wt sparsity', z1=z1, fname=fname, vmin=vmin, vmax=vmax,do_int=True, include_z=goodtrain )

        fname=savfolder+'out_sparsewt_pcs'+ext
        z1=np.rad2deg(Out_PCs[usef,1])
        vmin = min(20, min(z1))
        vmax=max(75, max(z1))
        plot_heatmap( x1=Wout[usef], x2=SparseWt[usef], xname='Wt Out scale', yname='Input wt sparsity', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )



    if nneufrac>1:
        fname=savfolder+'out_sparseneu_intD'+ext
        z1=np.rad2deg(Out_Int[usef,1])
        vmin = min(20, min(z1))
        vmax=max(75, max(z1))
        plot_heatmap( x1=Wout[usef], x2=SparseNeu[usef], xname='Wt Out scale', yname='Input sparsity (%neurons)', z1=z1, fname=fname, vmin=vmin, vmax=vmax,do_int=True, include_z=goodtrain )

        fname=savfolder+'out_sparseneu_pcs'+ext
        z1=np.rad2deg(Out_PCs[usef,1])
        vmin = min(20, min(z1))
        vmax=max(75, max(z1))
        plot_heatmap( x1=Wout[usef], x2=SparseNeu[usef], xname='Wt Out scale', yname='Input sparsity (%neurons)', z1=z1, fname=fname, vmin=vmin, vmax=vmax , do_int=True, include_z=goodtrain )


    #  ------------ Plot performance
    #----------------------------------------------------------------
    # vs wfbk and perturbation scale - on trained readout
    fname=savfolder+'fbk_amp_hitrate'+ext
    z1=HitRate_T_I[usef,0]
    vmin = min(0, min(z1))
    vmax=max(0.5, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=bumpamp[usef], xname='Wt Fbk scale', yname='Perturbation amp', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )

    # vs wfbk and perturbation scale - on intuitive decoder
    fname=savfolder+'fbk_amp_hitrate_int'+ext
    z1=HitRate_T_I[usef,1]
    vmin = min(0, min(z1))
    vmax=max(0.5, max(z1))
    plot_heatmap( x1=Wfbk[usef], x2=bumpamp[usef], xname='Wt Fbk scale', yname='Perturbation amp', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )

    #  ------------ Plot control analyses
    #----------------------------------------------------------------
    fname=savfolder+'henrici'+ext
    z1=henrici[usef]
    vmin = min(0, min(z1))
    vmax=max(10, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wout scale', yname='Wrec scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )


    fname=savfolder+'ctrb_ff'+ext
    z1=ctrb_ff[usef]
    vmin = min(0, min(z1))
    vmax=max(5, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wout scale', yname='Wrec scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )

    fname=savfolder+'ctrb_fb'+ext
    z1=ctrb_fb[usef]
    vmin = min(0, min(z1))
    vmax=max(10, max(z1))
    plot_heatmap( x1=Wout[usef], x2=Wrec[usef], xname='Wout scale', yname='Wrec scale', z1=z1, fname=fname, vmin=vmin, vmax=vmax, include_z=goodtrain  )

    pp.close('all')




############################################################################################################################################
#### ------------------------- Plotting helpers -----------------------------------------------------------------
############################################################################################################################################

def plot_heatmap( x1, x2, xname, yname, z1, fname='test.png', vmin=0, vmax=90, do_int=False, include_z=None ):

    x1U = np.unique(x1)
    x2U = np.unique(x2)

    X, Y = np.meshgrid( x1U, x2U )
    ctr= np.zeros( (len(x1U), len(x2U)) )
    Z= np.zeros( (len(x1U), len(x2U)) )
    files = np.arange(len(z1))

    if include_z is not None:
        files=files[include_z]
    

    for ff in files:
        wo = (x1U==x1[ff])
        wr = (x2U==x2[ff])
        ctr[wo,wr]+=1
        Z[wo,wr]+=z1[ff]

    ctr[ctr==0] = np.nan

    Z = np.divide(Z,ctr)


    fig, ax = pp.subplots()
    im = ax.imshow(Z, vmin=vmin, vmax=vmax)

    # Show all ticks and label them with the respective list entries
    ax.set_yticks(np.arange(len(x1U)), labels=[np.str_(yy) for yy in x1U])
    ax.set_ylabel( xname)
    ax.set_xticks(np.arange(len(x2U)),labels=[np.str_(xx) for xx in x2U])
    ax.set_xlabel( yname )
    
    for i in range(len(x1U)):
        for j in range(len(x2U)):
            if do_int:
                text = ax.text(j, i, np.int_(Z[i, j]),
                        ha="center", va="center", color="w")
            else:
                text = ax.text(j, i, "{:.2f}".format(Z[i, j]),
                        ha="center", va="center", color="w")
    
    fig.tight_layout()
    pp.savefig( fname )
    pp.close(fig)


def plot_principal_angles( xval, angles, savfolder, xlabel='output wt scale', ylabel='angle of readout with feedback wts', fname='out_fbk_', ext='.png'):
    f1 = pp.figure()
    pp.scatter( xval, angles[:,0], label='maxA')
    pp.scatter( xval, angles[:,1], label='minA')
    pp.xlabel(xlabel)
    pp.ylabel(ylabel)
    pp.legend()
    pp.savefig(savfolder+fname+ext)
    pp.close(f1)


def plot_curve( curve, savfolder='', xval=None, xlabel='# vectors', ylabel='angle with readout', fname='out_pccurve_', ext='.png', **kwargs ):
    f3 = pp.figure()
    if 'curve' in kwargs.keys():
        for jj in range(len(kwargs['curve'])):
            if xval is not None:
                pp.plot( xval, kwargs['curve'][jj].T, label=kwargs['clabel'][jj])
            else:
                pp.plot( kwargs['curve'][jj].T, label=kwargs['clabel'][jj])
    else:
        if xval is not None:
            pp.plot( xval, curve.T, label='minA')
        else:
            pp.plot( curve.T, label='minA')
    pp.xlabel(xlabel)
    pp.ylabel(ylabel)
    #pp.legend()
    pp.savefig(savfolder+fname+ext)
    pp.close(f3)


def plot_scatter( xval, yval_list, labels,  savfolder='', xlabel='Output wt scale', ylabel='angle with intuitive decoder', fname='out_intD_', ext='.png'):
    f11 = pp.figure()
    allx = np.unique(xval)
    meany = np.zeros(shape=allx.shape)
    sdy = np.zeros(shape=allx.shape)
    for jj in range(len(yval_list)):
        pp.scatter( xval, yval_list[jj], label=labels[jj])
        for xx in range(len(allx)):
            meany[xx] = np.mean(yval_list[jj][xval==allx[xx]])
            sdy[xx] = np.std(yval_list[jj][xval==allx[xx]])
        pp.errorbar(allx, meany, yerr=sdy, ecolor='k')
    pp.xlabel(xlabel)
    pp.ylabel(ylabel)
    pp.legend()
    pp.savefig(savfolder+fname+ext)
    pp.close(f11)

