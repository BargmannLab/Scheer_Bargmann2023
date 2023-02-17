# FUNCTIONS
import os
import pandas as pd
import numpy as np
import ssm
import copy
from sklearn.model_selection import train_test_split
import h5py

from commonFunctions import *

def preprocessData(dataPath, matFile, videonameh5file, binSize_in, minFillLen, RD_hmm_exog, RD_hmm_Cent_exog, Scalers, arHMM_Model,dataType):
    acceptableTypes = ['UL', 'LL', 'TwoDensity']
    assert dataType in acceptableTypes, "dataType not acceptable."
    if dataType == 'UL':
        boundaryVarsOK = False
        twoDensityVarsOK = False
    elif dataType == 'LL':
        boundaryVarsOK = True
        twoDensityVarsOK = False
    else:  # must be TwoDensity by elimination
        boundaryVarsOK = True
        twoDensityVarsOK = True

    print('Processing: ' + matFile[0:-2] + ' ...')
    print('Data type is '+dataType)

    global binSize  # make this available to all subroutines
    binSize = binSize_in
    Data = dict()

    #### parse dates and matfilenames to pandas structures
    d = h5py.File(os.path.join(dataPath, videonameh5file), 'r')
    videoname = pd.DataFrame(np.array([d['videoname'][i].decode("utf-8") for i in range(len(d['videoname']))]))
    print("Videoname size = " + str(videoname.shape))
    Data['videoname'] = videoname
    try:
        dates = videoname.T.iloc[0].str.extract(r'(_[0-9]{6}_)').T.iloc[0].map(lambda x: x.lstrip('_').rstrip('_')).astype(int)
        dates = pd.to_datetime(dates.astype(str).apply(lambda x: x.zfill(6)),format='%m%d%y').dt.date  # convert to a dataframe using datetime
    except:  # occasionally i fucked up the naming scheme, and this should always work (maybe i should just always do it this way?)
        print('oops! some datestrings were not manually included - switching to other method.')
        dates = videoname.T.iloc[0].str.extract(r'([0-9]{4}-[0-9]{2}-[0-9]{2})').squeeze().astype(str).str.replace('-','')
        dates = pd.to_datetime(dates, format='%Y%m%d').dt.date
    Data['dates'] = dates

    matfilename = pd.DataFrame(np.array([d['matfilename'][i].decode("utf-8") for i in range(len(d['matfilename']))]))
    matfilename = pd.DataFrame(np.array(list(map(lambda x: x[0], matfilename.to_numpy()))),columns=['matfilename'])  # get rid of array bracketing of strings (this is clunky but should work)
    Data['matfilename'] = matfilename

    #### begin reading in data from matfile
    f = h5py.File(os.path.join(dataPath, matFile), 'r')
    # define consensus missing data
    Data['pixpermm'] = np.array(f['PIXPERMM']).T
    Data['bgvidindex'] = np.array(f['BGVIDINDEX']).T
    missingdata = np.array(f['splineMissing']).astype(bool).T
    missingdata = np.logical_or(missingdata, np.isnan(np.array(f['Head_grayscale_v0']).T))  # sometimes there is a discrepancy between head grayscale and splineMissing.
    Data['missingdata'] = missingdata

    if boundaryVarsOK:
        Data['Center_Point_x'] = np.array(f['Center_Point_x']).T
        Data['Center_Point_y'] = np.array(f['Center_Point_y']).T
        Data['Lawn_Boundary_Pts_x'] = np.array(f['Lawn_Boundary_Pts_x']).T
        Data['Lawn_Boundary_Pts_y'] = np.array(f['Lawn_Boundary_Pts_y']).T
        if twoDensityVarsOK:
            Data['DenseLawn_Boundary_Pts_x'] = np.array(f['DenseLawn_Boundary_Pts_x']).T
            Data['DenseLawn_Boundary_Pts_y'] = np.array(f['DenseLawn_Boundary_Pts_y']).T
            Data['Seam_Pts_x'] = np.array(f['Seam_Pts_x']).T
            Data['Seam_Pts_y'] = np.array(f['Seam_Pts_y']).T

    #centroid measurements
    centMissing_tmp = np.array(f['centMissing']).astype(bool).T
    CentroidSpeed_tmp = readDataFillNA(np.array(f['Centroid_speed']).T, centMissing_tmp, minFillLen)
    # sometimes due to the jitter of this metric, we can get speeds that are unrealistic
    # simply censor these data by adding them to the centMissing, then read in data again.
    badCentroid = CentroidSpeed_tmp > 0.2
    Data['centMissing'] = np.logical_or(centMissing_tmp, badCentroid)

    #default values (OK for UL data, overwritten for LL and TwoDensity)
    Data['In_Or_Out'] = np.zeros_like(Data['missingdata']).astype(bool)
    Data['CentroidInLawn'] = np.ones_like(Data['missingdata']).astype(bool)
    if boundaryVarsOK:
        # read lawn-related data (don't fill missing data)
        print('Read in lawn-related data...')
        Data['Radial_Dist'] = np.array(f['Radial_Dist']).T
        Data['Lawn_Boundary_Dist'] = np.array(f['Lawn_Boundary_Dist']).T
        Data['In_Or_Out'] = np.array(f['In_Or_Out']).astype(bool).T
        Data['Lawn_Entry'] = np.array(f['Lawn_Entry']).astype(bool).T
        Data['Lawn_Exit'] = np.array(f['Lawn_Exit']).astype(bool).T
        Data['HeadInLawn'] = np.array(f['HeadInLawn']).astype(bool).T
        Data['MidbodyInLawn'] = np.array(f['MidbodyInLawn']).astype(bool).T
        Data['TailInLawn'] = np.array(f['TailInLawn']).astype(bool).T
        Data['HeadPokeAngle'] = np.array(f['HeadPokeAngle']).T
        Data['HeadPokeDist'] = np.array(f['HeadPokeDist']).T
        Data['HeadPokeFwd'] = np.array(f['HeadPokeFwd']).astype(bool).T
        Data['HeadPokeIntervals'] = np.array(f['HeadPokeIntervals']).astype(bool).T
        Data['HeadPokePause'] = np.array(f['HeadPokePause']).astype(bool).T
        Data['HeadPokeRev'] = np.array(f['HeadPokeRev']).astype(bool).T
        Data['HeadPokeSpeed'] = np.array(f['HeadPokeSpeed']).T
        Data['HeadPokesAll'] = np.array(f['HeadPokesAll']).astype(bool).T
        Data['radTrajAngle'] = readDataFillNA(np.array(f['radTrajAngle']).T, missingdata, minFillLen)
        Data['Centroid_Radial_Dist'] = readDataFillNA(np.array(f['Centroid_Radial_Dist']).T, Data['centMissing'], minFillLen)
        Data['Centroid_Lawn_Boundary_Dist'] = readDataFillNA(np.array(f['Centroid_Lawn_Boundary_Dist']).T,Data['centMissing'], minFillLen)
        Data['CentroidInLawn'] = readDataFillNA(np.array(f['CentroidInLawn']).T, Data['centMissing'], minFillLen)
        if twoDensityVarsOK:
            Data['CentroidDistToDenseSeam'] = readDataFillNA(np.array(f['CentroidDistToDenseSeam']).T, Data['centMissing'],minFillLen)
            Data['HeadDistToDenseSeam'] = readDataFillNA(np.array(f['HeadDistToDenseSeam']).T, Data['centMissing'], minFillLen)
            Data['MidbodyCentDistToDenseSeam'] = readDataFillNA(np.array(f['MidbodyCentDistToDenseSeam']).T,Data['centMissing'],minFillLen)
            Data['TrajAngleToDenseSeam'] = readDataFillNA(np.array(f['TrajAngleToDenseSeam']).T, Data['centMissing'],minFillLen)
            Data['CentroidRegion'] = np.array(f['CentroidRegion']).T  # in 10 sec bin format, make these into Centroid_fracInDense, Centroid_fracInSparse, Centroid_fracOutside, etc.
            Data['HeadRegion'] = np.array(f['HeadRegion']).T
            Data['MidbodyCentRegion'] = np.array(f['MidbodyCentRegion']).T
            Data['TailRegion'] = np.array(f['TailRegion']).T

    ##### read locomotion data
    print('Read in features and fill missing data for sufficiently short intervals...')
    Data['Midbody_cent_smth_x'] = readDataFillNA(np.array(f['Midbody_cent_smth_x']).T, missingdata, minFillLen)
    Data['Midbody_cent_smth_y'] = readDataFillNA(np.array(f['Midbody_cent_smth_y']).T, missingdata, minFillLen)
    Data['Midbody_cent_x'] = readDataFillNA(np.array(f['Midbody_cent_x']).T, missingdata, minFillLen)
    Data['Midbody_cent_y'] = readDataFillNA(np.array(f['Midbody_cent_y']).T, missingdata, minFillLen)
    Data['Head_cent_smth_x'] = readDataFillNA(np.array(f['Head_cent_smth_x']).T, missingdata, minFillLen)
    Data['Head_cent_smth_y'] = readDataFillNA(np.array(f['Head_cent_smth_y']).T, missingdata, minFillLen)
    Data['Head_cent_x'] = readDataFillNA(np.array(f['Head_cent_x']).T, missingdata, minFillLen)
    Data['Head_cent_y'] = readDataFillNA(np.array(f['Head_cent_y']).T, missingdata, minFillLen)
    Data['Tail_cent_smth_x'] = readDataFillNA(np.array(f['Tail_cent_smth_x']).T, missingdata, minFillLen)
    Data['Tail_cent_smth_y'] = readDataFillNA(np.array(f['Tail_cent_smth_y']).T, missingdata, minFillLen)
    Data['Tail_cent_x'] = readDataFillNA(np.array(f['Tail_cent_x']).T, missingdata, minFillLen)
    Data['Tail_cent_y'] = readDataFillNA(np.array(f['Tail_cent_y']).T, missingdata, minFillLen)
    Data['Midbody_speed'] = readDataFillNA(np.array(f['Midbody_speed']).T, missingdata, minFillLen)
    Data['Head_speed'] = readDataFillNA(np.array(f['Head_speed']).T, missingdata, minFillLen)
    Data['Tail_speed'] = readDataFillNA(np.array(f['Tail_speed']).T, missingdata, minFillLen)
    Data['Quirkiness'] = readDataFillNA(np.array(f['Quirkiness']).T, missingdata, minFillLen)
    Data['headAngVel_relMid'] = readDataFillNA(np.array(f['headAngVel_relMid']).T, missingdata, minFillLen)
    Data['headRadVel_relMid'] = readDataFillNA(np.array(f['headRadVel_relMid']).T, missingdata, minFillLen)
    Data['Omega'] = np.array(f['OMEGA']).astype(bool).T
    Data['Centroid_x'] = readDataFillNA(np.array(f['CENTROID_bbox_x']).T, Data['centMissing'], minFillLen)
    Data['Centroid_y'] = readDataFillNA(np.array(f['CENTROID_bbox_y']).T, Data['centMissing'], minFillLen)
    Data['Centroid_speed'] = readDataFillNA(np.array(f['Centroid_speed']).T, Data['centMissing'], minFillLen)

    ####
    # Calculate angular speed for Head, Midbody and Tail using the method used in O'Donnell et al (2018) - dot product/arc-cos method
    #Centroid
    angspeed = np.empty_like(Data['Centroid_x'])
    angspeed[:] = np.nan
    for i in range(Data['Centroid_x'].shape[0]):
        centroid = np.vstack([Data['Centroid_x'][i], Data['Centroid_y'][i]]).T
        angspeed[i, :] = angspeed_sengupta(centroid).ravel()
    Data['Centroid_angspeed'] = angspeed
    # Midbody
    cent_x = Data['Midbody_cent_x']
    cent_y = Data['Midbody_cent_y']
    angspeed = np.empty_like(Data['Midbody_speed'])
    angspeed[:] = np.nan
    for i in range(Data['Midbody_speed'].shape[0]):
        centroid = np.vstack([cent_x[i], cent_y[i]]).T
        angspeed[i, :] = angspeed_sengupta(centroid).ravel()
    Data['Midbody_angspeed'] = angspeed
    # Head
    cent_x = Data['Head_cent_x']
    cent_y = Data['Head_cent_y']
    angspeed = np.empty_like(Data['Head_speed'])
    angspeed[:] = np.nan
    for i in range(Data['Head_speed'].shape[0]):
        centroid = np.vstack([cent_x[i], cent_y[i]]).T
        angspeed[i, :] = angspeed_sengupta(centroid).ravel()
    Data['Head_angspeed'] = angspeed
    # Tail
    cent_x = Data['Tail_cent_x']
    cent_y = Data['Tail_cent_y']
    angspeed = np.empty_like(Data['Tail_speed'])
    angspeed[:] = np.nan
    for i in range(Data['Tail_speed'].shape[0]):
        centroid = np.vstack([cent_x[i], cent_y[i]]).T
        angspeed[i, :] = angspeed_sengupta(centroid).ravel()
    Data['Tail_angspeed'] = angspeed

    # Derive Moving Forward and Reverse based on speed threshold
    speed_thresh = 0.02  # mm/sec
    Data['MovingReverse'] = Data['Midbody_speed'] < -speed_thresh
    Data['MovingForward'] = Data['Midbody_speed'] > speed_thresh
    Data['Pause'] = abs(Data['Midbody_speed']) <= speed_thresh

    # Derive absolute speed from Midbody speed in order to disqualify any traces using the
    # removeIdxfromDataCriteria. Also include the stipulation any track containing only zeros (this was the MATLAB
    # initialization for every variable) should be removed
    # Head and Tail speed are left as absolute speeds, Midbody speed is + forward, - reverse, so separate these out.
    Midbody_absSpeed = abs(Data['Midbody_speed'])
    Midbody_fspeed = copy.deepcopy(Data['Midbody_speed'])
    Midbody_fspeed[Midbody_fspeed < 0] = 0
    Midbody_rspeed = copy.deepcopy(Data['Midbody_speed'])
    Midbody_rspeed[Midbody_rspeed >= 0] = 0
    Midbody_rspeed = -1 * Midbody_rspeed  # invert this so that it is also a positive number

    Data['Midbody_absSpeed'] = Midbody_absSpeed
    Data['Midbody_fspeed'] = Midbody_fspeed
    Data['Midbody_rspeed'] = Midbody_rspeed

    # my approach for modeling work defined midbody forward speed as speed when moving forward and 0 during reversals.
    # This is therefore slightly different than ignoring the reversal and paused frames. For the sake of interpretability,
    # we make new fields called midbody_fspeed_adj and midbody_rspeed_adj that use NaNs instead of zeros.
    # Note that this creates a ton of NaN's especially in the rspeed category. But the binned values are more reflective of the true measured values.
    midbody_fspeed_adj = copy.deepcopy(Data['Midbody_speed'])
    # midbody_fspeed_adj[~Data['MovingForward']] = np.nan #only forward movement allowed
    midbody_fspeed_adj[midbody_fspeed_adj <= 0] = np.nan  # only forward movement allowed (no 0)
    Data['Midbody_fspeed_adj'] = midbody_fspeed_adj
    midbody_rspeed_adj = copy.deepcopy(Data['Midbody_speed'])
    # midbody_rspeed_adj[~Data['MovingReverse']] = np.nan #only negative (not 0)
    midbody_rspeed_adj[midbody_rspeed_adj >= 0] = np.nan  # only reverse movement allowed (no 0)
    midbody_rspeed_adj = -1 * midbody_rspeed_adj  # invert so its also a positive number
    Data['Midbody_rspeed_adj'] = midbody_rspeed_adj

    #### Read in grayscale to compute bacterial density
    Data['Grayscale_bounds'] = np.array(f['Grayscale_bounds']).T
    Data['Head_grayscale'] = readDataFillNA(np.array(f['Head_grayscale']).T, missingdata, minFillLen)  # raw values of the head from the inverted original background (so lighter pixels are from more dense areas of bacterial food)

    # Rescale grayscale values to the establish grayscale bounds per video (then linearly scale from 0 to 1)
    Bacterial_Density = Data['Head_grayscale']
    for i in range(Bacterial_Density.shape[0]):
        min_gs = Data['Grayscale_bounds'][i][0]
        max_gs = Data['Grayscale_bounds'][i][-1]
        Bacterial_Density[i] = linscaleData_0to1(Bacterial_Density[i], min_gs, max_gs)
    Data['Bacterial_Density'] = Bacterial_Density
    Data['Centroid_grayscale'] = readDataFillNA(np.array(f['Centroid_grayscale']).T, Data['centMissing'], minFillLen)  # raw values of the Centroid from the inverted original background (so lighter pixels are from more dense areas of bacterial food)
    # Rescale grayscale values to the establish grayscale bounds per video (then linearly scale from 0 to 1)
    Bacterial_Density = Data['Centroid_grayscale']
    for i in range(Bacterial_Density.shape[0]):
        min_gs = Data['Grayscale_bounds'][i][0]
        max_gs = Data['Grayscale_bounds'][i][-1]
        Bacterial_Density[i] = linscaleData_0to1(Bacterial_Density[i], min_gs, max_gs)
    Data['Centroid_Bacterial_Density'] = Bacterial_Density

    if not twoDensityVarsOK:
        Data['Head_norm_grayscale'] = readDataFillNA(np.array(f['Head_norm_grayscale']).T, missingdata, minFillLen)  # based on an average radial profile of the lawn
        Data['Centroid_norm_grayscale'] = readDataFillNA(np.array(f['Centroid_norm_grayscale']).T, Data['centMissing'], minFillLen)  # based on an average radial profile of the lawn

    #### Bin Data
    print('bin Data...')
    bins = np.arange(0, Data['Head_speed'].shape[1], binSize)
    Data['bin_Head_grayscale'] = binData(Data['Head_grayscale'], binSize, np.nanmean)
    Data['bin_Centroid_grayscale'] = binData(Data['Centroid_grayscale'], binSize, np.nanmean)
    #Bin grayscale-related measures
    if not twoDensityVarsOK:
        tmp_binHNG = binData(Data['Head_norm_grayscale'], binSize, np.nanmean)
        for i in range(tmp_binHNG.shape[0]):
            tmp_binHNG[i] = minmaxnorm(tmp_binHNG[i])
        Data['bin_Head_norm_grayscale'] = tmp_binHNG

        tmp_binCNG = binData(Data['Centroid_norm_grayscale'], binSize, np.nanmean)
        for i in range(tmp_binCNG.shape[0]):
            tmp_binCNG[i] = minmaxnorm(tmp_binCNG[i])
        Data['bin_Centroid_norm_grayscale'] = tmp_binCNG

    tmp_binBD = binData(Data['Bacterial_Density'], binSize, np.nanmean)  # re-normalize 0-1 per track (the binned version)
    for i in range(tmp_binBD.shape[0]):
        tmp_binBD[i] = minmaxnorm(tmp_binBD[i])
    Data['bin_Bacterial_Density'] = tmp_binBD

    tmp_binBD = binData(Data['Centroid_Bacterial_Density'], binSize, np.nanmean)  # re-normalize 0-1 per track (the binned version)
    for i in range(tmp_binBD.shape[0]):
        tmp_binBD[i] = minmaxnorm(tmp_binBD[i])
    Data['bin_Centroid_Bacterial_Density'] = tmp_binBD

    # locomotory features, bin by averaging
    Data['bin_Head_speed'] = binData(Data['Head_speed'], binSize, np.nanmean)
    Data['bin_Midbody_absSpeed'] = binData(Data['Midbody_absSpeed'], binSize, np.nanmean)
    Data['bin_Midbody_fspeed'] = binData(Data['Midbody_fspeed'], binSize, np.nanmean)
    Data['bin_Midbody_rspeed'] = binData(Data['Midbody_rspeed'], binSize, np.nanmean)
    Data['bin_Tail_speed'] = binData(Data['Tail_speed'], binSize, np.nanmean)
    Data['bin_MovingForward'] = binData(Data['MovingForward'], binSize, np.nanmean)
    Data['bin_MovingReverse'] = binData(Data['MovingReverse'], binSize, np.nanmean)
    Data['bin_Pause'] = binData(Data['Pause'], binSize, np.nanmean)
    Data['bin_Quirkiness'] = binData(Data['Quirkiness'], binSize, np.nanmean)
    Data['bin_Centroid_speed'] = binData(Data['Centroid_speed'], binSize, np.nanmean)
    Data['bin_Centroid_angspeed'] = binData(Data['Centroid_angspeed'], binSize, np.nanmean)
    #bin using a different method that averages no matter how many NaNs there are

    Data['bin_Midbody_fspeed_adj'] = binDataNoNaNThresh(midbody_fspeed_adj, binSize, np.nanmean)
    Data['bin_Midbody_rspeed_adj'] = binDataNoNaNThresh(midbody_rspeed_adj, binSize, np.nanmean)
    # take absolute values before averaging
    Data['bin_Head_angspeed'] = binData(abs(Data['Head_angspeed']), binSize, np.nanmean)
    Data['bin_Midbody_angspeed'] = binData(abs(Data['Midbody_angspeed']), binSize, np.nanmean)
    Data['bin_Tail_angspeed'] = binData(abs(Data['Tail_angspeed']), binSize, np.nanmean)
    Data['bin_headAngVel_relMid'] = binData(abs(Data['headAngVel_relMid']), binSize, np.nanmean)
    Data['bin_headRadVel_relMid'] = binData(abs(Data['headRadVel_relMid']), binSize, np.nanmean)

    #all dataTypes get these 2 variables (set to False and True above, respectively)
    Data['bin_In_Or_Out'] = binData(Data['In_Or_Out'], binSize, np.nanmean)
    Data['bin_CentroidInLawn'] = binData(Data['CentroidInLawn'], binSize, np.nanmean)  # needs to be thresholded
    if boundaryVarsOK:
        Data['bin_radTrajAngle'] = binData(Data['radTrajAngle'], binSize, np.nanmean)
        Data['bin_Radial_Dist'] = binData(Data['Radial_Dist'], binSize, np.nanmean)
        Data['bin_Lawn_Boundary_Dist'] = binData(Data['Lawn_Boundary_Dist'], binSize, np.nanmean)
        Data['bin_LawnEntry'] = binData(Data['Lawn_Entry'], binSize, np.nanmean)
        Data['bin_LawnExit'] = binData(Data['Lawn_Exit'], binSize, np.nanmean)
        Data['bin_HeadPokeFwd'] = binData(Data['HeadPokeFwd'], binSize, np.nanmean)
        Data['bin_HeadPokeRev'] = binData(Data['HeadPokeRev'], binSize, np.nanmean)
        Data['bin_HeadPokePause'] = binData(Data['HeadPokePause'], binSize, np.nanmean)
        Data['bin_HeadPokesAll'] = binData(Data['HeadPokesAll'], binSize, np.nanmean)
        Data['bin_HeadPokeSpeed'] = binData(Data['HeadPokeSpeed'], binSize, np.nanmean)
        Data['bin_HeadPokeAngle'] = binData(Data['HeadPokeAngle'], binSize, np.nanmean)
        Data['bin_HeadPokeDist'] = binData(Data['HeadPokeDist'], binSize, np.nanmean)
        Data['bin_Centroid_Radial_Dist'] = binData(Data['Centroid_Radial_Dist'], binSize, np.nanmean)
        Data['bin_Centroid_Lawn_Boundary_Dist'] = binData(Data['Centroid_Lawn_Boundary_Dist'], binSize, np.nanmean)

    if twoDensityVarsOK:
        Data['bin_CentroidDistToDenseSeam'] = binData(Data['CentroidDistToDenseSeam'], binSize, np.nanmean)
        Data['bin_HeadDistToDenseSeam'] = binData(Data['HeadDistToDenseSeam'], binSize, np.nanmean)
        Data['bin_MidbodyCentDistToDenseSeam'] = binData(Data['MidbodyCentDistToDenseSeam'], binSize, np.nanmean)
        Data['bin_TrajAngleToDenseSeam'] = binData(Data['TrajAngleToDenseSeam'], binSize, np.nanmean)
        # turn these into frac in region (i.e. bin_CentroidInSparse = fraction of bin Centroid in Sparse)
        # Dense == 1, Sparse == 0, Outside ==2
        Data['CentroidInSparse'] = Data['CentroidRegion'] == 0
        Data['CentroidInDense'] = Data['CentroidRegion'] == 1
        Data['CentroidOutside'] = Data['CentroidRegion'] == 2
        Data['bin_CentroidInSparse'] = binData(Data['CentroidInSparse'], binSize, np.nanmean)
        Data['bin_CentroidInDense'] = binData(Data['CentroidInDense'], binSize, np.nanmean)
        Data['bin_CentroidOutside'] = binData(Data['CentroidOutside'], binSize, np.nanmean)

        Data['HeadInSparse'] = Data['HeadRegion'] == 0
        Data['HeadInDense'] = Data['HeadRegion'] == 1
        Data['HeadOutside'] = Data['HeadRegion'] == 2
        Data['bin_HeadInSparse'] = binData(Data['HeadInSparse'], binSize, np.nanmean)
        Data['bin_HeadInDense'] = binData(Data['HeadInDense'], binSize, np.nanmean)
        Data['bin_HeadOutside'] = binData(Data['HeadOutside'], binSize, np.nanmean)

        Data['MidbodyCentInSparse'] = Data['MidbodyCentRegion'] == 0
        Data['MidbodyCentInDense'] = Data['MidbodyCentRegion'] == 1
        Data['MidbodyCentOutside'] = Data['MidbodyCentRegion'] == 2
        Data['bin_MidbodyCentInSparse'] = binData(Data['MidbodyCentInSparse'], binSize, np.nanmean)
        Data['bin_MidbodyCentInDense'] = binData(Data['MidbodyCentInDense'], binSize, np.nanmean)
        Data['bin_MidbodyCentOutside'] = binData(Data['MidbodyCentOutside'], binSize, np.nanmean)

        Data['TailInSparse'] = Data['TailRegion'] == 0
        Data['TailInDense'] = Data['TailRegion'] == 1
        Data['TailOutside'] = Data['TailRegion'] == 2
        Data['bin_TailInSparse'] = binData(Data['TailInSparse'], binSize, np.nanmean)
        Data['bin_TailInDense'] = binData(Data['TailInDense'], binSize, np.nanmean)
        Data['bin_TailOutside'] = binData(Data['TailOutside'], binSize, np.nanmean)

    # get binMissing, goodIntervals for spline data
    binMissing = getGoodBinIdx(Data['bin_Head_speed'], Data['bin_MovingForward'], Data['bin_Head_angspeed'], Data['bin_headAngVel_relMid'], Data['bin_Head_angspeed'], Data['bin_Head_grayscale'])
    Data['binMissing_spline'] = binMissing
    # for centroid data
    Data['binMissing_centroid'] = np.isnan(Data['bin_Centroid_speed'])

    #### FIND RUNS INSIDE AND OUTSIDE THE LAWN
    # Permit longer stretches of data to remain included in analysis while graying out missing data from the InLawnRunMask
    InLawnMinRunLen = 5  # in 10 sec bins = 50 seconds
    OutLawnMinRunLen = 5
    NaNInterpThresh = 3  # the maximum interval length that we will interpolate in a later step
    strictBool = False
    InLawnRunMask, OutLawnRunMask = collectOnandOffLawnIntervals_decode(Data['bin_In_Or_Out'],
                                                                        Data['binMissing_spline'], strictBool,
                                                                        InLawnMinRunLen,
                                                                        OutLawnMinRunLen, NaNInterpThresh)
    Data['InLawnRunMask'] = InLawnRunMask
    Data['OutLawnRunMask'] = OutLawnRunMask

    # make a separate mask for bin centroid data - i.e. speed, angspeed, grayscale, bacterial density, and centroid R/D
    InLawnRunMask, OutLawnRunMask = collectOnandOffLawnIntervals_decode(Data['bin_In_Or_Out'],
                                                                        Data['binMissing_centroid'], strictBool,
                                                                        InLawnMinRunLen,
                                                                        OutLawnMinRunLen, NaNInterpThresh)
    Data['InLawnRunMask_centroid'] = InLawnRunMask
    Data['OutLawnRunMask_centroid'] = OutLawnRunMask

    # remove entries with aberrant speed, where entire track is NaNs or 0s
    Data, removeDataMask = removeIdxfromDataCriteria(Data)

    if boundaryVarsOK:
        # update the binLawnExit index so it matches this in lawn mask
        lookBackThresh = 3  # maximum time spacing between inlawnrunmask to end and to encounter a lawn exit
        binLawnExit_mostRecent = findMostRecentInLawnforLL(Data['bin_LawnExit'], lookBackThresh, ~Data['InLawnRunMask'])  # make sure that LL events occur at last in lawn moment
        Data['bin_LawnExit_mostRecent'] = binLawnExit_mostRecent  # this becomes now a boolean variable.

        # even if the animal isn't outside the lawn for a full 10 sec bin after lawn leaving, make sure that the bin after a lawn leaving event is set to False in the InLawnMask
        LEidx = np.where(Data['bin_LawnExit_mostRecent'])
        rowInQuestion = LEidx[0]
        nextFrame = LEidx[1] + 1
        rowInQuestion = rowInQuestion[nextFrame <= 239]
        nextFrame = nextFrame[nextFrame <= 239]
        LEidxToPutOut = [rowInQuestion, nextFrame]
        LEidxToPutOut_flat = np.ravel_multi_index(LEidxToPutOut, Data['InLawnRunMask'].shape)
        lawnrunmask_copy = copy.deepcopy(Data['InLawnRunMask'])
        np.ravel(lawnrunmask_copy)[LEidxToPutOut_flat] = False
        Data['InLawnRunMask'] = lawnrunmask_copy
        lawnrunmask_centroid_copy = copy.deepcopy(Data['InLawnRunMask_centroid'])
        np.ravel(lawnrunmask_centroid_copy)[LEidxToPutOut_flat] = False
        Data['InLawnRunMask_centroid'] = lawnrunmask_centroid_copy

    # now make an "_inLawn" versions of variables:
    # an InLawnRunMask - filtered, NaN-interpolated version of all binned data
    # these data should have NaNs only in the masked data segments (from InLawnRunMask)

    #These keys should work for all datatypes
    allDataKeys = ['bin_Head_speed', 'bin_Midbody_absSpeed', 'bin_Midbody_fspeed', 'bin_Midbody_rspeed',
                      'bin_Tail_speed', 'bin_Head_grayscale', 'bin_Bacterial_Density',
                      'bin_MovingForward', 'bin_MovingReverse', 'bin_Pause', 'bin_Quirkiness',
                      'bin_Head_angspeed', 'bin_Midbody_angspeed', 'bin_Tail_angspeed', 'bin_headAngVel_relMid',
                      'bin_headRadVel_relMid','bin_Midbody_fspeed_adj','bin_Midbody_rspeed_adj']
    keyWithInLawnName = [k + '_inLawn' for k in allDataKeys]
    for k_out, k_in in zip(keyWithInLawnName,allDataKeys):
        Data[k_out] = interpDataInLawn(Data[k_in], Data['InLawnRunMask']) #if its UL data "_inLawn" will be identical since its in the lawn all the time
    #and with variables that use the Centroid mask
    LawnBoundaryKeys = ['bin_Centroid_speed', 'bin_Centroid_angspeed', 'bin_Centroid_Bacterial_Density']
    keyWithInLawnName = [k + '_inLawn' for k in LawnBoundaryKeys]
    for k_out, k_in in zip(keyWithInLawnName, LawnBoundaryKeys):
        Data[k_out] = interpDataInLawn(Data[k_in], Data['InLawnRunMask_centroid'])

    if boundaryVarsOK:
            LawnBoundaryKeys = ['bin_radTrajAngle', 'bin_Radial_Dist', 'bin_Lawn_Boundary_Dist']
            keyWithInLawnName = [k + '_inLawn' for k in LawnBoundaryKeys]
            for k_out, k_in in zip(keyWithInLawnName,LawnBoundaryKeys):
                Data[k_out] = interpDataInLawn(Data[k_in], Data['InLawnRunMask'])
            # don't interpolate these ones
            LawnBoundaryKeys = ['bin_LawnEntry', 'bin_LawnExit', 'bin_HeadPokeFwd', 'bin_HeadPokeRev',
                                'bin_HeadPokePause','bin_HeadPokesAll', 'bin_HeadPokeSpeed', 'bin_HeadPokeAngle', 'bin_HeadPokeDist']
            keyWithInLawnName = [k + '_inLawn' for k in LawnBoundaryKeys]
            for k_out, k_in in zip(keyWithInLawnName, LawnBoundaryKeys):
                Data[k_out] = np.ma.masked_array(Data[k_in], mask=~Data['InLawnRunMask'])
            #now with the Centroid-based InLawnRunMask
            LawnBoundaryKeys = ['bin_Centroid_Radial_Dist','bin_Centroid_Lawn_Boundary_Dist']
            keyWithInLawnName = [k + '_inLawn' for k in LawnBoundaryKeys]
            for k_out, k_in in zip(keyWithInLawnName, LawnBoundaryKeys):
                Data[k_out] = interpDataInLawn(Data[k_in], Data['InLawnRunMask_centroid'])

    if twoDensityVarsOK:
        TwoDensityKeys = ['bin_HeadDistToDenseSeam', 'bin_MidbodyCentDistToDenseSeam']
        keyWithInLawnName = [k + '_inLawn' for k in TwoDensityKeys]
        for k_out, k_in in zip(keyWithInLawnName,TwoDensityKeys):
            Data[k_out] = interpDataInLawn(Data[k_in], Data['InLawnRunMask'])
        # now with the Centroid-based InLawnRunMask
        TwoDensityKeys = ['bin_CentroidDistToDenseSeam', 'bin_TrajAngleToDenseSeam']
        keyWithInLawnName = [k + '_inLawn' for k in TwoDensityKeys]
        for k_out, k_in in zip(keyWithInLawnName,TwoDensityKeys):
            Data[k_out] = interpDataInLawn(Data[k_in], Data['InLawnRunMask_centroid'])

    if not twoDensityVarsOK: #For uniform lawns and small lawns only
        Data['bin_Head_norm_grayscale_inLawn'] = interpDataInLawn(Data['bin_Head_norm_grayscale'], Data['InLawnRunMask'])
        Data['bin_Centroid_norm_grayscale_inLawn'] = interpDataInLawn(Data['bin_Centroid_norm_grayscale'], Data['InLawnRunMask_centroid'])

    # DECODE data with a supplied Roaming/Dwelling HMM and an endogenous HMM
    binMidbodyAbsspeed_InLawnRuns = splitIntoInLawnIntervals(Data['bin_Midbody_absSpeed_inLawn'],~Data['InLawnRunMask'], True)
    binMidbodyangspeed_InLawnRuns = splitIntoInLawnIntervals(Data['bin_Midbody_angspeed_inLawn'],~Data['InLawnRunMask'], True)

    decision_slope = 450
    x_offset = 0
    # observation object containing absolute speed and angular speed
    Obs_SpeedAngspeed = [np.hstack([binMidbodyAbsspeed_InLawnRuns[i].reshape(-1, 1), \
                                    binMidbodyangspeed_InLawnRuns[i].reshape(-1, 1)]) \
                         for i in range(len(binMidbodyAbsspeed_InLawnRuns))]

    # observation object of 1-dimensional roaming/dwelling
    Obs_Roaming_2D = [(binMidbodyAbsspeed_InLawnRuns[i] * decision_slope > binMidbodyangspeed_InLawnRuns[
        i] - x_offset).astype(int).reshape(-1, 1) \
                      for i in range(len(binMidbodyAbsspeed_InLawnRuns))]

    ind = np.arange(0, len(Obs_Roaming_2D))
    Obs_Roaming_2D_train, Obs_Roaming_2D_test, Obs_Roaming_2D_trainIdx, Obs_Roaming_2D_testIdx = train_test_split(
        Obs_Roaming_2D, ind, test_size=0.25, random_state=35)

    # train a categorical HMM to decode R/D states
    obs_dim = 1
    num_states = 2  # number of discrete states
    num_categories = 2  # number of output types/categories
    N_iters = 100

    # Make an HMM
    RD_hmm_endog = ssm.HMM(num_states, obs_dim, observations="categorical",observation_kwargs=dict(C=num_categories))
    ll = RD_hmm_endog.fit(Obs_Roaming_2D_train, method="em", num_iters=N_iters)
    RD_hmm_endog = permuteRDHMMStates(RD_hmm_endog, Obs_Roaming_2D,Data['bin_Midbody_absSpeed'],Data['InLawnRunMask'])  # ensure that roaming is the higher speed state

    # now decode the R/D states using the endogenous model
    RD_states_ALL_endog = decodeDatabyHMM_RD(Obs_Roaming_2D, RD_hmm_endog)
    RD_states_Matrix_endog = glomInLawnRunsToMatrix(RD_states_ALL_endog, Data['InLawnRunMask'])

    # and the exogenous model
    RD_states_ALL_exog = decodeDatabyHMM_RD(Obs_Roaming_2D, RD_hmm_exog)
    RD_states_Matrix_exog = glomInLawnRunsToMatrix(RD_states_ALL_exog, Data['InLawnRunMask'])

    # save new fields into Data
    Data['RD_decision_slope'] = decision_slope
    Data['RD_x_offset'] = x_offset
    Data['RD_Obs_SpeedAngspeed'] = Obs_SpeedAngspeed
    Data['RD_Obs_Roaming_2D'] = Obs_Roaming_2D
    Data['RD_hmm_endog'] = RD_hmm_endog
    Data['RD_states_Matrix_endog'] = RD_states_Matrix_endog
    Data['RD_hmm_exog'] = RD_hmm_exog
    Data['RD_states_Matrix_exog'] = RD_states_Matrix_exog

    ## Now, do the same but based on the Centroid measurements
    binCentspeed_InLawnRuns = splitIntoInLawnIntervals(Data['bin_Centroid_speed'], ~Data['InLawnRunMask_centroid'],True)
    binCentangspeed_InLawnRuns = splitIntoInLawnIntervals(Data['bin_Centroid_angspeed'],~Data['InLawnRunMask_centroid'], True)

    decision_slope = 450
    x_offset = 0
    # observation object containing absolute speed and angular speed
    Obs_SpeedAngspeed_Cent = [np.hstack([binCentspeed_InLawnRuns[i].reshape(-1, 1), \
                                         binCentangspeed_InLawnRuns[i].reshape(-1, 1)]) \
                              for i in range(len(binCentspeed_InLawnRuns))]

    # observation object of 1-dimensional roaming/dwelling
    Obs_Roaming_2D_centroid = [(binCentspeed_InLawnRuns[i] * decision_slope > binCentangspeed_InLawnRuns[i] - x_offset).astype(int).reshape(-1, 1) \
                               for i in range(len(binCentspeed_InLawnRuns))]

    ind = np.arange(0, len(Obs_Roaming_2D_centroid))
    Obs_Roaming_2D_centroid_train, Obs_Roaming_2D_centroid_test, Obs_Roaming_2D_centroid_trainIdx, Obs_Roaming_2D_centroid_testIdx = train_test_split(
        Obs_Roaming_2D_centroid, ind, test_size=0.25, random_state=35)

    # train a categorical HMM to decode R/D states
    obs_dim = 1
    num_states = 2  # number of discrete states
    num_categories = 2  # number of output types/categories
    N_iters = 100

    # Make an HMM
    RD_hmm_endog_Cent = ssm.HMM(num_states, obs_dim, observations="categorical",observation_kwargs=dict(C=num_categories))
    ll = RD_hmm_endog_Cent.fit(Obs_Roaming_2D_centroid_train, method="em", num_iters=N_iters)
    RD_hmm_endog_Cent = permuteRDHMMStates(RD_hmm_endog, Obs_Roaming_2D,Data['bin_Centroid_speed'],Data['InLawnRunMask'])  # ensure that roaming is the higher speed state

    # now decode the R/D states using the endogenous model
    RD_states_ALL_endog_Cent = decodeDatabyHMM_RD(Obs_Roaming_2D_centroid, RD_hmm_endog_Cent)
    RD_states_Matrix_endog_Cent = glomInLawnRunsToMatrix(RD_states_ALL_endog_Cent, Data['InLawnRunMask_centroid'])

    # and the exogenous model
    RD_states_ALL_exog_Cent = decodeDatabyHMM_RD(Obs_Roaming_2D_centroid, RD_hmm_Cent_exog)
    RD_states_Matrix_exog_Cent = glomInLawnRunsToMatrix(RD_states_ALL_exog_Cent, Data['InLawnRunMask_centroid'])

    # save new fields into Data
    Data['RD_Obs_SpeedAngspeed_Cent'] = Obs_SpeedAngspeed_Cent
    Data['RD_Obs_Roaming_2D_Cent'] = Obs_Roaming_2D_centroid
    Data['RD_hmm_endog_Cent'] = RD_hmm_endog_Cent
    Data['RD_states_Matrix_endog_Cent'] = RD_states_Matrix_endog_Cent
    Data['RD_states_Matrix_exog_Cent'] = RD_states_Matrix_exog_Cent

    ## Decode HMM states from an exogenous AR-HMM using exogenous scalers
    # Locomotion parameters that we will scale for ML state decoding
    binMidbodyFspeed = interpDataInLawn(Data['bin_Midbody_fspeed'], Data['InLawnRunMask'])  # True means IN LAWN
    binFracForward = interpDataInLawn(Data['bin_MovingForward'], Data['InLawnRunMask'])
    binMidbodyangspeed = interpDataInLawn(Data['bin_Midbody_angspeed'], Data['InLawnRunMask'])
    binHeadAngVelrelMid = interpDataInLawn(Data['bin_headAngVel_relMid'], Data['InLawnRunMask'])
    binHeadRadVelrelMid = interpDataInLawn(Data['bin_headRadVel_relMid'], Data['InLawnRunMask'])

    # using exogenous scalers
    midbodyF_scaler = Scalers['midbodyF_scaler']
    fracForward_scaler = Scalers['fracForward_scaler']
    midbodyAngspeed_scaler = Scalers['midbodyAngspeed_scaler']
    headAngVelrelMid_scaler = Scalers['headAngVelrelMid_scaler']
    headRadVelrelMid_scaler = Scalers['headRadVelrelMid_scaler']

    # scale data
    binMidbodyFspeed_ss = midbodyF_scaler.transform(binMidbodyFspeed.filled(np.nan))
    binFracForward_ss = fracForward_scaler.transform(binFracForward.filled(np.nan))
    binMidbodyangspeed_ss = midbodyAngspeed_scaler.transform(binMidbodyangspeed.filled(np.nan))
    binHeadAngVelrelMid_ss = headAngVelrelMid_scaler.transform(binHeadAngVelrelMid.filled(np.nan))
    binHeadRadVelrelMid_ss = headRadVelrelMid_scaler.transform(binHeadRadVelrelMid.filled(np.nan))

    # keep track of the original indices of in lawn runs
    rowIdx = np.tile(np.arange(0, binMidbodyFspeed_ss.shape[0]).reshape(-1, 1), (1, 240))
    timeIdx = np.tile(np.arange(0, 240), (binMidbodyFspeed_ss.shape[0], 1))
    rowIdx_InLawnRuns = splitIntoInLawnIntervals(rowIdx, ~Data['InLawnRunMask'], True)
    timeIdx_InLawnRuns = splitIntoInLawnIntervals(timeIdx, ~Data['InLawnRunMask'], True)

    binMidbodyFspeed_ss_InLawnRuns = splitIntoInLawnIntervals(binMidbodyFspeed_ss, ~Data['InLawnRunMask'], True)
    binFracForward_ss_InLawnRuns = splitIntoInLawnIntervals(binFracForward_ss, ~Data['InLawnRunMask'], True)
    binMidbodyangspeed_ss_InLawnRuns = splitIntoInLawnIntervals(binMidbodyangspeed_ss, ~Data['InLawnRunMask'], True)
    binHeadAngVelrelMid_ss_InLawnRuns = splitIntoInLawnIntervals(binHeadAngVelrelMid_ss, ~Data['InLawnRunMask'],True)
    binHeadRadVelrelMid_ss_InLawnRuns = splitIntoInLawnIntervals(binHeadRadVelrelMid_ss, ~Data['InLawnRunMask'],True)

    # generate Obs object for decoding
    Obs_forDecoding = [np.hstack([binMidbodyFspeed_ss_InLawnRuns[i].reshape(-1, 1), \
                                  binMidbodyangspeed_ss_InLawnRuns[i].reshape(-1, 1), \
                                  binFracForward_ss_InLawnRuns[i].reshape(-1, 1), \
                                  binHeadAngVelrelMid_ss_InLawnRuns[i].reshape(-1, 1), \
                                  binHeadRadVelrelMid_ss_InLawnRuns[i].reshape(-1, 1)])
                       for i in range(len(binMidbodyFspeed_ss_InLawnRuns))]

    MLstates = decodeMLStates_PerInLawnRun(arHMM_Model, Obs_forDecoding, rowIdx_InLawnRuns, timeIdx_InLawnRuns,~Data['InLawnRunMask'])

    Data['binMidbodyFspeed_ss'] = binMidbodyFspeed_ss
    Data['binFracForward_ss'] = binFracForward_ss
    Data['binMidbodyangspeed_ss'] = binMidbodyangspeed_ss
    Data['binHeadAngVelrelMid_ss'] = binHeadAngVelrelMid_ss
    Data['binHeadRadVelrelMid_ss'] = binHeadRadVelrelMid_ss
    Data['arHMM_MLstates'] = MLstates  # these are the most likely arHMM states

    return bins, Data







