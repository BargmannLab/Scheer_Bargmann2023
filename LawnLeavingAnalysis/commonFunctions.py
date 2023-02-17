# FUNCTIONS
import os
import pandas as pd
import numpy as np
import ssm
from ssm.util import one_hot
import re
import copy
from numpy.linalg import norm
import scipy.stats
from scipy.stats import multivariate_normal
from scipy.stats import mannwhitneyu
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint
from scipy.ndimage import convolve1d
from scipy import signal
from scipy.signal import find_peaks
from scipy.stats import zscore
from scipy.special import logit
from scipy.stats import ttest_ind
from functools import reduce
import itertools
import statsmodels.stats.proportion as prop

# Helper Functions
def stack_padding(it):
    """
    This function concatentates rows of different lengths into one 2d matrix of common dimension
    :param it: a tuple of two rows
    :return: a nan-padded version of concatenating the two rows
    """
    def resize(row, size):
        new = np.array(row)
        new = np.pad(new.astype(float),pad_width=(0,size-len(new)),mode='constant',constant_values=(np.nan,))
        return new

    # find longest row length
    row_length = max(it, key=len).__len__()
    mat = np.array( [resize(row, row_length) for row in it] )

    return mat
def readDataFillNA(orig, missingdata, minFillLen):
    """
    Read in data and fill in sufficiently short intervals defined by minFillLen
    """
    nafilled = []
    for i in range(0, orig.shape[0]):
        # print(i)
        curr_orig = orig[i]
        # if np.sum(np.isnan(curr_orig))/len(curr_orig)>0.75: #in case there is a discrepancy between the missingdata and this quantity we are supposed to read
        #     missingdata[i] = np.ones(len(curr_orig)).astype(bool)
        mean = np.nanmean(curr_orig)
        imputed_orig = copy.deepcopy(curr_orig)
        imputed_orig[np.isnan(imputed_orig)] = mean
        md = missingdata[i]
        if md.all():  # if the whole track is missing data, make it all NaNs
            tmp = np.empty(len(curr_orig))
            tmp[:] = np.nan
            curr_orig = tmp
        elif md.any():  # if there is any missing
            # identify intervals to interpolate
            vector_ints = get_intervals(md, 0)
            vector_ints = np.hstack((vector_ints[:, 0].reshape(-1, 1) - 1, vector_ints[:, 1].reshape(-1, 1) + 1))
            toinsertNaNs = vector_ints[:, 1] - vector_ints[:, 0] <= minFillLen
            vector_ints_toinsertNaNs = vector_ints[toinsertNaNs]
            for j in range(0, vector_ints_toinsertNaNs.shape[0]):
                imputed_orig[vector_ints_toinsertNaNs[j, 0]:vector_ints_toinsertNaNs[j, 1]] = np.nan
            nans, x = nan_helper(imputed_orig)
            # now we reintroduce consensus missing data to curr_orig, then fill in the interpolated values
            curr_orig[md] = np.nan
            curr_orig[nans] = np.interp(x(nans), x(~nans), imputed_orig[~nans])
        else:  # if there is nothing missing, do nothing to curr_orig
            curr_orig = curr_orig
        nafilled.append(curr_orig)

    stacked_filled_Data = np.vstack(nafilled)
    return stacked_filled_Data
def fillNA(orig):  # this version is just for pre-existing np.arrays (2D)
    for idx in range(0, orig.shape[0]):  # loop over rows
        curr_orig = orig[idx]
        nans, x = nan_helper(curr_orig)
        curr_orig[nans] = np.interp(x(nans), x(~nans), curr_orig[~nans])
        orig[idx] = curr_orig
    return orig
def fillNAwithInt(orig,toFill):
    for idx in range(0, orig.shape[0]):  # loop over rows
        curr_orig = orig[idx]
        nans, x = nan_helper(curr_orig)
        curr_orig[nans] = toFill
        orig[idx] = curr_orig
    return orig
def angspeed_sengupta(centroid):
    from numpy.linalg import norm
    """
    new method for calculating angular speed based on the arc-cos of the dot product between the vectors formed from 3
    consecutive points this gives better results and more closely matches the literature like Flavell (2013) and O'Donnell (2018)
    """
    interval = 1
    centroid_os1 = centroid[interval:, :]
    centroid_os2 = centroid[2 * interval:, :]

    angspeed = np.empty((centroid.shape[0], 1))
    angspeed[:] = np.nan

    for i in range(centroid_os2.shape[0]):
        #         print(i)
        # a,b,c are coordinates of 3 consecutive time points spaced by interval
        a = centroid[i, :]
        b = centroid_os1[i, :]
        c = centroid_os2[i, :]

        DirVector_v12 = b - a
        DirVector_v23 = c - b

        # by definition of dot product, calculate angles
        dotpdt = np.dot(DirVector_v12, DirVector_v23) / (norm(DirVector_v12) * norm(DirVector_v23))
        if dotpdt<-1:
            dotpdt = -1
        if dotpdt>1:
            dotpdt = 1
        #         print(np.degrees(np.arccos(dotpdt)))
        angspeed[i + interval] = np.degrees(np.arccos(dotpdt))

    return angspeed
def binData(dataToBin, binSize, function):
    """
    bin data if at least half of the bin is non-NaN
    """
    binDataOut = np.zeros((dataToBin.shape[0], np.floor((dataToBin.shape[1]) / binSize).astype(int)))
    bins = np.arange(0, dataToBin.shape[1], binSize)
    allIdx = np.arange(0, dataToBin.shape[1])
    binsPerData = np.digitize(allIdx, bins, right=False) - 1  # restore this to 0 indexing

    for vid in range(0, dataToBin.shape[0]):
        for i in range(0, len(bins)):
            currDataChunk = dataToBin[vid, np.where(binsPerData == i)[0]]
            if np.sum(np.isnan(
                    currDataChunk)) / binSize <= 0.5:  # if at least half of the data in the bin is non-NaN, process it
                binDataOut[vid, i] = function(currDataChunk)  # function, e.g. nanmean
            else:
                binDataOut[vid, i] = np.nan

    return binDataOut
def binDataNoNaNThresh(dataToBin, binSize, function):
    binDataOut = np.zeros((dataToBin.shape[0], np.floor((dataToBin.shape[1]) / binSize).astype(int)))

    bins = np.arange(0, dataToBin.shape[1], binSize)
    allIdx = np.arange(0, dataToBin.shape[1])
    binsPerData = np.digitize(allIdx, bins, right=False) - 1  # restore this to 0 indexing

    for vid in range(0, dataToBin.shape[0]):
        for i in range(0, len(bins)):
            currDataChunk = dataToBin[vid, np.where(binsPerData == i)[0]]
            binDataOut[vid, i] = function(currDataChunk)  # function, e.g. nanmean

    return binDataOut
def firstEventIdx(row):
    EventIdx = np.where(row)[0]
    if EventIdx.size>0:
        firstEvent = np.min(EventIdx)
    else:
        firstEvent = -1
    return firstEvent
def get_intervals(data, split_val):
    """
    finds the intervals of a consecutive array element within the given vector.
    :param data: a boolean array or array of 0s and 1s
    :param split_val: 0 or 1
    :return: intervals of array split by split_val
    """
    stepsize=0
    #first binarize the data -- True wherever the data is NOT split val
    data = data != split_val
    if (data==split_val).all(): #if the whole track is the split val (in other words there are no intervals of interest)
        consecIdx_firstLast = np.array([])
    elif (data != split_val).all():
        consecIdx_firstLast = np.array([0,len(data.ravel())-1])
    else:
        consecIdx = np.array(np.split(np.r_[:len(data)], np.where(np.diff(data) != stepsize)[0]+1),dtype='object')
        whichIntsToReturn = np.where(np.array([(data[consecIdx[i].astype(int)] != split_val).any() for i in range(len(consecIdx))]))[0]
        IntsToReturn = consecIdx[whichIntsToReturn]
        consecIdx_firstLast = np.vstack([np.hstack([IntsToReturn[i][0],IntsToReturn[i][-1]]) for i in range(len(IntsToReturn))])
    if len(consecIdx_firstLast.shape)==1:
        consecIdx_firstLast = consecIdx_firstLast.reshape(1,-1) #make sure that intervals have 2 dimensions
    return consecIdx_firstLast
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def weird_division(n, d):  # avoids division by zero
    return n / d if d else 0
def gaussBlurData(y, sigma):
    gaussian_func = lambda x, sigma: 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x ** 2) / (2 * sigma ** 2))
    gau_x = np.linspace(-2.7 * sigma, 2.7 * sigma, 6 * sigma)
    gau_mask = gaussian_func(gau_x, sigma)
    y_gau = np.convolve(y, gau_mask, 'same')
    return y_gau
def movMean(y, window):
    """
    Computes a moving average across an array
    :param y: input array
    :param window: length of window to be averaged
    :return: y_avg: y after applying moving average
    """
    avg_mask = np.ones(window) / window
    y_avg = np.convolve(y, avg_mask, 'same')
    return y_avg
def movSum(y, window):
    sum_mask = np.ones(window)
    y_sum = np.convolve(y, sum_mask, 'same')
    return y_sum
def movSum_masked(arrMasked, window):
    sum_mask = np.ones(window)
    y_sum = convolveMasked(arrMasked, sum_mask)
    return y_sum

# function to convolve a masked array
def convolveMasked(arrMasked, shorterToConvolve):
    arrConvolved = copy.deepcopy(arrMasked)
    OKints = get_intervals(~arrMasked.mask, 0)
    for i in range(OKints.shape[0]):
        #         print(i)
        intLen = OKints[i, 1] - OKints[i, 0]
        if intLen >= len(shorterToConvolve):
            arrConvolved[OKints[i, 0]:OKints[i, 1]] = convolve1d(arrMasked[OKints[i, 0]:OKints[i, 1]],
                                                                 weights=shorterToConvolve, mode='nearest')
        else:
            arrConvolved[OKints[i, 0]:OKints[i, 1]] = np.repeat(np.nan, intLen)
    return arrConvolved
def minmaxnorm(Data):
    """
    Scale data linearly from 0 to 1
    """
    dmin = np.nanmin(Data.ravel())  # normalize again
    dmax = np.nanmax(Data.ravel())
    normData = (Data - dmin) / (dmax - dmin)

    return normData
def linscaleData_0to1(Data, minval, maxval):
    """
    Scale data linearly between minval and maxval, then scale 0 to 1
    """
    tmpData = copy.deepcopy(Data)
    tmpData[tmpData < minval] = minval
    tmpData[tmpData > maxval] = maxval
    normData = (tmpData - minval) / (maxval - minval)
    return normData

def linscaleData_minmax(Data,minval,maxval):
    """
    Scale data linearly between minval and maxval
    """
    return np.interp(Data, (Data.min(), Data.max()), (minval, maxval))

def getGoodBinIdx(minBinInt,*binData):
    """
    finds valid contiguous intervals of data (without missing values)

    :param binData: using as many variables as you wish, the function finds the valid contiguous bins common to all inputs
    :param minBinInt: disregard intervals less than this many bins long
    :return:
    """
    binMissing = np.zeros_like(binData[0]).astype(bool)
    for bD in binData:
        binMissing = np.logical_or(binMissing, np.isnan(bD))
    return binMissing
def collectOnandOffLawnIntervals_decode(binInOrOut, binMissing, strictBool, InLawnMinRunLen, OutLawnMinRunLen, NaNInterpThresh):
    """
    collects runs where animal is either ON and OFF lawn intervals for extended periods
    :param binInOrOut: boolean variable. 0 means animal is ON lawn, 1 means animal is OFF lawn
    :param binMissing: boolean variable, True means no data exists in this bin
    :param strictBool: boolean flag variable -- whether to include data from bins where animal is partially inside or outside the lawn
    :param InLawnMinRunLen: minimum number of bins allowed inside lawn for inside data to be considered a run
    :param OutLawnMinRunLen: minimum number of bins allowed outside lawn for inside data to be considered a run
    :param NaNInterpThresh: the maximum interval length that we will interpolate in a later step
    :return: InRunLawnMask and OutRunLawn Mask, both boolean masks for the runs considered in or out of the lawn. True means it is a good run.
    """
    InLawnRunMask = np.zeros_like(binInOrOut).astype(bool)
    OutLawnRunMask = np.zeros_like(binInOrOut).astype(bool)
    binMissing = copy.deepcopy(binMissing) #don't edit the original
    for i in range(binInOrOut.shape[0]):
        # print(i)
        if strictBool:
            InLawnBool = binInOrOut[i] == 0.0
            OutLawnBool = binInOrOut[i] == 1.0
        else:
            InLawnBool = binInOrOut[i] < 1.0  # include bins that are mixed in and out (some bins will overlap)
            OutLawnBool = binInOrOut[i] > 0.0  # include bins that are mixed in and out

        #remove any binMissing intervals less than NaNInterThresh (they will be interpolated in a later step)
        MissingIntervals = get_intervals(binMissing[i],0)
        # print(MissingIntervals)
        if MissingIntervals.size>0:
            for j in range(MissingIntervals.shape[0]):
                if (MissingIntervals[j][1] - MissingIntervals[j][0] < NaNInterpThresh):  # if the missing interval is lower than the interpolation threshold, gray it out
                    binMissing[i][np.arange(MissingIntervals[j][0], MissingIntervals[j][1] + 1)] = False

        InLawnBool = np.logical_and(InLawnBool,~binMissing[i]) #logical and so the mask will reflect both whether animal is in lawn and whether there is missing data longer than interpolation threshold
        InLawnIntervals = get_intervals(InLawnBool, 0)  # in lawn intervals

        if InLawnIntervals.size>0:
            longEnoughIdx = np.where(InLawnIntervals[:, 1] - InLawnIntervals[:, 0] >= InLawnMinRunLen)[0]
            for j in longEnoughIdx:
                    InLawnRunMask[i][np.arange(InLawnIntervals[j][0], InLawnIntervals[j][1] + 1)] = True

        OutLawnBool = np.logical_and(OutLawnBool, ~binMissing[i])
        OutLawnIntervals = get_intervals(OutLawnBool, 0)  # in lawn intervals

        if OutLawnIntervals.size>0:
            longEnoughIdx = np.where(OutLawnIntervals[:, 1] - OutLawnIntervals[:, 0] >= OutLawnMinRunLen)[0]
            for j in longEnoughIdx:
                    OutLawnRunMask[i][np.arange(OutLawnIntervals[j][0], OutLawnIntervals[j][1] + 1)] = True

    return InLawnRunMask, OutLawnRunMask
def removeIdxfromDataCriteria(Data):
    """
    Function to remove entries with aberrant speed, or where the entire track is empty or missing.
    :param Data: dictionary of items
    :return: cleanData: a cleaned up version of Data, and mask
    """
    idxToRemoveNaN = np.union1d(np.unique(np.where(Data['Midbody_absSpeed'] > 0.45)[0]), np.where(np.array([np.isnan(Data['Midbody_absSpeed'][i]).all()
                                                                                                            for i in range(0, Data['Midbody_absSpeed'].shape[0])]))[0])
    idxToRemove0s = np.where(np.array([(Data['Midbody_absSpeed'][i] == 0).all()
                                       for i in range(0, Data['Midbody_absSpeed'].shape[0])]))[0] # where the whole track is 0s
    idxToRemoveAllOutside = np.where(np.array([(Data['InLawnRunMask'][i] == False).all()
                                               for i in range(0, Data['Midbody_absSpeed'].shape[0])]))[0] # where the whole track is outside the lawn

    idxToRemove = np.union1d(np.union1d(idxToRemoveNaN, idxToRemove0s),idxToRemoveAllOutside)

    cleanData = copy.deepcopy(Data)  # don't mess with the original
    mask = np.ones(cleanData['Midbody_absSpeed'].shape[0], dtype=bool)
    mask[idxToRemove] = False

    for key, val in cleanData.items():  # subselect only those entries that are not to be removed
        if val.shape[0]==len(mask):
            cleanData[key] = val[mask]
        else: #if we need to clean up one of the expanded dimensions like Lawn Boundary Points, first figure out what the multiplicative factor is then remove all of those rows
            xfactor = int(val.shape[0]/len(mask))
            startRows = xfactor * idxToRemove #start of the intervals to black out
            endRows = startRows + xfactor #end of the intervals to black out
            idxRunsToRemove = [np.arange(startRows[i], endRows[i]) for i in range(len(startRows))] #runs of indices to black out
            allIdxToRemove = list(flatten(idxRunsToRemove)) #all of those indices concatenated
            localmask = np.ones(val.shape[0], dtype=bool) #local version of mask scaled by xfactor
            localmask[allIdxToRemove] = False #set the black out indices to False
            cleanData[key] = val[localmask]


    return cleanData, mask
def flatten(l):
    """
    flattens any irregular list of lists. call list() on the generator object that is returned
    """
    from collections.abc import Iterable
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
def findMostRecentInLawnforLL(binLawnExit, numBinsBefore, inlawnrunmask):
    """
    useful to reassociate lawn exits that are not in lawn with the most recent in lawn datapoint
    :param binLawnExit: boolean indicating whether there was a lawn leaving event in a given bin
    :param numBinsBefore: maximum time spacing between inlawnrunmask to end and to encounter a lawn exit
    :param inlawnrunmask: boolean mask -- True means animal is in the lawn.
    :return: binLawnExit_minusBins, a revised version of binLawnExit so that lawn exits coincide with the inLawnRunMask
    """
    binLawnExit_minusBins = np.zeros_like(binLawnExit).astype(bool)
    for i in range(binLawnExit_minusBins.shape[0]):
        curr_LLIdx = np.where(binLawnExit[i])[0]
        curr_InLawnIdx = np.where(~inlawnrunmask[i])[0]
        for x in curr_LLIdx:
            if np.logical_and((x - curr_InLawnIdx) < numBinsBefore, (x - curr_InLawnIdx) >= 0).any():
                binLawnExit_minusBins[i][min(max(
                    curr_InLawnIdx[
                        np.logical_and((x - curr_InLawnIdx) < numBinsBefore, (x - curr_InLawnIdx) >= 0)]),
                    x)] = True
    return binLawnExit_minusBins
def interpDataInLawn(data, InLawnRunMask):
    """
    Interpolates data when animal is in an inlawn run so there are no missing values (if there are any).
    :param data: input data dimension
    :param InLawnRunMask: boolean mask -- True means animal is in the lawn.
    :return: a NaN interpolated version of data so that all data when animal is in lawn has values
    """
    maskedInterpData = np.full_like(InLawnRunMask, np.nan, dtype=np.double)
    for i in range(InLawnRunMask.shape[0]):
        InLawnIntervals = get_intervals(InLawnRunMask[i], 0) #get intervals when the animal is in the lawn
        #remove any intervals that are only a single frame
        InLawnIntervals = InLawnIntervals[(InLawnIntervals[:,1]-InLawnIntervals[:,0])>0,:]
        if InLawnIntervals.size>0:
            if InLawnIntervals.ndim==1:
                maskedInterpData[i][InLawnIntervals[0]:InLawnIntervals[1] + 1] = interp1darray(data[i][InLawnIntervals[0]:InLawnIntervals[1] + 1])
            else:
                for j in range(InLawnIntervals.shape[0]):
                    dataTointerpolate = data[i][InLawnIntervals[j, 0]:InLawnIntervals[j, 1] + 1]
                    maskedInterpData[i][InLawnIntervals[j, 0]:InLawnIntervals[j, 1] + 1] = interp1darray(dataTointerpolate)

    maskedInterpData = np.ma.masked_array(maskedInterpData, mask=~InLawnRunMask)  # invert the mask so that True means invalid data
    return maskedInterpData
def interp1darray(t):  # just for a single vector
    """
    Interpolates data where NaNs exist using the numpy interp function.
    :param t: a single array t
    :return: interpolated version of t
    """
    if not np.isnan(t).all():
        if np.isnan(t[0]):
            t[0] = np.nanmean(t)
        if np.isnan(t[-1]):
            t[-1] = np.nanmean(t)
        nans, x = nan_helper(t)
        t[nans] = np.interp(x(nans), x(~nans), t[~nans])
    # else:
        # print("entire array is nan!")
    return t
def splitIntoInLawnIntervals(Data, inlawnrunmask, fillNaNFlag):
    """
    Split up data into intervals based on pre-computed inlawnrunmask (the mask should be True == outside lawn)
    :param Data:
    :param inlawnrunmask:
    :param fillNaNFlag: whether to fill NaNs in the process
    :return: list of data in runs derived from InLawnRunMask
    """
    Data_InLawn_Runs = []  # list to collect inlawnruns
    if len(Data.shape)==1: #in case it's just a single row, reshape it first so that the i indexing will work
        Data = Data.reshape(1,-1)
        inlawnrunmask = inlawnrunmask.reshape(1,-1)
    for i in range(Data.shape[0]):
        # print(i)
        currData = Data[i]
        if isinstance(currData,np.ma.core.MaskedArray): #fill it with NaNs first if its a masked array
            currData = currData.filled(np.nan)
        InLawnIntervals = get_intervals(~inlawnrunmask[i], 0)  # in lawn intervals
        InLawnIntervals = InLawnIntervals[(InLawnIntervals[:,1]-InLawnIntervals[:,0])>0,:] #remove any intervals that are only a single frame long

        if len(InLawnIntervals.shape) == 1:
            if fillNaNFlag:
                dataChunk = interp1darray(currData[np.arange(InLawnIntervals[0], InLawnIntervals[
                    1] + 1)])  # make sure to add one so np.arange takes it up to the last frame.
            else:
                dataChunk = currData[np.arange(InLawnIntervals[0], InLawnIntervals[1] + 1)]
            Data_InLawn_Runs.append(dataChunk)
        else:
            for j in range(InLawnIntervals.shape[0]):
                if fillNaNFlag:
                    dataChunk = interp1darray(currData[np.arange(InLawnIntervals[j][0], InLawnIntervals[j][1] + 1)])
                else:
                    dataChunk = currData[np.arange(InLawnIntervals[j][0], InLawnIntervals[j][1] + 1)]
                Data_InLawn_Runs.append(dataChunk)
    return Data_InLawn_Runs
def permuteRDHMMStates(RD_hmm, Obs_Roaming_2D, Speed, InLawnRunMask):
    RD_states_ALL = decodeDatabyHMM_RD(Obs_Roaming_2D, RD_hmm)
    RD_states_Matrix = glomInLawnRunsToMatrix(RD_states_ALL,InLawnRunMask)
    state0speeds = (Speed[RD_states_Matrix == 0]).ravel()
    state1speeds = (Speed[RD_states_Matrix == 1]).ravel()
    if np.nanmean(state0speeds)>np.nanmean(state1speeds):
        RD_hmm.permute([1, 0])
    return RD_hmm
def decodeDatabyHMM_RD(obs,hmm):
    """
    decode roaming and dwelling states
    """
    decoded = []
    for i in range(0,len(obs)):
        curr_obs = obs[i]
        decoded.append(hmm.most_likely_states(curr_obs))
    return decoded
def glomInLawnRunsToMatrix(DataRuns,InLawnRunMask):
    """
    Turn lists into masked array
    :param DataRuns:
    :param InLawnRunMask:
    :return:
    """
    rowIdx = np.tile(np.arange(0, InLawnRunMask.shape[0]).reshape(-1, 1), (1, 240))
    timeIdx = np.tile(np.arange(0, 240), (InLawnRunMask.shape[0], 1))
    rowIdx_InLawnRuns = splitIntoInLawnIntervals(rowIdx, ~InLawnRunMask, True)
    timeIdx_InLawnRuns = splitIntoInLawnIntervals(timeIdx, ~InLawnRunMask, True)
    DataMatrix = np.zeros_like(InLawnRunMask)
    for i in range(len(DataRuns)):
        # print(len(rowIdx_InLawnRuns[i]))
        # print(len(timeIdx_InLawnRuns[i]))
        # print(len(DataRuns[i]))
        DataMatrix[np.unique(rowIdx_InLawnRuns[i]), timeIdx_InLawnRuns[i]] = DataRuns[i]
    DataMatrix = np.ma.masked_array(data=DataMatrix, mask=~InLawnRunMask)
    return DataMatrix
def decodeMLStates_PerInLawnRun(ChosenModel, Obs, rowIdx_InLawnRuns, timeIdx_InLawnRuns, InLawnRunMask):
    """
    Decodes most likely states using a supplied HMM object
    :param ChosenModel: the HMM used to decode states
    :param Obs: Data in list format
    :param rowIdx_InLawnRuns: the row indices from the original matrix that is the source of the runs
    :param timeIdx_InLawnRuns: the column indices from the original matrix that is the source of the runs
    :param InLawnRunMask:
    :return: Matrix version of the decoded most likely HMM states. Matches InLawnRunMask
    """
    MLstates = np.full_like(InLawnRunMask, np.nan, dtype=np.double)
    for i in range(len(Obs)):
        currRow = np.unique(rowIdx_InLawnRuns[i])
        currTimeIdx = timeIdx_InLawnRuns[i]
        MLstates[currRow, currTimeIdx] = ChosenModel.most_likely_states(Obs[i])
    MLstates = np.ma.masked_array(data=MLstates,mask=InLawnRunMask)

    return MLstates
def augmentULdata(Data):
    """
    adds the "inLawn" fields to make this data comparable to LL data, as well as lawn exits and head pokes
    """
    Data['bin_LawnExit_mostRecent'] = np.zeros_like(Data['bin_Midbody_fspeed'])
    Data['bin_HeadPokeFwd']= np.zeros_like(Data['bin_Midbody_fspeed'])
    Data['bin_HeadPokeRev']= np.zeros_like(Data['bin_Midbody_fspeed'])
    Data['bin_HeadPokePause']= np.zeros_like(Data['bin_Midbody_fspeed'])
    Data['bin_HeadPokesAll']= np.zeros_like(Data['bin_Midbody_fspeed'])
    Data['Lawn_Boundary_Dist'] = np.zeros_like(Data['Midbody_speed']) #fake field of 0s
    Data['bin_Lawn_Boundary_Dist'] = np.zeros_like(Data['bin_Midbody_fspeed'])
    Data['bin_Lawn_Boundary_Dist_inLawn'] = Data['bin_Lawn_Boundary_Dist']

    Data['bin_Centroid_speed_inLawn'] = Data['bin_Centroid_speed']
    Data['bin_Centroid_angspeed_inLawn'] = Data['bin_Centroid_angspeed']
    Data['bin_Centroid_Bacterial_Density_inLawn'] = Data['bin_Centroid_Bacterial_Density']

    return Data
def computeStateDurations(MLstates):
    """
    Computes the length of consecutive runs of each state (represented by integers) in MLstates
    :param MLstates: A matrix of states (e.g. decoded by a HMM)
    :return: inferred_durations, a list of lists of state durations for each integer represented in MLstates
    """
    inferred_durations = [[] for _ in range(0, MLstates.max() + 1)]  # list indexed by number of state
    for i in range(0, MLstates.shape[0]):
        inf_states, inf_durs = ssm.util.rle(MLstates[i])
        i=0
        for (s, d) in zip(inf_states, inf_durs):
            # print(i,s,d)
            inferred_durations[s].append(d)
            i=i+1
    return inferred_durations
def computeStateDurations_List(MLstates_list):
    """
    Same thing but for a list.
    :param MLstates_list:
    :return:
    """
    flat_list = [item for sublist in MLstates_list for item in sublist]
    inferred_durations = [[] for _ in range(0, max(flat_list) + 1)]  # list indexed by number of state
    for i in range(0, len(MLstates_list)):
        inf_states, inf_durs = ssm.util.rle(MLstates_list[i])
        for (s, d) in zip(inf_states, inf_durs):
            inferred_durations[s].append(d)
    return inferred_durations

def getStateRunsBeforeEvent(MLstates, chosenState, EventMatrix):
    """
    Finds the intervals comprising state runs that terminate in a discrete event (e.g. lawn exit)
    :param MLstates: state matrix
    :param chosenState: the state for which you want to know durations preceding events
    :param EventMatrix: boolean event matrix
    :return validRuns: a list of the state runs that terminate in the desired event
    :return: MLstatesRuns: a masked array of MLstates consisting only of the state runs that terminate in the desired event.
    """

    validRuns = []
    MLstatesRuns = copy.deepcopy(MLstates) #copy MLstates
    MLstatesRuns.mask = np.ones((MLstates.shape[0], MLstates.shape[1])).astype(bool) #mask out everything
    for i in range(0, MLstates.shape[0]):
        # for i in [0]:
        stateseq = MLstates[i] == chosenState
        if isinstance(stateseq, np.ma.core.MaskedArray):
            stateseq = stateseq.filled(False)
        eventseq = EventMatrix[i]
        eventIdx = np.where(eventseq)[0]  # the indices when discrete events occurred
        runIntervals = get_intervals(stateseq, True)
        # check which runs of ChosenState end in a discrete event of choice
        validRunIdx = np.unique(np.where(np.isin(runIntervals, eventIdx))[0])
        if len(validRunIdx) > 0:
            curr_interval = runIntervals[validRunIdx, :]
            validRuns.append([i,curr_interval])
            for c in curr_interval:
                MLstatesRuns.mask[i,np.arange(c[0],c[1]+1)] = False

    return validRuns, MLstatesRuns

def computeStateDurationsBeforeEvent(MLstates,chosenState,EventMatrix):
    """
    Computes the state durations of a given type that terminate in a discrete event (e.g. lawn exit)
    :param MLstates: state matrix
    :param chosenState: the state for which you want to know durations preceding events
    :param EventMatrix: boolean event matrix
    :return: inferred_durations: an array of state durations that terminate in a discrete event
    """
    inferred_durations = []  # list of state durations of the chosen type that terminate in the discrete event of choice (i.e. lawn exit)
    for i in range(0, MLstates.shape[0]):
        # for i in [0]:
        stateseq = MLstates[i] == chosenState
        if isinstance(stateseq, np.ma.core.MaskedArray):
            stateseq = stateseq.filled(False)
        eventseq = EventMatrix[i]
        eventIdx = np.where(eventseq)[0]  # the indices when discrete events occurred
        runIntervals = get_intervals(stateseq, True)
        # check which runs of ChosenState end in a discrete event of choice
        validRunIdx = np.unique(np.where(np.isin(runIntervals, eventIdx))[0])
        if len(validRunIdx) > 0:
            validRuns = runIntervals[validRunIdx, :]
            validRunLens = list(validRuns[:, 1] - validRuns[:, 0] + 1)
            inferred_durations.append(validRunLens)
    inferred_durations = list(flatten(inferred_durations))
    return inferred_durations

def getProbLeavingvsTimeinState(StateDurBeforeLeaving, overallStateDur, bins):
    OverallCounts, _ = np.histogram(overallStateDur, bins=bins)
    BeforeLeaveCounts, _ = np.histogram(StateDurBeforeLeaving, bins=bins)
    phat = BeforeLeaveCounts / OverallCounts
    recip = 1 - phat
    n = OverallCounts
    ste_prop = np.sqrt(phat * recip / n)

    return phat, ste_prop, OverallCounts, BeforeLeaveCounts

def compareGenotypes(Datas,Data_labels,keysToExtract,binSize,numStates, zscore_thresh, forbiddenDates, datesToInclude, subset=None):
    """
    takes in 4 lists:
    1) Datas - each element is a dict corresponding to a preloaded data
    2) Data_labels - each element is a string name for the corresponding element of Datas
    3) keysToExtract - these are the features that you wish to compare across genotypes
    4) forbiddenDates - these are dates of experiments that you wish to exclude from analysis.
    :param subset:
    """

    datesList = []
    for data in Datas:
        datesList.append(np.unique(data['dates']))
    commonDates = reduce(np.intersect1d, datesList)
    commonDates = np.setdiff1d(commonDates, forbiddenDates)
    commonDates = np.append(commonDates,datesToInclude)  # add back user-specified dates that should be included
    commonDates = np.unique(commonDates) #make sure that dates are unique again
    # commonDates = np.random.choice(commonDates, size=len(commonDates), replace=False)  # randomize the order of the dates used in case we do subsets later
    # print(commonDates)
    compareKeys = copy.deepcopy(keysToExtract)
    compareKeys.append('lowestDurLeavingEvents')
    compareKeys.append('highestDurLeavingEvents')
    if commonDates.size != 0:
        Data_subs = dict()
        Data_comparison_dfs = dict()
        for i, data in enumerate(Datas): #data is one of the Data_subs
            data_idx = np.sort(np.where([data['dates'] == commonDates[i] for i in range(0, len(commonDates))])[1]) #these are the indices for a particular genotype that belong to common date set
            if subset is not None:#if specified, take a random subset of the data without replacement.
                if subset < len(data_idx):#check that the subset size is less than the total amount of data per entry
                    np.random.seed(0)
                    # data_idx = np.random.choice(data_idx, size=subset, replace=False) #this way does not guarantee that data is compared correctly across days
                    new_data_idx = []
                    k = 0
                    while len(new_data_idx)<subset and k <= len(commonDates): #abs(len(new_data_idx)-subset)>10:
                        # print(commonDates[k])
                        new_data_idx = new_data_idx + list(np.where([data['dates'] == commonDates[k]])[1])
                        # print(new_data_idx)
                        # print("\n")
                        k+=1
                    data_idx = np.sort(new_data_idx)
                    # print(len(data_idx))

            Data_subs[Data_labels[i]] = dict()
            Data_subs[Data_labels[i]]['indexInAll'] = data_idx #this can be useful just to check against all data from the given genotype
            Data_subs[Data_labels[i]]['label'] = Data_labels[i] #associate data with label
            Data_subs[Data_labels[i]]['dates'] = data['dates'].iloc[data_idx]
            Data_subs[Data_labels[i]]['matfilename'] = data['matfilename'].iloc[data_idx]
            #subselect features corresponding to the keys provided
            for key in keysToExtract:
                if key in data: #this allows you to just select the keys that exist in a dataset (useful when comparing UL to LL data)
                    if isinstance(data[key], pd.core.frame.DataFrame) or isinstance(data[key], pd.core.series.Series):
                        Data_subs[Data_labels[i]][key] = data[key].iloc[data_idx]
                    else:
                        Data_subs[Data_labels[i]][key] = data[key][data_idx]
            #add state transitions as another set of keys so they can be compared in the dfs below
            arHMM_stateTrans = getStateTransitionsfromInto(Data_subs[Data_labels[i]]['arHMM_MLstates'], numStates)
            for fromState, subdict in arHMM_stateTrans.items():
                for toState, transMatrix in subdict.items():
                    id = 'arHMM_StateTransitions_' + str(fromState) + '_' + str(toState)
                    # print(id)
                    Data_subs[Data_labels[i]][id] = transMatrix
                    if id not in compareKeys:
                        compareKeys.append(id)
            RD_stateTrans = getStateTransitionsfromInto(Data_subs[Data_labels[i]]['RD_states_Matrix_exog'], 2)
            for fromState, subdict in RD_stateTrans.items():
                for toState, transMatrix in subdict.items():
                    id = 'RD_StateTransitions_' + str(fromState) + '_' + str(toState)
                    # print(id)
                    Data_subs[Data_labels[i]][id] = transMatrix
                    if id not in compareKeys:
                        compareKeys.append(id)

            RD_stateTrans = getStateTransitionsfromInto(Data_subs[Data_labels[i]]['RD_states_Matrix_exog_Cent'], 2)
            for fromState, subdict in RD_stateTrans.items():
                for toState, transMatrix in subdict.items():
                    id = 'RD_Cent_StateTransitions_' + str(fromState) + '_' + str(toState)
                    # print(id)
                    Data_subs[Data_labels[i]][id] = transMatrix
                    if id not in compareKeys:
                        compareKeys.append(id)

            #derive out durations and lawn leaving event matrices that correspond to short and long durations
            if 'bin_In_Or_Out' in Data_subs[Data_labels[i]] and 'bin_LawnExit_mostRecent' in Data_subs[Data_labels[i]]:
                outDurations = getOutLawnDurations(Data_subs[Data_labels[i]], 40 * 6, 40 * 6)
                Data_subs[Data_labels[i]]['outDurations'] = outDurations
                if outDurations.size>0:
                    lowestDurLeavingEvents, lowestDurations, highestDurLeavingEvents, highestDurations = getLeavingEvents_perDurationPercentile(Data_subs[Data_labels[i]], outDurations,25, 75)
                    Data_subs[Data_labels[i]]['lowestDurLeavingEvents'] = lowestDurLeavingEvents
                    Data_subs[Data_labels[i]]['highestDurLeavingEvents'] = highestDurLeavingEvents
                else:
                    Data_subs[Data_labels[i]]['lowestDurLeavingEvents'] = []
                    Data_subs[Data_labels[i]]['highestDurLeavingEvents'] = []

            # add keys corresponding to upsteps and downsteps in bacterial density for aligning
            if 'bin_Bacterial_Density_inLawn' in Data_subs[Data_labels[i]]:
                BD = Data_subs[Data_labels[i]]['bin_Bacterial_Density_inLawn']
                BD_diff = np.hstack([np.zeros((BD.shape[0], 1)), np.diff(BD, axis=1)])  # calculate the 1st derivative
                BD_diff_zscore = zscore(BD_diff.ravel(), nan_policy='omit').reshape(BD_diff.shape)

                Data_subs[Data_labels[i]]['BD_diff_upsteps'] = BD_diff_zscore >= zscore_thresh
                Data_subs[Data_labels[i]]['BD_diff_downsteps'] = BD_diff_zscore <= -1 * zscore_thresh
                if 'BD_diff_upsteps' not in compareKeys:
                    compareKeys.append('BD_diff_upsteps')
                if 'BD_diff_downsteps' not in compareKeys:
                    compareKeys.append('BD_diff_downsteps')

            if ('bin_Midbody_fspeed_inLawn' in Data_subs[Data_labels[i]]) and ('bin_MovingForward' in Data_subs[Data_labels[i]]): #divide this by fraction forward to make the values uncorrupted by reversals
                fSpeed = Data_subs[Data_labels[i]]['bin_Midbody_fspeed_inLawn']
                ffwd = Data_subs[Data_labels[i]]['bin_MovingForward']
                fSpeed_adjust = fSpeed/ffwd

            if 'bin_Midbody_fspeed_inLawn' in Data_subs[Data_labels[i]]:
                # add keys corresponding to upsteps and downsteps in Midbody F. Speed
                fSpeed = Data_subs[Data_labels[i]]['bin_Midbody_fspeed_inLawn']
                fSpeed_diff = np.hstack(
                    [np.zeros((fSpeed.shape[0], 1)), np.diff(fSpeed, axis=1)])  # calculate the 1st derivative
                fSpeed_diff_zscore = zscore(fSpeed_diff.ravel(), nan_policy='omit').reshape(fSpeed_diff.shape)

                Data_subs[Data_labels[i]]['fSpeed_diff_upsteps'] = fSpeed_diff_zscore >= zscore_thresh
                Data_subs[Data_labels[i]]['fSpeed_diff_downsteps'] = fSpeed_diff_zscore <= -1 * zscore_thresh
                if 'fSpeed_diff_upsteps' not in compareKeys:
                    compareKeys.append('fSpeed_diff_upsteps')
                if 'fSpeed_diff_downsteps' not in compareKeys:
                    compareKeys.append('fSpeed_diff_downsteps')

            if 'bin_Centroid_speed_inLawn' in Data_subs[Data_labels[i]]:
                # add keys corresponding to upsteps and downsteps in Midbody F. Speed
                cSpeed = Data_subs[Data_labels[i]]['bin_Centroid_speed_inLawn']
                cSpeed_diff = np.hstack(
                    [np.zeros((cSpeed.shape[0], 1)), np.diff(cSpeed, axis=1)])  # calculate the 1st derivative
                cSpeed_diff_zscore = zscore(cSpeed_diff.ravel(), nan_policy='omit').reshape(cSpeed_diff.shape)

                Data_subs[Data_labels[i]]['cSpeed_diff_upsteps'] = cSpeed_diff_zscore >= zscore_thresh
                Data_subs[Data_labels[i]]['cSpeed_diff_downsteps'] = cSpeed_diff_zscore <= -1 * zscore_thresh
                if 'cSpeed_diff_upsteps' not in compareKeys:
                    compareKeys.append('cSpeed_diff_upsteps')
                if 'cSpeed_diff_downsteps' not in compareKeys:
                    compareKeys.append('cSpeed_diff_downsteps')

        for key in compareKeys:
            # print(key)
            data_comparisons = []
            for i, label in enumerate(Data_labels):
                # print(label)
                if key in Data_subs[label]:
                    # print(key)
                    data_tmp_df = pd.DataFrame(data=Data_subs[label][key])
                    # print(data_tmp_df.shape)
                    data_tmp_df.insert(loc=0, column='genotype', value=np.repeat(label, len(Data_subs[label]['dates']), axis=0))
                    data_tmp_df.insert(loc=0, column='dates', value=Data_subs[label]['dates'].reset_index(drop=True))
                    data_tmp_df.insert(loc=0, column='matfilename', value=Data_subs[label]['matfilename'].reset_index(drop=True))
                    data_comparisons.append(data_tmp_df) #append data from the selected genotypes
            if len(data_comparisons) != 0:
                Data_comparison_dfs[key] = pd.concat(data_comparisons,axis=0,ignore_index=True)
    else:
        Data_subs = []
        Data_comparison_dfs = []

    return Data_subs, Data_comparison_dfs
#associate a list of genotypes with wild type data according to each (don't compare everything in the list to each other)
def associateControlData(currentData,currentDataNames,Control_Datas,Control_datanames,keysToExtract,binSize,forbiddenDates,datesToInclude, subset=None):
    Comparisons = dict()
    for i, Data in enumerate(currentData):  # for each genotype, get the data and accompanying wild type data
        print(currentDataNames[i])
        datesList = []
        uniqueDates = np.unique(Data['dates'])
        datesList.append(uniqueDates)
        datesList = np.setdiff1d(datesList, forbiddenDates)  # get rid of forbidden
        datesList = np.append(datesList, datesToInclude)  # add back included
        # choose the wild type data to use on a given day
        whichControl = np.zeros(len(Control_Datas)).astype(bool)
        for j in range(len(Control_Datas)):
            overlap = np.intersect1d(datesList, np.unique(Control_Datas[j]['dates']))
            if overlap.size > 0:
                whichControl[j] = True
        # just in case there was ever a day when both N2 and PD1074 were recorded, which I doubt, choose PD1074 (highest priority)
        if whichControl[0]:
            ControlData = Control_Datas[0]
            CDidx = 0
        else:
            CDidx = np.where(whichControl)[0]
            if len(CDidx) == 1:
                CDidx = CDidx[0]
                ControlData = Control_Datas[CDidx]
            else:
                print("warning: no single wild type dataset could be associated with" + currentDataNames[i])
                continue  # skip this date entirely if no wild type dataset or multiple non priority can be found

        DataToCompare = np.append(Data, ControlData)
        genotypesToCompare = np.append(currentDataNames[i], Control_datanames[CDidx])
        Data_subs, Data_comparison_dfs = compareGenotypes(DataToCompare, genotypesToCompare, keysToExtract, binSize, 4,
                                                          2, forbiddenDates, datesToInclude, subset) #compare the data, can subset the data
        Comparisons[currentDataNames[i]] = dict() #store the comparisons in a dictionary
        Comparisons[currentDataNames[i]]['Data_subs'] = Data_subs
        Comparisons[currentDataNames[i]]['Data_comparison_dfs'] = Data_comparison_dfs

    Comparisons_df = pd.DataFrame()
    for i, name in enumerate(currentDataNames): #for each set of genotype + wild type, add a few more fields and make a summary dataframe
        df = generateComparisonsDF(Comparisons[name]['Data_comparison_dfs'])
        num_entries = Comparisons[name]['Data_comparison_dfs']['bin_LawnExit_mostRecent'].shape[0]
        df.insert(0, 'group', pd.Series(np.repeat(name, num_entries)))
        # check which indices in this comparison are experimental or wt (or whatever your control group is)
        wtindices = np.zeros_like(df['genotype']).astype(bool)
        for i in range(len(df['genotype'])): #long-winded but it works...
            gn = [df['genotype'].iloc[i]]
            intersection = np.intersect1d(gn, Control_datanames)
            if intersection.size>0:
                wtindices[i] = True
        wtindices = np.where(wtindices)[0]
        # wtindices = np.where(np.logical_or(df['genotype'] == 'N2', df['genotype'] == 'PD1074'))[0]
        expindices = np.setdiff1d(np.arange(len(df['genotype'])), wtindices)
        identifier = np.array([None] * (len(df['genotype'])))
        identifier[wtindices] = 'control'
        identifier[expindices] = 'experimental'
        df.insert(1, 'identifier', identifier)
        Comparisons_df = pd.concat([Comparisons_df, df], axis=0, ignore_index=True)

    # #generate 2 new fields that are useful in subsequent analyses
    # Comparisons_df['numLLevent'] = Comparisons_df['#lawn exits/min'] * 40
    # Comparisons_df['containsLLevent'] = Comparisons_df['numLLevent'] > 0

    #do statistics for proportion of traces containing lawn leaving event
    df_containsLL, df_containsLL_stats = propLawnLeavingStats(Comparisons_df, currentDataNames)

    return Comparisons, Comparisons_df, df_containsLL, df_containsLL_stats
#do statistics and multiple test correction to compare proportion of different genotype groups that contain lawn leaving events
def propLawnLeavingStats(Comparisons_df,datanames):
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    if ("group" not in Comparisons_df) or ("identifier" not in Comparisons_df):
        raise TypeError("this function only works for multiple genotypes compared against paired controls")
    df_containsLL = getPropLawnLeaving(Comparisons_df, ['group', 'identifier'])
    df_containsLL_stats = df_containsLL.groupby("group", sort=False).apply(lambda dft: pd.Series(
        {'prop_diff': abs(dft.proportion_containing_LL.iloc[0] - dft.proportion_containing_LL.iloc[1]),
         'n0': dft.n.iloc[0], 'n1': dft.n.iloc[1], 'total_n': dft.n.sum(),
         'overall_prop': dft.contains_LL.sum() / dft.n.sum()})).reindex(datanames).reset_index() #this is brittle - only permits a pairwise comparison

    df_containsLL_stats['stderror'] = np.sqrt(
        df_containsLL_stats['overall_prop'] * (1 - df_containsLL_stats['overall_prop']) * (
                    (1 / df_containsLL_stats['n0']) + (1 / df_containsLL_stats['n1'])))
    #comparing two proportions
    #https://online.stat.psu.edu/stat415/lesson/9/9.4
    df_containsLL_stats['zstat'] = df_containsLL_stats['prop_diff'] / df_containsLL_stats['stderror']
    df_containsLL_stats['pvalue'] = norm.sf(abs(df_containsLL_stats['zstat'])) * 2  # for two-tailed test
    reject, pvals_adj, alphacSidak, alphacBonf = multipletests(df_containsLL_stats['pvalue'], alpha=0.05,
                                                               method='bonferroni')
    df_containsLL_stats['pvalue_adj'] = pvals_adj

    return df_containsLL, df_containsLL_stats
def getStateTransitionsfromInto(MLstates,numstates):
    """
    Mark time bins when animals change from one state to another.
    :param MLstates: HMM state matrix
    :param numstates: overall number of states used in MLstates
    :return: StateTransitions: 2D array of boolean matrices where True indicates a state transition. StateTransitions is indexed by the source and destination state.
    """
    state_permutations = list(itertools.permutations(range(numstates), 2))
    StateTransitions = dict() #make a lookup dictionary for the from state and the into state
    for i in range(numstates):
        StateTransitions[i] = dict()
    #get permutations of all state transitions
    for j in range(0, len(state_permutations)):
        fromState = state_permutations[j][0]
        intoState = state_permutations[j][1]

        fromState_marked = MLstates == fromState
        IntoState_marked = MLstates == intoState
        detectChangeMatrix = np.ones_like(MLstates).astype(float)*50 #just make it a big number so that there can't be an diff=1

        detectChangeMatrix[fromState_marked] = 0.0
        detectChangeMatrix[IntoState_marked] = 1.0

        StateTransitions[fromState][intoState] = np.hstack((np.zeros((detectChangeMatrix.shape[0], 1)),
                                     np.diff(detectChangeMatrix, axis=1) == 1.0))
    return StateTransitions
def getOutLawnDurations(Data,binTimeBefore,binTimeAfter):
    """
    Get duration of bouts outside the lawn boundary.
    :param Data:
    :param binTimeBefore:
    :param binTimeAfter:
    :return:
    """
    rowMat = np.where(np.ones_like(Data['bin_Midbody_fspeed']))[0].reshape(Data['bin_Midbody_fspeed'].shape)
    colMat = np.where(np.ones_like(Data['bin_Midbody_fspeed']))[1].reshape(Data['bin_Midbody_fspeed'].shape)

    ColIdx_LLaligned, binTimeLine, binAlignIdx = alignData(colMat, Data['bin_LawnExit_mostRecent'],
                                                           binTimeBefore, binTimeAfter)
    RowIdx_LLaligned, binTimeLine, binAlignIdx = alignData(rowMat, Data['bin_LawnExit_mostRecent'],
                                                           binTimeBefore, binTimeAfter)

    binInOrOut_LLaligned, binTimeLine, binAlignIdx = alignData(Data['bin_In_Or_Out'],
                                                               Data['bin_LawnExit_mostRecent'], binTimeBefore,
                                                               binTimeAfter)

    sortOrder = np.arange(0, ColIdx_LLaligned.shape[0])
    outDurations = np.zeros((len(sortOrder), 3)).astype(int)  # row of lawn exit, column of lawn exit, outside duration
    idxToDelete = []
    for i in range(len(sortOrder)):
        outInts = get_intervals(np.nan_to_num(binInOrOut_LLaligned[i]), 0)
        #check if there is an out interval close enough to the align index, if not then skip
        if (binAlignIdx - outInts[:, 0] < 4).any():
            centeredInt = outInts[np.where(binAlignIdx - outInts[:, 0] < 4)[0], :][0]  # get the closest out interval to align index
            outDur = centeredInt[1] - centeredInt[0] + 2  # to extend the interval to the adjacent in-lawn indices
            outDurations[i] = np.hstack([RowIdx_LLaligned[i][binAlignIdx], ColIdx_LLaligned[i][binAlignIdx], outDur])
        else:
            idxToDelete.append(i)

    outDurations = np.delete(outDurations, (idxToDelete), axis=0) #remove any aberrant data

    return outDurations
def alignData(data, lawnBool, timeBefore, timeAfter):
    """
    Align input data to discrete events within the dataset.

    :param data: Matrix of data to align
    :param lawnBool: Boolean Matrix of matching size, where True indicates frames for aligning
    :param timeBefore: number of bins before for alignment
    :param timeAfter: number of bins after for alignment
    :return: LL_aligned: data aligned to places where lawnBool was True. Matrix is dimension sum(lawnBool) x timeBefore+timeAfter
    """
    numBins = data.shape[1]
    lawnExit_idx = np.array(np.where(lawnBool))
    numexits = len(lawnExit_idx[0])

    timeLine = np.arange(-1 * timeBefore, timeAfter).astype(int)
    alignIdx = np.where(timeLine == 0)[0][0]
    timeLineIdx = np.arange(0, len(timeLine)).astype(int)

    # pre-allocate
    LL_aligned = np.empty((numexits, len(timeLine)))
    LL_aligned[:] = np.NaN

    for i in range(0, len(lawnExit_idx[0])):
        currLLidx = [lawnExit_idx[0][i], lawnExit_idx[1][i]]
        intToExtract = timeLine + currLLidx[1]
        currTimeLineIdx = timeLineIdx[np.logical_and(intToExtract >= 0, intToExtract <= numBins - 1)]
        intToExtract = intToExtract[np.logical_and(intToExtract >= 0, intToExtract <= numBins - 1)]
        LL_aligned[i, currTimeLineIdx] = data[currLLidx[0], intToExtract]

    return LL_aligned, timeLine, alignIdx

def alignData_masked(data, mask, lawnBool, timeBefore, timeAfter, dtype): #TODO see if you can merge this function with other alignData so we only have 1
    """
    Align input data to discrete events within the dataset. Use this function if you wish to include a boolean masking variable
    :param data: Matrix of data to align
    :param mask: Boolean mask of data. True means mask it OUT.
    :param lawnBool: Boolean Matrix of matching size, where True indicates frames for aligning
    :param timeBefore: number of bins before for alignment
    :param timeAfter: number of bins after for alignment
    :return: LL_aligned: data aligned to places where lawnBool was True. Matrix is dimension sum(lawnBool) x timeBefore+timeAfter
    """
    numBins = data.shape[1]
    lawnExit_idx = np.array(np.where(lawnBool)) #where lawn exits occur
    numexits = len(lawnExit_idx[0])
    # print(numexits)

    timeLine = np.arange(-1 * timeBefore, timeAfter).astype(int) #timeLine to which we will align data, centered at 0
    alignIdx = np.where(timeLine == 0)[0][0]
    timeLineIdx = np.arange(0, len(timeLine)).astype(int)  #timeLine indices in consecutive order

    LL_aligned = np.empty((numexits, len(timeLine)), dtype=dtype)
    mask_aligned = np.ones_like(LL_aligned, dtype=bool)  # default everything is masked
    LL_aligned = np.ma.masked_array(data=LL_aligned, mask=mask_aligned)  # pre-allocate it as a masked array

    # print("LL_aligned.shape "+str(LL_aligned.shape))
    # print("mask_aligned.shape "+str(mask_aligned.shape))
    # print("data.shape "+str(data.shape))
    # print("mask.shape "+str(mask.shape))
    # print("lawnBool.shape"+str(lawnBool.shape))

    for i in range(0, numexits):
        currLLidx = [lawnExit_idx[0][i], lawnExit_idx[1][i]] #the current lawn leaving event in row, column indices
        intToExtract = timeLine + currLLidx[1] #add the column where the lawn leaving event is to the timeline as an offset so we can grab that interval
        currTimeLineIdx = timeLineIdx[np.logical_and(intToExtract >= 0, intToExtract <= numBins - 1)] #find the corresponding indices in the timeline, taking care not to go out of bounds
        intToExtract = intToExtract[np.logical_and(intToExtract >= 0, intToExtract <= numBins - 1)]
        currMask = mask[currLLidx[0], intToExtract] #grab the corresponding interval of mask
        # if np.nansum(currMask[0:timeBefore]) < timeBefore/2: #(why is this here ?)
        LL_aligned[i, currTimeLineIdx] = data[currLLidx[0], intToExtract] #insert the bit of data into the aligned matrix
        mask_aligned[i, currTimeLineIdx] = mask[currLLidx[0], intToExtract] #insert the bit of mask into the aligned matrix

    #remove aligned entries where the the mask is True for the entire row (was initialized to be)
    idxToKeep = np.where(np.nansum(mask_aligned[:, 0:timeBefore], axis=1) != timeBefore)[0]
    LL_aligned = LL_aligned[idxToKeep]
    mask_aligned = mask_aligned[idxToKeep]
    LL_aligned = np.ma.masked_array(data=LL_aligned, mask=mask_aligned)

    return LL_aligned, mask_aligned, timeLine, alignIdx

def alignData_masked_lastRunOnly(data, mask, lawnBool, timeBefore, timeAfter, dtype): #TODO see if you can merge this function with other alignData so we only have 1
    """
    Align input data to discrete events within the dataset. Use this function if you wish to include a boolean masking variable
    :param data: Matrix of data to align
    :param mask: Boolean mask of data. True means mask it OUT.
    :param lawnBool: Boolean Matrix of matching size, where True indicates frames for aligning
    :param timeBefore: number of bins before for alignment
    :param timeAfter: number of bins after for alignment
    :return: LL_aligned: data aligned to places where lawnBool was True. Matrix is dimension sum(lawnBool) x timeBefore+timeAfter
    """
    numBins = data.shape[1]
    lawnExit_idx = np.array(np.where(lawnBool)) #where lawn exits occur
    numexits = len(lawnExit_idx[0])

    timeLine = np.arange(-1 * timeBefore, timeAfter).astype(int) #timeLine to which we will align data, centered at 0
    alignIdx = np.where(timeLine == 0)[0][0]
    timeLineIdx = np.arange(0, len(timeLine)).astype(int)  #timeLine indices in consecutive order

    LL_aligned = np.empty((numexits, len(timeLine)), dtype=dtype)
    mask_aligned = np.ones_like(LL_aligned, dtype=bool)  # default everything is masked
    LL_aligned = np.ma.masked_array(data=LL_aligned, mask=mask_aligned)  # pre-allocate it as a masked array

    for i in range(0, len(lawnExit_idx[0])):
        currLLidx = [lawnExit_idx[0][i], lawnExit_idx[1][i]] #the current lawn leaving event in row, column indices
        intToExtract = timeLine + currLLidx[1] #add the column where the lawn leaving event is to the timeline as an offset so we can grab that interval

        currTimeLineIdx = timeLineIdx[np.logical_and(intToExtract >= 0, intToExtract <= numBins - 1)] #find the corresponding indices in the timeline, taking care not to go out of bounds
        intToExtract = intToExtract[np.logical_and(intToExtract >= 0, intToExtract <= numBins - 1)]
        currMask = mask[currLLidx[0], intToExtract] #grab the corresponding interval of mask

        LL_aligned[i, currTimeLineIdx] = data[currLLidx[0], intToExtract] #insert the bit of data into the aligned matrix
        #find only the run that ends at alignIdx
        maskInts = get_intervals(~currMask,0)
        chosenRunIdx_before = np.unique(np.where(np.isin(maskInts, np.arange(alignIdx-1,alignIdx+1)))[0])
        chosenRunIdx_after = np.unique(np.where(np.isin(maskInts, np.arange(alignIdx,alignIdx+timeAfter)))[0])
        chosenRunIdx = np.unique(np.union1d(chosenRunIdx_before,chosenRunIdx_after))

        lastrunmask = np.ones_like(currMask).astype(bool) #True means we dont want to show it (it is masked)
        for c in chosenRunIdx:
            chosenRun = np.arange(maskInts[c,0],maskInts[c,1]+1)
            lastrunmask[chosenRun]=False #False means we want to show it
        mask_aligned[i, currTimeLineIdx] = lastrunmask

    #remove aligned entries where the the mask is True for the entire row (was initialized to be)
    idxToKeep = np.where(np.nansum(mask_aligned[:, 0:timeBefore], axis=1) != timeBefore)[0]
    LL_aligned = LL_aligned[idxToKeep]
    mask_aligned = mask_aligned[idxToKeep]

    LL_aligned = np.ma.masked_array(data=LL_aligned, mask=mask_aligned)

    return LL_aligned, mask_aligned, timeLine, alignIdx

def getLeavingEvents_perDurationPercentile(Data,outDurations,lowerPercentile,upperPercentile):
    """
    sort in order of out-duration and compare the lowest quartile and the highest quartile of out duration
    """
    sortOrder = np.arange(0, outDurations.shape[0])
    durSortOrder = np.argsort(outDurations[:, 2].ravel())
    lowestDurIdx = durSortOrder[np.arange(np.round(np.percentile(sortOrder, lowerPercentile))).astype(int)]  # lowest X percentile
    lowestDurations = outDurations[lowestDurIdx,:]
    highestDurIdx = durSortOrder[
        np.arange(np.round(np.percentile(sortOrder, upperPercentile)), len(sortOrder)).astype(int)]  # Xth percentile and above
    highestDurations = outDurations[highestDurIdx,:]

    # re-associate these indices to full matrix indices --  which lawn leaving events lead to the highest duration and the lowest outside lawn
    lowestDurLeavingEvents = np.zeros_like(Data['bin_LawnExit_mostRecent']).astype(bool)
    for i in range(len(lowestDurIdx)):
        row = outDurations[lowestDurIdx[i], 0]
        col = outDurations[lowestDurIdx[i], 1]
        lowestDurLeavingEvents[row, col] = True

    highestDurLeavingEvents = np.zeros_like(Data['bin_LawnExit_mostRecent']).astype(bool)
    for i in range(len(highestDurIdx)):
        row = outDurations[highestDurIdx[i], 0]
        col = outDurations[highestDurIdx[i], 1]
        highestDurLeavingEvents[row, col] = True

    return lowestDurLeavingEvents, lowestDurations, highestDurLeavingEvents, highestDurations

def generateComparisonsDF(Data_comparison_dfs): #TODO: fix this to accept arbitrary number of states, timebins
    """
    Makes a dataframe that summarizes data across several key dimensions for cross-genotype comparison
    :param Data_comparison_dfs: the output of compareGenotypes
    :return: Comparisons_df
    """
    Comparisons_df = pd.DataFrame()

    num_entries = Data_comparison_dfs['bin_LawnExit_mostRecent'].shape[0]
    lawnexit_df = Data_comparison_dfs['bin_LawnExit_mostRecent'].loc[:, 0:239].astype(float).copy()
    midbody_fspeed_df = Data_comparison_dfs['bin_Midbody_fspeed_inLawn'].loc[:, 0:239].astype(float).copy()
    headspeed_df = Data_comparison_dfs['bin_Head_speed_inLawn'].loc[:, 0:239].astype(float).copy()
    roamdwell_df = Data_comparison_dfs['RD_states_Matrix_exog_Cent'].loc[:, 0:239].astype(float).copy()
    arhmm_df = Data_comparison_dfs['arHMM_MLstates'].loc[:, 0:239].astype(float).copy()
    lawnboundarydist_df = Data_comparison_dfs['bin_Lawn_Boundary_Dist_inLawn'].loc[:, 0:239].astype(float).copy()
    bacdensity_df = Data_comparison_dfs['bin_Bacterial_Density_inLawn'].loc[:, 0:239].astype(float).copy()

    # Comparisons_df['group'] = pd.Series(np.repeat('group', num_entries))
    Comparisons_df['matfilename'] = Data_comparison_dfs['bin_LawnExit_mostRecent']['matfilename']
    Comparisons_df['genotype'] = Data_comparison_dfs['bin_LawnExit_mostRecent']['genotype']
    Comparisons_df['dates'] = Data_comparison_dfs['bin_LawnExit_mostRecent']['dates']
    Comparisons_df['#lawn exits/min'] = lawnexit_df.apply(lambda x: np.nansum(x) / 40, axis=1)  # lawn exit rate
    # generate 2 new fields that are useful in subsequent analyses
    Comparisons_df['numLLevent'] = Comparisons_df['#lawn exits/min'] * 40
    Comparisons_df['containsLLevent'] = Comparisons_df['numLLevent'] > 0

    Comparisons_df['midbody forward speed'] = pd.Series(np.nanmean(midbody_fspeed_df,axis=1))  # midbody_speed_df.apply(lambda x: np.nanmean(x),axis=1) #mean forward speed
    Comparisons_df['head speed'] = pd.Series(np.nanmean(headspeed_df,axis=1))

    Comparisons_df['fraction roaming'] = roamdwell_df.apply(
        lambda x: (np.nansum(x == 1.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 0'] = arhmm_df.apply(
        lambda x: (np.nansum(x == 0.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 1'] = arhmm_df.apply(
        lambda x: (np.nansum(x == 1.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 2'] = arhmm_df.apply(
        lambda x: (np.nansum(x == 2.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 3'] = arhmm_df.apply(
        lambda x: (np.nansum(x == 3.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['lawn boundary dist'] = lawnboundarydist_df.apply(lambda x: np.nanmean(x),
                                                                     axis=1)  # mean lawn boundary dist
    Comparisons_df['bacterial density'] = bacdensity_df.apply(lambda x: np.nanmean(x),
                                                              axis=1)  # mean lawn boundary dist

    return Comparisons_df

def generateComparisonsDF_lightOFFON(Data_comparison_dfs,lightStim): #TODO: fix this to accept arbitrary number of states, timebins
    """
    Makes a dataframe that summarizes data across several key dimensions for cross-genotype comparison
    :param Data_comparison_dfs: the output of compareGenotypes
    :param lightStim: a vector of zeros and ones (which indicate when the light was OFF and ON, respectively)
    :return: Comparisons_df
    """
    Comparisons_df = pd.DataFrame()
    lightOFFidx = np.where(~lightStim)[0]
    lightONidx = np.where(lightStim)[0]

    num_entries = Data_comparison_dfs['bin_LawnExit_mostRecent'].shape[0]

    lawnexit_lightOFF = Data_comparison_dfs['bin_LawnExit_mostRecent'].loc[:, lightOFFidx].astype(float).copy()
    lawnexit_lightON = Data_comparison_dfs['bin_LawnExit_mostRecent'].loc[:, lightONidx].astype(float).copy()

    midbody_fspeed_lightOFF = Data_comparison_dfs['bin_Midbody_fspeed_inLawn'].loc[:, lightOFFidx].astype(float).copy()
    midbody_fspeed_lightON = Data_comparison_dfs['bin_Midbody_fspeed_inLawn'].loc[:, lightONidx].astype(float).copy()

    headspeed_lightOFF = Data_comparison_dfs['bin_Head_speed_inLawn'].loc[:, lightOFFidx].astype(float).copy()
    headspeed_lightON = Data_comparison_dfs['bin_Head_speed_inLawn'].loc[:, lightONidx].astype(float).copy()

    roamdwell_lightOFF = Data_comparison_dfs['RD_states_Matrix_exog_Cent'].loc[:, lightOFFidx].astype(float).copy()
    roamdwell_lightON = Data_comparison_dfs['RD_states_Matrix_exog_Cent'].loc[:, lightONidx].astype(float).copy()

    arhmm_lightOFF = Data_comparison_dfs['arHMM_MLstates'].loc[:, lightOFFidx].astype(float).copy()
    arhmm_lightON = Data_comparison_dfs['arHMM_MLstates'].loc[:, lightONidx].astype(float).copy()

    lawnboundarydist_lightOFF = Data_comparison_dfs['bin_Lawn_Boundary_Dist_inLawn'].loc[:, lightOFFidx].astype(float).copy()
    lawnboundarydist_lightON = Data_comparison_dfs['bin_Lawn_Boundary_Dist_inLawn'].loc[:, lightONidx].astype(float).copy()

    bacdensity_lightOFF = Data_comparison_dfs['bin_Bacterial_Density_inLawn'].loc[:, lightOFFidx].astype(float).copy()
    bacdensity_lightON = Data_comparison_dfs['bin_Bacterial_Density_inLawn'].loc[:, lightONidx].astype(float).copy()

    # Comparisons_df['group'] = pd.Series(np.repeat('group', num_entries))
    Comparisons_df['matfilename'] = Data_comparison_dfs['bin_LawnExit_mostRecent']['matfilename']
    Comparisons_df['genotype'] = Data_comparison_dfs['bin_LawnExit_mostRecent']['genotype']
    Comparisons_df['dates'] = Data_comparison_dfs['bin_LawnExit_mostRecent']['dates']

    Comparisons_df['#lawn exits/min light OFF'] = lawnexit_lightOFF.apply(lambda x: np.nansum(x) / 40, axis=1)  # lawn exit rate
    Comparisons_df['#lawn exits/min light ON'] = lawnexit_lightON.apply(lambda x: np.nansum(x) / 40, axis=1)  # lawn exit rate

    # generate 2 new fields that are useful in subsequent analyses
    Comparisons_df['numLLevent light OFF'] = Comparisons_df['#lawn exits/min light OFF'] * 40
    Comparisons_df['numLLevent light ON'] = Comparisons_df['#lawn exits/min light ON'] * 40

    Comparisons_df['containsLLevent light OFF'] = Comparisons_df['numLLevent light OFF'] > 0
    Comparisons_df['containsLLevent light ON'] = Comparisons_df['numLLevent light ON'] > 0

    Comparisons_df['midbody forward speed light OFF'] = pd.Series(np.nanmean(midbody_fspeed_lightOFF,axis=1))
    Comparisons_df['midbody forward speed light ON'] = pd.Series(np.nanmean(midbody_fspeed_lightON, axis=1))

    Comparisons_df['head speed light OFF'] = pd.Series(np.nanmean(headspeed_lightOFF,axis=1))
    Comparisons_df['head speed light ON'] = pd.Series(np.nanmean(headspeed_lightON, axis=1))

    Comparisons_df['fraction roaming light OFF'] = roamdwell_lightOFF.apply(
        lambda x: (np.nansum(x == 1.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction roaming light ON'] = roamdwell_lightON.apply(
        lambda x: (np.nansum(x == 1.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)

    Comparisons_df['fraction state 0 light OFF'] = arhmm_lightOFF.apply(
        lambda x: (np.nansum(x == 0.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 0 light ON'] = arhmm_lightON.apply(
        lambda x: (np.nansum(x == 0.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)

    Comparisons_df['fraction state 1 light OFF'] = arhmm_lightOFF.apply(
        lambda x: (np.nansum(x == 1.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 1 light ON'] = arhmm_lightON.apply(
        lambda x: (np.nansum(x == 1.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)

    Comparisons_df['fraction state 2 light OFF'] = arhmm_lightOFF.apply(
        lambda x: (np.nansum(x == 2.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 2 light ON'] = arhmm_lightON.apply(
        lambda x: (np.nansum(x == 2.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)

    Comparisons_df['fraction state 3 light OFF'] = arhmm_lightOFF.apply(
        lambda x: (np.nansum(x == 3.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)
    Comparisons_df['fraction state 3 light ON'] = arhmm_lightON.apply(
        lambda x: (np.nansum(x == 3.0) + 1) / (np.sum(~np.isnan(x)) + 1), axis=1)

    Comparisons_df['lawn boundary dist light OFF'] = lawnboundarydist_lightOFF.apply(lambda x: np.nanmean(x),
                                                                     axis=1)  # mean lawn boundary dist
    Comparisons_df['lawn boundary dist light ON'] = lawnboundarydist_lightON.apply(lambda x: np.nanmean(x),
                                                                     axis=1)  # mean lawn boundary dist
    Comparisons_df['bacterial density light OFF'] = bacdensity_lightOFF.apply(lambda x: np.nanmean(x),
                                                              axis=1)  # mean lawn boundary dist
    Comparisons_df['bacterial density light ON'] = bacdensity_lightON.apply(lambda x: np.nanmean(x),
                                                              axis=1)  # mean lawn boundary dist

    return Comparisons_df

def getPropLawnLeaving(Comparisons_df,groupbykeys):
    """
    Get the proportion of animals leaving the lawn across conditions
    Use this function if you have just one set of genotypes to compare (not a whole bunch with paired controls)
    :param Comparisons_df: output of generateComparisonsDF
    :param groupbykeys: keys to compare. often "genotype"
    :return: df_containsLL, a new pandas dataframe
    """
    df_containsLL = Comparisons_df.groupby(groupbykeys, sort=False).apply(lambda dft: pd.Series(
        {'contains_LL': dft.containsLLevent.sum(), 'notContains_LL': (~dft.containsLLevent).sum()}))
    # df_containsLL['num_animals'] = Comparisons_df.groupby(groupbykeys, sort=False)["dates"].count()
    df_containsLL['n'] = df_containsLL['contains_LL'] + df_containsLL['notContains_LL']
    df_containsLL['proportion_containing_LL'] = df_containsLL['contains_LL'] / df_containsLL['n']
    df_containsLL = df_containsLL.reset_index()
    return df_containsLL

def genProbStateAligned_masked(dataToAlign, maskToAlign, IdxToAlign, timeBefore, timeAfter, num_states):
    """
    #generate probability of finding HMM states aligned to a particular event

    :param dataToAlign: MLstates
    :param maskToAlign: boolean mask
    :param IdxToAlign: boolean matrix of events to align data to
    :param timeBefore: bins before for alignment
    :param timeAfter: bins after for alignment
    :param num_states: total number of HMM states used in dataToAlign
    :return:
    """

    StateByAligned, MaskByAligned, timeLine, alignIdx = alignData_masked(dataToAlign, maskToAlign, IdxToAlign, timeBefore, timeAfter, dtype=float)

    numer = np.array([np.sum(StateByAligned == i, axis=0) for i in range(0, num_states)]).T
    denom = np.sum(~MaskByAligned, axis=0)
    probStateAligned = np.array([(numer[k] / denom[k]) for k in range(0, len(timeLine))]).reshape(len(timeLine), num_states).T
    mult_prop_ci = np.array([prop.multinomial_proportions_confint(numer[k], alpha=0.05, method='goodman') for k in range(0, len(timeLine))]).T

    return StateByAligned, MaskByAligned, probStateAligned, timeLine, alignIdx, mult_prop_ci

def genProbStatePerParam(Data, States, bins):
    """
    This function bins a particular data dimension by provided HMM states
    :param Data: dimension of choice, e.g. Bacterial Density or Lawn Boundary Dist.
    :param States: HMM states
    :param bins: bins across the Data dimension
    :return: Return the fraction of time State X was found within a particular bin of data dimension (Data)
    """
    maxstateIdx = int(np.nanmax(States)) + 1
    # for each bin (corresponding to a range of Param), get the probability of "var" for each state used
    ParambinPerData = np.digitize(Data, bins=bins)
    StatebyParam = np.array([States[ParambinPerData == k] for k in range(0, len(bins))],
                            dtype='object')  # get the states used in each bin of Param
    nState = np.array(
        [np.sum(StatebyParam[k] == i) for k in range(0, len(bins)) for i in range(0, maxstateIdx)]).reshape(len(bins),
                                                                                                            maxstateIdx)  # get the number of times each state is used in each bin of Param (numerator)
    nObs = np.array(
        [np.sum(~np.isnan(StatebyParam[k])) + 1 for k in range(0, len(bins))])  # denominator, give a pseudocount

    # generate confidence intervals for proportions
    probStatebyParam = np.array([(nState[k] / nObs[k]) for k in range(0, len(bins))]).reshape(len(bins), maxstateIdx)
    mult_prop_ci = np.array(
        [prop.multinomial_proportions_confint(nState[k], alpha=0.05, method='goodman') for k in range(0, len(bins))])

    # better to reshape these for plotting (numstates,numbins)
    probStatebyParam = probStatebyParam.T.reshape(maxstateIdx, len(bins))
    mult_prop_ci = mult_prop_ci.T.reshape(2, maxstateIdx, len(bins))

    return StatebyParam, probStatebyParam, mult_prop_ci

def mean_confidence_interval(data, confidence=0.95):
    """
    Computes the confidence interval around the mean at the provided confidence level
    :param data:
    :param confidence:
    :return:
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h
#TODO: can these functions be streamlined?
#to be used after running compareGenotypes to align features across genotypes
def alignSeveralDatas(Datas, Data_labels, featureKey, maskKey, alignKey, binTimeBefore, binTimeAfter):
    Datas_aligned = dict()
    for i, data in enumerate(Datas.values()):
        # print(i)
        data_sub_aligned, Mask_aligned, binTimeLine, binAlignIdx = alignData_masked(
            data[featureKey], ~data[maskKey], data[alignKey],
            binTimeBefore, binTimeAfter, dtype=float)
        Datas_aligned[Data_labels[i]] = data_sub_aligned
    return Datas_aligned, Mask_aligned, binTimeLine, binAlignIdx
#to be used for aligning different genotypes to light ON or OFF transitions
def alignSeveralDatasToLight(Datas, Data_labels, featureKey, maskKey, lightStimChanges, binTimeBefore, binTimeAfter):
    Datas_aligned = dict()
    Masks_aligned = dict()
    for i, data in enumerate(Datas.values()):
        # print(i)
        lightChangePoints = np.repeat(lightStimChanges.reshape(1,-1),data[featureKey].shape[0],axis=0)
        data_sub_aligned, Mask_aligned, binTimeLine, binAlignIdx = alignData_masked(
            data[featureKey], ~data[maskKey], lightChangePoints,
            binTimeBefore, binTimeAfter, dtype=float)
        Datas_aligned[Data_labels[i]] = data_sub_aligned
        Masks_aligned[Data_labels[i]] = Mask_aligned
    return Datas_aligned, Masks_aligned, binTimeLine, binAlignIdx
#this version only shows data from the states that correspond to the given state transition
# (i.e. even if you show data going back several minutes before roam to dwell, we only see roaming data represented in the average before the transition)
def alignSeveralDatas_StateTransitions(Datas, Data_labels, featureKey, maskKey, alignKey, stateKey, stateBefore, stateAfter, binTimeBefore, binTimeAfter):
    Datas_aligned = dict()
    for i, data in enumerate(Datas.values()):
        # print(i)
        data_sub_aligned, Mask_aligned, binTimeLine, binAlignIdx = alignData_masked(
            data[featureKey], ~data[maskKey], data[alignKey],
            binTimeBefore, binTimeAfter, dtype=float)
        bin_State_aligned, _, _, _ = alignData_masked(
            data[stateKey], ~data[maskKey], data[alignKey],
            binTimeBefore, binTimeAfter, dtype=float)
        beforeMask = ~(bin_State_aligned.filled(-9999)[:, 0:binAlignIdx] == stateBefore)
        afterMask = ~(bin_State_aligned.filled(-9999)[:, binAlignIdx:] == stateAfter)
        # print(Mask_aligned.dtype)
        Mask_aligned = np.logical_or(np.hstack([beforeMask, afterMask]), Mask_aligned)
        data_sub_aligned_censored = np.ma.MaskedArray(data=data_sub_aligned.data, mask=Mask_aligned)
        Datas_aligned[Data_labels[i]] = data_sub_aligned_censored

    return Datas_aligned, Mask_aligned, binTimeLine, binAlignIdx

def alignStates_SeveralDatas(Datas, Data_labels, stateKey, num_states, maskKey, alignKey, binTimeBefore, binTimeAfter):
    Datas_aligned = dict()
    for i, data in enumerate(Datas.values()):
        statesToAlign = data[stateKey]
        maskToAlign = ~data[maskKey]
        IdxToAlign = data[alignKey]

        StateByAligned, MaskByAligned, probStateAligned, binTimeLine, binAlignIdx, mult_prop_ci =\
            genProbStateAligned_masked(statesToAlign, maskToAlign, IdxToAlign, binTimeBefore, binTimeAfter, num_states)

        Datas_aligned[Data_labels[i]] = dict()
        Datas_aligned[Data_labels[i]]['StateByAligned'] = StateByAligned
        Datas_aligned[Data_labels[i]]['MaskByAligned'] = MaskByAligned
        Datas_aligned[Data_labels[i]]['probStateAligned'] = probStateAligned
        Datas_aligned[Data_labels[i]]['mult_prop_ci'] = mult_prop_ci

    return Datas_aligned, binTimeLine, binAlignIdx
#this version allows arbitrary matrix to be passed in as the event to align
def alignStates_SeveralDatas_EventToAlign(Datas, Data_labels, stateKey, num_states, maskKey, IdxToAlign, binTimeBefore, binTimeAfter):
    Datas_aligned = dict()
    for i, data in enumerate(Datas.values()):
        statesToAlign = data[stateKey]
        maskToAlign = ~data[maskKey]

        StateByAligned, MaskByAligned, probStateAligned, binTimeLine, binAlignIdx, mult_prop_ci =\
            genProbStateAligned_masked(statesToAlign, maskToAlign, IdxToAlign, binTimeBefore, binTimeAfter, num_states)

        Datas_aligned[Data_labels[i]] = dict()
        Datas_aligned[Data_labels[i]]['StateByAligned'] = StateByAligned
        Datas_aligned[Data_labels[i]]['MaskByAligned'] = MaskByAligned
        Datas_aligned[Data_labels[i]]['probStateAligned'] = probStateAligned
        Datas_aligned[Data_labels[i]]['mult_prop_ci'] = mult_prop_ci

    return Datas_aligned, binTimeLine, binAlignIdx

def genAutoCrossCorr_SeveralDatas(Datas, Data_labels, featureKeys, commonKey):
    Datas_autocorr = dict()
    Datas_xcorr = dict()
    featureCombinations = [[fKey,commonKey] for fKey in featureKeys]
    for i, data in enumerate(Datas.values()):
        Datas_autocorr[Data_labels[i]] = dict()
        Datas_xcorr[Data_labels[i]] = dict()
        for j in range(len(featureCombinations)):
            featureKey1 = featureCombinations[j][0]
            featureKey2 = featureCombinations[j][1]
            Mean_feature1_autocorr,SEM_feature1_autocorr,Mean_feature2_autocorr,SEM_feature2_autocorr,Mean_feature1_feature2_xcorr,SEM_feature1_feature2_xcorr = \
                generateAutoCrossCorr(data[featureKey1], data[featureKey2], data['InLawnRunMask'])
            Datas_autocorr[Data_labels[i]][featureKey1 + ' mean'] = Mean_feature1_autocorr
            Datas_autocorr[Data_labels[i]][featureKey1 + ' SEM'] = SEM_feature1_autocorr
            Datas_autocorr[Data_labels[i]][featureKey2 + ' mean'] = Mean_feature2_autocorr
            Datas_autocorr[Data_labels[i]][featureKey2 + ' SEM'] = SEM_feature2_autocorr
            Datas_xcorr[Data_labels[i]][featureKey1 + '_' + featureKey2 + '_mean'] = Mean_feature1_feature2_xcorr
            Datas_xcorr[Data_labels[i]][featureKey1 + '_' + featureKey2 + '_SEM'] = SEM_feature1_feature2_xcorr
    return Datas_autocorr, Datas_xcorr
#generate autocorrelation and cross correlations between two data features
def generateAutoCrossCorr(feature1,feature2,inlawnrunmask):
    from scipy.stats.mstats import sem
    #1) zscore data natively for this selection and make sure that there are no missing values
    feature1_zs_inLawn = np.ma.masked_array(
        data=(feature1-np.nanmean(feature1,axis=1).reshape(-1,1))/(np.nanstd(feature1)),
        mask=~inlawnrunmask) #divide by the length of the vector to make this a scale between -1 and 1
    feature2_zs_inLawn = np.ma.masked_array(
        data=(feature2 - np.nanmean(feature2, axis=1).reshape(-1, 1)) / (np.nanstd(feature2)),
        mask=~inlawnrunmask)

    # #alternatively: just mean-subtract but don't divide out stdev -- looks like it yields the same result
    # feature1_zs_inLawn = np.ma.masked_array(
    #     data=(feature1 - np.nanmean(feature1, axis=1).reshape(-1, 1)),
    #     mask=~inlawnrunmask)
    # feature2_zs_inLawn = np.ma.masked_array(
    #     data=(feature2 - np.nanmean(feature2, axis=1).reshape(-1, 1)),
    #     mask=~inlawnrunmask)

    feature1_zs_inLawn_filled = feature1_zs_inLawn.filled(np.nanmean(feature1_zs_inLawn))
    feature2_zs_inLawn_filled = feature2_zs_inLawn.filled(np.nanmean(feature2_zs_inLawn))
    #interpolate if any NaNs remain:
    if np.sum(np.isnan(feature1_zs_inLawn_filled)) > 0:
        feature1_zs_inLawn_filled = fillNA(feature1_zs_inLawn_filled)
    if np.sum(np.isnan(feature2_zs_inLawn_filled)) > 0:
        feature2_zs_inLawn_filled = fillNA(feature2_zs_inLawn_filled)

    #2) pre-allocate arrays for auto and cross correlations
    feature1_autocorr = np.zeros((feature1_zs_inLawn_filled.shape[0],(feature1_zs_inLawn_filled.shape[1]*2)-1))
    feature2_autocorr = np.zeros_like(feature1_autocorr)
    feature1_feature2_xcorr = np.zeros_like(feature1_autocorr)
    #3) compute correlations per animal
    for i in range(0, feature1_zs_inLawn_filled.shape[0]):
        feature1_AC = np.correlate(feature1_zs_inLawn_filled[i],feature1_zs_inLawn_filled[i], "full")
        # Normalize by the zero-lag value:
        feature1_AC /= feature1_AC[feature1_zs_inLawn.shape[1] - 1]
        feature1_autocorr[i, :] = feature1_AC

        feature2_AC = np.correlate(feature2_zs_inLawn_filled[i], feature2_zs_inLawn_filled[i], "full")
        # Normalize by the zero-lag value:
        feature2_AC /= feature2_AC[feature2_zs_inLawn.shape[1] - 1]
        feature2_autocorr[i, :] = feature2_AC

        feature1_feature2_XCov = np.correlate(feature1_zs_inLawn_filled[i],feature2_zs_inLawn_filled[i], "full")
        # denom = max(1e-6,feature2_zs_inLawn.shape[1] * feature1_zs_inLawn_filled[i].std() * feature2_zs_inLawn_filled[i].std())
        denom = feature2_zs_inLawn.shape[1] * feature1_zs_inLawn_filled[i].std() * feature2_zs_inLawn_filled[i].std()
        feature1_feature2_XCor = feature1_feature2_XCov / denom
        feature1_feature2_xcorr[i, :] = feature1_feature2_XCor

    # #4) compute mean and std. error for each auto and cross correlation
    Mean_feature1_autocorr = np.mean(feature1_autocorr, 0)
    SEM_feature1_autocorr = sem(feature1_autocorr, axis=0)

    Mean_feature2_autocorr = np.mean(feature2_autocorr, 0)
    SEM_feature2_autocorr = sem(feature2_autocorr, axis=0)

    Mean_feature1_feature2_xcorr = np.mean(feature1_feature2_xcorr, 0)
    SEM_feature1_feature2_xcorr = sem(feature1_feature2_xcorr, axis=0)

    return Mean_feature1_autocorr,SEM_feature1_autocorr,Mean_feature2_autocorr,SEM_feature2_autocorr,Mean_feature1_feature2_xcorr,SEM_feature1_feature2_xcorr

def fisher_exact_pvals(Comparisons_df, datakey, pairs):
    """
    Calculates the pvalues of all pairs of variables by Fisher's Exact Test. Should only be used for Bernoulli data.
    :param Comparisons_df: big dataframe containing all genotypes to be compared with requisite fields
    :param datakey: the parameter to compare
    :param pairs: pairs of genoyptes, for which to compute pvalues of statistical testing (two tailed), default Boneferroni corrected
    :return: pval_df, a new dataframe containing all adjusted pvalues for each pair to be tested.
    """
    pvals = dict()
    oddsratio = dict()
    data_0_ci = dict()
    data_1_ci = dict()

    for idx, d in enumerate(pairs):
        df_0 = Comparisons_df.where(Comparisons_df.genotype == d[0]).dropna()
        df_1 = Comparisons_df.where(Comparisons_df.genotype == d[1]).dropna()
        data_0 = df_0[datakey]
        data_1 = df_1[datakey]
        data_0_numSuccess = np.sum(data_0 == 1.0)
        data_0_numFail = np.sum(data_0 == 0.0)
        data_1_numSuccess = np.sum(data_1 == 1.0)
        data_1_numFail = np.sum(data_1 == 0.0)
        # top row = control, bottom row = experimental; left column = # success, right column = # fail
        contTable = np.array([[data_0_numSuccess, data_0_numFail], \
                              [data_1_numSuccess, data_1_numFail]])
        oddsr, p = fisher_exact(contTable, alternative='two-sided')
        # get 95% confidence intervals
        data_0_ci_lower, data_0_ci_upper = \
            proportion_confint(data_0_numSuccess, data_0_numSuccess + data_0_numFail, alpha=0.05)
        data_1_ci_lower, data_1_ci_upper = \
            proportion_confint(data_1_numSuccess, data_1_numSuccess + data_1_numFail, alpha=0.05)

        pvals[d[0] + ', ' + d[1]] = p
        oddsratio[d[0] + ', ' + d[1]] = oddsr
        data_0_ci[d[0] + ', ' + d[1]] = (
                                                    data_0_ci_upper - data_0_ci_lower) / 2  # report the size of the errorbar for symmetric size reporting
        data_1_ci[d[0] + ', ' + d[1]] = (data_1_ci_upper - data_1_ci_lower) / 2

    pval_df = pd.DataFrame.from_dict(pvals, orient='index').rename(columns={0: "p value"})
    #     print(pvals.values())
    pvals_adj = np.array(list(pvals.values())) * len(pairs)  # bonferroni correct
    #     reject, pvals_adj, alphacSidak, alphacBonf = multipletests(pval_df['p value'], alpha=0.05,
    #                                                                    method='bonferroni')
    pval_df.insert(loc=0, column="data 0", value=[p_[0] for p_ in pairs])
    pval_df.insert(loc=1, column="data 1", value=[p_[1] for p_ in pairs])
    pval_df.insert(loc=2, column="type", value=["fishersexact" for p_ in pairs])
    pval_df.insert(loc=3, column="pvals_adj", value=pvals_adj)
    pval_df.insert(loc=4, column="pvals_adj sig?", value=pvals_adj < 0.05)
    pval_df.insert(loc=5, column="data_0 ci", value=data_0_ci.values())
    pval_df.insert(loc=6, column="data_1 ci", value=data_1_ci.values())

    return pval_df

# do statistical test on last X minutes before and after leaving, bonferroni corrected
def mwuCompareCurves(stateAligned, whichState, binAlignIdx, pairs, numMins, binsPerMin): #compares usage of a state on either side of an aligned index
    binXmin = numMins * binsPerMin

    p_Before = []
    p_After = []
    p_Before_starStrings = []
    p_After_starStrings = []

    numPairs = len(pairs) #figure out how many statistical tests you will do
    for i, p in enumerate(pairs):
        if stateAligned[p[0]].shape[0] < 3 or stateAligned[p[1]].shape[0] < 3: #if there are less than 3 traces that align with the index in either element of the pair, skip this pair
            numPairs = numPairs - 1

    for i, p in enumerate(pairs):
        if stateAligned[p[0]].shape[0] < 3 or stateAligned[p[1]].shape[0] < 3: #if there are less than 3 traces that align with the index in either element of the pair, skip this pair
            p_Before.append('NA')
            p_Before_starStrings.append('NA')
            p_After.append('NA')
            p_After_starStrings.append('NA')
            continue

        fracInState = np.nanmean((stateAligned[p[0]] == whichState)[:, binAlignIdx - binXmin:binAlignIdx], axis=1)
        fracInState = linscaleData_minmax(fracInState, (1/binXmin), 1-(1/binXmin))
        g0_Xminbefore = logit(fracInState)

        fracInState = np.nanmean((stateAligned[p[1]] == whichState)[:, binAlignIdx - binXmin:binAlignIdx], axis=1)
        fracInState = linscaleData_minmax(fracInState, (1/binXmin), 1-(1/binXmin))
        g1_Xminbefore = logit(fracInState)

        # _, p_Bf = mannwhitneyu(g0_Xminbefore, g1_Xminbefore)
        _, p_Bf = ttest_ind(g0_Xminbefore, g1_Xminbefore)
        p_Bf = p_Bf * numPairs # bonferroni correction
        p_Before.append(p_Bf)
        p_Before_starStrings.append(categorizePValue(p_Bf))

        fracInState = np.nanmean((stateAligned[p[0]] == whichState)[:, binAlignIdx:binAlignIdx + binXmin], axis=1)
        fracInState = linscaleData_minmax(fracInState, (1/binXmin), 1-(1/binXmin))
        g0_Xminafter = logit(fracInState)

        fracInState = np.nanmean((stateAligned[p[1]] == whichState)[:, binAlignIdx:binAlignIdx + binXmin], axis=1)
        fracInState = linscaleData_minmax(fracInState, (1/binXmin), 1-(1/binXmin))
        g1_Xminafter = logit(fracInState)
        # _, p_Af = mannwhitneyu(g0_Xminafter, g1_Xminafter)
        _, p_Af = ttest_ind(g0_Xminafter, g1_Xminafter)
        p_Af = p_Af * numPairs
        p_After.append(p_Af)  # bonferroni correction
        p_After_starStrings.append(categorizePValue(p_Af))

    return p_Before, p_Before_starStrings, p_After, p_After_starStrings

def logit_ttest_CompareCurves_anytime(stateAligned, whichState, binAlignIdx, timeIntervals, pairs): #compares usage of a state during arbitrarily defined time intervals (relative to alignIdx)
    """
    :param stateAligned: integer valued state matrix aligned to a particular event
    :param whichState: the state number or value for which you would like to compare the prevalence in the given time intervals
    :param timeIntervals: the time bins for which you would like to compare state usage across genotypes/conditions. should be expressed as an array of arrays, each one specifying a time interval with a start and end bin (in terms of bins relative to 0)
    :param pairs: the genotype or condition combinations you wish to compare to each other.
    :return: pvalues regarding comparisons of state usage across specified time intervals
    """

    givenpairs = copy.deepcopy(pairs)
    pairs = []
    for i, p in enumerate(givenpairs): #edit pairs to make sure all have enough data to compare
        if stateAligned[p[0]].shape[0] >= 3 and stateAligned[p[1]].shape[0] >= 3:  # if there are less than 3 traces that align with the index in either element of the pair, skip this pair
            pairs.append(p)
    numPairs = len(pairs)  # figure out how many statistical tests you will do
    # print(pairs)

    p_vals = np.empty(shape=(len(pairs),len(timeIntervals))) #rows correspond to pairs, columns correspond to timeIntervals
    p_vals_starStrings = np.empty(shape=(len(pairs),len(timeIntervals)),dtype="U16")

    for i, p in enumerate(pairs):
        # print(p)
        #loop over time Intervals to compare
        for j, t in enumerate(timeIntervals):
            binXmin = abs(t[1]-t[0])
            # print(binXmin)
            fracInState = np.nanmean((stateAligned[p[0]] == whichState)[:, binAlignIdx+t[0]:binAlignIdx+t[1]], axis=1)
            fracInState = linscaleData_minmax(fracInState, (1 / binXmin), 1 - (1 / binXmin))
            # print(fracInState)
            g0 = logit(fracInState)
            # print(g0)

            fracInState = np.nanmean((stateAligned[p[1]] == whichState)[:, binAlignIdx+t[0]:binAlignIdx+t[1]], axis=1)
            fracInState = linscaleData_minmax(fracInState, (1 / binXmin), 1 - (1 / binXmin))
            # print(fracInState)
            g1 = logit(fracInState)
            # print(g1)

            # _, p_val = mannwhitneyu(g0, g1)
            _, p_val = ttest_ind(g0, g1)
            # print('pval = '+str(p_val))
            p_val = p_val * numPairs
            p_vals[i][j] = p_val # bonferroni correction
            p_vals_starStrings[i][j] = categorizePValue(p_val)

    return pairs,p_vals, p_vals_starStrings

def mwuCompareCurves_other(stateAligned, binAlignIdx, pairs, numMins, binsPerMin):  # for quantities that arent states
    binXmin = numMins * binsPerMin

    p_Before = []
    p_After = []
    p_Before_starStrings = []
    p_After_starStrings = []

    numPairs = len(pairs)
    for i, p in enumerate(pairs):
        if stateAligned[p[0]].size == 0 or stateAligned[p[1]].size == 0:
            numPairs = numPairs - 1
            p_Before.append('NA')
            p_Before_starStrings.append('NA')
            p_After.append('NA')
            p_After_starStrings.append('NA')
            continue

        g0_Xminbefore = np.nanmean(stateAligned[p[0]][:, binAlignIdx - binXmin:binAlignIdx], axis=1)
        g1_Xminbefore = np.nanmean(stateAligned[p[1]][:, binAlignIdx - binXmin:binAlignIdx], axis=1)
        _, p_Bf = mannwhitneyu(g0_Xminbefore, g1_Xminbefore)
        p_Bf = p_Bf * numPairs
        p_Before.append(p_Bf)  # bonferroni correction
        p_Before_starStrings.append(categorizePValue(p_Bf))

        g0_Xminafter = np.nanmean(stateAligned[p[0]][:, binAlignIdx:binAlignIdx + binXmin], axis=1)
        g1_Xminafter = np.nanmean(stateAligned[p[1]][:, binAlignIdx:binAlignIdx + binXmin], axis=1)
        _, p_Af = mannwhitneyu(g0_Xminafter, g1_Xminafter)
        p_Af = p_Af * numPairs
        p_After.append(p_Af)  # bonferroni correction
        p_After_starStrings.append(categorizePValue(p_Af))

    return p_Before, p_Before_starStrings, p_After, p_After_starStrings

def mwuCompareCurves_other_anytime(dataAligned, binAlignIdx, timeIntervals, pairs):
    """
        :param dataAligned: continuous valued state matrix aligned to a particular event
        :param timeIntervals: the time bins for which you would like to compare state usage across genotypes/conditions. should be expressed as an array of arrays, each one specifying a time interval with a start and end bin (in terms of bins relative to 0)
        :param pairs: the genotype or condition combinations you wish to compare to each other.
        :return: pvalues regarding comparisons of state usage across specified time intervals
        """
    p_vals = np.empty(shape=(len(pairs), len(timeIntervals)))  # rows correspond to pairs, columns correspond to timeIntervals
    p_vals_starStrings = np.empty(shape=(len(pairs), len(timeIntervals)), dtype="U16")

    numPairs = len(pairs)
    for i, p in enumerate(pairs):
        # print(p)
        if dataAligned[p[0]].size == 0 or dataAligned[p[1]].size == 0:  # if there are no traces that align with the index in either element of the pair, skip this pair
            numPairs = numPairs - 1
            p_vals.append('NA')
            p_vals_starStrings.append('NA')
            p_After.append('NA')
            p_After_starStrings.append('NA')
            continue

        # loop over time Intervals to compare
        for j, t in enumerate(timeIntervals):
            # print(t)
            t = t.astype(int)
            # print(t)
            g0 = np.nanmean((dataAligned[p[0]])[:, binAlignIdx + t[0]:binAlignIdx + t[1]], axis=1)
            g1 = np.nanmean((dataAligned[p[1]])[:, binAlignIdx + t[0]:binAlignIdx + t[1]], axis=1)
            # print(np.mean(g0),np.mean(g1))
            _, p_val = mannwhitneyu(g0, g1)
            # print(p_val)
            p_val = p_val * numPairs
            # print(p_val)
            p_vals[i][j] = p_val  # bonferroni correction
            p_vals_starStrings[i][j] = categorizePValue(p_val)

    return p_vals, p_vals_starStrings

def categorizePValue(p):
    # print(p)
    starString = None
    if p > 5.00e-02:
        starString = 'ns'
    elif 1.00e-02 < p <= 5.00e-02:
        starString = '*'
    elif 1.00e-03 < p <= 1.00e-02:
        starString = '**'
    elif 1.00e-04 < p <= 1.00e-03:
        starString = '***'
    elif p <= 1.00e-04:
        starString = '****'
    elif np.isnan(p):
        starString = 'nan'
    else:
        raise ValueError("p value must be numeric in the specified ranges.")
    return starString

def getFracARHMMstate_light(d, lightOnIdx, lightOffIdx):
    matfilenames = d['matfilename'].astype(str)
    dates = pd.DataFrame(d['dates'].to_numpy(), columns=['dates'])

    state0 = pd.DataFrame(np.repeat('state0', len(dates)), columns=['state'])
    state1 = pd.DataFrame(np.repeat('state1', len(dates)), columns=['state'])
    state2 = pd.DataFrame(np.repeat('state2', len(dates)), columns=['state'])
    state3 = pd.DataFrame(np.repeat('state3', len(dates)), columns=['state'])

    lightOff = pd.DataFrame(np.repeat('light OFF', len(dates)), columns=['light'])
    lightOn = pd.DataFrame(np.repeat('light ON', len(dates)), columns=['light'])

    # calculate the same metrics for light ON and light OFF and enter them as separate entries into a common dataframe
    # light OFF
    states = (d['arHMM_MLstates']).filled(10).astype(int)[:, lightOffIdx]
    fracState_lightOff = np.array(
        [[np.sum(states[k] == j) / states.shape[1] for j in range(0, 4)] for k in range(0, states.shape[0])])

    state0_df_lightOff = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOff, state0, fracState_lightOff[:, 0].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    state1_df_lightOff = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOff, state1, fracState_lightOff[:, 1].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    state2_df_lightOff = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOff, state2, fracState_lightOff[:, 2].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    state3_df_lightOff = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOff, state3, fracState_lightOff[:, 3].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    # calculate the same metrics for light ON and light ON and enter them as separate entries into a common dataframe
    # light ON
    states = (d['arHMM_MLstates']).filled(10).astype(int)[:, lightOnIdx]
    fracState_lightOn = np.array(
        [[np.sum(states[k] == j) / states.shape[1] for j in range(0, 4)] for k in range(0, states.shape[0])])

    state0_df_lightOn = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOn, state0, fracState_lightOn[:, 0].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    state1_df_lightOn = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOn, state1, fracState_lightOn[:, 1].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    state2_df_lightOn = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOn, state2, fracState_lightOn[:, 2].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    state3_df_lightOn = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOn, state3, fracState_lightOn[:, 3].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    #######
    # combine it all togethers
    df = pd.concat([state0_df_lightOff, state1_df_lightOff, state2_df_lightOff, state3_df_lightOff, \
                    state0_df_lightOn, state1_df_lightOn, state2_df_lightOn, state3_df_lightOn], axis=0)

    return df
def getFracRoamDwell_light(d, lightOnIdx, lightOffIdx):
    matfilenames = d['matfilename'].astype(str)
    dates = pd.DataFrame(d['dates'].to_numpy(), columns=['dates'])

    dwell = pd.DataFrame(np.repeat('dwell', len(dates)), columns=['state'])
    roam = pd.DataFrame(np.repeat('roam', len(dates)), columns=['state'])

    lightOff = pd.DataFrame(np.repeat('light OFF', len(dates)), columns=['light'])
    lightOn = pd.DataFrame(np.repeat('light ON', len(dates)), columns=['light'])

    # calculate the same metrics for light ON and light OFF and enter them as separate entries into a common dataframe
    # light OFF
    states = (d['RD_states_Matrix_exog_Cent']).filled(10).astype(int)[:, lightOffIdx]
    fracState_lightOff = np.array(
        [[np.sum(states[k] == j) / states.shape[1] for j in range(0, 2)] for k in range(0, states.shape[0])])

    dwell_df_lightOff = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOff, dwell, fracState_lightOff[:, 0].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    roam_df_lightOff = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOff, roam, fracState_lightOff[:, 1].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    # calculate the same metrics for light ON and light ON and enter them as separate entries into a common dataframe
    # light ON
    states = (d['RD_states_Matrix_exog_Cent']).filled(10).astype(int)[:, lightOnIdx]
    fracState_lightOn = np.array(
        [[np.sum(states[k] == j) / states.shape[1] for j in range(0, 2)] for k in range(0, states.shape[0])])

    dwell_df_lightOn = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOn, dwell, fracState_lightOn[:, 0].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    roam_df_lightOn = pd.DataFrame(
        np.hstack([matfilenames, dates, lightOn, roam, fracState_lightOn[:, 1].reshape(-1, 1)]),
        columns=['matfilename', 'dates', 'light', 'state', 'frac state'])

    #######
    # combine it all togethers
    df = pd.concat([dwell_df_lightOff, roam_df_lightOff, dwell_df_lightOn, roam_df_lightOn], axis=0)

    return df
