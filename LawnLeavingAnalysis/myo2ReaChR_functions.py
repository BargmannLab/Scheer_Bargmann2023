
#functions from "Figure 1 and 2 - Foraging decisions emanate from high arousal states.ipynb" originally worked on this in December 2020
import os
import pandas as pd
import numpy as np
import ssm
from ssm.util import one_hot
import re
import copy
from numpy.linalg import norm
from scipy.stats import multivariate_normal
from scipy.ndimage import convolve1d
from functools import reduce
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.model_selection import train_test_split
import statsmodels.stats.proportion as prop
from collections.abc import Iterable
# from sklearn.cluster import DBSCAN
from scipy.signal import find_peaks
from scipy.stats import zscore
import h5py
from scipy import signal
from scipy.stats.mstats import sem
import itertools
from functools import reduce

def findCommonDateEntriesMultiple(*Datas):
    datesList = []
    for data in Datas:
        datesList.append(np.unique(data['dates']))
    commonDates = reduce(np.intersect1d, datesList)
    dataIdxList = []
    dataSubselectList = []
    for data in Datas:
        dataIdx = np.sort(np.where([data['dates']==commonDates[i] for i in range(0,len(commonDates))])[1])
        dataSubselect = subselectData(data,dataIdx)
        dataIdxList.append(dataIdx)
        dataSubselectList.append(dataSubselect)
    return dataIdxList, dataSubselectList

#function to return a desired subset of the Data
import copy

def subselectData(Data,idxToReturn):
    tmpData = copy.deepcopy(Data) #don't mess with the original
    mask = np.zeros(tmpData['speed'].shape[0], dtype=bool)
    mask[idxToReturn] = True

    for key, val in tmpData.items(): #subselect only those entries that are not to be removed
        if key!="goodIntervals" and key!="PCAObj" and key!="MLstates": #just resave these as IS (i.e. do nothing)
#             print(key)
            tmpData[key] = val[mask]
    return tmpData


# makes an row entry for each animalxeach state - convenient for plotting.
def makeDataFramePerStatePerAnimal_light(*data, lightOnIdx, lightOffIdx, genotypeList, odList, experimentList):
    dataIdxList, dataSubselectList = findCommonDateEntriesMultiple(*data)

    #     df = pd.DataFrame(columns = ['dates','genotype','od','experiment',
    #     frac_state0','frac_state1','frac_state2','frac_state3'
    #     'mean_PC1_state0'...,'mean_PC2_state0'...,'mean_binAbsSpeed_state0...','mean_binfspeed_state0...',
    #      'mean_binrspeed_state0'...,'mean_binangspeed_state0...'
    #       binForward_state0...,binReverse_state0...,binPause_state0... ])

    df_list = []
    for i, d in enumerate(data):
        videonames = d['videoname'].iloc[dataIdxList[i]].astype(str)
        dates = pd.DataFrame(d['dates'].iloc[dataIdxList[i]].to_numpy(), columns=['dates'])
        genotype = pd.DataFrame(np.repeat(genotypeList[i], len(dates)), columns=['genotype'])
        od = pd.DataFrame(np.repeat(odList[i], len(dates)), columns=['od'])
        experiment = pd.DataFrame(np.repeat(experimentList[i], len(dates)), columns=['experiment'])

        state0 = pd.DataFrame(np.repeat('state0', len(dates)), columns=['state'])
        state1 = pd.DataFrame(np.repeat('state1', len(dates)), columns=['state'])
        state2 = pd.DataFrame(np.repeat('state2', len(dates)), columns=['state'])
        state3 = pd.DataFrame(np.repeat('state3', len(dates)), columns=['state'])

        lightOff = pd.DataFrame(np.repeat('light OFF', len(dates)), columns=['light'])
        lightOn = pd.DataFrame(np.repeat('light ON', len(dates)), columns=['light'])

        # calculate the same metrics for light ON and light OFF and enter them as separate entries into a common dataframe
        # light OFF
        states = d['MLstates'][dataIdxList[i]][:, lightOffIdx]
        binAbsSpeed = d['binAbsSpeed'][dataIdxList[i]][:, lightOffIdx]
        binfspeed = d['binfspeed'][dataIdxList[i]][:, lightOffIdx]
        binrspeed = d['binrspeed'][dataIdxList[i]][:, lightOffIdx]
        binangspeed = d['binangspeed'][dataIdxList[i]][:, lightOffIdx]
        binMSD = d['binMSD'][dataIdxList[i]][:, lightOffIdx]
        binForward = d['binForward'][dataIdxList[i]][:, lightOffIdx]
        binReverse = d['binReverse'][dataIdxList[i]][:, lightOffIdx]
        binPause = d['binPause'][dataIdxList[i]][:, lightOffIdx]

        fracState_lightOff = np.array(
            [[np.sum(states[k] == j) / states.shape[1] for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binAbsSpeed_lightOff = np.array(
            [[np.nanmean(binAbsSpeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binfspeed_lightOff = np.array(
            [[np.nanmean(binfspeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binrspeed_lightOff = np.array(
            [[np.nanmean(binrspeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binangspeed_lightOff = np.array(
            [[np.nanmean(binangspeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binMSD_lightOff = np.array(
            [[np.nanmean(binMSD[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binForward_lightOff = np.array(
            [[np.nanmean(binForward[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binReverse_lightOff = np.array(
            [[np.nanmean(binReverse[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binPause_lightOff = np.array(
            [[np.nanmean(binPause[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])

        #         set_trace()

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOff[:, 0], mean_binAbsSpeed_lightOff[:, 0],
                                           mean_binfspeed_lightOff[:, 0], mean_binrspeed_lightOff[:, 0],
                                           mean_binangspeed_lightOff[:, 0], mean_binMSD_lightOff[:, 0],
                                           mean_binForward_lightOff[:, 0], mean_binReverse_lightOff[:, 0],
                                           mean_binPause_lightOff[:, 0])).T)
        state0_df_lightOff = pd.DataFrame(
            np.hstack([videonames, dates, genotype, od, experiment, lightOff, state0, tmp]),
            columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light', 'state', 'frac state',
                     'mean binAbsSpeed', 'mean binfspeed', 'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                     'mean binForward', 'mean binReverse', 'mean binPause'])

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOff[:, 1], mean_binAbsSpeed_lightOff[:, 1],
                                           mean_binfspeed_lightOff[:, 1], mean_binrspeed_lightOff[:, 1],
                                           mean_binangspeed_lightOff[:, 1], mean_binMSD_lightOff[:, 1],
                                           mean_binForward_lightOff[:, 1], mean_binReverse_lightOff[:, 1],
                                           mean_binPause_lightOff[:, 1])).T)
        state1_df_lightOff = pd.DataFrame(
            np.hstack([videonames, dates, genotype, od, experiment, lightOff, state1, tmp]),
            columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light', 'state', 'frac state',
                     'mean binAbsSpeed', 'mean binfspeed', 'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                     'mean binForward', 'mean binReverse', 'mean binPause'])

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOff[:, 2], mean_binAbsSpeed_lightOff[:, 2],
                                           mean_binfspeed_lightOff[:, 2], mean_binrspeed_lightOff[:, 2],
                                           mean_binangspeed_lightOff[:, 2], mean_binMSD_lightOff[:, 2],
                                           mean_binForward_lightOff[:, 2], mean_binReverse_lightOff[:, 2],
                                           mean_binPause_lightOff[:, 2])).T)
        state2_df_lightOff = pd.DataFrame(
            np.hstack([videonames, dates, genotype, od, experiment, lightOff, state2, tmp]),
            columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light', 'state', 'frac state',
                     'mean binAbsSpeed', 'mean binfspeed', 'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                     'mean binForward', 'mean binReverse', 'mean binPause'])

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOff[:, 3], mean_binAbsSpeed_lightOff[:, 3],
                                           mean_binfspeed_lightOff[:, 3], mean_binrspeed_lightOff[:, 3],
                                           mean_binangspeed_lightOff[:, 3], mean_binMSD_lightOff[:, 3],
                                           mean_binForward_lightOff[:, 3], mean_binReverse_lightOff[:, 3],
                                           mean_binPause_lightOff[:, 3])).T)
        state3_df_lightOff = pd.DataFrame(
            np.hstack([videonames, dates, genotype, od, experiment, lightOff, state3, tmp]),
            columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light', 'state', 'frac state',
                     'mean binAbsSpeed', 'mean binfspeed', 'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                     'mean binForward', 'mean binReverse', 'mean binPause'])

        #### light ON
        states = d['MLstates'][dataIdxList[i]][:, lightOnIdx]
        binAbsSpeed = d['binAbsSpeed'][dataIdxList[i]][:, lightOnIdx]
        binfspeed = d['binfspeed'][dataIdxList[i]][:, lightOnIdx]
        binrspeed = d['binrspeed'][dataIdxList[i]][:, lightOnIdx]
        binangspeed = d['binangspeed'][dataIdxList[i]][:, lightOnIdx]
        binMSD = d['binMSD'][dataIdxList[i]][:, lightOnIdx]
        binForward = d['binForward'][dataIdxList[i]][:, lightOnIdx]
        binReverse = d['binReverse'][dataIdxList[i]][:, lightOnIdx]
        binPause = d['binPause'][dataIdxList[i]][:, lightOnIdx]

        fracState_lightOn = np.array(
            [[np.sum(states[k] == j) / states.shape[1] for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binAbsSpeed_lightOn = np.array(
            [[np.nanmean(binAbsSpeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binfspeed_lightOn = np.array(
            [[np.nanmean(binfspeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binrspeed_lightOn = np.array(
            [[np.nanmean(binrspeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binangspeed_lightOn = np.array(
            [[np.nanmean(binangspeed[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binMSD_lightOn = np.array(
            [[np.nanmean(binMSD[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binForward_lightOn = np.array(
            [[np.nanmean(binForward[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binReverse_lightOn = np.array(
            [[np.nanmean(binReverse[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])
        mean_binPause_lightOn = np.array(
            [[np.nanmean(binPause[k][states[k] == j]) for j in range(0, 4)] for k in range(0, states.shape[0])])

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOn[:, 0], mean_binAbsSpeed_lightOn[:, 0],
                                           mean_binfspeed_lightOn[:, 0], mean_binrspeed_lightOn[:, 0],
                                           mean_binangspeed_lightOn[:, 0], mean_binMSD_lightOn[:, 0],
                                           mean_binForward_lightOn[:, 0], mean_binReverse_lightOn[:, 0],
                                           mean_binPause_lightOn[:, 0])).T)
        state0_df_lightOn = pd.DataFrame(np.hstack([videonames, dates, genotype, od, experiment, lightOn, state0, tmp]),
                                         columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light',
                                                  'state', 'frac state', 'mean binAbsSpeed', 'mean binfspeed',
                                                  'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                                                  'mean binForward', 'mean binReverse', 'mean binPause'])

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOn[:, 1], mean_binAbsSpeed_lightOn[:, 1],
                                           mean_binfspeed_lightOn[:, 1], mean_binrspeed_lightOn[:, 1],
                                           mean_binangspeed_lightOn[:, 1], mean_binMSD_lightOn[:, 1],
                                           mean_binForward_lightOn[:, 1], mean_binReverse_lightOn[:, 1],
                                           mean_binPause_lightOn[:, 1])).T)
        state1_df_lightOn = pd.DataFrame(np.hstack([videonames, dates, genotype, od, experiment, lightOn, state1, tmp]),
                                         columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light',
                                                  'state', 'frac state', 'mean binAbsSpeed', 'mean binfspeed',
                                                  'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                                                  'mean binForward', 'mean binReverse', 'mean binPause'])

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOn[:, 2], mean_binAbsSpeed_lightOn[:, 2],
                                           mean_binfspeed_lightOn[:, 2], mean_binrspeed_lightOn[:, 2],
                                           mean_binangspeed_lightOn[:, 2], mean_binMSD_lightOn[:, 2],
                                           mean_binForward_lightOn[:, 2], mean_binReverse_lightOn[:, 2],
                                           mean_binPause_lightOn[:, 2])).T)
        state2_df_lightOn = pd.DataFrame(np.hstack([videonames, dates, genotype, od, experiment, lightOn, state2, tmp]),
                                         columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light',
                                                  'state', 'frac state', 'mean binAbsSpeed', 'mean binfspeed',
                                                  'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                                                  'mean binForward', 'mean binReverse', 'mean binPause'])

        tmp = pd.DataFrame(data=np.vstack((fracState_lightOn[:, 3], mean_binAbsSpeed_lightOn[:, 3],
                                           mean_binfspeed_lightOn[:, 3], mean_binrspeed_lightOn[:, 3],
                                           mean_binangspeed_lightOn[:, 3], mean_binMSD_lightOn[:, 3],
                                           mean_binForward_lightOn[:, 3], mean_binReverse_lightOn[:, 3],
                                           mean_binPause_lightOn[:, 3])).T)
        state3_df_lightOn = pd.DataFrame(np.hstack([videonames, dates, genotype, od, experiment, lightOn, state3, tmp]),
                                         columns=['videoname', 'dates', 'genotype', 'od', 'experiment', 'light',
                                                  'state', 'frac state', 'mean binAbsSpeed', 'mean binfspeed',
                                                  'mean binrspeed', 'mean binangspeed', 'mean binMSD',
                                                  'mean binForward', 'mean binReverse', 'mean binPause'])

        #######
        # combine it all togethers
        tmp_df = pd.concat([state0_df_lightOff, state1_df_lightOff, state2_df_lightOff, state3_df_lightOff, \
                            state0_df_lightOn, state1_df_lightOn, state2_df_lightOn, state3_df_lightOn], axis=0)

        df_list.append(tmp_df)

    df = pd.concat(df_list, axis=0).reset_index(drop=True)

    return df


def getFracARHMMstate_light(d, lightOnIdx, lightOffIdx):
    #     matfilenames = d['matfilename'].astype(str)
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
    #     matfilenames = d['matfilename'].astype(str)
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
