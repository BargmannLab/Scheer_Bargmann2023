# dependencies
import itertools as it
import copy
import numpy as np
from scipy.stats.mstats import sem
import matplotlib as mpl
import matplotlib.pyplot as plt
from ssm.plots import white_to_color_cmap
from matplotlib import cm
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap as lcmap

# PLOTTING FUNCTIONS
from commonFunctions import *


###### helper functions
def annotateBars(row, ax, labels):
    for i,p in enumerate(ax.patches):
         ax.annotate(str(labels[i]), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=14, color='gray', rotation=0, xytext=(0, 20),
             textcoords='offset points')

def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])

def sortByFirstDeriv(toPlot,alignIdx,numLastBins): #sorts a matrix based on the average first derivative in a given aligned interval
    lastbins = toPlot[:, (alignIdx - numLastBins):alignIdx] #this defines the aligned interval to check
    lastbins_1stderiv = np.nanmean(np.gradient(lastbins, axis=1), axis=1)
    sortOrder = np.flip(np.ma.argsort(lastbins_1stderiv, endwith=False))
    toPlot_sorted = toPlot[sortOrder]
    return toPlot_sorted, sortOrder

#generate 1 dimensional histograms for a given feature
def plot1dhistograms(ax,feature,featureLabel,dataRange,numBins,places2round):
    faceColor = (0.5, 0.5, 0.5, 0)
    colorForEdge = 'black'
    lineWeight = 1

    BinNumel = len(feature.ravel())
    _ = ax.hist(feature.ravel(), bins=np.linspace(dataRange[0], dataRange[-1], numBins), fc=faceColor,
                    edgecolor=colorForEdge, lw=lineWeight, weights=np.ones(BinNumel) / BinNumel)
    _ = ax.set_xticks(np.linspace(dataRange[0], dataRange[-1], 4))
    _ = ax.set_xticklabels(np.round(np.linspace(dataRange[0], dataRange[-1], 4),places2round))
    _ = ax.set_xlabel(featureLabel,fontsize=14)
    _ = ax.set_ylabel('normalized frequency',fontsize=14)
    return ax

def plot1dand2dhistograms(feature1,feature1Label,feature1dataRange,feature2,feature2Label,feature2dataRange,norm):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    faceColor = (0.5, 0.5, 0.5, 0)
    colorForEdge = 'black'
    lineWeight = 1


    BinNumel = len(feature1.ravel())
    _ = axs[0].hist(feature1.ravel(), bins=np.linspace(feature1dataRange[0], feature1dataRange[-1], 50), fc=faceColor,
                    edgecolor=colorForEdge, lw=lineWeight, weights=np.ones(BinNumel) / BinNumel)
    _ = axs[0].set_xlabel(feature1Label)
    _ = axs[0].set_ylabel('normalized frequency')

    BinNumel = len(feature2.ravel())
    _ = axs[1].hist(feature2.ravel(), bins=np.linspace(feature2dataRange[0], feature2dataRange[-1], 50), fc=faceColor,
                    edgecolor=colorForEdge, lw=lineWeight, weights=np.ones(BinNumel) / BinNumel)
    _ = axs[1].set_xlabel(feature2Label)
    _ = axs[1].set_ylabel('normalized frequency')

    _, h = plot2dhistosPC1vPC2(axs[2], norm, '', feature1.ravel(), feature1Label, feature1dataRange, feature2.ravel(), feature2Label, feature2dataRange)
    # divider = make_axes_locatable(axs[2])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = plt.colorbar(h, cax=cax)
    # cbar.set_label('normalized frequency', rotation=90)

    return fig,axs

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N

###### plotting properties of an HMM
def plot2dgaussians_emiss_HMM_range(HMM, dataRanges, labels, spacing, colors, figsize):
    combos = np.array(list(it.combinations(np.unique(np.arange(0, HMM.D)), 2)))  # all combinations of dimensions
    fig, axs = plt.subplots(1, len(combos), sharey=False, tight_layout=True, figsize=figsize)
    for i in range(combos.shape[0]):
        dimToCompare = combos[i]
        range0 = np.linspace(dataRanges[dimToCompare[0]][0], dataRanges[dimToCompare[0]][1], spacing)
        range1 = np.linspace(dataRanges[dimToCompare[1]][0], dataRanges[dimToCompare[1]][1], spacing)
        X, Y = np.meshgrid(range0, range1)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y
        for j in reversed(range(HMM.K)):
            if hasattr(HMM.observations,'mus'):
                mu = HMM.observations.mus[j] #gaussian
            elif hasattr(HMM.observations,'bs'):
                mu = HMM.observations.bs[j]  # autoregressive

            Sigma = HMM.observations.Sigmas[j]
            # select out entries relating to the dimToCompare
            m = np.array([mu[dimToCompare[0]], mu[dimToCompare[1]]])
            s = np.array([[Sigma[dimToCompare[0], dimToCompare[0]], Sigma[dimToCompare[0], dimToCompare[1]]], \
                          [Sigma[dimToCompare[1], dimToCompare[0]], Sigma[dimToCompare[1], dimToCompare[1]]]])
            Z = multivariate_gaussian(pos, m, s)
            if len(combos) > 1:
                axs[i].contour(X, Y, Z, cmap=white_to_color_cmap(colors[j]))
                axs[i].set_xlabel(labels[dimToCompare[0]])
                axs[i].set_ylabel(labels[dimToCompare[1]])
            else:
                axs.contour(X, Y, Z, cmap=white_to_color_cmap(colors[j]))
                axs.set_xlabel(labels[dimToCompare[0]])
                axs.set_ylabel(labels[dimToCompare[1]])
    return fig, axs

def plot2dhistosPC1vPC2(ax, norm, genotypeString, PC1, xlabel, pc1datarange, PC2, ylabel, pc2datarange, my_colormap=None):
    xdata = PC1.ravel()
    ydata = PC2.ravel()

    if my_colormap==None:
        my_colormap = copy.copy(mpl.cm.get_cmap('rocket'))  # copy the default cmap
        my_colormap.set_bad((0, 0, 0))
    counts, xedges, yedges, h = ax.hist2d(xdata, ydata, bins=[pc1datarange, pc2datarange], norm=norm, weights=np.ones_like(xdata) / len(ydata), rasterized=True,cmap=my_colormap)
    # cm = h.get_cmap()
    # print(cm.name)
    _ = ax.set_title(genotypeString)
    _ = ax.set_xlabel(xlabel)
    _ = ax.set_ylabel(ylabel)
    _ = plt.colorbar(h, ax=ax)
    return ax, h

#plotting decoded MLstates from an HMM
def plotHMMDecodedData(dataToDecode, MLstates, figsize, colors, cmap, xlabel,yticklabels, title):
    time_bins = dataToDecode.shape[0]
    obs_dim = dataToDecode.shape[1]

    ytickplaces = []
    lim = 2 * abs(np.nanmax(dataToDecode))
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=figsize)
    axs.imshow(MLstates[None, :],
               aspect="auto",
               cmap=cmap,
               vmin=0,
               vmax=len(colors) - 1,
               extent=(0, time_bins, -lim, (obs_dim) * lim))
    for d in range(obs_dim):
        axs.plot(dataToDecode[:, d] + lim * d, '-k', linewidth=2)
        axs.hlines(lim * d, 0, time_bins, colors='k', linestyles='dashed', linewidth=1)
        ytickplaces.append(lim * d)

    axs.set_xlim(0, time_bins)
    xtickplaces = np.linspace(0, time_bins, 5)
    tickMins = list((xtickplaces / 6).astype(int).astype(str))
    axs.set_xticks(xtickplaces)
    axs.set_xticklabels(tickMins)
    axs.set_xlabel(xlabel)
    axs.set_yticks(ytickplaces)
    axs.set_yticklabels(yticklabels)

    axs.set_title(title)
    axs.grid(b=None)
    return fig, axs

def plotGLMHMMDecodedData(dataToDecode,Choices, MLstates, figsize, colors, cmap, xlabel, title):
    from sklearn.preprocessing import StandardScaler
    time_bins = dataToDecode.shape[0]
    obs_dim = dataToDecode.shape[1]

    choiceColor_names = ["black","aquamarine","hot pink"]
    choiceColors = sns.xkcd_palette(choiceColor_names)
    #scale each dimension of dataToDecode for visualization purposes
    dataToDecode = StandardScaler().fit_transform(dataToDecode)

    lim = 2 * abs(dataToDecode).max()
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True, figsize=figsize)
    axs.imshow(MLstates[None, :],
               aspect="auto",
               cmap=cmap,
               vmin=0,
               vmax=len(colors) - 1,
               extent=(0, time_bins, -lim, (obs_dim) * lim))
    for d in range(obs_dim):
        axs.plot(dataToDecode[:, d] + lim * d, '-k', linewidth=3)
        axs.hlines(lim * d, 0, time_bins, colors='k', linestyles='dashed', linewidth=2)
    for k in range(1, Choices.max() + 1):  # indicate the animal's choices
        axs.vlines(np.where(Choices == k)[0], -lim, (lim * obs_dim), color=choiceColors[k - 1],
                   linestyles='dotted')
    axs.set_xlim(0, time_bins)
    axs.set_xlabel(xlabel)
    axs.set_yticks([])

    axs.set_title(title)
    axs.grid(b=None)
    return fig, axs

###### plotting various things aligned to other things
def plotMLstates_prealigned(ax,MLstates_aligned,probStateAligned,probStateAligned_prop_ci,numstates,inlawnrunmask_aligned,binTimeLine,alignIdx,binTimeBefore,maskMeanBool,colors):
    statelabels = ['state ' + str(i) for i in range(0, numstates)]
    numLastBins = binTimeBefore
    binsPerMin = 6
    # first each trace in a heatmap style using native colors
    discreteCmap = lcmap(colors[0:numstates])

    toPlot,  = sortByFirstDeriv(MLstates_aligned, alignIdx, numLastBins)
    # rearrange this by which state the trace ends on before leaving
    lastStatesBeforeLeaving = toPlot[:, 0:alignIdx]
    lastState = np.empty((lastStatesBeforeLeaving.shape[0], 1))
    lastState[:] = np.nan
    for i in range(0, len(lastState)):
        nonNaNidx = ~np.isnan(lastStatesBeforeLeaving[i])
        if nonNaNidx.any():
            lastState[i] = lastStatesBeforeLeaving[i][nonNaNidx][-1]
    lastStateIdx = np.concatenate([np.where(lastState == i)[0] for i in range(0, numstates)])
    toPlot = toPlot[lastStateIdx, :]

    dfToPlot = pd.DataFrame(toPlot)
    sns.heatmap(dfToPlot, mask=toPlot.mask, cmap=discreteCmap, cbar_kws={"ticks": []}, ax=ax[0], rasterized=True)
    xticksToPlot = np.hstack((binTimeLine[0:-1:binsPerMin], binTimeLine[-1]))
    xtickLabels = np.ceil(xticksToPlot / binsPerMin).astype(int)
    yticksToPlot = np.linspace(0, toPlot.shape[0] - 1, 5, dtype=np.int)
    ax[0].set_xticks(xticksToPlot + binTimeBefore)
    ax[0].set_xticklabels(xtickLabels)
    ax[0].set_yticks(yticksToPlot)
    ax[0].set_yticklabels(yticksToPlot)
    ax[0].set_xlabel("time aligned to event (min)")
    # ax[0].set_title(titleToPlot, fontsize=12)

    # then the fraction of animals in each state using same colors
    for i in range(0, numstates):
        if maskMeanBool:
            Mean = np.ma.masked_array(data=probStateAligned[i], mask=inlawnrunmask_aligned[i])
            stErr_lower = np.ma.masked_array(data=probStateAligned_prop_ci[0][i], mask=inlawnrunmask_aligned[i])
            stErr_upper = np.ma.masked_array(data=probStateAligned_prop_ci[1][i], mask=inlawnrunmask_aligned[i])
        else:
            Mean = probStateAligned[i]
            stErr_lower = probStateAligned_prop_ci[0][i]
            stErr_upper = probStateAligned_prop_ci[1][i]
        ax[1].plot(binTimeLine / binsPerMin, Mean, color=colors[i], label=statelabels[i])
        ax[1].fill_between(binTimeLine / binsPerMin, y1=stErr_lower, y2=stErr_upper, facecolor=colors[i], edgecolor=None, linewidth=0.0, alpha=0.4)

    # Shrink current axis by 20%
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax[1].set_title(titleToPlot, fontsize=14)
    ax[1].set_ylabel('p(State)')
    ax[1].set_xlabel('time aligned to event (min)')
    ax[1].vlines(0, 0, 1, linestyles='dashed',color="black")

    plt.tight_layout()

    return ax

def plot_MeanSingleState_preAligned_noHeatmap(ax,datalabel,stateIdx,probStateAligned,probStateAligned_prop_ci,inlawnrunmask_aligned,binTimeLine,maskMeanBool,color):
    i = stateIdx
    binsPerMin = 6
    # then the fraction of animals in each state using same colors
    if maskMeanBool:
        Mean = np.ma.masked_array(data=probStateAligned[i], mask=inlawnrunmask_aligned[i])
        stErr_lower = np.ma.masked_array(data=probStateAligned_prop_ci[0][i], mask=inlawnrunmask_aligned[i])
        stErr_upper = np.ma.masked_array(data=probStateAligned_prop_ci[1][i], mask=inlawnrunmask_aligned[i])
    else:
        Mean = probStateAligned[i]
        stErr_lower = probStateAligned_prop_ci[0][i]
        stErr_upper = probStateAligned_prop_ci[1][i]
    ax.plot(binTimeLine / binsPerMin, Mean, color=color, label=datalabel)
    ax.fill_between(binTimeLine / binsPerMin, y1=stErr_lower, y2=stErr_upper, facecolor=color, edgecolor=None, linewidth=0.0, alpha=0.4)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('p(State)')
    ax.set_xlabel('time aligned to event (min)')
    ax.vlines(0, 0, 2,linestyles='dashed', color="black")

    plt.tight_layout()
    return ax

def plotAligned_Heatmap(ax, data, framesPerMin,binTimeLine, binTimeBefore, vmin, vmax, titleToPlot,sortFlag,symmetricFlag,alignIdx,numLastBins):
    from scipy.stats.mstats import sem
    #reorder entries by ascending first derivative in the last 5 bins before lawn leaving (if they are present)
    if sortFlag:
        data, _ = sortByFirstDeriv(data, alignIdx, numLastBins)

    dfToPlot = pd.DataFrame(data)

    if symmetricFlag:
        offset = mpl.colors.TwoSlopeNorm(vcenter=0., vmin=vmin, vmax=vmax)
        sns.heatmap(dfToPlot, mask=data.mask, cmap="coolwarm", norm=offset, ax=ax, rasterized=True)
    else:
        sns.heatmap(dfToPlot, mask=data.mask, cmap="rocket", vmin=vmin, vmax=vmax, ax=ax, rasterized=True)

    xticksToPlot = np.hstack((binTimeLine[0:-1:framesPerMin], binTimeLine[-1]))
    xtickLabels = np.ceil(xticksToPlot / framesPerMin).astype(int)
    yticksToPlot = np.linspace(0, data.shape[0] - 1, 5, dtype=np.int)
    ax.set_xticks(xticksToPlot + binTimeBefore)
    ax.set_xticklabels(xtickLabels)
    ax.set_yticks(yticksToPlot)
    ax.set_yticklabels(yticksToPlot)
    ax.set_xlabel("time aligned to event (min)")
    ax.set_title(titleToPlot)

    return ax
#plots a particular quantity that has been pre-aligned to another event - assumes data does not come from a masked array
def plotAligned(ax, toPlot,binTimeLine,binTimeBefore,vmin, vmax, ylabelToPlot,titleToPlot,colors,sortFlag,symmetricFlag,alignIdx,numLastBins):
    #toPlot is a normal numpy ndarray
    from scipy.stats import sem
    from matplotlib import colors as MPLcolors

    #reorder entries by ascending first derivative in the last 5 bins before lawn leaving (if they are present)
    if sortFlag:
        toPlot, _ = sortByFirstDeriv(toPlot, alignIdx, numLastBins)

    dfToPlot = pd.DataFrame(toPlot)
    if symmetricFlag:
        offset = MPLcolors.TwoSlopeNorm(vcenter=0., vmin=np.nanmin(toPlot), vmax=np.nanmax(toPlot))
        sns.heatmap(dfToPlot, cmap="coolwarm", norm=offset, ax=ax[0], rasterized=True)
    else:
        sns.heatmap(dfToPlot, cmap="rocket", ax=ax[0], vmin=vmin, vmax=vmax, rasterized=True)

    xticksToPlot = np.hstack((binTimeLine[0:-1:6], binTimeLine[-1]))
    xtickLabels = np.round(xticksToPlot / 6, 1)
    yticksToPlot = np.linspace(0, toPlot.shape[0] - 1, 5, dtype=np.int)
    ax[0].set_xticks(xticksToPlot + binTimeBefore)
    ax[0].set_xticklabels(xtickLabels)
    ax[0].set_yticks(yticksToPlot)
    ax[0].set_yticklabels(yticksToPlot)
    ax[0].set_xlabel("time aligned to event (min)")
    ax[0].set_title(titleToPlot)

    mean = np.nanmean(toPlot, axis=0)
    stderr = sem(toPlot, nan_policy='omit',axis=0)

    ax[1].plot(binTimeLine / 6, mean, color='k', lw=1)
    ax[1].fill_between(binTimeLine / 6, y1=mean - stderr, y2=mean + stderr, facecolor=colors[0], edgecolor=None, linewidth=0.0, alpha=0.5)
    # ax[1].vlines(0, np.nanmin(mean) - .01, np.nanmax(mean) + .01, linestyle="--", color="red")
    ax[1].vlines(0, 0, 1, linestyle="--", color="black")
    # ax[1].set_ylim(-0.1,1.1)
    ax[1].set_ylabel(ylabelToPlot)
    ax[1].set_xlabel("time aligned to event (min)")

    return ax

#plots a particular quantity that has been pre-aligned to another event - assumes data comes from a masked array
def plotAligned_masked(ax, data, binTimeLine, binTimeBefore, vmin, vmax, framesPerMin, ylabelToPlot, titleToPlot,colors,sortFlag,symmetricFlag,alignIdx,numLastBins,interpFlag,missingFracThresh):
    from scipy.stats.mstats import sem
    #reorder entries by ascending first derivative in the last 5 bins before lawn leaving (if they are present)
    if sortFlag=="firstDeriv":
        #first sort by first derivative
        data, _ = sortByFirstDeriv(data, alignIdx, numLastBins)
    elif sortFlag=="length":
        #sort first on the length of the prealigned data
        sortIdx = np.argsort(np.sum(~data.mask, axis=1))
        data = data[sortIdx]
    #otherwise we won't sort

    dfToPlot = pd.DataFrame(data)

    if symmetricFlag:
        offset = mpl.colors.TwoSlopeNorm(vcenter=0., vmin=vmin, vmax=vmax)
        sns.heatmap(dfToPlot, mask=data.mask, cmap="coolwarm", norm=offset, ax=ax[0], rasterized=True)
    else:
        sns.heatmap(dfToPlot, mask=data.mask, cmap="rocket", vmin=vmin, vmax=vmax, ax=ax[0], rasterized=True)
        # sns.heatmap(dfToPlot, mask=data.mask, cmap="viridis", vmin=vmin, vmax=vmax, ax=ax[0])

    xticksToPlot = np.hstack((binTimeLine[0:-1:framesPerMin], binTimeLine[-1]))
    xtickLabels = np.ceil(xticksToPlot / framesPerMin).astype(int)
    yticksToPlot = np.linspace(0, data.shape[0] - 1, 5, dtype=np.int)
    ax[0].set_xticks(xticksToPlot + binTimeBefore)
    ax[0].set_xticklabels(xtickLabels)
    ax[0].set_yticks(yticksToPlot)
    ax[0].set_yticklabels(yticksToPlot)
    ax[0].set_xlabel("time aligned to event (min)")
    ax[0].set_title(titleToPlot)

    if interpFlag:
        missingdatafilled = np.vstack(
            np.array([interp1darray(data[i].filled(np.nanmean(data[i]))) for i in range(data.shape[0])]))
        tmp = (np.mean(data.mask, axis=0) > missingFracThresh).reshape(1,-1)  # if more than frac is missing, don't show this data
        commonMask = np.repeat(tmp, data.mask.shape[0], axis=0)
        maskedData = np.ma.masked_array(data=missingdatafilled, mask=commonMask)
    else:
        maskedData = data
        commonMask = data.mask

    mean_1 = np.nanmean(maskedData, 0)
    stderr_1 = sem(maskedData, axis=0)

    timeLine2 = binTimeLine / framesPerMin

    ax[1].plot(timeLine2, mean_1, color=colors[0], lw=1.5)
    ax[1].fill_between(binTimeLine / framesPerMin, y1=mean_1 - stderr_1, y2=mean_1 + stderr_1, facecolor=colors[0], edgecolor=None, linewidth=0.0, alpha=0.5)

      # ax[1].vlines(0, -200, 200, linestyle="--", color="black")
    xticksToPlot2 = np.round(np.linspace(timeLine2[0],timeLine2[-1],len(xticksToPlot))).astype(int)
    ax[1].set_xticks(xticksToPlot2)
    ax[1].set_xticklabels(xtickLabels)
    ax[1].set_ylabel(ylabelToPlot)
    ax[1].set_xlabel("time aligned to event (min)")

    return ax

#aligns and plots MLstates
def plotMLstates_Aligned_heatmap(ax,MLstates,ChosenModel,binTimeLine,binTimeBefore,binsPerMin,titleToPlot,colors):
    # first each trace in a heatmap style using native colors
    discreteCmap = lcmap(colors[0:ChosenModel.K])
    toPlot = MLstates
    dfToPlot = pd.DataFrame(toPlot)
    sns.heatmap(dfToPlot, mask=toPlot.mask, cmap=discreteCmap, cbar_kws={"ticks": []}, ax=ax, cbar=False, rasterized=True)
    xticksToPlot = np.hstack((binTimeLine[0:-1:binsPerMin], binTimeLine[-1]))
    xtickLabels = np.ceil(xticksToPlot / binsPerMin).astype(int)
    yticksToPlot = np.linspace(0, toPlot.shape[0] - 1, 5, dtype=np.int)
    ax.set_xticks(xticksToPlot + binTimeBefore)
    ax.set_xticklabels(xtickLabels)
    ax.set_yticks(yticksToPlot)
    ax.set_yticklabels(yticksToPlot)
    ax.set_xlabel("time aligned to event (min)")
    ax.set_title(titleToPlot)

    return ax

def plotMLstates_Aligned_masked(ax,MLstates,ChosenModel,inlawnrunmask,EventToAlign,binTimeLine,binTimeBefore,binTimeAfter,binsPerMin,titleToPlot,colors,numLastBins,maskMeanBool,figsize):
    from matplotlib import gridspec
    statelabels = ['state ' + str(i) for i in range(0, ChosenModel.K)]

    StateByAligned, MaskByAligned, probStateAligned, timeLine, alignIdx, probStateAligned_prop_ci = \
        genProbStateAligned_masked(MLstates, inlawnrunmask, EventToAlign, binTimeBefore, binTimeAfter, ChosenModel.K)

    #first each trace in a heatmap style using native colors
    discreteCmap = lcmap(colors[0:ChosenModel.K])
    toPlot, _ = sortByFirstDeriv(StateByAligned, alignIdx, numLastBins)
    #rearrange this by which state the trace ends on before leaving
    lastStatesBeforeLeaving = toPlot[:, 0:alignIdx]
    lastState = np.empty((lastStatesBeforeLeaving.shape[0], 1))
    lastState[:] = np.nan
    for i in range(0, len(lastState)):
        nonNaNidx = ~np.isnan(lastStatesBeforeLeaving[i])
        if nonNaNidx.any():
            lastState[i] = lastStatesBeforeLeaving[i][nonNaNidx][-1]
    lastStateIdx = np.concatenate([np.where(lastState==i)[0] for i in range(0,ChosenModel.K)])
    toPlot = toPlot[lastStateIdx,:]

    dfToPlot = pd.DataFrame(toPlot)
    sns.heatmap(dfToPlot, mask=toPlot.mask, cmap=discreteCmap, cbar_kws={"ticks": []}, ax=ax[0], cbar=False, rasterized=True)
    xticksToPlot = np.hstack((binTimeLine[0:-1:binsPerMin], binTimeLine[-1]))
    xtickLabels = np.ceil(xticksToPlot / binsPerMin).astype(int)
    yticksToPlot = np.linspace(0, toPlot.shape[0] - 1, 5, dtype=np.int)
    ax[0].set_xticks(xticksToPlot + binTimeBefore)
    ax[0].set_xticklabels(xtickLabels)
    ax[0].set_yticks(yticksToPlot)
    ax[0].set_yticklabels(yticksToPlot)
    ax[0].set_xlabel("time aligned to event (min)")
    ax[0].set_title(titleToPlot)

    # then the fraction of animals in each state using same colors
    for i in range(0, ChosenModel.K):
        if maskMeanBool:
            Mean = np.ma.masked_array(data=probStateAligned[i],mask=MaskByAligned[i])
            stErr_lower = np.ma.masked_array(data=probStateAligned_prop_ci[0][i],mask=MaskByAligned[i])
            stErr_upper = np.ma.masked_array(data=probStateAligned_prop_ci[1][i],mask=MaskByAligned[i])
        else:
            Mean = probStateAligned[i]
            stErr_lower = probStateAligned_prop_ci[0][i]
            stErr_upper = probStateAligned_prop_ci[1][i]
        ax[1].plot(timeLine / binsPerMin, Mean, color=colors[i], label=statelabels[i])
        ax[1].fill_between(timeLine / binsPerMin, y1=stErr_lower, y2=stErr_upper, facecolor=colors[i], edgecolor=None, linewidth=0.0, alpha=0.4)
    # Shrink current axis by 20%
    # box = ax[1].get_position()
    # ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    # ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax[1].set_title(titleToPlot, fontsize=14)
    ax[1].set_ylabel('fraction of animals in state')
    ax[1].set_xlabel('time aligned to event (min)')
    # ax[1].vlines(0, 0, 1, linestyles='dashed', color='black')
    ax[1].set_xticks(np.arange(-int(np.round((binTimeBefore/binsPerMin))),int(np.round((binTimeAfter/binsPerMin)+1))))
    ax[1].set_xticklabels(xtickLabels)
    ax[1].set_ylim([0, 1])
    # ax[2].set_ylim([0, 1])
    # ax[2].set_yticklabels([])

    return ax, dfToPlot

#Special code for sorting MLstates by the duration of the last state (of choice) before leaving
def plotMLstateHeatmap_durSort(ax, MLstates, sortState, InLawnRunMask, EventToAlign, binsPerMin, binTimeBefore, binTimeAfter, ChosenModel,colors):

    StateByAligned, MaskByAligned, probStateAligned, timeLine, alignIdx, probStateAligned_prop_ci = \
            genProbStateAligned_masked(MLstates,
                                       InLawnRunMask,
                                       EventToAlign,
                                       binTimeBefore,
                                       binTimeAfter,
                                       ChosenModel.K)

    #order StateByAligned by the duration of the final stateing state
    lastStateDur = np.zeros((StateByAligned.shape[0],1))
    for i in range(StateByAligned.shape[0]):
        stateInts = get_intervals(StateByAligned[i]==sortState,0)
        chosenRunIdx_before = np.unique(np.where(np.isin(stateInts, np.arange(alignIdx-1,alignIdx+1)))[0]).tolist()
        if len(chosenRunIdx_before)>0:
            lastState = stateInts[chosenRunIdx_before[0]]
            lastStateDur[i] = lastState[1]-lastState[0]
    sortIdx = np.argsort(lastStateDur.ravel())

    ax = plotMLstates_Aligned_heatmap(ax,
                                 StateByAligned[sortIdx],
                                 ChosenModel,
                                 timeLine,
                                 binTimeBefore,
                                 binsPerMin,
                                 '',
                                 colors)

    return ax, StateByAligned[sortIdx]

#aligns and plots other kinds of data
def AlignAndPlot(ax, Data,alignIdx,mask,binTimeBefore,binTimeAfter,ymin,ymax,ylabel,figTitle,colors,interpFlag,symmetricFlag,missingFracThresh): #here ax is actually two side by side axes
    Data_aligned, Mask_aligned, binTimeLine, binAlignIdx = alignData_masked(
        Data, mask, alignIdx, binTimeBefore, binTimeAfter, dtype=float)

    # ax = plotAligned_masked(ax, Data_aligned, binTimeLine, binTimeBefore, ymin, ymax, 6, ylabel, figTitle, colors, True,
    #                         symmetricFlag, binAlignIdx, binTimeBefore)
    ax = plotAligned_masked(ax, Data_aligned, binTimeLine, binTimeBefore, ymin, ymax, 6, ylabel, figTitle,colors,interpFlag,symmetricFlag,binAlignIdx,binTimeBefore,interpFlag,missingFracThresh)

    ax[1].set_ylim([ymin, ymax])

    return ax, Data_aligned, Mask_aligned, binTimeLine, binAlignIdx

# align any variable to a state transition but censor any data on either side that doesnt come from the beforestate or afterstate, respectively.
def AlignAndPlot_stateTransitions(ax, Data, StateMatrix, alignIdx, stateBefore, stateAfter, mask,binTimeBefore,binTimeAfter,ymin,ymax,ylabel,figTitle,colors,symmetricFlag):

    Data_aligned, Mask_aligned, binTimeLine, binAlignIdx = alignData_masked(
        Data, mask, alignIdx, binTimeBefore, binTimeAfter, dtype=float)

    State_aligned, _, _, _ = alignData_masked(
        StateMatrix, mask, alignIdx, binTimeBefore, binTimeAfter, dtype=float)

    beforeMask = ~(State_aligned.filled(-9999)[:, 0:binAlignIdx] == stateBefore)
    afterMask = ~(State_aligned.filled(-9999)[:, binAlignIdx:] == stateAfter)

    Mask_aligned = np.logical_or(np.hstack([beforeMask, afterMask]), Mask_aligned)
    Data_aligned_censored = np.ma.MaskedArray(data=Data_aligned.data, mask=Mask_aligned)

    ax = plotAligned_masked(ax, Data_aligned_censored, binTimeLine, binTimeBefore, ymin, ymax, 6, ylabel, figTitle, colors, True,
                            symmetricFlag, binAlignIdx, binTimeBefore)

    ax[1].set_ylim([ymin, ymax])

    return ax, Data_aligned, Mask_aligned, binTimeLine, binAlignIdx

###### plotting data across multiple conditions
# this function plots time series means of two datasets on the same plot along with their stderr
def compareMeanCurves(ax,Data1,Data1_label,Data2,Data2_label,binTimeLine,binsPerMin,ymin,ymax,ylabel,figTitle,colors):
    timeLine2 = binTimeLine / binsPerMin
    xticksToPlot = np.hstack((binTimeLine[0:-1:binsPerMin], binTimeLine[-1]))
    xtickLabels = np.ceil(xticksToPlot / binsPerMin).astype(int)
    xticksToPlot2 = np.round(np.linspace(timeLine2[0], timeLine2[-1], len(xticksToPlot))).astype(int)
    #Data1
    # make sure stderr_1 and mean_1 line up
    mean_1 = np.nanmean(Data1, axis=0)
    stderr_1 = sem(Data1, axis=0)
    commonMask = np.logical_or(mean_1.mask, stderr_1.mask)
    mean_1 = np.ma.masked_array(data=mean_1, mask=commonMask)
    stderr_1 = np.ma.masked_array(data=stderr_1, mask=commonMask)
    ax.plot(timeLine2, mean_1, color=colors[0], lw=1.5,label=Data1_label)
    ax.fill_between(binTimeLine / binsPerMin, y1=mean_1 - stderr_1, y2=mean_1 + stderr_1, facecolor=colors[0], edgecolor=None, linewidth=0.0, alpha=0.5)
    ax.vlines(0, -200, 200, linestyle="--", color="black")
    #Data2
    # make sure stderr_2 and mean_2 line up
    mean_2 = np.nanmean(Data2, axis=0)
    stderr_2 = sem(Data2, axis=0)
    commonMask = np.logical_or(mean_2.mask, stderr_2.mask)
    mean_2 = np.ma.masked_array(data=mean_2, mask=commonMask)
    stderr_2 = np.ma.masked_array(data=stderr_2, mask=commonMask)
    ax.plot(timeLine2, mean_2, color=colors[1], lw=1.5,label=Data2_label)
    ax.fill_between(binTimeLine / binsPerMin, y1=mean_2 - stderr_2, y2=mean_2 + stderr_2, facecolor=colors[1], edgecolor=None, linewidth=0.0, alpha=0.5)
    ax.vlines(0, -200, 200, linestyle="--", color="black")

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=14)
    ax.set_ylim(ymin,ymax)
    ax.set_xticks(xticksToPlot2)
    ax.set_xticklabels(xtickLabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time aligned to event (min)")
    ax.set_title(figTitle)

    return ax

def plotMeanSEMCurve(ax,data,missingFracThresh,Data_label,binTimeLine,framesPerMin,color):
    from scipy.stats.mstats import sem

    missingdatafilled = np.vstack(
        np.array([interp1darray(data[i].filled(np.nanmean(data[i]))) for i in range(data.shape[0])]))
    tmp = (np.mean(data.mask, axis=0) > missingFracThresh).reshape(1, -1) #if more than frac is missing, don't show this data
    commonMask = np.repeat(tmp, data.mask.shape[0], axis=0)
    maskedData = np.ma.masked_array(data=missingdatafilled, mask=commonMask)

    mean_1 = np.nanmean(maskedData, 0)
    stderr_1 = sem(maskedData, axis=0)

    timeLine2 = binTimeLine / framesPerMin

    ax.plot(timeLine2, mean_1, color=color, lw=1.5, label=Data_label)
    ax.fill_between(binTimeLine / framesPerMin, y1=mean_1 - stderr_1, y2=mean_1 + stderr_1, facecolor=color, edgecolor=None, linewidth=0.0, alpha=0.5)

    return ax

def pairedScatterPlot(ax, Data_t0, Data_t1,leftColor,middleColor,rightColor,middleAlpha):
    ax.set_facecolor('w')
    ax.grid(b=None)
    xpos = np.hstack(
        [np.ones_like(Data_t0).reshape(-1, 1), 2 * np.ones_like(Data_t1).reshape(-1, 1)])
    change = np.hstack([Data_t0,Data_t1])
    for i in range(xpos.shape[0]):
        _ = ax.plot(xpos[i, :], change[i, :], color=middleColor,alpha=middleAlpha,zorder=0)
        _ = ax.scatter(xpos[i, 0], change[i, 0], s=5, color=leftColor, zorder=1)
        _ = ax.scatter(xpos[i, 1], change[i, 1], s=5, color=rightColor,zorder=2)
    mean_before = np.nanmean(change[:, 0])
    mean_leave = np.nanmean(change[:, 1])
    _ = ax.plot([1, 2], [mean_before, mean_leave], lw=5, color='k',zorder=3)

    return ax

def compareStateDurations(axs,ChosenModel,MLstates,RDstates,colors):
    k = ChosenModel.K

    stateDur = computeStateDurations(fillNAwithInt(MLstates, 25).astype(int))  # fill missing data with a much larger int
    infDurations_RD_ALL = computeStateDurations(RDstates.filled(25).astype(int)) #RDstates is masked array

    statelabels = ["state " + str(i) for i in range(0, ChosenModel.K)]
    RDlabels = ['dwell', 'roam']

    durBins = np.insert(np.linspace(0, 20 * 6, 50), 0, -1 * np.inf)
    ticklabels = np.round((10 / 60 * np.linspace(0, durBins[-1], 5))).astype(str)
    [axs.hist(stateDur[s], bins=durBins, color=colors[s], density=True, histtype='step', cumulative=-1, lw=2,
              label=statelabels[s]) for s in range(0, ChosenModel.K)]
    [axs.hist(infDurations_RD_ALL[s], bins=durBins, color=colors[s], density=True, linestyle='dashed', alpha=0.5,
              histtype='step', cumulative=-1, lw=2, label=RDlabels[s]) for s in range(0, 2)]

    axs.set_xticks(ticks=np.linspace(0, durBins[-1], 5))
    axs.set_xticklabels(ticklabels)
    axs.set_xlabel('Duration (min)')
    axs.set_ylabel('P(state duration > x)')
    # axs.legend(bbox_to_anchor=(1, 1), loc="upper left")
    axs.legend(loc="upper right")

    if hasattr(ChosenModel.transitions, 'kappa'):
        kappa = str(int(ChosenModel.transitions.kappa))
        axs.set_title('state durations, k = ' + str(k) + ', kappa = ' + kappa)
    else:
        axs.set_title('state durations, k = ' + str(k))

    return axs


# make a function that does the violinplot and boxplot
def customViolinBoxPlot(df, colors, axs):
    """
    works with a dataframe where each column has the data for the violinplots
    """
    # violinplot, outline only
    g = sns.violinplot(ax=axs, data=df, showfliers=False, palette=colors, dodge=True, linewidth=1, saturation=1,
                       inner=None, cut=0, scale='width', zorder=0)
    for violin in g.collections:
        violin.set_edgecolor(violin.get_facecolor())
        violin.set_facecolor('None')

    # DIY custom boxplot
    # make the medians pop, plot the quartiles a bit narrower
    m_width = 0.5
    q_width = 0.2
    medians = df.median()
    lower_q = df.quantile(0.25)
    upper_q = df.quantile(0.75)

    if len(medians) == 1:
        medians = [medians]
        lower_q = [lower_q]
        upper_q = [upper_q]

    for i, xtick in enumerate(g.get_xticks()):
        g.plot([xtick - q_width / 2, xtick + q_width / 2], [lower_q[i], lower_q[i]], color=colors[i], linestyle='-',
               linewidth=1, zorder=3)
        g.plot([xtick - q_width / 2, xtick + q_width / 2], [upper_q[i], upper_q[i]], color=colors[i], linestyle='-',
               linewidth=1, zorder=3)
        g.plot([xtick - m_width / 2, xtick + m_width / 2], [medians[i], medians[i]], color=colors[i], linestyle='-',
               linewidth=1, zorder=3)
        g.plot([xtick, xtick], [lower_q[i], upper_q[i]], color=colors[i], linestyle='-', linewidth=1, zorder=3)

    return g
