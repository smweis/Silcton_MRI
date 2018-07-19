# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:54:03 2018

@author: stweis
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from statsmodels.formula.api import ols
from scipy.spatial.distance import mahalanobis



def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    fig = plt.figure(figsize=(size,size))
    ax1 = fig.add_subplot(111)
    cmap = plt.cm.get_cmap('jet')
    a = df.corr()
    np.fill_diagonal(a.values,np.nan)
    masked_array = np.ma.array (a, mask=np.isnan(a))
    cmap.set_bad('black',1.)
    ax1.imshow(masked_array, interpolation='nearest', cmap=cmap)
    cax = ax1.imshow(a, interpolation="nearest", cmap=cmap)
    plt.xticks(range(len(df.columns)), df.columns);
    plt.yticks(range(len(df.columns)), df.columns);
    fig.colorbar(cax)
    return fig


def zscore(column, data):
    '''Calculate and return the z-score value of a variable from a DataFrame'''
    zscoredVariable = (data[column] - data[column].mean())/data[column].std(ddof=0)
    return zscoredVariable

def doAnova(dependentVar,data):
    '''Calculate and return oneway ANOVA of a variable from 3 groups'''
    data = data[np.isfinite(data[dependentVar])]
    grps = pd.unique(data.groups.values)
    d_data = {grp:data[dependentVar][data.groups == grp] for grp in grps}
    F, p = scipy.stats.f_oneway(d_data[1], d_data[2], d_data[3])
    print('F = ',F,'p = ',p)
    
def stderror(dependentVar):
    std = np.std(dependentVar)
    n = len(dependentVar)
    sem = std / np.sqrt(n)
    
    return sem

def saveTheFig(name1,name2,f):
    '''save a figure as a pdf with 2 axes as names'''
    figName = 'scatter'+name1.capitalize()+name2.capitalize()+'.pdf'
    prompt = 'save figure with ' + name1.capitalize() + name2.capitalize() + '? y/n '
    a = input(prompt)

    if a.lower() == 'y':
        f.savefig(figName, format='pdf',transparent=True,)


def formatPlot(x,y,plot,ax1):
    ax1.set_xlabel(x.name,fontsize=20)
    ax1.set_ylabel(y.name,fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    fit = np.polyfit(x, y, deg=1) # calculate regression line
    # print lines as black
    ax1.plot(x, fit[0] * x + fit[1], color='gray',label='',linewidth=1) 
    pearsonR = 'r = ' + "{0:.2f}".format(fit[0])
    locationOffset = (max(y) - min(y)) / 20
    location = (max(x)*fit[0] + fit[1]) + locationOffset
    
    ax1.text(max(x),location,pearsonR,horizontalalignment='right',fontsize=14)
    ax1.legend(edgecolor='black',fontsize=14)
    fig = plt.gcf()
    fig.set_size_inches(10.5, 10.5)
    plt.show()


def mahalanobisR(X,meanCol,IC):
    m = []
    for i in range(X.shape[0]):
        m.append(mahalanobis(X.iloc[i,:],meanCol,IC) ** 2)
    return(m)
    
def scatter_with_trendline_groups(x,y,groups,data):
    f, ax1 = plt.subplots(1)
    group_dict = {1:['#00B050','Integrators'],2:['#E46C0A','Non-Integrators'],3:['#4BACC6','Imprecise Navigators']}
    for kind in group_dict:
        d = data[data.groups==kind]
        plt.scatter(d[x], d[y], c = group_dict[kind][0],label=group_dict[kind][1],s=60)
        plt.hlines(np.mean(d[y]), np.mean(d[x]) - stderror(d[x]), 
                   np.mean(d[x]) + stderror(d[x]), linestyle='dashed')
        plt.vlines(np.mean(d[x]), np.mean(d[y]) - stderror(d[y]), 
                   np.mean(d[y]) + stderror(d[y]), linestyle='dashed')
        plt.plot(np.mean(d[x]), np.mean(d[y]), marker='o', markersize=20,
                 color=group_dict[kind][0], markeredgecolor='black')
        
    return f, ax1


def make_scatter(x,y,groups,data):
    f, ax1 = scatter_with_trendline_groups(x,y,groups,data)
    formatPlot(data[x], data[y],f,ax1)
    
# <codecell>

#dirname = os.path.dirname(__file__) #expects data in the same folder as script
filename = 'C:\\Users\\stweis\\Dropbox\\Penn Post Doc\\Silcton_FMRI\\DataAnalysisWith70Participants_Jupyter.xlsx'

data = pd.read_excel(filename)

#variables
normedData = pd.DataFrame()
normedData['brainVol']= zscore('BrainSegVol',data)
normedData['cortexVol']= zscore('CortexVol',data)
normedData['lhc']= zscore('Left_Hippocampus',data)
normedData['rhc'] = zscore('Right_Hippocampus',data)
normedData['lamyg'] = zscore('Left_Amygdala',data)
normedData['ramyg'] = zscore('Right_Amygdala',data)
normedData['csf'] = zscore('Cerebrospinal_Fluid',data)
normedData['pointBetween'] = zscore('Pointing_Between',data)
normedData['pointWithin'] = zscore('Pointing_Within',data)
normedData['pointTotal'] = zscore('Pointing_Total',data)
normedData['rcaud'] = zscore('Right_Caudate',data)
normedData['lcaud'] = zscore('Left_Caudate',data)
normedData['sbsod'] = zscore('SBSOD',data)
normedData['wrat'] = zscore('WRAT',data)
normedData['mrt'] = zscore('MRT',data)
normedData['gender'] = data['Gender']
normedData['groups'] = data['int1non2imp3']
normedData['rhc_head'] = zscore('Right_Head',data)
normedData['lhc_head'] = zscore('Left_Head',data)
normedData['rhc_bodytail'] = zscore('Right_BodyTail',data)
normedData['lhc_bodytail'] = zscore('Left_BodyTail',data)



#reverse the pointing ones to be higher = good score
normedData['pointBetween'] = -1*(normedData['pointBetween'])
normedData['pointWithin'] = -1*(normedData['pointWithin'])
normedData['pointTotal'] = -1*(normedData['pointTotal'])

normedData['totalhc'] = normedData['rhc'] + normedData['lhc']

# <codecell>

x = normedData.loc[:,['rhc','pointTotal','rcaud','sbsod']]
mean = x.mean().values
Sx = x.cov().values

mR = mahalanobisR(x,mean,Sx)

normedData['outliers'] = mR;

msk = normedData['outliers'] < 13

normedData = normedData.loc[msk,:]
# <codecell>


make_scatter('rhc_head','wrat',normedData.groups,normedData)
#saveTheFig(x.name,y.name,f)

# <codecell>
doAnova('rhc_bodytail',normedData)
# <codecell>

nonannormedData = normedData[~np.isnan(normedData).any(axis=1)]

print(np.corrcoef(nonannormedData.wrat,nonannormedData.rhc))
# <codecell>

model = ols("pointTotal ~ rhc_head + wrat", normedData).fit()
print(model.summary())

print("\nRetrieving manually the parameter estimates:")
print(model._results.params)
# should be array([-4.99754526,  3.00250049, -0.50514907])

# <codecell>
order = [0,2,3,10,11,9,12,13,14]
cols = normedData.columns.tolist()

mylist = [ cols[i] for i in order]

orderedNormedData = normedData[mylist]
corrMat = plot_corr(orderedNormedData,20)
plt.show()

#saveTheFig('brain','behavior',corrMat)


