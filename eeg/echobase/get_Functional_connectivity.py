"""
2020.05.06
Andy Revell and Alex Silva

Purpose:
    Calculate functional correlations between given time series data and channels.

Logic of code:
    1. Calculate correlations within a given time window: Window data in 1 second
    2. Calculate broadband functional connectivity with echobase broadband_conn
    3. Calculate other band functional connectivity with echobase multiband_conn

Input:
    inputfile: a pickled list. See get_iEEG_data.py and https://docs.python.org/3/library/pickle.html for more information
        index 0: time series data N x M : row x column : time x channels
        index 1: fs, sampling frequency of time series data


Output:
    outputfile: saves output as a numpy array npz. Array is organized by names of bands:
    broadband_CC, alphatheta, beta, lowgamma, highgamma

Example:

inputfile = '/Users/andyrevell/Box/01_papers/sub-RID0278_HUP138_phaseII_415723190000_6000000.pickle'
outputfile = '/Users/andyrevell/Box/01_papers/sub-RID0278_HUP138_phaseII_415723190000_6000000.npz'
getFuncConn(inputfile,outputfile)

Please use this naming convention if data is from iEEG.org
sub-RIDXXXX_iEEGFILENAME_STARTTIME_DURATION
example: 'sub-RID0278_HUP138_phaseII_415723190000_6000000'

"""
from echobase import broadband_conn, multiband_conn
import numpy as np
import pickle
""""
Note on Pycharm users to import echobase:
Right click the folder in which get_Functional_connectivity.py and echobase.py (MUST BE IN SAME DIRECTORY)
Select "Mark Directory As" --> "Mark as Source Root" 
Can also go to preferences --> Project structure to change source root 
"""

def getFuncConn(inputfile,outputfile):

    with open(inputfile, 'rb') as f: data, fs = pickle.load(f)#un-pickles files
    data_array = np.array(data)
    fs = float(fs)
    totalSecs = np.floor(np.size(data_array,0)/fs)
    totalSecs = int(totalSecs)
    alphatheta = np.zeros((np.size(data_array,1),np.size(data_array,1),totalSecs))
    beta = np.zeros((np.size(data_array,1),np.size(data_array,1),totalSecs))
    broadband_CC = np.zeros((np.size(data_array,1),np.size(data_array,1),totalSecs))
    highgamma = np.zeros((np.size(data_array,1),np.size(data_array,1),totalSecs))
    lowgamma = np.zeros((np.size(data_array,1),np.size(data_array,1),totalSecs))
    for t in range(0,totalSecs):
        startInd = int(t*fs)
        endInd = int(((t+1)*fs) - 1)
        window = data_array[startInd:endInd,:]
        broad = broadband_conn(window,int(fs),avgref=True)
        adj_alphatheta, adj_beta, adj_lowgamma, adj_highgamma = multiband_conn(window,int(fs),avgref=True)
        alphatheta[:,:,t] = adj_alphatheta
        beta[:,:,t] = adj_beta
        broadband_CC[:,:,t] = broad
        highgamma[:,:,t] = adj_highgamma
        lowgamma[:,:,t] = adj_lowgamma
        print(t)

    np.savez(outputfile, broadband_CC=broadband_CC, alphatheta=alphatheta, beta=beta, lowgamma=lowgamma, highgamma=highgamma)



