"""
2020.05.06
Ankit Khambhati, updated by Andy Revell

Purpose:
Function pipelines for filtering time-varying data

Logic of code:
    1. Common average reference (common_avg_ref)
    2. Fit an AR(1) model to the data and retains the residual as the pre-whitened data (ar_one)
    3. bandpass, lowpass, highpass filtering (Notch at 60Hz, HPF at 5Hz, LPF at 115Hz, XCorr at 0.25) (elliptic)
    4. Calculate cross-correlation similarity function for functional connectivity (xcorr_mag)
    5. Calculate a band-specific functional network, coherence. (multitaper)

Table of Contents:
A. Main
    1. broadband_conn
    2. multiband_conn
B. Supporting Code:
    3. common_avg_ref
    4. ar_one
    5. elliptic
    6. xcorr_mag
    7. xcorr
C. Utilities
    8. check_path
    9. make_path
    10. check_path_overwrite
    11. check_has_key
    12. check_dims
    13. check_type
    14. check_function

See individual function comments for inputs and outputs

Change Log
----------
2016/12/11 - Implemented broadband_conn
2020/05/06 - updated code to work with python 3.7, numpy 1.18.4, scipy 1.4.1, mtspec 0.3.2. Added header to python code above
    Added note how to install mtsepc below.
"""

from __future__ import division
import numpy as np
import os
import inspect
import scipy.signal as spsig
from mtspec import mt_coherence, mtspec
""""
Note 2020.05.06
To install mtspec:
See https://krischer.github.io/mtspec/ for more documentation
1. Need to have gfortran installed on computer
2. It is different for Linux and Mac

Linux:
#apt-get install gfortran
#pip install mtspec

Mac OS:
Need homebrew, then do:
#brew install gcc
#brew cask install gfortran
#pip install mtspec
"""

"""
A. Main
"""

def broadband_conn(data, fs, avgref=True):
    """
    Pipeline function for computing a broadband functional network from ECoG.

    See: Khambhati, A. N. et al. (2015).
    Dynamic Network Drivers of Seizure Generation, Propagation and Termination in
    Human Neocortical Epilepsy. PLOS Computational Biology, 11(12).

    Data --> CAR Filter --> Notch Filter --> Band-pass Filter --> Cross-Correlation

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        reref: True/False
            Re-reference data to the common average (default: True)

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
    """

    # Standard param checks
    check_type(data,np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)

    # Parameter set
    param = {}
    param['Notch_60Hz'] = {'wpass': [58.0, 62.0],
                           'wstop': [59.0, 61.0],
                           'gpass': 0.1,
                           'gstop': 60.0}
    param['HPF_5Hz'] = {'wpass': [5.0],
                        'wstop': [4.0],
                        'gpass': 0.1,
                        'gstop': 60.0}
    param['LPF_115Hz'] = {'wpass': [115.0],
                          'wstop': [120.0],
                          'gpass': 0.1,
                          'gstop': 60.0}
    param['XCorr'] = {'tau': 0.25}

    # Build pipeline
    if avgref:
        data_hat = common_avg_ref(data)
    else:
        data_hat = data.copy()
    data_hat = ar_one(data_hat)
    data_hat = elliptic(data_hat, fs, **param['Notch_60Hz'])
    data_hat = elliptic(data_hat, fs, **param['HPF_5Hz'])
    data_hat = elliptic(data_hat, fs, **param['LPF_115Hz'])
    adj = xcorr_mag(data_hat, fs, **param['XCorr'])

    return adj


def multiband_conn(data, fs, avgref=True):
    """
    Pipeline function for computing a band-specific functional network from ECoG.

    See: Khambhati, A. N. et al. (2016).
    Virtual Cortical Resection Reveals Push-Pull Network Control
    Preceding Seizure Evolution. Neuron, 91(5).

    Data --> CAR Filter --> Multi-taper Coherence

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        reref: True/False
            Re-reference data to the common average (default: True)

    Returns
    -------
        adj_alphatheta: ndarray, shape (N, N)
            Adjacency matrix for N variates (Alpha/Theta Band 5-15 Hz)

        adj_beta: ndarray, shape (N, N)
            Adjacency matrix for N variates (Beta Band 15-25 Hz)

        adj_lowgamma: ndarray, shape (N, N)
            Adjacency matrix for N variates (Low Gamma Band 30-40 Hz)

        adj_highgamma: ndarray, shape (N, N)
            Adjacency matrix for N variates (High Gamma Band 95-105 Hz)
    """

    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)

    # Parameter set
    param = {}
    param['time_band'] = 5.
    param['n_taper'] = 9
    param['AlphaTheta_Band'] = [5., 15.]
    param['Beta_Band'] = [15., 25.]
    param['LowGamma_Band'] = [30., 40.]
    param['HighGamma_Band'] = [95., 105.]

    # Build pipeline
    if avgref:
        data_hat = common_avg_ref(data)
    else:
        data_hat = data.copy()
    adj_alphatheta = multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['AlphaTheta_Band'])
    adj_beta = multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['Beta_Band'])
    adj_lowgamma = multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['LowGamma_Band'])
    adj_highgamma = multitaper(data_hat, fs, param['time_band'], param['n_taper'], param['HighGamma_Band'])

    return adj_alphatheta, adj_beta, adj_lowgamma, adj_highgamma


"""
B. Supporting functions
"""

def common_avg_ref(data):
    """
    The common_avg_ref function subtracts the common mode signal from the original
    signal. Suggested for removing correlated noise, broadly over a sensor array.

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

    Returns
    -------
        data_reref: ndarray, shape (T, N)
            Referenced signal with common mode removed
    """
    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    # Remove common mode signal
    data_reref = (data.T - data.mean(axis=1)).T
    return data_reref


def ar_one(data):
    """
    The ar_one function fits an AR(1) model to the data and retains the residual as
    the pre-whitened data

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

    Returns
    -------
        data_white: ndarray, shape (T, N)
            Whitened signal with reduced autocorrelative structure
    """
    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    # Retrieve data attributes
    n_samp, n_chan = data.shape
    # Apply AR(1)
    data_white = np.zeros((n_samp-1, n_chan))
    for i in range(n_chan):
        win_x = np.vstack((data[:-1, i], np.ones(n_samp-1)))
        w = np.linalg.lstsq(win_x.T, data[1:, i], rcond=None)[0]
        data_white[:, i] = data[1:, i] - (data[:-1, i]*w[0] + w[1])
    return data_white


def elliptic(data, fs, wpass, wstop, gpass, gstop):
    """
    The elliptic function implements bandpass, lowpass, highpass filtering

    This implements zero-phase filtering to pre-process and analyze
    frequency-dependent network structure. Implements Elliptic IIR filter.

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        wpass: tuple, shape: (1,) or (1,1)
            Pass band cutoff frequency (Hz)

        wstop: tuple, shape: (1,) or (1,1)
            Stop band cutoff frequency (Hz)

        gpass: float
            Pass band maximum loss (dB)

        gstop: float
            Stop band minimum attenuation (dB)

    Returns
    -------
        data_filt: ndarray, shape (T, N)
            Filtered signal with T samples over N variates
    """

    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)
    check_type(wpass, list)
    check_type(wstop, list)
    check_type(gpass, float)
    check_type(gstop, float)
    if not len(wpass) == len(wstop):
        raise Exception('Frequency criteria mismatch for wpass and wstop')
    if not (len(wpass) < 3):
        raise Exception('Must only be 1 or 2 frequency cutoffs in wpass and wstop')

    # Design filter
    nyq = fs / 2.0

    # new code. Works with scipy 1.4 (2020.05.06)
    wpass_nyq = [iter*0 for iter in range(len(wpass))]
    for m in range(0, len(wpass)):
        wpass_nyq[m] = wpass[m] / nyq

    # new code. Works with scipy 1.4 (2020.05.06)
    wstop_nyq = [iter*0 for iter in range(len(wstop))]
    for m in range(0, len(wstop)):
        wstop_nyq[m] = wstop[m] / nyq

    #wpass_nyq = map(lambda f: f/nyq, wpass) #old code. Works with scipy 0.18
    #wstop_nyq = map(lambda f: f/nyq, wstop) #old code. Works with scipy 0.18
    coef_b, coef_a = spsig.iirdesign(wp=wpass_nyq,
                                     ws=wstop_nyq,
                                     gpass=gpass,
                                     gstop=gstop,
                                     analog=0, ftype='ellip',
                                     output='ba')
    # Perform filtering and dump into signal_packet
    data_filt = spsig.filtfilt(coef_b, coef_a, data, axis=0)
    return data_filt


def xcorr_mag(data, fs, tau):
    """
    The xcorr_mag function implements a cross-correlation similarity function
    for computing functional connectivity -- maximum magnitude cross-correlation

    This function implements an FFT-based cross-correlation (using convolution).

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        tau: float
            The max lag limits of cross-correlation in seconds

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
    """

    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)
    check_type(tau, float)

    # Get data attributes
    n_samp, n_chan = data.shape
    tau_samp = int(tau*fs)
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Normalize the signal
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    lags = np.hstack((range(0, n_samp, 1),
                      range(-n_samp, 0, 1)))
    tau_ix = np.flatnonzero(lags <= tau_samp)

    # Use FFT to compute cross-correlation
    data_fft = np.fft.rfft(
        np.vstack((data, np.zeros_like(data))),
        axis=0)

    # Iterate over all edges
    for n1, n2 in zip(triu_ix, triu_iy):
        xc = 1 / n_samp * np.fft.irfft(
            data_fft[:, n1] * np.conj(data_fft[:, n2]))
        adj[n1, n2] = np.max(np.abs(xc[tau_ix]))
    adj += adj.T

    return adj


def xcorr(data, fs, tau):
    """
    The xcorr function implements a cross-correlation similarity function
    for computing functional connectivity.

    This function implements an FFT-based cross-correlation (using convolution).

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        tau: float
            The max lag limits of cross-correlation in seconds

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
    """

    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    check_type(fs, int)
    check_type(tau, float)

    # Get data attributes
    n_samp, n_chan = data.shape
    tau_samp = int(tau*fs)
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Normalize the signal
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))
    lags = np.hstack((range(0, n_samp, 1),
                      range(-n_samp, 0, 1)))
    tau_ix = np.flatnonzero(lags <= tau_samp)

    # Use FFT to compute cross-correlation
    data_fft = np.fft.rfft(
        np.vstack((data, np.zeros_like(data))),
        axis=0)

    # Iterate over all edges
    for n1, n2 in zip(triu_ix, triu_iy):
        xc = 1 / n_samp * np.fft.irfft(
            data_fft[:, n1] * np.conj(data_fft[:, n2]))

        if xc[tau_ix].max() > np.abs(xc[tau_ix].min()):
            adj[n1, n2] = xc[tau_ix].max()
        else:
            adj[n1, n2] = xc[tau_ix].min()
    adj += adj.T

    return adj


def multitaper(data, fs, time_band, n_taper, cf):
    """
    The multitaper function windows the signal using multiple Slepian taper
    functions and then computes coherence between windowed signals.

    Parameters
    ----------
        data: ndarray, shape (T, N)
            Input signal with T samples over N variates

        fs: int
            Sampling frequency

        time_band: float
            The time half bandwidth resolution of the estimate [-NW, NW];
            such that resolution is 2*NW

        n_taper: int
            Number of Slepian sequences to use (Usually < 2*NW-1)

        cf: list
            Frequency range over which to compute coherence [-NW+C, C+NW]

    Returns
    -------
        adj: ndarray, shape (N, N)
            Adjacency matrix for N variates
    """

    # Standard param checks
    check_type(data, np.ndarray)
    check_dims(data, 2)
    check_type(time_band, float)
    check_type(n_taper, int)
    check_type(cf, list)
    if n_taper >= 2*time_band:
        raise Exception('Number of tapers must be less than 2*time_band')
    if not len(cf) == 2:
        raise Exception('Must give a frequency range in list of length 2')

    # Get data attributes
    n_samp, n_chan = data.shape
    triu_ix, triu_iy = np.triu_indices(n_chan, k=1)

    # Initialize adjacency matrix
    adj = np.zeros((n_chan, n_chan))

    # Compute all coherences
    for n1, n2 in zip(triu_ix, triu_iy):
        if (data[:, n1] == data[:, n2]).all():
            adj[n1, n2] = np.nan
        else:
            out = mt_coherence(1.0/fs,
                               data[:, n1],
                               data[:, n2],
                               time_band,
                               n_taper,
                               int(n_samp/2.), 0.95,
                               iadapt=1,
                               cohe=True, freq=True)

            # Find closest frequency to the desired center frequency
            cf_idx = np.flatnonzero((out['freq'] >= cf[0]) &
                                    (out['freq'] <= cf[1]))

            # Store coherence in association matrix
            adj[n1, n2] = np.mean(out['cohe'][cf_idx])
    adj += adj.T

    return adj


"""
C. Utilities:
"""

def check_path(path):
    '''
    Check if path exists

    Parameters
    ----------
        path: str
            Check if valid path
    '''
    if not os.path.exists(path):
        raise IOError('%s does not exists' % path)


def make_path(path):
    '''
    Make new path if path does not exist

    Parameters
    ----------
        path: str
            Make the specified path
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        raise IOError('Path: %s, already exists' % path)


def check_path_overwrite(path):
    '''
    Prevent overwriting existing path

    Parameters
    ----------
        path: str
            Check if path exists
    '''
    if os.path.exists(path):
        raise IOError('%s cannot be overwritten' % path)


def check_has_key(dictionary, key_ref):
    '''
    Check whether the dictionary has the specified key

    Parameters
    ----------
        dictionary: dict
            The dictionary to look through

        key_ref: str
            The key to look for
    '''
    if key_ref not in dictionary.keys():
        raise KeyError('%r should contain the %r key' % (dictionary, key_ref))


def check_dims(arr, nd):
    '''
    Check if numpy array has specific number of dimensions

    Parameters
    ----------
        arr: numpy.ndarray
            Input array for dimension checking

        nd: int
            Number of dimensions to check against
    '''
    if not arr.ndim == nd:
        raise Exception('%r has %r dimensions. Must have %r' % (arr, arr.ndim, nd))


def check_type(obj, typ):
    '''
    Check if obj is of correct type

    Parameters
    ----------
        obj: any
            Input object for type checking

        typ: type
            Reference object type (e.g. str, int)
    '''
    if not isinstance(obj, typ):
        raise TypeError('%r is %r. Must be %r' % (obj, type(obj), typ))


def check_function(obj):
    '''
    Check if obj is a function

    Parameters
    ----------
        obj: any
            Input object for type checking
    '''
    if not inspect.isfunction(obj):
        raise TypeError('%r must be a function.' % (obj))
