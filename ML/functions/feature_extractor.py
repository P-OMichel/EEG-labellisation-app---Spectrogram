'''
File to compute all features for the classifier model
'''

import numpy as np
import scipy as sc
from numpy.lib.stride_tricks import sliding_window_view


#NOTE: functions to compute features landscapes
# ------------------------------------  compute slope --------------------------------------------
def local_slope(y, window=15, x=None, pad_mode="reflect"):
    """
    Local slope via sliding-window least squares line fit.

    Parameters
    ----------
    y : (N,) array_like
        1D signal / time series.
    window : int
        Odd window size (>= 3). Larger => smoother slope.
    x : (N,) array_like or None
        Optional x-values (time). If None, uses np.arange(N).
        If your sampling is non-uniform, pass x.
    pad_mode : str
        np.pad mode for borders: 'reflect', 'edge', 'symmetric', etc.

    Returns
    -------
    slope : (N,) ndarray
        Local slope at each point (same length as y).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    n = y.size

    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x must have the same shape as y")

    half = window // 2

    # Pad both x and y so output length matches input length
    yp = np.pad(y, (half, half), mode=pad_mode)
    xp = np.pad(x, (half, half), mode=pad_mode)

    # Sliding window views
    Y = np.lib.stride_tricks.sliding_window_view(yp, window_shape=window)  # (N, window)
    X = np.lib.stride_tricks.sliding_window_view(xp, window_shape=window)  # (N, window)

    # Center x per window for numerical stability
    Xc = X - X.mean(axis=1, keepdims=True)

    # Least squares slope for each window: cov(X,Y)/var(X)
    denom = np.sum(Xc * Xc, axis=1)
    # Avoid divide-by-zero if x is constant in a window (rare unless x is weird/padded oddly)
    denom = np.where(denom == 0, np.nan, denom)

    slope = np.sum(Xc * (Y - Y.mean(axis=1, keepdims=True)), axis=1) / denom
    return slope

# -------------------------------------  compute median  ------------------------------------------
def local_median(y, window=15, pad_mode="reflect"):
    """
    Compute local median in a sliding window.

    Parameters
    ----------
    y : (N,) array_like
        1D signal / time series.
    window : int
        Odd window size (>= 3).
    pad_mode : str
        Padding mode for borders ('reflect', 'edge', etc.)

    Returns
    -------
    med : (N,) ndarray
        Local median (same length as input).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")

    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    half = window // 2

    # Pad to preserve length
    yp = np.pad(y, (half, half), mode=pad_mode)

    # Sliding windows
    Y = np.lib.stride_tricks.sliding_window_view(yp, window_shape=window)

    # Median along window axis
    med = np.median(Y, axis=1)

    return med

# -------------------------------------  compute median  ------------------------------------------
def local_std(y, window=128, pad_mode="reflect"):
    """
    Compute local median in a sliding window.

    Parameters
    ----------
    y : (N,) array_like
        1D signal / time series.
    window : int
        Odd window size (>= 3).
    pad_mode : str
        Padding mode for borders ('reflect', 'edge', etc.)

    Returns
    -------
    med : (N,) ndarray
        Local median (same length as input).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1D")

    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")

    half = window // 2

    # Pad to preserve length
    yp = np.pad(y, (half, half), mode=pad_mode)

    # Sliding windows
    Y = np.lib.stride_tricks.sliding_window_view(yp, window_shape=window)

    # Median along window axis
    std = np.std(Y, axis = 1)

    return std

# NOTE: edge frequency on spectrogram
# ------------------------------------ Edge frequency ---------------------------------------------
def edge_frequencies_limit_value(M, f, min_val = 0.001, max_val=20, threshold=None, T=5, q=0.75, threshold_min=0.01, threshold_max=20, factor_hf=5):
    '''
    M: time_frequency matrix
    f: associated frequencies
    min_val: minimum value to clip M
    max_val: maximum value to clip M 
    T: Value by which to multiply to get the threshold
    q: quantile value to get the on the M elements
    threshold_min: minimum value for the threshold
    threshold_max: maximum value for the threshold
    smooth: when True the edge_frequency is smoothed with a (3,1) Savitsky-Golay filter
    '''

    # mask of high frequency high power parts
    T_M = (M >= max_val/10).astype(int)
    count = np.sum(T_M[35:,:], axis=0)
    mask_hf = 1 - (count >= 1).astype(int)
    

    # thresholding   
    if threshold == None:
        if np.sum(mask_hf) != 0 and np.prod(mask_hf) !=1:
            threshold = T * np.quantile(np.clip(M[:, mask_hf == 1], min_val, max_val), q)
        else:
            threshold = T * np.quantile(np.clip(M, min_val, max_val), q)
        # min max values 
        threshold = min(threshold_max, threshold)
        threshold = max(threshold_min, threshold)

    # get list of frequencies:
    edge_frequencies = get_edge_limit_value(M, f, threshold)

    # estimation of the hf edge frequency
    threshold_hf = threshold / factor_hf

    # get list of frequencies:
    edge_frequencies_hf = get_edge_limit_value(M, f, threshold_hf)

    return edge_frequencies, edge_frequencies_hf, threshold

def get_edge_limit_value(spectro, f_spectro, threshold):
    '''
    Inputs:
    spectro: clipped spectrogram  
    N_max: maximum number of point per colums higher than the threshold

    Ouput:
    '''

    # cumulative of each colums
    reversed_spectro = np.flipud(spectro)
    reversed_cumulative_spectro = np.cumsum(reversed_spectro, axis=0)
    cumulative_spectro = np.flipud(reversed_cumulative_spectro)

    cond = cumulative_spectro <= threshold
    indices = np.argmax(cond, axis = 0)
    # check where the condition cannot be reached as it return 0 and change to -1 to get highest frequency later
    valid = cond.any(axis=0)
    indices[~valid] = -1

    edge_frequencies = f_spectro[indices]

    return edge_frequencies

# NOTE: PSD slope on spectrogram
# ------------------------------------ PSD slopes ---------------------------------------------
def linear_fit(x, y):
    """
    Linear regression using numpy.polyfit (degree=1).
    Returns slope and intercept and plots the fit.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ---- Fit ----
    slope, intercept = np.polyfit(x, y, 1)

    # Fitted values
    #y_fit = slope * x + intercept

    return slope, intercept

def log_f_log_psd_slope_per_column(spectro, f_spectro, f_start, f_end):

    mask = (f_spectro >= f_start) & (f_spectro <= f_end)

    N = len(spectro[0,:])
    slopes = np.zeros(N)

    for i in range(len(spectro[0,:])):
        slopes[i] = linear_fit(np.log2(f_spectro[mask] + 0.000000001), spectro[mask, i])[0]

    return slopes

# NOTE: resample time series features to length of spectrogram time axis
# --------------------------------- resample fine to coarse time series ---------------------------
def fast_resample_fine_to_coarse(t_fine, y_fine, t_coarse):
    mask = (t_coarse >= t_fine[0]) & (t_coarse <= t_fine[-1])
    t_common = t_coarse[mask]
    y_fine_on_coarse = np.interp(t_common, t_fine, y_fine)
    return t_common, y_fine_on_coarse

def resample_to_mask_length(y: np.ndarray, mask: np.ndarray, method: str = "linear") -> np.ndarray:
    """
    Resample 1D signal y to have exactly len(mask) samples using interpolation.

    Parameters
    ----------
    y : np.ndarray
        1D input signal.
    mask : np.ndarray
        Array whose length defines the target length (values are ignored).
    method : str
        Interpolation method: "linear" (default). ("nearest" also supported.)

    Returns
    -------
    y_resampled : np.ndarray
        1D array of length len(mask).
    """
    y = np.asarray(y, dtype=float).ravel()
    n_src = y.size
    n_tgt = int(np.asarray(mask).size)

    if n_tgt <= 0:
        raise ValueError("mask must have length >= 1")
    if n_src == 0:
        raise ValueError("y must have length >= 1")
    if n_src == 1:
        return np.full(n_tgt, y[0], dtype=float)
    if n_tgt == 1:
        return np.array([y[0]], dtype=float)

    # Map both signals onto [0, 1] then interpolate
    x_src = np.linspace(0.0, 1.0, n_src)
    x_tgt = np.linspace(0.0, 1.0, n_tgt)

    if method == "linear":
        return np.interp(x_tgt, x_src, y)

    if method == "nearest":
        idx = np.rint(x_tgt * (n_src - 1)).astype(int)
        return y[idx]

    raise ValueError('method must be "linear" or "nearest"')

# NOTE  functions to extract time series features
def extract_features_time_series(y, window, mask):

    y = y - np.median(y)

    windows = sliding_window_view(y, window)

    # Mean & Std
    mean = np.mean(windows, axis=1)
    std = np.std(windows, axis=1)

    # Waveform length (classical EEG feature)
    dx = np.diff(windows, axis=1)
    linelen = np.sum(np.abs(dx), axis=1)

    # quantile envelope
    q = 0.95
    upper = np.quantile(windows, q, axis=1)
    lower = np.quantile(windows, 1 - q, axis=1)    

    # Padding to keep same size as input
    pad = window // 2
    pad_width = (pad, window - pad - 1)

    mean = np.pad(mean, pad_width, mode="edge")
    std = np.pad(std, pad_width, mode="edge")
    linelen = np.pad(linelen, pad_width, mode="edge")
    upper = np.pad(upper, pad_width, mode="edge")
    lower = np.pad(lower, pad_width, mode="edge")

    env = upper -  lower

    # std of env (compute before smoothing)
    std_env = local_std(env, window = 33) # fs / 4  + 1

    # smooth
    fs = 128
    n_smooth = int(fs /  4)
    mean =  np.abs(sc.signal.savgol_filter(mean, n_smooth, 1))
    std =  sc.signal.savgol_filter(std, n_smooth, 1)
    linelen =  sc.signal.savgol_filter(linelen, n_smooth, 1)
    env =  sc.signal.savgol_filter(env, n_smooth, 1)

    # convert to same size as spectro
    mean = resample_to_mask_length(mean, mask)
    std = resample_to_mask_length(std, mask)
    linelen = resample_to_mask_length(linelen, mask)
    env = resample_to_mask_length(env, mask)
    std_env = resample_to_mask_length(std_env, mask)

    # divide by quantile
    q_std = std / np.quantile(std, 0.8)
    q_linelen = linelen / np.quantile(linelen, 0.8)
    q_env = env / np.quantile(env, 0.8)
    q_std_env = std_env / np.quantile(std_env, 0.8)

    # get local median
    med_mean = local_median(mean, window = 15)
    med_q_std = local_median(q_std, window = 15)
    med_q_linelen = local_median(q_linelen, window = 15)
    med_q_env = local_median(q_env, window = 15)
    med_q_std_env = local_median(q_std_env, window = 15)

    return mean, med_mean, q_std, med_q_std, q_linelen, med_q_linelen, q_env, med_q_env, q_std_env, med_q_std_env

# NOTE  functions to extract spectrogram features
def extract_features_spectrogram(f_spectro, spectro):

    #--- edge frequency
    ef = edge_frequencies_limit_value(spectro, f_spectro, factor_hf = 5, threshold_min = 5, threshold_max=12, threshold=None, T=1, q=0.95)[0]
    
    #--- edge frequency recovery
    ef_recovery = edge_frequencies_limit_value(spectro, f_spectro, factor_hf = 5, threshold_min = 0.01, threshold_max=12, threshold=None, T=1, q=0.95)[0]    

    # local ef 
    med_ef = local_median(ef, window = 15)
    med_ef_recovery = local_median(ef_recovery, window = 15)

    #--- power proportions
    mask_delta = (f_spectro >= 0.5) & (f_spectro <= 4)
    mask_alpha = (f_spectro >= 7) & (f_spectro <= 14)
    mask_beta = (f_spectro >= 15) & (f_spectro <= 30)
    mask_gamma = (f_spectro >= 30) & (f_spectro <= 45)

    P_delta = np.sum(spectro[mask_delta, :], axis = 0)
    P_alpha = np.sum(spectro[mask_alpha, :], axis = 0)
    P_beta = np.sum(spectro[mask_beta, :], axis = 0)
    P_gamma = np.sum(spectro[mask_gamma, :], axis = 0)
    P_tot = P_delta + P_alpha + P_beta + P_gamma

    # power divided by quantile
    q_P_tot = P_tot / np.quantile(P_tot, 0.8)

    # local median of power divided by quantile
    med_q_P_tot = local_median(q_P_tot, window = 15)

    # power prop
    prop_delta = P_delta / P_tot
    prop_alpha = P_alpha/ P_tot
    prop_beta = P_beta / P_tot
    prop_gamma = P_gamma / P_tot

    # psd slopes
    slopes = log_f_log_psd_slope_per_column(spectro, f_spectro, 20, 45)

    return ef, ef_recovery, med_ef, med_ef_recovery, prop_delta, prop_alpha, prop_beta, prop_gamma, q_P_tot, med_q_P_tot, slopes