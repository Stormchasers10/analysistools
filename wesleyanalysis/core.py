#AI was used to help write comments, making it easier to understand the function of each variable.

import numpy as np
import ugradio


def capture_data(sample_rate, nsamples, nblocks, direct=True):
    """
    Create an SDR object and capture data.

    Parameters
    ----------
    sample_rate : float
        Sampling rate in Hz
    nsamples : int
        Number of samples per block
    nblocks : int
        Number of blocks
    direct : bool
        Use direct sampling mode

    Returns
    -------
    data : np.ndarray
        Captured data array of shape (nblocks, nsamples)
    """
    # Create SDR Object.
    sdr = ugradio.sdr.SDR(direct=direct, sample_rate=sample_rate)
    data = sdr.capture_data(nsamples=nsamples, nblocks=nblocks)
    sdr.close()
    return data


def voltage_spectrum(data):
    """
    Voltage Spectrum (verbatim extraction from original script)

    Returns
    -------
    fft_data : array
        Complex FFT of the data
    """
    # Voltage Spectrum
    data_len = len(data)
    fft_data = np.fft.fft(data)
    # Vmag = np.abs(fft_data)
    return fft_data


def power_spectrum(data):
    """
    Power Spectrum (verbatim extraction from original script)

    Returns
    -------
    power : array
        Power spectrum |FFT|^2
    """
    # Power Spectrum
    fft_data = np.fft.fft(data)
    power = np.abs(fft_data)**2
    return power


###### This is the start of the theory plot of f_obs vs f_samp plot ######

import numpy as np

def alias_peak(f0, fs):
    """
    Return aliased frequency in the FIRST Nyquist zone [0, fs/2].
    f0 and fs must be in the same units.
    Works with scalars or numpy arrays.
    """
    f0 = np.asarray(f0, dtype=float)
    fs = np.asarray(fs, dtype=float)

    r = np.mod(f0, fs)                 # in [0, fs)
    return np.where(r <= fs/2, r, fs - r)

def theory_fobs_vs_fs(f0, fs_vals):
    """
    Returns (fs_vals, fobs_vals) for the aliasing theory curve.
    No plotting. Units pass through unchanged.

    Parameters
    ----------
    f0 : float
        True signal frequency (same units as fs_vals)
    fs_vals : array-like
        Sampling frequencies (same units as f0)

    Returns
    -------
    fs_vals : np.ndarray
    fobs_vals : np.ndarray
    """
    fs_vals = np.asarray(fs_vals, dtype=float)
    fobs_vals = alias_peak(f0, fs_vals)
    return fs_vals, fobs_vals

def logspace_fs(fs_max, fs_min, npts=400):
    """
    Convenience: returns log-spaced sampling frequencies.
    """
    return np.logspace(np.log10(fs_max), np.log10(fs_min), int(npts))

    ###### This is the end of the theory plot of f_obs vs f_samp plot ######
