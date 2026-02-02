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
    power = np.abs(fft_data**2)
    return power