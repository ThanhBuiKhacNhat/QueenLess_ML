import os
import librosa
import numpy as np
from .audio_features import AudioFeatures


def collect_features(mfcc):
    # Calculate the mean along the time axis (axis=1)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Calculate the standard deviation along the time axis (axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Concatenate the mean and standard deviation arrays
    mfcc_merged = np.concatenate((mfcc_mean, mfcc_std))

    return mfcc_mean, mfcc_std, mfcc_merged


def calc_mfcc(audio_file):
    # Load the audio file, do not specify the sample rate
    y, sr = librosa.load(audio_file, sr=None)

    # Pre-emphasis with alpha = 0.97
    emphasized_signal = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Pre-compute log-power Mel spectrogram
    S = librosa.feature.melspectrogram(y=emphasized_signal, sr=sr, n_mels=128)

    # Calculate MFCCs
    s_mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S),  # Pre-computed Mel spectrogram
        n_mfcc=40,           # Number of MFCC coefficients to compute
        dct_type=2,          # Type of Discrete Cosine Transform
        norm='ortho',        # Type of normalization to use ('ortho', 'slaney', or None)
        lifter=22,           # Lifter parameter to apply to MFCCs
        n_fft=512,           # Length of the FFT window
        hop_length=512,      # Hop length for the STFT
        win_length=256,      # Length of the window function
        window='hann',       # Window function type ('hann', 'hamming', etc.)
        center=True,         # Whether to center the frames on time or not
        pad_mode='reflect',  # Padding mode for short frames
        power=2.0,           # Exponent for the magnitude spectrogram (typically 2 for power)
    )

    # Collect the features for the mel spectrogram method
    s_mean, s_std, s_merged = collect_features(s_mfcc)

    # Calculate MFCCs
    mfcc = librosa.feature.mfcc(
        y=emphasized_signal,  # The audio signal (time series)
        sr=sr,                # Sample rate of the audio file
        n_mfcc=13,            # Number of MFCC coefficients to compute
        dct_type=2,           # Type of Discrete Cosine Transform
        norm='ortho',         # Type of normalization to use ('ortho', 'slaney', or None)
        lifter=22,            # Lifter parameter to apply to MFCCs
        n_fft=512,            # Length of the FFT window
        hop_length=512,       # Hop length for the STFT
        win_length=256,       # Length of the window function
        window='hann',        # Window function type ('hann', 'hamming', etc.)
        center=True,          # Whether to center the frames on time or not
        pad_mode='reflect',   # Padding mode for short frames
        power=2.0,            # Exponent for the magnitude spectrogram (typically 2 for power)
        n_mels=40,            # Number of mel filter banks
    )

    # Calculate delta features
    delta = librosa.feature.delta(mfcc)

    # Calculate delta-delta features
    delta_delta = librosa.feature.delta(mfcc, order=2)

    # Stack the MFCCs, deltas, and delta-deltas together
    d_mfcc = np.vstack([mfcc, delta, delta_delta])

    # Collect the features for the delta method
    d_mean, d_std, d_merged = collect_features(d_mfcc)

    return AudioFeatures(s_mean, s_std, s_merged, d_mean, d_std, d_merged)
