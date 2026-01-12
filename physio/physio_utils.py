#!/usr/bin/env python3
"""
Shared utility functions for physiological signal processing.

Author: Joe Bathelt
Date: December 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import neurokit2 as nk


def detect_flatlines(signal: np.ndarray, sampling_rate: float, threshold_seconds: float = 10.0,
                     window_seconds: float = 1.0) -> dict:
    """
    Detect flat line segments in a physiological signal using a windowed variance approach.
    
    A flat line is defined as a segment where the signal variance within sliding windows
    is near zero, indicating no meaningful physiological variation.
    
    Parameters:
    -----------
    signal : np.ndarray
        The physiological signal to analyze
    sampling_rate : float
        Sampling rate in Hz
    threshold_seconds : float
        Minimum duration in seconds to consider as a problematic flat line
    window_seconds : float
        Window size in seconds for computing local variance (default 1s)
        
    Returns:
    --------
    dict with:
        - max_flatline_duration: Maximum flat line duration in seconds
        - n_flatline_segments: Number of flat line segments exceeding threshold
        - total_flatline_duration: Total duration of all flat lines in seconds
        - flatline_percentage: Percentage of recording that is flat
    """
    if len(signal) == 0:
        return {
            'max_flatline_duration': 0.0,
            'n_flatline_segments': 0,
            'total_flatline_duration': 0.0,
            'flatline_percentage': 0.0
        }
    
    signal = signal.astype(float)
    window_samples = int(window_seconds * sampling_rate)
    
    # Compute rolling variance using a sliding window
    n_windows = len(signal) - window_samples + 1
    if n_windows <= 0:
        return {
            'max_flatline_duration': 0.0,
            'n_flatline_segments': 0,
            'total_flatline_duration': 0.0,
            'flatline_percentage': 0.0
        }
    
    # Calculate variance for each window position using cumsum for efficiency
    cumsum = np.cumsum(signal)
    cumsum2 = np.cumsum(signal ** 2)
    
    # Variance = E[X^2] - E[X]^2
    window_sum = cumsum[window_samples-1:] - np.concatenate([[0], cumsum[:-window_samples]])
    window_sum2 = cumsum2[window_samples-1:] - np.concatenate([[0], cumsum2[:-window_samples]])
    
    window_mean = window_sum / window_samples
    window_var = (window_sum2 / window_samples) - (window_mean ** 2)
    window_var = np.maximum(window_var, 0)  # Handle numerical precision issues
    
    # Define "flat" as variance below a threshold
    overall_var = np.var(signal)
    var_threshold = overall_var * 0.001 if overall_var > 0 else 1e-10
    
    is_flat = window_var < var_threshold
    
    # Find runs of flat windows
    threshold_samples = int(threshold_seconds * sampling_rate)
    
    # Find start and end of flat segments
    flat_segments_samples = []
    in_flat = False
    flat_start = 0
    
    for i, flat in enumerate(is_flat):
        if flat and not in_flat:
            in_flat = True
            flat_start = i
        elif not flat and in_flat:
            in_flat = False
            segment_length_samples = i - flat_start
            if segment_length_samples >= threshold_samples:
                flat_segments_samples.append(segment_length_samples)
    
    # Check if we ended in a flat segment
    if in_flat:
        segment_length_samples = len(is_flat) - flat_start
        if segment_length_samples >= threshold_samples:
            flat_segments_samples.append(segment_length_samples)
    
    # Calculate statistics
    total_duration_seconds = len(signal) / sampling_rate
    total_flat_seconds = sum(flat_segments_samples) / sampling_rate
    
    return {
        'max_flatline_duration': max(flat_segments_samples) / sampling_rate if flat_segments_samples else 0.0,
        'n_flatline_segments': len(flat_segments_samples),
        'total_flatline_duration': total_flat_seconds,
        'flatline_percentage': (total_flat_seconds / total_duration_seconds) * 100 if total_duration_seconds > 0 else 0.0
    }


def calculate_hrv_regressor(cardiac_signal: np.ndarray, sampling_rate: float, 
                            tr: float, n_volumes: int, window_sec: float = 30) -> np.ndarray:
    """
    Calculate Heart Rate Variability (HRV) regressor for fMRI.
    
    Computes standard deviation of heart rate over sliding windows,
    which captures HRV across multiple frequency bands.
    
    Parameters:
    -----------
    cardiac_signal : np.ndarray
        Raw cardiac (PPG) signal
    sampling_rate : float
        Sampling rate of cardiac signal in Hz
    tr : float
        fMRI repetition time in seconds
    n_volumes : int
        Number of fMRI volumes
    window_sec : float
        Window size for HRV calculation in seconds (default: 30s)
        
    Returns:
    --------
    np.ndarray
        HRV regressor resampled to TR (length = n_volumes)
    """
    # Process PPG signal
    signals, info = nk.ppg_process(cardiac_signal, sampling_rate=sampling_rate)
    peaks = np.where(signals['PPG_Peaks'] == 1)[0]
    
    if len(peaks) < 2:
        return np.full(n_volumes, np.nan)
    
    # Get instantaneous heart rate timeseries
    peak_times = peaks / sampling_rate
    ibi = np.diff(peak_times)
    ibi_times = peak_times[:-1] + ibi / 2
    instantaneous_hr = 60 / ibi  # bpm
    
    # Interpolate to high resolution for smooth windowing
    physio_times = np.arange(0, len(cardiac_signal)) / sampling_rate
    interp_func = interp1d(ibi_times, instantaneous_hr, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
    hr_continuous = interp_func(physio_times)
    
    # Calculate HRV as rolling standard deviation of HR
    window_samples = int(window_sec * sampling_rate)
    hr_series = pd.Series(hr_continuous)
    hrv_continuous = hr_series.rolling(
        window=window_samples, 
        center=True, 
        min_periods=window_samples//3  # Allow calculation with fewer samples at edges
    ).std()
    
    # Fill any remaining NaNs
    hrv_continuous = hrv_continuous.fillna(method='bfill').fillna(method='ffill').values
    
    # Resample to fMRI timepoints
    fmri_times = np.arange(n_volumes) * tr + tr / 2
    valid_range = (fmri_times >= physio_times[0]) & (fmri_times <= physio_times[-1])
    
    if not np.any(valid_range):
        return np.full(n_volumes, np.nan)
    
    hrv_interp = interp1d(physio_times, hrv_continuous, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
    hrv_resampled = hrv_interp(fmri_times)
    
    # Demean only (don't z-score)
    hrv_resampled = hrv_resampled - np.nanmean(hrv_resampled)
    
    return hrv_resampled

def calculate_rvt_regressor(resp_signal: np.ndarray, sampling_rate: float,
                            tr: float, n_volumes: int) -> np.ndarray:
    """
    Calculate Respiratory Volume per Time (RVT) regressor for fMRI.
    
    RVT is calculated as: (peak - trough) / breath_period
    This follows the approach described in Birn et al. (2006).
    
    Parameters:
    -----------
    resp_signal : np.ndarray
        Raw respiratory signal
    sampling_rate : float
        Sampling rate of respiratory signal in Hz
    tr : float
        fMRI repetition time in seconds
    n_volumes : int
        Number of fMRI volumes
        
    Returns:
    --------
    np.ndarray
        RVT regressor resampled to TR (length = n_volumes)
    """
    # Process respiratory signal
    resp_signals, resp_info = nk.rsp_process(resp_signal, sampling_rate=sampling_rate)
    
    # Get peaks and troughs
    peaks = resp_info['RSP_Peaks']
    troughs = resp_info['RSP_Troughs']
    
    if len(peaks) < 2 or len(troughs) < 2:
        return np.full(n_volumes, np.nan)
    
    # Calculate RVT for each breath cycle
    rvt_values = []
    rvt_times = []
    
    for i, peak_idx in enumerate(peaks[:-1]):
        # Find the trough before this peak
        preceding_troughs = troughs[troughs < peak_idx]
        if len(preceding_troughs) == 0:
            continue
        trough_idx = preceding_troughs[-1]
        
        # Find the next peak for breath period
        next_peak_idx = peaks[i + 1]
        
        # Calculate amplitude and period
        amplitude = resp_signal[peak_idx] - resp_signal[trough_idx]
        period = (next_peak_idx - peak_idx) / sampling_rate  # in seconds
        
        if period > 0:
            rvt = amplitude / period
            rvt_values.append(rvt)
            rvt_times.append(peak_idx / sampling_rate)
    
    if len(rvt_values) < 2:
        return np.full(n_volumes, np.nan)
    
    rvt_values = np.array(rvt_values)
    rvt_times = np.array(rvt_times)
    
    # Create time vector for fMRI volumes
    fmri_times = np.arange(n_volumes) * tr + tr / 2
    
    # Interpolate RVT to fMRI time points
    interp_func = interp1d(rvt_times, rvt_values, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
    rvt_resampled = interp_func(fmri_times)
    
    # Demean the regressor
    rvt_resampled = rvt_resampled - np.nanmean(rvt_resampled)
    
    return rvt_resampled


def get_n_volumes_from_fmri(bids_folder: Path, subject: str, run_id: str) -> int:
    """
    Get the number of volumes from the corresponding fMRI file.
    
    Looks for any BOLD file matching the run number.
    """
    func_folder = bids_folder / subject / 'func'
    if not func_folder.exists():
        return None
    
    # Find matching BOLD JSON file
    for f in func_folder.iterdir():
        if f.name.endswith('_bold.json') and run_id in f.name:
            with open(f, 'r') as jf:
                meta = json.load(jf)
                if 'NumberOfVolumes' in meta:
                    return meta['NumberOfVolumes']
    
    # If no metadata, try to infer from NIfTI file
    import nibabel as nib
    for f in func_folder.iterdir():
        if f.name.endswith('_bold.nii.gz') and run_id in f.name:
            try:
                img = nib.load(f)
                return img.shape[3] if len(img.shape) == 4 else None
            except Exception:
                pass
    
    return None


def get_tr_from_fmri(bids_folder: Path, subject: str, run_id: str, default_tr: float = 1.7) -> float:
    """
    Get the TR (RepetitionTime) from the corresponding fMRI JSON sidecar.
    
    Parameters:
    -----------
    bids_folder : Path
        Path to BIDS folder
    subject : str
        Subject ID (e.g., 'sub-001')
    run_id : str
        Run identifier (e.g., 'run-01')
    default_tr : float
        Default TR to use if not found in JSON
        
    Returns:
    --------
    float
        TR in seconds
    """
    func_folder = bids_folder / subject / 'func'
    if not func_folder.exists():
        return default_tr
    
    # Find matching BOLD JSON file
    for f in func_folder.iterdir():
        if f.name.endswith('_bold.json') and run_id in f.name:
            try:
                with open(f, 'r') as jf:
                    meta = json.load(jf)
                    if 'RepetitionTime' in meta:
                        return meta['RepetitionTime']
            except Exception:
                pass
    
    return default_tr


def load_physio_data(physio_path: Path) -> tuple:
    """
    Load physiological data and metadata from BIDS-formatted files.
    
    Parameters:
    -----------
    physio_path : Path
        Path to the physio.tsv.gz file
        
    Returns:
    --------
    tuple: (phys_data: pd.DataFrame or None, sampling_rate: float)
        Returns (None, 0) if loading fails
    """
    json_path = physio_path.with_suffix('').with_suffix('.json')
    
    if not json_path.exists():
        print(f'    Warning: Missing JSON sidecar, skipping')
        return None, 0
    
    try:
        with open(json_path, 'r') as f:
            phys_meta = json.load(f)
    except Exception as e:
        print(f'    Error reading JSON sidecar: {e}')
        return None, 0
    
    sampling_rate = phys_meta.get('SamplingFrequency', 496)
    if isinstance(sampling_rate, list):
        sampling_rate = sampling_rate[0]
    
    try:
        phys_data = pd.read_csv(physio_path, sep='\t', compression='gzip')
    except Exception as e:
        print(f'    Error reading physio file: {e}')
        return None, 0
    
    # Rename columns based on metadata
    if 'Columns' in phys_meta:
        column_names = [col.split('_')[-1] for col in phys_meta['Columns']]
        phys_data.columns = column_names
    
    return phys_data, sampling_rate


def extract_run_id(filename: str) -> str:
    """
    Extract run identifier from a BIDS filename.
    
    Parameters:
    -----------
    filename : str
        BIDS-formatted filename
        
    Returns:
    --------
    str: Run ID (e.g., 'run-01')
    """
    for part in filename.replace('_physio.tsv.gz', '').split('_'):
        if part.startswith('run-'):
            return part
    return None


# Default TR for fMRI
DEFAULT_TR = 1.7  # seconds

# Expected physiological ranges for healthy young adults
HR_MIN, HR_MAX = 50, 100  # bpm (resting heart rate)
RESP_MIN, RESP_MAX = 8, 25  # breaths/min (resting respiratory rate)

# Quality thresholds
QUALITY_THRESHOLD = 0.70  # Ratio of good samples
PPG_FLATLINE_THRESHOLD = 10.0  # seconds
RSP_FLATLINE_THRESHOLD = 45.0  # seconds
