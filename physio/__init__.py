"""
Physiological Signal Processing Pipeline for fMRI Analysis.

This package provides tools for:
- Quality control of cardiac (PPG) and respiratory signals (physio_qc)
- HRV and RVT regressor calculation (physio_regressors)
- GLM analysis with physiological regressors (physio_glm)

Usage:
    # Run QC first to identify problematic recordings
    python -m physio.physio_qc
    
    # Review the physio_qc_review.tsv file and mark useable recordings
    
    # Calculate regressors for useable recordings
    python -m physio.physio_regressors
    
    # Run GLM analysis
    python -m physio.physio_glm

Author: Joe Bathelt
Date: December 2025
"""

from .physio_utils import (
    detect_flatlines,
    calculate_hrv_regressor,
    calculate_rvt_regressor,
    get_n_volumes_from_fmri,
    get_tr_from_fmri,
    load_physio_data,
    extract_run_id,
    DEFAULT_TR,
    HR_MIN, HR_MAX,
    RESP_MIN, RESP_MAX,
    QUALITY_THRESHOLD,
    PPG_FLATLINE_THRESHOLD,
    RSP_FLATLINE_THRESHOLD,
)

__all__ = [
    'detect_flatlines',
    'calculate_hrv_regressor',
    'calculate_rvt_regressor',
    'get_n_volumes_from_fmri',
    'get_tr_from_fmri',
    'load_physio_data',
    'extract_run_id',
    'DEFAULT_TR',
    'HR_MIN', 'HR_MAX',
    'RESP_MIN', 'RESP_MAX',
    'QUALITY_THRESHOLD',
    'PPG_FLATLINE_THRESHOLD',
    'RSP_FLATLINE_THRESHOLD',
]
