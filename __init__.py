"""
Diffusion-Weighted Imaging (DWI) Quality Control and Visualization.

This package provides tools for:
- Quality control map extraction (qc_maps)
- Structural connectome visualization (plot_connectome)

Usage:
    # Extract QC maps from processed data
    python -m dwi.qc_maps
    
    # Visualize structural connectomes
    python -m dwi.plot_connectome

Author: Joe Bathelt
Date: December 2025
"""

from .dwi_utils import (
    run_cmd,
    run_docker_cmd,
    is_subject_processed,
    get_total_mem_mb,
)

__all__ = [
    'run_cmd',
    'run_docker_cmd',
    'is_subject_processed',
    'get_total_mem_mb',
]
