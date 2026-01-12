#!/usr/bin/env python3
"""
Shared utility functions for DWI processing.

Author: Joe Bathelt
Date: December 2025
"""

import os
import subprocess
from pathlib import Path


def get_total_mem_mb() -> int:
    """Return total system memory in MB (Linux /proc/meminfo)."""
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith('MemTotal:'):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return max(1, int(int(parts[1]) / 1024))  # kB -> MB
    # Fallback conservative default (64 GB)
    return 64 * 1024


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command with error handling.
    
    Parameters:
    -----------
    cmd : str
        The command to run
    check : bool
        Whether to raise an error if the command fails
        
    Returns:
    --------
    subprocess.CompletedProcess
    """
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def run_docker_cmd(image: str, cmd: str, bind_paths: list = None, 
                   check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a command inside a Docker container.
    
    Parameters:
    -----------
    image : str
        Docker image name (e.g., 'mrtrix3/mrtrix3:3.0.8')
    cmd : str
        The command to run inside the container
    bind_paths : list
        List of paths to bind mount into the container
    check : bool
        Whether to raise an error if the command fails
        
    Returns:
    --------
    subprocess.CompletedProcess
    """
    docker_cmd = f"docker run --rm"
    
    if bind_paths:
        for path in bind_paths:
            docker_cmd += f" -v {path}:{path}"
    
    docker_cmd += f" {image} {cmd}"
    
    print(f"  Running: {cmd}")
    result = subprocess.run(docker_cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  Error: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return result


def is_subject_processed(out_dir: Path, subject: str) -> bool:
    """
    Heuristic check if subject appears already processed.
    
    Parameters:
    -----------
    out_dir : Path
        Output directory to check
    subject : str
        Subject ID (e.g., 'sub-001')
        
    Returns:
    --------
    bool
        True if subject appears processed
    """
    # Check for report HTML
    report = out_dir / f"{subject}.html"
    if report.exists():
        return True
    
    # Check if subject directory has files
    subj_dir = out_dir / subject
    if not subj_dir.exists():
        return False
    
    for root, _, files in os.walk(subj_dir):
        if files:
            return True
    
    return False


# Docker image configurations
MRTRIX_DOCKER_IMAGE = "mrtrix3/mrtrix3:3.0.8"
ANTS_DOCKER_IMAGE = "antsx/ants:v2.6.3"
QSIPREP_DOCKER_IMAGE = "pennlinc/qsiprep:1.0.1"
QSIRECON_DOCKER_IMAGE = "pennlinc/qsirecon:1.1.1"

# Default paths
DEFAULT_PROJECT_FOLDER = Path('/home/ASDPrecision/')
