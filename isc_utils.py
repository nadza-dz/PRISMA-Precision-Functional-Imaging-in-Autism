#!/usr/bin/env python3
"""
Shared utility functions for ISC analysis.
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def load_participants(participants_tsv):
    """Load participant info and split by diagnosis."""
    df = pd.read_csv(participants_tsv, sep='\t')
    asc_subs = df[df['autism_diagnosis'] == 'yes']['participant_id'].tolist()
    cmp_subs = df[df['autism_diagnosis'] == 'no']['participant_id'].tolist()
    return asc_subs, cmp_subs

def get_timeseries_file(subject, task, xcpd_dir):
    """Find XCP-D timeseries file for a subject and task."""
    pattern = f"{subject}_task-{task}_run-*_space-fsLR_seg-4S456Parcels_stat-mean_timeseries.tsv"
    files = glob.glob(str(Path(xcpd_dir) / subject / "func" / pattern))
    if len(files) == 1:
        return files[0]
    elif len(files) > 1:
        return sorted(files)[0]
    return None

def compute_isc(data_list):
    """
    Compute leave-one-out ISC for a list of subjects using fully vectorized operations.
    """
    n_subjects = len(data_list)
    if n_subjects < 2:
        return None
    data_stack = np.array(data_list)
    means = data_stack.mean(axis=1, keepdims=True)
    stds = data_stack.std(axis=1, keepdims=True) + 1e-8
    data_zscore = (data_stack - means) / stds
    total_sum = data_zscore.sum(axis=0)
    isc_per_subject = np.zeros((n_subjects, data_stack.shape[2]))
    for i in range(n_subjects):
        mean_others = (total_sum - data_zscore[i]) / (n_subjects - 1)
        mean_others_zscore = (mean_others - mean_others.mean(axis=0)) / (mean_others.std(axis=0) + 1e-8)
        isc_per_subject[i] = np.mean(data_zscore[i] * mean_others_zscore, axis=0)
    isc_values = isc_per_subject.mean(axis=0)
    return isc_values
