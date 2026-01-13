#!/usr/bin/env python3
"""
Script to extract Freesurfer quality metrics: Euler numbers and grey/white boundary CNR
Author: Joe Bathelt, December 2025

Extracts:
1. Euler numbers for left and right hemispheres from recon-all log files
2. Grey/white matter contrast-to-noise ratio (CNR) from autodet.gw.stats files

The Euler number is a topological measure indicating the quality of cortical surface
reconstruction. Lower (more negative) values indicate more topological defects.

CNR is calculated as: (white_mean - gray_mean) / sqrt(white_std^2 + gray_std^2)
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
freesurfer_dir = "/home/ASDPrecision/data/bids/derivatives/fmriprep/sourcedata/freesurfer"
participants_file = "/home/ASDPrecision/data/bids/participants.tsv"
output_dir = "/home/ASDPrecision/quality_metrics/freesurfer"
output_file = os.path.join(output_dir, "freesurfer_qc_metrics.tsv")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def extract_euler_number(log_file, hemisphere):
    """
    Extract Euler number from Freesurfer recon-all log file.

    Parameters:
    -----------
    log_file : str
        Path to recon-all-{lh,rh}.log file
    hemisphere : str
        'lh' or 'rh'

    Returns:
    --------
    int or None
        Euler number extracted from the log
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Look for the Euler number calculation for orig.premesh
        # Pattern: "euler # = v-e+f = 2g-2: VERTICES - EDGES + FACES = EULER"
        pattern = r'euler # = v-e\+f = 2g-2: \d+ - \d+ \+ \d+ = (-?\d+)'
        matches = re.findall(pattern, content)

        if matches:
            # Take the last occurrence (final Euler number after topology correction)
            euler = int(matches[-1])
            return euler
        else:
            print(f"Warning: Could not find Euler number in {log_file}")
            return None

    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_file}")
        return None
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
        return None


def extract_cnr(stats_file, hemisphere):
    """
    Extract grey/white matter CNR from Freesurfer autodet.gw.stats file.

    CNR = (white_mean - gray_mean) / sqrt(white_std^2 + gray_std^2)

    Parameters:
    -----------
    stats_file : str
        Path to autodet.gw.stats.{lh,rh}.dat file
    hemisphere : str
        'lh' or 'rh'

    Returns:
    --------
    float or None
        Calculated CNR value
    """
    try:
        # Read the stats file
        stats = {}
        with open(stats_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) == 2:
                        key, value = parts
                        try:
                            stats[key] = float(value)
                        except ValueError:
                            stats[key] = value

        # Extract required values
        white_mean = stats.get('white_mean')
        white_std = stats.get('white_std')
        gray_mean = stats.get('gray_mean')
        gray_std = stats.get('gray_std')

        # Check if all values are present
        if all(v is not None for v in [white_mean, white_std, gray_mean, gray_std]):
            # Calculate CNR
            cnr = (white_mean - gray_mean) / np.sqrt(white_std**2 + gray_std**2)
            return cnr
        else:
            print(f"Warning: Missing required values in {stats_file}")
            return None

    except FileNotFoundError:
        print(f"Warning: Stats file not found: {stats_file}")
        return None
    except Exception as e:
        print(f"Error processing {stats_file}: {e}")
        return None


def report_group_descriptives(df, participants_df):
    """
    Report group-level descriptive statistics for Freesurfer QC metrics.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Freesurfer QC metrics
    participants_df : pd.DataFrame
        DataFrame with participant information including autism_diagnosis

    Returns:
    --------
    None
        Prints group-level statistics to console
    """
    # Merge with participant data to get group assignment
    df_merged = df.merge(participants_df[['participant_id', 'autism_diagnosis']],
                         on='participant_id', how='left')

    # Map autism diagnosis to group labels
    df_merged['group'] = df_merged['autism_diagnosis'].map({
        'yes': 'ASC',
        'no': 'CMP'
    })

    # Remove subjects without group assignment
    df_merged = df_merged.dropna(subset=['group'])

    print(f"\n{'='*70}")
    print("GROUP-LEVEL DESCRIPTIVE STATISTICS")
    print(f"{'='*70}")

    # Count subjects per group
    group_counts = df_merged['group'].value_counts()
    print(f"\nSample sizes:")
    for group in ['ASC', 'CMP']:
        if group in group_counts.index:
            print(f"  {group}: n = {group_counts[group]}")

    # Euler Number descriptives (range and median per hemisphere per group)
    print(f"\n{'-'*70}")
    print("EULER NUMBER (Range and Median by Group)")
    print(f"{'-'*70}")
    print(f"{'Group':<8} {'Hemisphere':<12} {'Min':<8} {'Max':<8} {'Median':<8}")
    print(f"{'-'*70}")

    for group in ['ASC', 'CMP']:
        group_data = df_merged[df_merged['group'] == group]
        if len(group_data) > 0:
            for hemi in ['lh', 'rh']:
                col = f'euler_{hemi}'
                min_val = group_data[col].min()
                max_val = group_data[col].max()
                median_val = group_data[col].median()
                print(f"{group:<8} {hemi.upper():<12} {min_val:<8.0f} {max_val:<8.0f} {median_val:<8.1f}")

    # CNR descriptives (mean and SD per hemisphere per group)
    print(f"\n{'-'*70}")
    print("GREY/WHITE CNR (Mean ± SD by Group)")
    print(f"{'-'*70}")
    print(f"{'Group':<8} {'Hemisphere':<12} {'Mean':<12} {'SD':<12}")
    print(f"{'-'*70}")

    for group in ['ASC', 'CMP']:
        group_data = df_merged[df_merged['group'] == group]
        if len(group_data) > 0:
            for hemi in ['lh', 'rh']:
                col = f'cnr_{hemi}'
                mean_val = group_data[col].mean()
                sd_val = group_data[col].std()
                print(f"{group:<8} {hemi.upper():<12} {mean_val:<12.4f} {sd_val:<12.4f}")

    print(f"{'='*70}\n")


def main():
    """
    Main function to extract Freesurfer QC metrics for all subjects.
    """
    print("Extracting Freesurfer QC metrics...")
    print(f"Freesurfer directory: {freesurfer_dir}")
    print(f"Output file: {output_file}\n")

    # Initialize list to store results
    results = []

    # Get list of subject directories
    subject_dirs = sorted([d for d in Path(freesurfer_dir).iterdir()
                          if d.is_dir() and d.name.startswith('sub-')])

    print(f"Found {len(subject_dirs)} subjects\n")

    # Process each subject
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        print(f"Processing {subject_id}...")

        # Initialize result dictionary
        result = {'participant_id': subject_id}

        # Extract Euler numbers
        for hemi in ['lh', 'rh']:
            log_file = subject_dir / 'scripts' / f'recon-all-{hemi}.log'
            euler = extract_euler_number(log_file, hemi)
            result[f'euler_{hemi}'] = euler

        # Extract CNR values
        for hemi in ['lh', 'rh']:
            stats_file = subject_dir / 'surf' / f'autodet.gw.stats.{hemi}.dat'
            cnr = extract_cnr(stats_file, hemi)
            result[f'cnr_{hemi}'] = cnr

        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Calculate mean values across hemispheres
    df['euler_mean'] = df[['euler_lh', 'euler_rh']].mean(axis=1)
    df['cnr_mean'] = df[['cnr_lh', 'cnr_rh']].mean(axis=1)

    # Reorder columns
    columns = ['participant_id', 'euler_lh', 'euler_rh', 'euler_mean',
               'cnr_lh', 'cnr_rh', 'cnr_mean']
    df = df[columns]

    # Save to TSV
    df.to_csv(output_file, sep='\t', index=False, float_format='%.4f')

    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    print(f"Total subjects processed: {len(df)}")
    print(f"\nEuler Number (lower values indicate more topological defects):")
    print(df[['euler_lh', 'euler_rh', 'euler_mean']].describe())
    print(f"\nGrey/White CNR (higher values indicate better contrast):")
    print(df[['cnr_lh', 'cnr_rh', 'cnr_mean']].describe())
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    # Check for potential quality issues
    low_euler_threshold = -20  # Typical threshold for poor quality
    low_cnr_threshold = 2.0    # Typical threshold for poor contrast

    subjects_low_euler = df[df['euler_mean'] < low_euler_threshold]['participant_id'].tolist()
    subjects_low_cnr = df[df['cnr_mean'] < low_cnr_threshold]['participant_id'].tolist()

    if subjects_low_euler:
        print(f"Warning: {len(subjects_low_euler)} subjects with low Euler numbers (<{low_euler_threshold}):")
        print(f"  {', '.join(subjects_low_euler)}\n")

    if subjects_low_cnr:
        print(f"Warning: {len(subjects_low_cnr)} subjects with low CNR (<{low_cnr_threshold}):")
        print(f"  {', '.join(subjects_low_cnr)}\n")

    # Load participant data and report group-level descriptives
    print("Loading participant data for group-level analysis...")
    try:
        participants_df = pd.read_csv(participants_file, sep='\t')
        report_group_descriptives(df, participants_df)
    except FileNotFoundError:
        print(f"Warning: Participants file not found at {participants_file}")
        print("Skipping group-level analysis.\n")
    except Exception as e:
        print(f"Error loading participants file: {e}")
        print("Skipping group-level analysis.\n")

    print("Done!")


if __name__ == "__main__":
    main()
