#!/usr/bin/env python3
"""
Calculate temporal Signal-to-Noise Ratio (tSNR) maps for all participants.

This script:
1. Calculates individual tSNR maps for each functional run
2. Averages tSNR maps across all runs for each subject
3. Creates group-average tSNR maps (ASC, CMP) to visualize spatial patterns

tSNR is calculated as: mean(timeseries) / std(timeseries)
Higher tSNR values indicate better data quality with more stable signal over time.

Author: Joe Bathelt
Date: December 2025
"""

import os
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.masking import apply_mask, unmask

# Set CMU Sans Serif as the default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['CMU Sans Serif', 'DejaVu Sans']

# Configuration
base_dir = Path("/home/ASDPrecision")
bids_dir = base_dir / "data" / "bids"
fmriprep_dir = bids_dir / "derivatives" / "fmriprep"
participants_file = bids_dir / "participants.tsv"
output_dir = base_dir / "quality_metrics" / "tsnr"
log_file = output_dir / "tsnr_calculation_log.txt"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir / "individual_runs", exist_ok=True)
os.makedirs(output_dir / "subject_averages", exist_ok=True)
os.makedirs(output_dir / "group_averages", exist_ok=True)
os.makedirs(output_dir / "summary", exist_ok=True)

# Remove old log file if exists
if log_file.exists():
    os.remove(log_file)


def log(message, print_terminal=True):
    """Write a message to log file and optionally print to terminal."""
    with open(log_file, "a") as f:
        f.write(message + "\n")
    if print_terminal:
        print(message)


def calculate_tsnr(bold_file):
    """
    Calculate tSNR map from a 4D BOLD NIfTI file.

    Parameters:
    -----------
    bold_file : str or Path
        Path to the preprocessed BOLD NIfTI file

    Returns:
    --------
    tsnr_data : numpy array
        3D array of tSNR values
    affine : numpy array
        Affine transformation matrix from the input file
    """
    # Load the 4D BOLD image
    img = nib.load(bold_file)
    data = img.get_fdata()

    # Calculate mean and std across time (4th dimension)
    mean_signal = np.mean(data, axis=3)
    std_signal = np.std(data, axis=3)

    # Calculate tSNR (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        tsnr_data = mean_signal / std_signal
        tsnr_data[~np.isfinite(tsnr_data)] = 0

    return tsnr_data, img.affine


def process_single_run(bold_file, output_dir):
    """
    Process a single functional run to calculate and save tSNR map.

    Parameters:
    -----------
    bold_file : Path
        Path to the preprocessed BOLD NIfTI file
    output_dir : Path
        Directory to save individual run tSNR maps

    Returns:
    --------
    dict with tsnr_data and affine, or None if error
    """
    try:
        # Check if tSNR map already exists
        filename = bold_file.name
        tsnr_filename = filename.replace('_desc-preproc_bold.nii.gz', '_desc-tsnr.nii.gz')
        tsnr_output_path = output_dir / "individual_runs" / tsnr_filename

        if tsnr_output_path.exists():
            # Load existing tSNR map instead of recalculating
            tsnr_img = nib.load(tsnr_output_path)
            tsnr_data = tsnr_img.get_fdata()
            affine = tsnr_img.affine
            return {'tsnr_data': tsnr_data, 'affine': affine, 'filename': filename, 'skipped': True}

        # Calculate tSNR
        tsnr_data, affine = calculate_tsnr(bold_file)

        # Save individual run tSNR map
        tsnr_img = nib.Nifti1Image(tsnr_data, affine)
        nib.save(tsnr_img, tsnr_output_path)

        return {'tsnr_data': tsnr_data, 'affine': affine, 'filename': filename, 'skipped': False}

    except Exception as e:
        return {'error': str(e), 'filename': bold_file.name}


def plot_group_tsnr_maps(asc_map_path, comparison_map_path, output_path, log_func):
    """
    Create a figure showing group-average tSNR maps for ASC and Comparison groups.

    Parameters:
    -----------
    asc_map_path : Path
        Path to the ASC group average tSNR map
    comparison_map_path : Path
        Path to the Comparison group average tSNR map
    output_path : Path
        Path to save the output figure
    log_func : function
        Logging function to use for messages
    """
    log_func("\nCreating group tSNR visualization...")

    # Load the MNI152NLin2009cAsym brain mask from fMRIPrep output
    # Using a mask from the actual fMRIPrep output ensures it matches the exact
    # space and resolution of the preprocessed data (2x2x2.2mm anisotropic)
    mni_mask_path = Path("/home/ASDPrecision/data/bids/derivatives/fmriprep/sub-001/func/sub-001_task-fomo_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
    mni_mask = nib.load(mni_mask_path)

    # Load the tSNR maps
    asc_img = nib.load(asc_map_path)
    comparison_img = nib.load(comparison_map_path)

    # Apply MNI mask to the tSNR maps
    asc_masked = apply_mask(asc_img, mni_mask)
    comparison_masked = apply_mask(comparison_img, mni_mask)

    # Unmask back to image
    asc_img_masked = unmask(asc_masked, mni_mask)
    comparison_img_masked = unmask(comparison_masked, mni_mask)

    # Determine common color scale based on both groups
    vmin = 10
    vmax = 120

    log_func(f"  Color scale range: {vmin:.1f} - {vmax:.1f}")

    # Create figure with GridSpec for custom layout
    # Convert 90mm to inches (1 inch = 25.4mm)
    fig_width = 90 / 25.4  # ~3.54 inches
    fig_height = fig_width * 0.6  # Reduced height for 2 rows

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create a grid: 2 rows for plots, 1 column for main plot, 1 for colorbar
    from matplotlib.gridspec import GridSpec
    # Main grid with space for colorbar on the right
    gs_main = GridSpec(2, 2, figure=fig, width_ratios=[1, 0.08], hspace=0.1, wspace=0.05)

    # Nested grid for colorbar to center it vertically (2/3 height)
    # The colorbar should span 2/3 of total height, centered
    gs_cbar = gs_main[0:2, 1].subgridspec(6, 1, hspace=0)

    # Create subplot axes
    ax1 = fig.add_subplot(gs_main[0, 0])  # Top row: ASC
    ax2 = fig.add_subplot(gs_main[1, 0])  # Bottom row: Comparison

    # Colorbar in middle 4 cells (2/3 of 6 cells), with 1 cell padding on each side
    cbar_ax = fig.add_subplot(gs_cbar[1:5, 0])

    # Plot ASC group (top) - ortho view without colorbar
    display_asc = plotting.plot_stat_map(
        asc_img_masked,
        display_mode='ortho',
        vmin=vmin,
        vmax=vmax,
        cmap='inferno',
        colorbar=False,
        axes=ax1,
        annotate=False,
        draw_cross=False
    )

    # Plot Comparison group (bottom) - ortho view without colorbar
    display_comparison = plotting.plot_stat_map(
        comparison_img_masked,
        display_mode='ortho',
        vmin=vmin,
        vmax=vmax,
        cmap='inferno',
        colorbar=False,
        axes=ax2,
        annotate=False,
        draw_cross=False
    )

    # Add a single shared colorbar
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('tSNR', rotation=270, labelpad=15, fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    log_func(f"  Group tSNR visualization saved to: {output_path}")


def main():
    log("=" * 80)
    log("Starting tSNR calculation and group averaging")
    log("=" * 80)

    # Determine number of CPUs to use
    n_cpus = max(1, cpu_count() - 1)  # Leave one CPU free
    log(f"Using {n_cpus} CPUs for parallel processing")

    # Load participants data
    log(f"\nLoading participants data from: {participants_file}")
    participants_df = pd.read_csv(participants_file, sep='\t')
    log(f"Found {len(participants_df)} participants")

    # Get list of all subjects in fmriprep directory
    subject_dirs = sorted([d for d in fmriprep_dir.glob("sub-*") if d.is_dir()])
    log(f"\nFound {len(subject_dirs)} subject directories in fmriprep output")

    # Storage for group-level averaging
    asc_maps = []
    comparison_maps = []
    affine_template = None

    # Storage for summary statistics
    results = []

    # Process each subject
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        subject_id = subject_dir.name

        # Get autism diagnosis for this subject
        subj_data = participants_df[participants_df['participant_id'] == subject_id]
        if subj_data.empty:
            log(f"Warning: {subject_id} not found in participants.tsv, skipping")
            continue

        autism_diagnosis = subj_data['autism_diagnosis'].values[0]

        # Find all functional runs for this subject
        func_dir = subject_dir / "func"
        if not func_dir.exists():
            log(f"Warning: No func directory for {subject_id}")
            continue

        # Find all preprocessed BOLD files
        bold_files = sorted(func_dir.glob("*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))

        if len(bold_files) == 0:
            log(f"Warning: No preprocessed BOLD files found for {subject_id}")
            continue

        log(f"\n{subject_id} ({autism_diagnosis}): Processing {len(bold_files)} functional runs in parallel")

        # Storage for this subject's tSNR maps
        subject_tsnr_maps = []

        # Process all runs in parallel
        process_func = partial(process_single_run, output_dir=output_dir)

        with Pool(n_cpus) as pool:
            run_results = pool.map(process_func, bold_files)

        # Collect results
        n_skipped = 0
        n_calculated = 0
        for result in run_results:
            if 'error' in result:
                log(f"  ERROR processing {result['filename']}: {result['error']}")
            else:
                # Store affine for later use (should be same for all files)
                if affine_template is None:
                    affine_template = result['affine']

                # Store for subject averaging
                subject_tsnr_maps.append(result['tsnr_data'])

                # Track skipped vs calculated
                if result.get('skipped', False):
                    n_skipped += 1
                else:
                    n_calculated += 1

        if n_skipped > 0 or n_calculated > 0:
            log(f"  Processed {n_calculated} runs, loaded {n_skipped} existing maps")

        if len(subject_tsnr_maps) == 0:
            log(f"  No valid tSNR maps for {subject_id}")
            continue

        # Calculate subject-average tSNR map
        subject_avg_tsnr = np.mean(subject_tsnr_maps, axis=0)

        # Calculate summary statistics for this subject
        brain_mask = subject_avg_tsnr > 0
        tsnr_brain = subject_avg_tsnr[brain_mask]

        mean_tsnr = np.mean(tsnr_brain)
        median_tsnr = np.median(tsnr_brain)
        std_tsnr = np.std(tsnr_brain)

        log(f"  Subject average - Mean tSNR: {mean_tsnr:.2f}, Median: {median_tsnr:.2f}")

        # Save subject-average tSNR map
        subject_avg_filename = f"{subject_id}_desc-tsnr_average.nii.gz"
        subject_avg_path = output_dir / "subject_averages" / subject_avg_filename
        subject_avg_img = nib.Nifti1Image(subject_avg_tsnr, affine_template)
        nib.save(subject_avg_img, subject_avg_path)

        # Add to group lists
        if autism_diagnosis.lower() == 'yes':
            asc_maps.append(subject_avg_tsnr)
        else:
            comparison_maps.append(subject_avg_tsnr)

        # Store results
        results.append({
            'subject_id': subject_id,
            'autism_diagnosis': autism_diagnosis,
            'n_runs': len(subject_tsnr_maps),
            'mean_tsnr': mean_tsnr,
            'median_tsnr': median_tsnr,
            'std_tsnr': std_tsnr,
            'n_voxels': np.sum(brain_mask)
        })

    # Create group-average maps
    log("\n" + "=" * 80)
    log("Creating group-average tSNR maps")
    log("=" * 80)

    asc_group_path = None
    comparison_group_path = None

    if len(asc_maps) > 0:
        asc_group_avg = np.mean(asc_maps, axis=0)
        asc_group_path = output_dir / "group_averages" / "group-ASC_desc-tsnr_average.nii.gz"
        asc_group_img = nib.Nifti1Image(asc_group_avg, affine_template)
        nib.save(asc_group_img, asc_group_path)

        brain_mask_asc = asc_group_avg > 0
        log(f"ASC group (n={len(asc_maps)}): Mean tSNR = {np.mean(asc_group_avg[brain_mask_asc]):.2f}")
        log(f"  Saved to: {asc_group_path}")
    else:
        log("Warning: No ASC subjects processed!")

    if len(comparison_maps) > 0:
        comparison_group_avg = np.mean(comparison_maps, axis=0)
        comparison_group_path = output_dir / "group_averages" / "group-Comparison_desc-tsnr_average.nii.gz"
        comparison_group_img = nib.Nifti1Image(comparison_group_avg, affine_template)
        nib.save(comparison_group_img, comparison_group_path)

        brain_mask_comparison = comparison_group_avg > 0
        log(f"Comparison group (n={len(comparison_maps)}): Mean tSNR = {np.mean(comparison_group_avg[brain_mask_comparison]):.2f}")
        log(f"  Saved to: {comparison_group_path}")
    else:
        log("Warning: No comparison subjects processed!")

    # Create difference map (ASC - Comparison)
    if len(asc_maps) > 0 and len(comparison_maps) > 0:
        diff_map = asc_group_avg - comparison_group_avg
        diff_path = output_dir / "group_averages" / "group-ASC-vs-Comparison_desc-tsnr_difference.nii.gz"
        diff_img = nib.Nifti1Image(diff_map, affine_template)
        nib.save(diff_img, diff_path)
        log(f"\nDifference map (ASC - Comparison) saved to: {diff_path}")

        # Create visualization of group maps
        figure_path = output_dir / "group_averages" / "group_tsnr_comparison.png"
        try:
            plot_group_tsnr_maps(asc_group_path, comparison_group_path, figure_path, log)
        except Exception as e:
            log(f"Warning: Could not create group visualization: {str(e)}")

    # Save summary statistics
    results_df = pd.DataFrame(results)
    summary_file = output_dir / "summary" / "tsnr_summary_by_subject.tsv"
    results_df.to_csv(summary_file, sep='\t', index=False)
    log(f"\nSaved subject summary statistics to: {summary_file}")

    # Calculate group statistics
    group_stats = results_df.groupby('autism_diagnosis').agg({
        'mean_tsnr': ['mean', 'std', 'count'],
        'median_tsnr': ['mean', 'std']
    }).round(2)

    group_stats_file = output_dir / "summary" / "tsnr_summary_by_group.tsv"
    group_stats.to_csv(group_stats_file, sep='\t')
    log(f"Saved group summary statistics to: {group_stats_file}")

    log("\n" + "=" * 80)
    log("tSNR calculation completed successfully!")
    log(f"Total subjects processed: {len(results_df)}")
    log(f"  - ASC: {len(asc_maps)}")
    log(f"  - Comparison: {len(comparison_maps)}")
    log(f"\nOutputs:")
    log(f"  - Individual run tSNR maps: {output_dir / 'individual_runs'}")
    log(f"  - Subject-average tSNR maps: {output_dir / 'subject_averages'}")
    log(f"  - Group-average tSNR maps: {output_dir / 'group_averages'}")
    log(f"  - Summary statistics: {output_dir / 'summary'}")
    log("=" * 80)


if __name__ == "__main__":
    main()
