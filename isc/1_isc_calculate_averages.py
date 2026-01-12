#!/usr/bin/env python3
"""
Calculate parcel-wise inter-subject correlation (ISC) for movie-watching fMRI.

Uses XCP-D preprocessed timeseries (4S456 parcellation).
Computes ISC separately for ASC and CMP groups, then averages by category:
- News: farmers, poland, royals
- Reality TV: interviews, fomo, firstdates

Author: Joe Bathelt
Date: December 2025
"""

import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from nilearn import plotting, datasets, surface

# Import shared ISC utilities
from isc_utils import load_participants, get_timeseries_file, compute_isc

# Set CMU Sans Serif as the default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['CMU Sans Serif', 'DejaVu Sans']

# Paths
XCPD_DIR = Path("/home/ASDPrecision/data/bids/derivatives/xcpd")
PARTICIPANTS_TSV = Path("/home/ASDPrecision/data/bids/participants.tsv")
OUTPUT_DIR = Path("/home/ASDPrecision/quality_metrics/isc_maps")
ATLAS_FILE = Path("/home/ASDPrecision/data/bids/derivatives/qsirecon/atlases/atlas-4S456Parcels/atlas-4S456Parcels_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz")
ATLAS_LABELS = Path("/home/ASDPrecision/data/bids/derivatives/qsirecon/atlases/atlas-4S456Parcels/atlas-4S456Parcels_dseg.tsv")

# Task categories
NEWS_TASKS = ['farmers', 'poland', 'royals']
REALITY_TASKS = ['interviews', 'fomo', 'firstdates']
ALL_TASKS = NEWS_TASKS + REALITY_TASKS


def load_participants():
    """Load participant info and split by diagnosis."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    
    asc_subs = df[df['autism_diagnosis'] == 'yes']['participant_id'].tolist()
    cmp_subs = df[df['autism_diagnosis'] == 'no']['participant_id'].tolist()
    
    print(f"ASC subjects: {len(asc_subs)}")
    print(f"CMP subjects: {len(cmp_subs)}")
    
    return asc_subs, cmp_subs


def get_timeseries_file(subject, task):
    """Find XCP-D timeseries file for a subject and task."""
    pattern = f"{subject}_task-{task}_run-*_space-fsLR_seg-4S456Parcels_stat-mean_timeseries.tsv"
    files = glob.glob(str(XCPD_DIR / subject / "func" / pattern))
    
    if len(files) == 1:
        return files[0]
    elif len(files) > 1:
        return sorted(files)[0]
    return None


def compute_isc(data_list):
    """
    Compute leave-one-out ISC for a list of subjects using fully vectorized operations.
    
    Parameters:
    -----------
    data_list : list of np.ndarray
        List of 2D arrays (timepoints x parcels) for each subject
    
    Returns:
    --------
    isc_values : np.ndarray
        1D array of ISC values for each parcel
    """
    n_subjects = len(data_list)
    if n_subjects < 2:
        return None
    
    # Stack all subjects: shape (n_subjects, timepoints, parcels)
    data_stack = np.array(data_list)
    n_timepoints = data_stack.shape[1]
    
    # Z-score each subject's data along time axis (axis=1)
    # Shape: (n_subjects, timepoints, parcels)
    means = data_stack.mean(axis=1, keepdims=True)
    stds = data_stack.std(axis=1, keepdims=True) + 1e-8
    data_zscore = (data_stack - means) / stds
    
    # Compute sum of all subjects for efficient leave-one-out mean calculation
    total_sum = data_zscore.sum(axis=0)  # (timepoints, parcels)
    
    # For each subject, mean of others = (total_sum - subject) / (n_subjects - 1)
    # Correlation with mean of others for all subjects at once
    # r = (1/T) * sum(z_i * z_others_mean) for each subject
    
    isc_per_subject = np.zeros((n_subjects, data_stack.shape[2]))
    
    for i in range(n_subjects):
        # Mean of others (excluding subject i)
        mean_others = (total_sum - data_zscore[i]) / (n_subjects - 1)
        
        # Re-standardize the mean of others
        mean_others_zscore = (mean_others - mean_others.mean(axis=0)) / (mean_others.std(axis=0) + 1e-8)
        
        # Correlation = mean of element-wise product
        isc_per_subject[i] = np.mean(data_zscore[i] * mean_others_zscore, axis=0)
    
    # Average ISC across subjects
    isc_values = isc_per_subject.mean(axis=0)
    
    return isc_values


def compute_isc_for_task(subjects, task):
    """
    Compute ISC for a specific task and group of subjects.
    
    Returns:
    --------
    isc_values : np.ndarray or None
        ISC values per parcel, or None if insufficient data
    parcel_names : list or None
        List of parcel names
    n_subjects : int
        Number of subjects with data for this task
    """
    data_list = []
    parcel_names = None
    
    for sub in subjects:
        ts_file = get_timeseries_file(sub, task, XCPD_DIR)
        if ts_file is None:
            continue
        
        try:
            df = pd.read_csv(ts_file, sep='\t')
            if parcel_names is None:
                parcel_names = df.columns.tolist()
            
            data_list.append(df.values)
        except Exception as e:
            print(f"  Error loading {sub} {task}: {e}")
            continue
    
    n_subjects = len(data_list)
    if n_subjects < 2:
        print(f"  Insufficient subjects for {task}: {n_subjects}")
        return None, None, n_subjects
    
    # Truncate to minimum timepoints
    min_timepoints = min(d.shape[0] for d in data_list)
    data_list = [d[:min_timepoints, :] for d in data_list]
    
    print(f"  {task}: {n_subjects} subjects, {min_timepoints} timepoints, {len(parcel_names)} parcels")
    
    isc_values = compute_isc(data_list)
    
    return isc_values, parcel_names, n_subjects


def main():
    """Main function to compute ISC."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load participant groups
    asc_subs, cmp_subs = load_participants()
    
    # Store ISC results
    isc_results = {
        'ASC': {task: None for task in ALL_TASKS},
        'CMP': {task: None for task in ALL_TASKS}
    }
    parcel_names = None
    
    # Compute ISC for each task and group
    for group_name, subjects in [('ASC', asc_subs), ('CMP', cmp_subs)]:
        print(f"\nProcessing {group_name} group...")
        
        for task in ALL_TASKS:
            print(f"  Computing ISC for {task}...")
            isc_values, parcels, n_subs = compute_isc_for_task(subjects, task)
            
            if isc_values is not None:
                isc_results[group_name][task] = isc_values
                if parcel_names is None:
                    parcel_names = parcels
                
                # Save individual task ISC
                out_df = pd.DataFrame({
                    'parcel': parcel_names,
                    'isc': isc_values
                })
                out_file = OUTPUT_DIR / f"isc_{group_name}_{task}.tsv"
                out_df.to_csv(out_file, sep='\t', index=False)
                print(f"    Saved: {out_file.name}")
    
    # Average by category
    print("\nAveraging by category...")
    for group_name in ['ASC', 'CMP']:
        # News category
        news_iscs = [isc_results[group_name][t] for t in NEWS_TASKS 
                     if isc_results[group_name][t] is not None]
        if news_iscs:
            news_avg = np.mean(news_iscs, axis=0)
            out_df = pd.DataFrame({'parcel': parcel_names, 'isc': news_avg})
            out_file = OUTPUT_DIR / f"isc_{group_name}_news_average.tsv"
            out_df.to_csv(out_file, sep='\t', index=False)
            print(f"  Saved: {out_file.name} (n={len(news_iscs)} tasks)")
        
        # Reality TV category
        reality_iscs = [isc_results[group_name][t] for t in REALITY_TASKS 
                        if isc_results[group_name][t] is not None]
        if reality_iscs:
            reality_avg = np.mean(reality_iscs, axis=0)
            out_df = pd.DataFrame({'parcel': parcel_names, 'isc': reality_avg})
            out_file = OUTPUT_DIR / f"isc_{group_name}_reality_average.tsv"
            out_df.to_csv(out_file, sep='\t', index=False)
            print(f"  Saved: {out_file.name} (n={len(reality_iscs)} tasks)")
        
        # Overall average
        all_iscs = [isc_results[group_name][t] for t in ALL_TASKS 
                    if isc_results[group_name][t] is not None]
        if all_iscs:
            all_avg = np.mean(all_iscs, axis=0)
            out_df = pd.DataFrame({'parcel': parcel_names, 'isc': all_avg})
            out_file = OUTPUT_DIR / f"isc_{group_name}_all_average.tsv"
            out_df.to_csv(out_file, sep='\t', index=False)
            print(f"  Saved: {out_file.name} (n={len(all_iscs)} tasks)")
    
    print(f"\nDone! ISC results saved to: {OUTPUT_DIR}")
    
    # Visualize the average ISC maps
    visualize_isc_maps(parcel_names)
    
    # Create CIFTI dscalar files for surface visualization
    create_cifti_dscalar_files()


def visualize_isc_maps(parcel_names):
    """Create brain visualizations of average ISC maps using nilearn."""
    print("\nCreating ISC visualizations...")
    
    # Load atlas
    atlas_img = nib.load(ATLAS_FILE)
    atlas_data = atlas_img.get_fdata().astype(int)
    
    # Load atlas labels to get index mapping
    labels_df = pd.read_csv(ATLAS_LABELS, sep='\t')
    label_to_index = dict(zip(labels_df['label'], labels_df['index']))
    
    # Figure dimensions (mm to inches at 300 DPI)
    fig_width_mm = 180
    fig_width_in = fig_width_mm / 25.4
    
    # Visualizations to create
    viz_configs = [
        ('ASC', 'all_average', 'ASC - All Videos'),
        ('CMP', 'all_average', 'CMP - All Videos'),
        ('ASC', 'news_average', 'ASC - News'),
        ('CMP', 'news_average', 'CMP - News'),
        ('ASC', 'reality_average', 'ASC - Reality TV'),
        ('CMP', 'reality_average', 'CMP - Reality TV'),
    ]
    
    for group, category, title in viz_configs:
        isc_file = OUTPUT_DIR / f"isc_{group}_{category}.tsv"
        if not isc_file.exists():
            continue
        
        # Load ISC values
        isc_df = pd.read_csv(isc_file, sep='\t')
        
        # Create ISC volume by replacing parcel indices with ISC values
        isc_volume = np.zeros_like(atlas_data, dtype=np.float64)
        for _, row in isc_df.iterrows():
            parcel = row['parcel']
            isc_val = row['isc']
            if parcel in label_to_index:
                idx = label_to_index[parcel]
                isc_volume[atlas_data == idx] = isc_val
        
        # Create NIfTI image with ISC values
        isc_img = nib.Nifti1Image(isc_volume, atlas_img.affine)
        
        # Save as NIfTI
        nii_file = OUTPUT_DIR / f"isc_{group}_{category}.nii.gz"
        nib.save(isc_img, nii_file)
        
        # Create visualization using nilearn's plot_img for continuous data
        # Use the ISC image itself as the background to avoid resampling issues
        fig = plt.figure(figsize=(fig_width_in, fig_width_in / 3))
        
        display = plotting.plot_img(
            isc_img,
            display_mode='ortho',
            cut_coords=(0, -20, 10),
            cmap='YlOrRd',
            colorbar=True,
            title=title,
            vmin=0,
            vmax=0.6,
            figure=fig,
            black_bg=False,
            resampling_interpolation='nearest'
        )
        
        # Save figure
        out_file = OUTPUT_DIR / f"isc_{group}_{category}.png"
        fig.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {out_file.name}")
    
    # Create comparison figure (ASC vs CMP for all videos)
    create_comparison_figure(label_to_index, atlas_data, atlas_img)


def create_comparison_figure(label_to_index, atlas_data, atlas_img):
    """Create side-by-side comparison of ASC vs CMP ISC."""
    print("\nCreating comparison figure...")
    
    # Load both group averages
    asc_df = pd.read_csv(OUTPUT_DIR / "isc_ASC_all_average.tsv", sep='\t')
    cmp_df = pd.read_csv(OUTPUT_DIR / "isc_CMP_all_average.tsv", sep='\t')
    
    # Create ISC volumes
    asc_volume = np.zeros_like(atlas_data, dtype=np.float64)
    cmp_volume = np.zeros_like(atlas_data, dtype=np.float64)
    
    for _, row in asc_df.iterrows():
        parcel = row['parcel']
        if parcel in label_to_index:
            idx = label_to_index[parcel]
            asc_volume[atlas_data == idx] = row['isc']
    
    for _, row in cmp_df.iterrows():
        parcel = row['parcel']
        if parcel in label_to_index:
            idx = label_to_index[parcel]
            cmp_volume[atlas_data == idx] = row['isc']
    
    asc_img = nib.Nifti1Image(asc_volume, atlas_img.affine)
    cmp_img = nib.Nifti1Image(cmp_volume, atlas_img.affine)
    
    # Figure dimensions
    fig_width_mm = 180
    fig_width_in = fig_width_mm / 25.4
    
    # Common parameters
    vmax = 0.6
    cut_coords = (0, -20, 10)
    
    # ASC
    fig_asc = plt.figure(figsize=(fig_width_in, fig_width_in / 4))
    plotting.plot_img(
        asc_img,
        display_mode='ortho',
        cut_coords=cut_coords,
        cmap='YlOrRd',
        colorbar=True,
        title='ASC',
        vmin=0,
        vmax=vmax,
        figure=fig_asc,
        black_bg=False,
        resampling_interpolation='nearest'
    )
    fig_asc.savefig(OUTPUT_DIR / "isc_ASC_all_average_ortho.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_asc)
    
    # CMP
    fig_cmp = plt.figure(figsize=(fig_width_in, fig_width_in / 4))
    plotting.plot_img(
        cmp_img,
        display_mode='ortho',
        cut_coords=cut_coords,
        cmap='YlOrRd',
        colorbar=True,
        title='CMP',
        vmin=0,
        vmax=vmax,
        figure=fig_cmp,
        black_bg=False,
        resampling_interpolation='nearest'
    )
    fig_cmp.savefig(OUTPUT_DIR / "isc_CMP_all_average_ortho.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_cmp)
    print("  Saved: isc_ASC_all_average_ortho.png")
    print("  Saved: isc_CMP_all_average_ortho.png")
    
    # Also create difference map
    diff_volume = asc_volume - cmp_volume
    diff_img = nib.Nifti1Image(diff_volume, atlas_img.affine)
    nib.save(diff_img, OUTPUT_DIR / "isc_difference_ASC_minus_CMP.nii.gz")
    
    fig_diff = plt.figure(figsize=(fig_width_in, fig_width_in / 4))
    plotting.plot_img(
        diff_img,
        display_mode='ortho',
        cut_coords=cut_coords,
        cmap='RdBu_r',
        colorbar=True,
        title='ISC Difference (ASC - CMP)',
        vmin=-0.15,
        vmax=0.15,
        figure=fig_diff,
        black_bg=False,
        resampling_interpolation='nearest'
    )
    
    out_file = OUTPUT_DIR / "isc_difference_ASC_minus_CMP.png"
    fig_diff.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_diff)
    print(f"  Saved: {out_file.name}")


def create_cifti_dscalar_files():
    """Create CIFTI dscalar files for surface visualization in Workbench."""
    print("\nCreating CIFTI dscalar files for surface visualization...")
    
    from nibabel.cifti2 import Cifti2Image, Cifti2Header
    from nibabel.cifti2.cifti2_axes import ScalarAxis
    
    # Load the parcellation CIFTI
    dlabel_file = XCPD_DIR / "atlases/atlas-4S456Parcels/atlas-4S456Parcels_space-fsLR_den-91k_dseg.dlabel.nii"
    dlabel = nib.load(dlabel_file)
    parc_data = dlabel.get_fdata().squeeze()
    
    # Get label mapping from CIFTI
    header = dlabel.header
    label_axis = header.get_axis(0)
    label_dict = label_axis.label[0]
    brain_axis = header.get_axis(1)
    
    # Create mapping: ISC parcel name -> CIFTI index
    cifti_name_to_idx = {}
    for idx, (name, color) in label_dict.items():
        if idx == 0:
            continue
        isc_name = name.replace('7Networks_', '')
        cifti_name_to_idx[isc_name] = idx
    
    # Files to convert
    isc_files = [
        ('ASC', 'all_average'),
        ('CMP', 'all_average'),
        ('ASC', 'news_average'),
        ('CMP', 'news_average'),
        ('ASC', 'reality_average'),
        ('CMP', 'reality_average'),
    ]
    
    for group, category in isc_files:
        tsv_file = OUTPUT_DIR / f"isc_{group}_{category}.tsv"
        if not tsv_file.exists():
            continue
        
        # Load ISC data
        isc_df = pd.read_csv(tsv_file, sep='\t')
        isc_dict = dict(zip(isc_df['parcel'], isc_df['isc']))
        
        # Map ISC values to vertices
        isc_data = np.zeros_like(parc_data, dtype=np.float32)
        for parcel_name, isc_val in isc_dict.items():
            if parcel_name in cifti_name_to_idx:
                parcel_idx = cifti_name_to_idx[parcel_name]
                isc_data[parc_data == parcel_idx] = isc_val
        
        # Create scalar axis
        map_name = f"ISC_{group}_{category}"
        scalar_axis = ScalarAxis([map_name])
        
        # Create new header and image
        new_header = Cifti2Header.from_axes((scalar_axis, brain_axis))
        isc_cifti = Cifti2Image(isc_data.reshape(1, -1), new_header)
        
        # Save
        out_file = OUTPUT_DIR / f"isc_{group}_{category}.dscalar.nii"
        nib.save(isc_cifti, out_file)
        print(f"  Saved: {out_file.name}")
    
    # Also create difference map
    asc_df = pd.read_csv(OUTPUT_DIR / "isc_ASC_all_average.tsv", sep='\t')
    cmp_df = pd.read_csv(OUTPUT_DIR / "isc_CMP_all_average.tsv", sep='\t')
    
    asc_dict = dict(zip(asc_df['parcel'], asc_df['isc']))
    cmp_dict = dict(zip(cmp_df['parcel'], cmp_df['isc']))
    
    diff_data = np.zeros_like(parc_data, dtype=np.float32)
    for parcel_name in asc_dict:
        if parcel_name in cifti_name_to_idx and parcel_name in cmp_dict:
            parcel_idx = cifti_name_to_idx[parcel_name]
            diff_data[parc_data == parcel_idx] = asc_dict[parcel_name] - cmp_dict[parcel_name]
    
    scalar_axis = ScalarAxis(["ISC_difference_ASC_minus_CMP"])
    new_header = Cifti2Header.from_axes((scalar_axis, brain_axis))
    diff_cifti = Cifti2Image(diff_data.reshape(1, -1), new_header)
    
    out_file = OUTPUT_DIR / "isc_difference_ASC_minus_CMP.dscalar.nii"
    nib.save(diff_cifti, out_file)
    print(f"  Saved: {out_file.name}")


if __name__ == "__main__":
    main()
