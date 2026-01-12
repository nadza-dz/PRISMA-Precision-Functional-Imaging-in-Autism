#!/usr/bin/env python3
"""
Extract group-level CSD-based QC maps from qsirecon output.

Creates maps analogous to direction-modulated FA for FOD/CSD data:
- Multi-tissue normalization maps (mean, std, CV)
- Apparent Fiber Density (AFD) maps
- Peak FOD amplitude maps
- Direction-Encoded Color (DEC) maps

Usage:
    python -m dwi.qc_maps

Author: Joe Bathelt
Date: December 2025
"""

import os
import tempfile

import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import plotting

from .dwi_utils import run_docker_cmd, MRTRIX_DOCKER_IMAGE, ANTS_DOCKER_IMAGE

# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['CMU Sans Serif', 'DejaVu Sans']

# MNI template for reference
MNI_TEMPLATE = "/home/ASDPrecision/data/bids/derivatives/qsirecon/atlases/atlas-4S156Parcels/atlas-4S156Parcels_space-MNI152NLin2009cAsym_res-01_dseg.nii.gz"


def run_mrtrix_cmd(cmd: str, bind_paths: list = None, check: bool = True):
    """Run an MRtrix3 command using Docker."""
    return run_docker_cmd(MRTRIX_DOCKER_IMAGE, cmd, bind_paths, check)


def run_ants_cmd(cmd: str, bind_paths: list = None, check: bool = True):
    """Run an ANTs command using Docker."""
    return run_docker_cmd(ANTS_DOCKER_IMAGE, cmd, bind_paths, check)


def warp_to_mni(input_file: Path, output_file: Path, transform_file: Path, 
                reference_file: Path) -> bool:
    """Warp an image from ACPC space to MNI space using ANTs."""
    bind_paths = list(set([
        str(input_file.parent),
        str(output_file.parent),
        str(transform_file.parent),
        str(reference_file.parent)
    ]))
    
    img = nib.load(input_file)
    is_4d = img.ndim == 4 and img.shape[3] > 1
    
    cmd = f"antsApplyTransforms -d 3 "
    if is_4d:
        cmd += "-e 3 "
    
    cmd += (
        f"-i {input_file} "
        f"-r {reference_file} "
        f"-o {output_file} "
        f"-t {transform_file} "
        f"-n Linear"
    )
    
    try:
        run_ants_cmd(cmd, bind_paths=bind_paths)
        return True
    except RuntimeError as e:
        print(f"    Warning: Warp to MNI failed: {e}")
        return False


def extract_afd_and_peak(fod_file: Path, output_dir: Path, subject: str) -> dict:
    """Extract AFD and peak amplitude maps from WM FOD image using MRtrix3."""
    output_files = {}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        fixel_dir = Path(tmpdir) / 'fixels'
        bind_paths = [str(fod_file.parent), str(output_dir), tmpdir]
        
        # Convert FOD to fixel format
        cmd = f"fod2fixel {fod_file} {fixel_dir} -afd afd.mif -peak_amp peak.mif -force -quiet"
        try:
            run_mrtrix_cmd(cmd, bind_paths=bind_paths)
        except RuntimeError:
            print(f"    Warning: fod2fixel failed for {subject}")
            return output_files
        
        # Convert AFD fixels to voxel-wise map
        afd_output = output_dir / f'{subject}_afd_total.nii.gz'
        cmd = f"fixel2voxel {fixel_dir}/afd.mif sum {afd_output} -force -quiet"
        try:
            run_mrtrix_cmd(cmd, bind_paths=bind_paths)
            output_files['afd'] = afd_output
            print(f"    ✓ Created AFD map")
        except RuntimeError:
            print(f"    Warning: fixel2voxel (AFD) failed")
        
        # Convert peak amplitude to voxel-wise map
        peak_output = output_dir / f'{subject}_peak_amplitude.nii.gz'
        cmd = f"fixel2voxel {fixel_dir}/peak.mif max {peak_output} -force -quiet"
        try:
            run_mrtrix_cmd(cmd, bind_paths=bind_paths)
            output_files['peak'] = peak_output
            print(f"    ✓ Created peak amplitude map")
        except RuntimeError:
            print(f"    Warning: fixel2voxel (peak) failed")
        
        # Extract peak directions for DEC map
        peaks_output = output_dir / f'{subject}_peaks.nii.gz'
        cmd = f"fod2fixel {fod_file} {fixel_dir}_peaks -peak_amp peak.mif -force -quiet"
        try:
            run_mrtrix_cmd(cmd, bind_paths=bind_paths)
            cmd = f"fixel2peaks {fixel_dir}_peaks/directions.mif {peaks_output} -number 1 -force -quiet"
            run_mrtrix_cmd(cmd, bind_paths=bind_paths)
            output_files['peaks'] = peaks_output
            print(f"    ✓ Created peaks direction map")
        except RuntimeError:
            print(f"    Warning: Peak direction extraction failed")
    
    return output_files


def create_dec_map(peaks_file: Path, amplitude_file: Path, output_file: Path) -> bool:
    """Create a Direction-Encoded Color (DEC) map from peak directions and amplitude."""
    try:
        peaks_img = nib.load(peaks_file)
        peaks_data = peaks_img.get_fdata()
        amp_img = nib.load(amplitude_file)
        amp_data = amp_img.get_fdata()
        
        if peaks_data.ndim == 4 and peaks_data.shape[3] >= 3:
            r = np.abs(peaks_data[..., 0])
            g = np.abs(peaks_data[..., 1])
            b = np.abs(peaks_data[..., 2])
        else:
            print(f"    Warning: Unexpected peaks shape: {peaks_data.shape}")
            return False
        
        amp_norm = amp_data / (np.percentile(amp_data[amp_data > 0], 99) + 1e-10)
        amp_norm = np.clip(amp_norm, 0, 1)
        
        rgb = np.stack([r * amp_norm, g * amp_norm, b * amp_norm], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        
        dec_img = nib.Nifti1Image(rgb, peaks_img.affine)
        nib.save(dec_img, output_file)
        return True
    except Exception as e:
        print(f"    Warning: DEC map creation failed: {e}")
        return False


def compute_group_statistics(image_list: list, output_prefix: Path, metric_name: str):
    """Compute group-level statistics (mean, std, CV) from a list of images."""
    if not image_list:
        print(f"  No {metric_name} images to process")
        return
    
    print(f"\n  Computing group statistics for {metric_name} ({len(image_list)} subjects)...")
    
    ref_img = nib.load(image_list[0])
    affine = ref_img.affine
    
    all_data = []
    for img_path in image_list:
        try:
            img = nib.load(img_path)
            data = img.get_fdata()
            if data.ndim == 4:
                data = data[..., 0]
            all_data.append(data)
        except Exception as e:
            print(f"    Warning: Could not load {img_path.name}: {e}")
    
    if not all_data:
        return
    
    all_data = np.stack(all_data, axis=-1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_map = np.nanmean(all_data, axis=-1)
        std_map = np.nanstd(all_data, axis=-1)
        cv_map = np.where(mean_map > 0, std_map / mean_map, 0)
    
    mean_file = Path(str(output_prefix) + '_mean.nii.gz')
    std_file = Path(str(output_prefix) + '_std.nii.gz')
    cv_file = Path(str(output_prefix) + '_cv.nii.gz')
    
    nib.save(nib.Nifti1Image(mean_map.astype(np.float32), affine), mean_file)
    nib.save(nib.Nifti1Image(std_map.astype(np.float32), affine), std_file)
    nib.save(nib.Nifti1Image(cv_map.astype(np.float32), affine), cv_file)
    
    print(f"    ✓ Saved: {mean_file.name}, {std_file.name}, {cv_file.name}")


def compute_mean_dec_map(dec_maps: list, output_file: Path, metric_name: str):
    """Compute mean DEC map across subjects."""
    if not dec_maps:
        return
    
    print(f"\n  Computing mean DEC map for {metric_name} ({len(dec_maps)} subjects)...")
    
    ref_img = nib.load(dec_maps[0])
    affine = ref_img.affine
    
    all_data = []
    for img_path in dec_maps:
        try:
            img = nib.load(img_path)
            all_data.append(img.get_fdata())
        except Exception as e:
            print(f"    Warning: Could not load {img_path.name}: {e}")
    
    if not all_data:
        return
    
    all_data = np.stack(all_data, axis=0)
    mean_dec = np.mean(all_data, axis=0).astype(np.uint8)
    
    nib.save(nib.Nifti1Image(mean_dec, affine), output_file)
    print(f"    ✓ Saved: {output_file.name}")


def plot_qc_maps(output_folder: Path, metric_name: str, plot_folder: Path):
    """Generate visualization plots for QC maps using nilearn."""
    plot_folder.mkdir(exist_ok=True, parents=True)
    
    files = {
        'all_mean': output_folder / f'group_{metric_name}_mean.nii.gz',
        'autism_mean': output_folder / f'group_{metric_name}_autism_mean.nii.gz',
        'control_mean': output_folder / f'group_{metric_name}_control_mean.nii.gz',
        'autism_cv': output_folder / f'group_{metric_name}_autism_cv.nii.gz',
        'control_cv': output_folder / f'group_{metric_name}_control_cv.nii.gz',
    }
    
    existing_files = {k: v for k, v in files.items() if v.exists()}
    if not existing_files:
        return
    
    metric_label = metric_name.upper().replace('_', ' ')
    
    # Group comparison plot
    if 'autism_mean' in existing_files and 'control_mean' in existing_files:
        print(f"  Creating group comparison plot for {metric_name}...")
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        autism_data = nib.load(existing_files['autism_mean']).get_fdata()
        control_data = nib.load(existing_files['control_mean']).get_fdata()
        vmax = max(np.percentile(autism_data[autism_data > 0], 99),
                   np.percentile(control_data[control_data > 0], 99))
        
        plotting.plot_stat_map(
            str(existing_files['autism_mean']),
            display_mode='z', cut_coords=7,
            title=f'{metric_label} Mean - Autism Group',
            colorbar=True, vmax=vmax, axes=axes[0], cmap='hot'
        )
        
        plotting.plot_stat_map(
            str(existing_files['control_mean']),
            display_mode='z', cut_coords=7,
            title=f'{metric_label} Mean - Control Group',
            colorbar=True, vmax=vmax, axes=axes[1], cmap='hot'
        )
        
        plt.tight_layout()
        plt.savefig(plot_folder / f'{metric_name}_group_comparison_mean.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {metric_name}_group_comparison_mean.png")


def main():
    """Main processing function."""
    import subprocess
    
    project_folder = Path('/home/ASDPrecision/')
    qsirecon_folder = project_folder / 'data/bids/derivatives/qsirecon'
    mrtrix_folder = qsirecon_folder / 'derivatives/qsirecon-MRtrix3_fork-SS3T_act-HSVS'
    output_folder = project_folder / 'quality_metrics' / 'dwi_qc_maps'
    subject_output_folder = output_folder / 'subjects'
    qsiprep_folder = project_folder / 'data/bids/derivatives/qsiprep'
    
    output_folder.mkdir(exist_ok=True, parents=True)
    subject_output_folder.mkdir(exist_ok=True, parents=True)
    
    print("=" * 70)
    print("DWI Group QC Map Extraction")
    print("=" * 70)
    
    # Check Docker
    docker_check = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
    if docker_check.returncode != 0:
        print("Error: Docker not found.")
        return
    
    # Check MNI reference
    mni_reference = Path(MNI_TEMPLATE)
    ants_available = mni_reference.exists()
    if not ants_available:
        print(f"Warning: MNI reference not found: {mni_reference}")
    
    # Load participants
    participants_file = project_folder / 'data/bids/participants.tsv'
    if not participants_file.exists():
        print(f"Error: participants.tsv not found")
        return
    
    participants_df = pd.read_csv(participants_file, sep='\t')
    
    subject_to_group = {}
    for _, row in participants_df.iterrows():
        subject_id = row['participant_id']
        autism_diagnosis = str(row['autism_diagnosis']).lower().strip()
        subject_to_group[subject_id] = 'autism' if autism_diagnosis == 'yes' else 'control'
    
    # Find subjects
    if not mrtrix_folder.exists():
        print(f"Error: MRtrix3 output folder not found: {mrtrix_folder}")
        return
    
    subject_list = sorted([d.name.split('.')[0] for d in mrtrix_folder.iterdir() if d.name.endswith('.html')])
    print(f"\nFound {len(subject_list)} subjects")
    
    # Collect files
    norm_maps = {'all': [], 'autism': [], 'control': []}
    afd_maps = {'all': [], 'autism': [], 'control': []}
    peak_maps = {'all': [], 'autism': [], 'control': []}
    dec_maps = {'all': [], 'autism': [], 'control': []}
    
    mni_output_folder = subject_output_folder / 'mni'
    mni_output_folder.mkdir(exist_ok=True, parents=True)
    
    for subject in subject_list:
        print(f"\nProcessing {subject}...")
        subject_dwi_dir = mrtrix_folder / subject / 'dwi'
        
        group = subject_to_group.get(subject, 'unknown')
        if group == 'unknown':
            continue
        
        if not subject_dwi_dir.exists():
            continue
        
        transform_file = qsiprep_folder / subject / 'anat' / f'{subject}_from-ACPC_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
        if not transform_file.exists() or not ants_available:
            print(f"  Skipping: transform or ANTs not available")
            continue
        
        # MT-normalization map
        norm_file = subject_dwi_dir / f'{subject}_space-ACPC_model-mtnorm_param-norm_dwimap.nii.gz'
        if norm_file.exists():
            norm_mni_file = mni_output_folder / f'{subject}_space-MNI_mtnorm.nii.gz'
            if not norm_mni_file.exists():
                if warp_to_mni(norm_file, norm_mni_file, transform_file, mni_reference):
                    norm_maps['all'].append(norm_mni_file)
                    norm_maps[group].append(norm_mni_file)
            else:
                norm_maps['all'].append(norm_mni_file)
                norm_maps[group].append(norm_mni_file)
        
        # Extract AFD and peak from WM FOD
        wm_fod_file = subject_dwi_dir / f'{subject}_space-ACPC_model-ss3t_param-fod_label-WM_dwimap.mif.gz'
        if wm_fod_file.exists():
            output_files = extract_afd_and_peak(wm_fod_file, subject_output_folder, subject)
            
            if 'afd' in output_files:
                afd_mni_file = mni_output_folder / f'{subject}_space-MNI_afd.nii.gz'
                if not afd_mni_file.exists():
                    if warp_to_mni(output_files['afd'], afd_mni_file, transform_file, mni_reference):
                        afd_maps['all'].append(afd_mni_file)
                        afd_maps[group].append(afd_mni_file)
                else:
                    afd_maps['all'].append(afd_mni_file)
                    afd_maps[group].append(afd_mni_file)
            
            if 'peak' in output_files:
                peak_mni_file = mni_output_folder / f'{subject}_space-MNI_peak.nii.gz'
                if not peak_mni_file.exists():
                    if warp_to_mni(output_files['peak'], peak_mni_file, transform_file, mni_reference):
                        peak_maps['all'].append(peak_mni_file)
                        peak_maps[group].append(peak_mni_file)
                else:
                    peak_maps['all'].append(peak_mni_file)
                    peak_maps[group].append(peak_mni_file)
    
    # Compute group statistics
    print("\n" + "=" * 70)
    print("Computing Group-Level Statistics")
    print("=" * 70)
    
    for group_name in ['all', 'autism', 'control']:
        group_suffix = '' if group_name == 'all' else f'_{group_name}'
        group_label = 'All subjects' if group_name == 'all' else f'{group_name.capitalize()} group'
        
        print(f"\n--- {group_label} ---")
        
        compute_group_statistics(norm_maps[group_name], output_folder / f'group_mtnorm{group_suffix}', f'MT-norm ({group_label})')
        compute_group_statistics(afd_maps[group_name], output_folder / f'group_afd{group_suffix}', f'AFD ({group_label})')
        compute_group_statistics(peak_maps[group_name], output_folder / f'group_peak{group_suffix}', f'Peak ({group_label})')
        compute_mean_dec_map(dec_maps[group_name], output_folder / f'group_dec{group_suffix}_mean.nii.gz', f'DEC ({group_label})')
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating Visualization Plots")
    print("=" * 70)
    
    plot_folder = output_folder / 'plots'
    plot_folder.mkdir(exist_ok=True)
    
    if norm_maps['all']:
        plot_qc_maps(output_folder, 'mtnorm', plot_folder)
    if afd_maps['all']:
        plot_qc_maps(output_folder, 'afd', plot_folder)
    if peak_maps['all']:
        plot_qc_maps(output_folder, 'peak', plot_folder)
    
    print("\n" + "=" * 70)
    print("QC Map Extraction Complete")
    print("=" * 70)
    print(f"\nOutput folder: {output_folder}")
    print(f"Subjects processed: {len(norm_maps['all'])}")


if __name__ == '__main__':
    main()
