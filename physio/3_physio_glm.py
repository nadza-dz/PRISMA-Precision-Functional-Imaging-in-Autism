"""
GLM analysis using physiological regressors.

This script runs:
- First level: HRV, RVT, HRV×RVT interaction, motion parameters
- Second level: Fixed effects to combine runs within subject
- Group level: Average effects for autism and comparison groups

Usage:
    python physio_glm.py

Author: Joe Bathelt
Date: December 2025
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn import plotting

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Set CMU Sans Serif as the default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['CMU Sans Serif', 'DejaVu Sans']

# Maximum parallel workers
MAX_WORKERS_SUBJECTS = 16

def load_regressor_summary(regressor_folder: Path) -> list:
    """
    Load regressor summary from TSV file.
    
    Parameters:
    -----------
    regressor_folder : Path
        Path to regressor folder
        
    Returns:
    --------
    list
        List of regressor summary dictionaries
    """
    summary_file = regressor_folder / 'regressor_summary.tsv'
    if not summary_file.exists():
        print(f'Error: regressor_summary.tsv not found at {summary_file}')
        return []
    
    summary_df = pd.read_csv(summary_file, sep='\t')
    return summary_df.to_dict('records')


def _process_single_run(run_info: dict, subject: str, bids_folder: Path,
                        regressor_folder: Path, first_level_folder: Path) -> dict:
    """
    Process a single run for first-level GLM (for parallel execution).
    
    Returns:
    --------
    dict
        Result containing run_id, contrast_maps, model info, and status
    """
    run_id = run_info['run_id']
    tr = run_info['tr']
    n_volumes = run_info['n_volumes']
    
    result = {
        'run_id': run_id,
        'contrast_maps': {'hrv': None, 'rvt': None, 'hrv_x_rvt': None},
        'model_info': None,
        'status': 'skipped',
        'message': ''
    }
    
    # Check if both HRV and RVT are available
    if not (run_info['hrv_calculated'] and run_info['rvt_calculated']):
        result['message'] = f'Skipping {run_id}: missing HRV or RVT'
        return result
    
    # Use fMRIPrep preprocessed data
    fmriprep_folder = bids_folder / 'derivatives' / 'fmriprep' / subject / 'func'
    if not fmriprep_folder.exists():
        result['message'] = f'fMRIPrep folder not found for {subject}'
        return result
    
    subject_first_level = first_level_folder / subject
    os.makedirs(subject_first_level, exist_ok=True)
    
    # Find preprocessed BOLD file and brain mask
    bold_file = None
    mask_file = None
    confounds_file = None
    
    for f in fmriprep_folder.iterdir():
        if (f.name.endswith('_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz') 
            and run_id in f.name):
            bold_file = f
        elif (f.name.endswith('_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz') 
              and run_id in f.name):
            mask_file = f
        elif ('confounds' in f.name and run_id in f.name 
              and f.name.endswith('.tsv')):
            confounds_file = f
    
    if bold_file is None:
        result['message'] = f'Skipping {run_id}: preprocessed BOLD not found'
        return result
    
    # Check if outputs already exist
    expected_outputs = [
        subject_first_level / f'{subject}_{run_id}_hrv_effect.nii.gz',
        subject_first_level / f'{subject}_{run_id}_rvt_effect.nii.gz',
        subject_first_level / f'{subject}_{run_id}_hrv_x_rvt_effect.nii.gz'
    ]
    if all(f.exists() for f in expected_outputs):
        result['status'] = 'cached'
        result['message'] = f'Skipping {run_id}: outputs already exist'
        for f in expected_outputs:
            contrast_name = f.name.replace(f'{subject}_{run_id}_', '').replace('_effect.nii.gz', '')
            result['contrast_maps'][contrast_name] = str(f)
        return result
    
    # Find regressor file
    regressor_file = None
    subject_regressor_folder = regressor_folder / subject
    if subject_regressor_folder.exists():
        for f in subject_regressor_folder.iterdir():
            if f.name.endswith('_physio-regressors.tsv') and run_id in f.name:
                regressor_file = f
                break
    
    if regressor_file is None:
        result['message'] = f'Skipping {run_id}: regressor file not found'
        return result
    
    physio_regressors = pd.read_csv(regressor_file, sep='\t')
    
    # Build design matrix
    design_matrix = pd.DataFrame()
    design_matrix['hrv'] = physio_regressors['hrv']
    design_matrix['rvt'] = physio_regressors['rvt']
    
    # Add interaction term (z-scored)
    hrv_z = (design_matrix['hrv'] - design_matrix['hrv'].mean()) / design_matrix['hrv'].std()
    rvt_z = (design_matrix['rvt'] - design_matrix['rvt'].mean()) / design_matrix['rvt'].std()
    design_matrix['hrv_x_rvt'] = hrv_z * rvt_z
    
    # Add motion parameters if available
    if confounds_file is not None:
        try:
            confounds = pd.read_csv(confounds_file, sep='\t')
            motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
            for col in motion_cols:
                if col in confounds.columns:
                    design_matrix[col] = confounds[col].values[:n_volumes]
        except Exception as e:
            pass  # Silently skip motion parameters if they fail
    
    # Add intercept
    design_matrix['intercept'] = 1
    design_matrix = design_matrix.fillna(0)
    
    # Fit first-level model
    try:
        first_level_model = FirstLevelModel(
            t_r=tr,
            noise_model='ar1',
            hrf_model=None,
            standardize=True,
            drift_model=None,
            minimize_memory=True,
            mask_img=str(mask_file) if mask_file else None
        )
        
        first_level_model.fit(str(bold_file), design_matrices=design_matrix)
        
        # Compute contrasts
        contrasts = {'hrv': 'hrv', 'rvt': 'rvt', 'hrv_x_rvt': 'hrv_x_rvt'}
        
        for contrast_name, contrast_def in contrasts.items():
            try:
                contrast_map = first_level_model.compute_contrast(
                    contrast_def, output_type='effect_size'
                )
                output_file = subject_first_level / f'{subject}_{run_id}_{contrast_name}_effect.nii.gz'
                nib.save(contrast_map, output_file)
                result['contrast_maps'][contrast_name] = str(output_file)
            except Exception as e:
                result['message'] += f' Warning: Could not compute {contrast_name}: {e}'
        
        result['status'] = 'completed'
        result['message'] = f'✓ {run_id} completed'
        
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f'Error fitting model for {run_id}: {e}'
    
    return result


def run_first_level(subject: str, subject_runs: list, 
                    bids_folder: Path, regressor_folder: Path,
                    first_level_folder: Path, reports_folder: Path) -> dict:
    """
    Run first-level GLM for a subject (runs processed in parallel).
    
    Parameters:
    -----------
    subject : str
        Subject ID
    subject_runs : list
        List of run info dictionaries
    bids_folder : Path
        Path to BIDS folder
    regressor_folder : Path
        Path to regressor folder
    first_level_folder : Path
        Path to first-level output folder
    reports_folder : Path
        Path to reports folder
        
    Returns:
    --------
    dict
        Dictionary mapping contrast names to list of output files
    """
    subject_contrast_maps = {'hrv': [], 'rvt': [], 'hrv_x_rvt': []}
    
    # Use fMRIPrep preprocessed data in MNI152NLin2009cAsym space
    fmriprep_folder = bids_folder / 'derivatives' / 'fmriprep' / subject / 'func'
    if not fmriprep_folder.exists():
        print(f'    fMRIPrep folder not found for {subject}')
        return subject_contrast_maps
    
    subject_first_level = first_level_folder / subject
    os.makedirs(subject_first_level, exist_ok=True)
    
    # Process runs in parallel
    n_workers = min(len(subject_runs), os.cpu_count() or 4)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _process_single_run, run_info, subject, bids_folder,
                regressor_folder, first_level_folder
            ): run_info['run_id']
            for run_info in subject_runs
        }
        
        for future in as_completed(futures):
            result = future.result()
            print(f'    {result["message"]}')
            
            # Collect successful contrast maps
            for contrast_name, map_path in result['contrast_maps'].items():
                if map_path is not None:
                    subject_contrast_maps[contrast_name].append(map_path)
    
    return subject_contrast_maps



def _process_subject_second_level(subject: str, contrast_dict: dict,
                                   second_level_folder: Path,
                                   reports_folder: Path) -> tuple:
    """
    Process second-level analysis for a single subject (for parallel execution).
    
    Returns:
    --------
    tuple
        (subject, {contrast: combined_map_path}, messages)
    """
    result = {}
    messages = []
    
    subject_second_level = second_level_folder / subject
    os.makedirs(subject_second_level, exist_ok=True)
    
    for contrast_name, run_maps in contrast_dict.items():
        # Filter out None values from run_maps
        valid_maps = [m for m in run_maps if m is not None and os.path.exists(m)]
        
        if len(valid_maps) == 0:
            continue
        elif len(valid_maps) == 1:
            result[contrast_name] = valid_maps[0]
        else:
            try:
                n_runs = len(valid_maps)
                design_matrix = pd.DataFrame({'intercept': np.ones(n_runs)})
                
                second_level_model = SecondLevelModel()
                second_level_model.fit(valid_maps, design_matrix=design_matrix)
                
                combined_map = second_level_model.compute_contrast(
                    second_level_contrast='intercept',
                    output_type='effect_size'
                )
                
                output_file = subject_second_level / f'{subject}_{contrast_name}_fixed_effects.nii.gz'
                nib.save(combined_map, output_file)
                result[contrast_name] = str(output_file)
                
                # Generate report
                try:
                    report = second_level_model.generate_report(
                        contrasts='intercept',
                        title=f'{subject} - {contrast_name} (Fixed Effects)',
                        plot_type='slice'
                    )
                    report_file = reports_folder / f'{subject}_second_level_{contrast_name}_report.html'
                    report.save_as_html(str(report_file))
                except Exception as e:
                    messages.append(f'Warning: Could not generate report for {contrast_name}: {e}')
                
            except Exception as e:
                messages.append(f'Warning: Fixed effects failed for {contrast_name}: {e}')
                result[contrast_name] = valid_maps[0]
    
    return subject, result, messages


def run_second_level(subject_contrast_maps: dict, second_level_folder: Path,
                     reports_folder: Path) -> dict:
    """
    Run second-level (fixed effects) analysis within subject (parallel across subjects).
    
    Parameters:
    -----------
    subject_contrast_maps : dict
        {subject: {contrast: [run_maps]}}
    second_level_folder : Path
        Output folder
    reports_folder : Path
        Reports folder
        
    Returns:
    --------
    dict
        {subject: {contrast: combined_map_path}}
    """
    second_level_maps = {}
    subjects = list(subject_contrast_maps.keys())
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_SUBJECTS) as executor:
        futures = {
            executor.submit(
                _process_subject_second_level, subject,
                subject_contrast_maps[subject],
                second_level_folder, reports_folder
            ): subject
            for subject in subjects
        }
        
        for future in as_completed(futures):
            subject, result, messages = future.result()
            second_level_maps[subject] = result
            
            for msg in messages:
                print(f'    {subject}: {msg}')
            
            print(f'    ✓ {subject} fixed effects completed')
    
    return second_level_maps


def run_group_level(second_level_maps: dict, participants_df: pd.DataFrame,
                    group_level_folder: Path, reports_folder: Path):
    """
    Run group-level analysis for autism and comparison groups.
    
    Parameters:
    -----------
    second_level_maps : dict
        {subject: {contrast: map_path}}
    participants_df : pd.DataFrame
        Participants dataframe with diagnosis
    group_level_folder : Path
        Output folder
    reports_folder : Path
        Reports folder
    """
    # Organize subjects by group
    group_maps = {'autism': {}, 'comparison': {}}
    
    for subject in second_level_maps.keys():
        subj_data = participants_df[participants_df['participant_id'] == subject]
        if len(subj_data) == 0:
            continue
        
        diagnosis = str(subj_data['autism_diagnosis'].values[0]).lower()
        group = 'autism' if 'yes' in diagnosis else 'comparison'
        
        for contrast_name, map_file in second_level_maps[subject].items():
            if contrast_name not in group_maps[group]:
                group_maps[group][contrast_name] = []
            group_maps[group][contrast_name].append(map_file)
    
    # Fit group-level models
    for group in ['autism', 'comparison']:
        print(f'\n    {group.upper()} group:')
        group_output = group_level_folder / group
        os.makedirs(group_output, exist_ok=True)
        
        for contrast_name, subject_maps in group_maps[group].items():
            if len(subject_maps) < 2:
                print(f'      Skipping {contrast_name}: not enough subjects ({len(subject_maps)})')
                continue
            
            try:
                n_subjects = len(subject_maps)
                design_matrix = pd.DataFrame({'intercept': np.ones(n_subjects)})
                
                group_model = SecondLevelModel()
                group_model.fit(subject_maps, design_matrix=design_matrix)
                
                # Compute statistics
                group_effect = group_model.compute_contrast(
                    second_level_contrast='intercept', output_type='effect_size'
                )
                group_tstat = group_model.compute_contrast(
                    second_level_contrast='intercept', output_type='stat'
                )
                group_zstat = group_model.compute_contrast(
                    second_level_contrast='intercept', output_type='z_score'
                )
                
                # Save maps
                nib.save(group_effect, group_output / f'{group}_{contrast_name}_effect.nii.gz')
                nib.save(group_tstat, group_output / f'{group}_{contrast_name}_tstat.nii.gz')
                nib.save(group_zstat, group_output / f'{group}_{contrast_name}_zstat.nii.gz')
                
                print(f'      ✓ {contrast_name}: n={len(subject_maps)} subjects')
                
                # Generate report
                try:
                    report = group_model.generate_report(
                        contrasts='intercept',
                        title=f'Group {group} - {contrast_name}',
                        plot_type='glass'
                    )
                    report_file = reports_folder / f'group_{group}_{contrast_name}_report.html'
                    report.save_as_html(str(report_file))
                except Exception as e:
                    print(f'      Warning: Could not generate group report: {e}')
                
            except Exception as e:
                print(f'      Error for {contrast_name}: {e}')

def plot_group_level_stat_maps(group_level_folder: Path):
    """
    Plot group-level z-stat maps for HRV and RVT regressors side-by-side for autism and comparison groups.
    Shows glass brain view with a single central colour bar.
    """
    contrasts = ['hrv', 'rvt']
    groups = ['autism', 'comparison']
    
    # Collect z-stat file paths
    zstat_paths = {}
    for group in groups:
        zstat_paths[group] = {}
        group_dir = group_level_folder / group
        for contrast in contrasts:
            zfile = group_dir / f"{group}_{contrast}_zstat.nii.gz"
            if zfile.exists():
                zstat_paths[group][contrast] = str(zfile)
            else:
                zstat_paths[group][contrast] = None
    
    # Create figure with dimensions 180mm x 60mm (converted to inches: 1 inch = 25.4mm)
    fig_width = 180 / 25.4
    fig_height = 60 / 25.4  
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create gridspec: 2 rows (groups) x 2 cols (contrasts)
    gs = gridspec.GridSpec(2, 2, figure=fig, 
                          hspace=0.15,  # Vertical space
                          wspace=0.05,  # Horizontal space
                          left=0.05, right=0.88, top=0.94, bottom=0.05)
    
    # Plot parameters
    threshold = 2.3
    vmin = 2.3
    vmax = 6
    
    # Plot each panel
    for i, group in enumerate(groups):
        for j, contrast in enumerate(contrasts):
            ax = fig.add_subplot(gs[i, j])
            zmap = zstat_paths[group][contrast]
            
            if zmap:
                display = plotting.plot_glass_brain(
                    zmap,
                    display_mode='ortho',
                    colorbar=False,
                    threshold=threshold,
                    vmin=vmin,
                    vmax=vmax,
                    plot_abs=True,
                    resampling_interpolation='continuous',
                    title=None,
                    cmap='inferno',
                    axes=ax,
                    annotate=False
                )
            else:
                ax.axis('off')
    
    # Add single colorbar on the right side
    # Calculate colorbar position: 2/3 height, centered vertically
    cbar_height = 0.7 * (2/3)  # 2/3 of available height
    cbar_bottom = 0.15 + (0.7 - cbar_height) / 2  # Center vertically
    
    cbar_ax = fig.add_axes([0.90, cbar_bottom, 0.02, cbar_height])
    
    # Create a mappable for the colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap='inferno', norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('|z|', rotation=270, labelpad=15, fontsize=10)
    
    return fig

def main():
    """Main GLM analysis function."""
    
    project_folder = Path('/home/ASDPrecision/')
    bids_folder = project_folder / 'data' / 'bids'
    regressor_folder = project_folder / 'derivatives' / 'physio_regressors'
    glm_folder = project_folder / 'derivatives' / 'physio_glm'
    
    # Output folders
    first_level_folder = glm_folder / 'first_level'
    second_level_folder = glm_folder / 'second_level'
    group_level_folder = glm_folder / 'group_level'
    reports_folder = glm_folder / 'reports'
    
    for folder in [first_level_folder, second_level_folder, group_level_folder, reports_folder]:
        os.makedirs(folder, exist_ok=True)
    
    # Load participants data
    participants_file = bids_folder / 'participants.tsv'
    if not participants_file.exists():
        print(f'Error: participants.tsv not found')
        return
    
    participants_df = pd.read_csv(participants_file, sep='\t')
    
    # Check for autism_diagnosis column
    if 'autism_diagnosis' not in participants_df.columns:
        for col in ['diagnosis', 'group', 'Diagnosis', 'Group']:
            if col in participants_df.columns:
                participants_df['autism_diagnosis'] = participants_df[col]
                break
    
    # Load regressor summary
    regressor_summary = load_regressor_summary(regressor_folder)
    if not regressor_summary:
        print('No regressors found. Run physio_regressors.py first.')
        return
    
    print(f'Loaded {len(regressor_summary)} regressor records')
    
    # =========================================================================
    # First level GLM
    # =========================================================================
    print('\n' + '='*60)
    print('FIRST-LEVEL GLM ANALYSIS')
    print('='*60)
    
    subjects_with_regressors = sorted(set(item['subject_id'] for item in regressor_summary))
    all_subject_contrast_maps = {}
    
    # Process subjects sequentially, but runs within each subject in parallel
    print(f'\n  Processing {len(subjects_with_regressors)} subjects (runs in parallel per subject)...')
    
    for subject in subjects_with_regressors:
        print(f'\n  {subject}...')
        
        subject_runs = [item for item in regressor_summary if item['subject_id'] == subject]
        
        contrast_maps = run_first_level(
            subject, subject_runs,
            bids_folder, regressor_folder,
            first_level_folder, reports_folder
        )
        
        if any(contrast_maps.values()):
            all_subject_contrast_maps[subject] = contrast_maps
    
    # =========================================================================
    # Second level (fixed effects)
    # =========================================================================
    print('\n' + '='*60)
    print('SECOND-LEVEL (FIXED EFFECTS) ANALYSIS')
    print('='*60)
    
    second_level_maps = run_second_level(
        all_subject_contrast_maps, second_level_folder, reports_folder
    )
    
    # =========================================================================
    # Group level
    # =========================================================================
    print('\n' + '='*60)
    print('GROUP-LEVEL ANALYSIS')
    print('='*60)
    
    run_group_level(
        second_level_maps, participants_df, group_level_folder, reports_folder
    )
    
    print('\n' + '='*60)
    print('GLM ANALYSIS COMPLETE')
    print('='*60)
    print(f'Results saved to: {glm_folder}')

    # =========================================================================
    # Plot group-level stat maps
    # =========================================================================
    print('\nPlotting group-level z-stat maps...')
    fig = plot_group_level_stat_maps(group_level_folder)
    fig.savefig(reports_folder / 'physio_validation.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
