#!/usr/bin/env python3
"""
Calculate physiological regressors (HRV and RVT) for fMRI analysis.

This script:
1. Reads the physio_qc_review.tsv to identify useable recordings
2. Calculates useable data percentage by autism diagnosis group
3. Calculates average heart rate and breathing rate by group
4. Generates HRV and RVT regressors for fMRI analysis

Usage:
    python physio_regressors.py

Author: Joe Bathelt
Date: December 2025
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pathlib import Path

# Maximum parallel workers for subject processing
MAX_WORKERS = 16

from physio_utils import (
    calculate_hrv_regressor,
    calculate_rvt_regressor,
    get_n_volumes_from_fmri,
    get_tr_from_fmri,
    load_physio_data,
    extract_run_id,
    DEFAULT_TR
)


def load_useable_status(review_file: Path) -> tuple:
    """
    Load useable status from review file.
    
    Parameters:
    -----------
    review_file : Path
        Path to physio_qc_review.tsv
        
    Returns:
    --------
    tuple
        (useable_ppg, useable_rsp) dictionaries mapping (subject, run_id) -> bool
    """
    useable_ppg = {}
    useable_rsp = {}
    
    if not review_file.exists():
        return useable_ppg, useable_rsp
    
    review_df = pd.read_csv(review_file, sep='\t')
    
    for _, row in review_df.iterrows():
        key = (row['subject_id'], row['file_id'])
        ppg_status = str(row.get('ppg_useable', '')).lower().strip()
        rsp_status = str(row.get('rsp_useable', '')).lower().strip()
        
        useable_ppg[key] = ppg_status in ['yes', 'partial']
        useable_rsp[key] = rsp_status in ['yes', 'partial']
    
    return useable_ppg, useable_rsp


def calculate_useable_percentages(bids_folder: Path, participants_df: pd.DataFrame,
                                  useable_ppg: dict, useable_rsp: dict) -> dict:
    """
    Calculate percentage of useable data by diagnosis group.
    
    Only counts recordings where physio data exists (was attempted).
    Useable status is based on QC review file.
    
    Parameters:
    -----------
    bids_folder : Path
        Path to BIDS folder
    participants_df : pd.DataFrame
        Participants dataframe with diagnosis info
    useable_ppg : dict
        Dictionary of PPG useable status
    useable_rsp : dict
        Dictionary of RSP useable status
        
    Returns:
    --------
    dict
        Statistics by group including participant counts
    """
    ppg_stats = {'autism': {'total': 0, 'useable': 0}, 
                 'comparison': {'total': 0, 'useable': 0}}
    rsp_stats = {'autism': {'total': 0, 'useable': 0}, 
                 'comparison': {'total': 0, 'useable': 0}}
    
    # Track participants with and without physio data
    participants_with_physio = {'autism': set(), 'comparison': set()}
    participants_without_physio = {'autism': set(), 'comparison': set()}
    
    # Track session counts
    total_fmri_sessions = {'autism': 0, 'comparison': 0}
    fmri_with_physio = {'autism': 0, 'comparison': 0}
    fmri_missing_physio = {'autism': 0, 'comparison': 0}
    physio_without_fmri = {'autism': 0, 'comparison': 0}
    
    # Get all subjects
    all_subjects = [s for s in os.listdir(bids_folder) if s.startswith('sub-')]
    
    for subject in all_subjects:
        # Get diagnosis
        subj_data = participants_df[participants_df['participant_id'] == subject]
        if len(subj_data) == 0:
            continue
        
        diagnosis = str(subj_data['autism_diagnosis'].values[0]).lower().strip()
        group = 'autism' if diagnosis == 'yes' else 'comparison'
        
        # Get func folder
        func_folder = bids_folder / subject / 'func'
        if not func_folder.exists():
            participants_without_physio[group].add(subject)
            continue
        
        # Get all fMRI BOLD files and physio files
        bold_files = [f for f in os.listdir(func_folder) if f.endswith('_bold.nii.gz')]
        physio_files = [f for f in os.listdir(func_folder) if f.endswith('_physio.tsv.gz')]
        
        # Extract run IDs from BOLD files
        bold_run_ids = set()
        for bold_file in bold_files:
            for part in bold_file.replace('_bold.nii.gz', '').split('_'):
                if part.startswith('run-'):
                    bold_run_ids.add(part)
                    break
        
        # Extract run IDs from physio files
        physio_run_ids = set()
        for physio_file in physio_files:
            run_id = extract_run_id(physio_file)
            if run_id:
                physio_run_ids.add(run_id)
        
        # Count sessions
        total_fmri_sessions[group] += len(bold_run_ids)
        fmri_with_physio[group] += len(bold_run_ids & physio_run_ids)  # Both exist
        fmri_missing_physio[group] += len(bold_run_ids - physio_run_ids)  # BOLD but no physio
        physio_without_fmri[group] += len(physio_run_ids - bold_run_ids)  # Physio but no BOLD
        
        if len(physio_files) == 0:
            participants_without_physio[group].add(subject)
            continue
        
        participants_with_physio[group].add(subject)
        
        # Only count physio files that have corresponding BOLD files for QC stats
        for physio_file in physio_files:
            run_id = extract_run_id(physio_file)
            if run_id is None:
                continue
            
            # Only count if corresponding BOLD exists
            if run_id not in bold_run_ids:
                continue
            
            key = (subject, run_id)
            
            # Total = number of fMRI sessions with physio recordings
            ppg_stats[group]['total'] += 1
            rsp_stats[group]['total'] += 1
            
            # Useable = marked as useable in review file, or not flagged at all
            ppg_is_useable = key not in useable_ppg or useable_ppg[key]
            if ppg_is_useable:
                ppg_stats[group]['useable'] += 1
            
            rsp_is_useable = key not in useable_rsp or useable_rsp[key]
            if rsp_is_useable:
                rsp_stats[group]['useable'] += 1
    
    # Convert sets to counts
    participant_counts = {
        'with_physio': {
            'autism': len(participants_with_physio['autism']),
            'comparison': len(participants_with_physio['comparison'])
        },
        'without_physio': {
            'autism': len(participants_without_physio['autism']),
            'comparison': len(participants_without_physio['comparison'])
        }
    }
    
    session_counts = {
        'total_fmri': total_fmri_sessions,
        'fmri_with_physio': fmri_with_physio,
        'fmri_missing_physio': fmri_missing_physio,
        'physio_without_fmri': physio_without_fmri
    }
    
    return {'ppg': ppg_stats, 'rsp': rsp_stats, 'participants': participant_counts, 'sessions': session_counts}


def calculate_physio_means_by_group(bids_folder: Path, qc_folder: Path,
                                    participants_df: pd.DataFrame,
                                    useable_ppg: dict, useable_rsp: dict) -> dict:
    """
    Calculate mean heart rate and respiratory rate by group for useable recordings.
    
    Returns:
    --------
    dict
        HR and RR values by group
    """
    hr_by_group = {'autism': [], 'comparison': []}
    rr_by_group = {'autism': [], 'comparison': []}
    
    # Load QC summaries
    ppg_file = qc_folder / 'physio_qc_ppg.tsv'
    rsp_file = qc_folder / 'physio_qc_rsp.tsv'
    
    if not ppg_file.exists() or not rsp_file.exists():
        print('Warning: QC summary files not found')
        return {'hr': hr_by_group, 'rr': rr_by_group}
    
    ppg_df = pd.read_csv(ppg_file, sep='\t')
    rsp_df = pd.read_csv(rsp_file, sep='\t')
    
    all_subjects = set(ppg_df['subject_id'].unique()) | set(rsp_df['subject_id'].unique())
    
    for subject in all_subjects:
        # Get diagnosis
        subj_data = participants_df[participants_df['participant_id'] == subject]
        if len(subj_data) == 0:
            continue
        
        diagnosis = str(subj_data['autism_diagnosis'].values[0]).lower().strip()
        group = 'autism' if diagnosis == 'yes' else 'comparison'
        
        # Get recordings
        func_folder = bids_folder / subject / 'func'
        if not func_folder.exists():
            continue
        
        physio_files = [f for f in os.listdir(func_folder) if f.endswith('_physio.tsv.gz')]
        
        for physio_file in physio_files:
            run_id = extract_run_id(physio_file)
            if run_id is None:
                continue
            
            key = (subject, run_id)
            
            # Check PPG useability and get HR
            ppg_is_useable = key not in useable_ppg or useable_ppg[key]
            if ppg_is_useable:
                subj_ppg = ppg_df[ppg_df['subject_id'] == subject]
                if len(subj_ppg) > 0:
                    run_col = f'{run_id.replace("-", "")}_mean_hr'
                    if run_col in subj_ppg.columns:
                        hr_val = subj_ppg[run_col].values[0]
                        if pd.notna(hr_val) and hr_val != 'n/a':
                            hr_by_group[group].append(float(hr_val))
            
            # Check RSP useability and get RR
            rsp_is_useable = key not in useable_rsp or useable_rsp[key]
            if rsp_is_useable:
                subj_rsp = rsp_df[rsp_df['subject_id'] == subject]
                if len(subj_rsp) > 0:
                    run_col = f'{run_id.replace("-", "")}_mean_rate'
                    if run_col in subj_rsp.columns:
                        rr_val = subj_rsp[run_col].values[0]
                        if pd.notna(rr_val) and rr_val != 'n/a':
                            rr_by_group[group].append(float(rr_val))
    
    return {'hr': hr_by_group, 'rr': rr_by_group}


def calculate_regressors_for_subject(subject: str, bids_folder: Path, 
                                     output_folder: Path,
                                     useable_ppg: dict, useable_rsp: dict) -> list:
    """
    Calculate HRV and RVT regressors for all runs of a subject.
    
    Parameters:
    -----------
    subject : str
        Subject ID
    bids_folder : Path
        Path to BIDS folder
    output_folder : Path
        Path to output folder
    useable_ppg : dict
        Dictionary of PPG useable status
    useable_rsp : dict
        Dictionary of RSP useable status
        
    Returns:
    --------
    list
        List of regressor summary dictionaries
    """
    func_folder = bids_folder / subject / 'func'
    if not func_folder.exists():
        return []
    
    subject_output = output_folder / subject
    os.makedirs(subject_output, exist_ok=True)
    
    physio_files = sorted([f for f in os.listdir(func_folder) if f.endswith('_physio.tsv.gz')])
    
    regressor_summary = []
    
    for physio_file in physio_files:
        run_id = extract_run_id(physio_file)
        if run_id is None:
            continue
        
        key = (subject, run_id)
        
        # Check if useable
        ppg_is_useable = key not in useable_ppg or useable_ppg[key]
        rsp_is_useable = key not in useable_rsp or useable_rsp[key]
        
        if not ppg_is_useable and not rsp_is_useable:
            continue
        
        # Check if regressor file already exists
        base_name = physio_file.replace('_physio.tsv.gz', '')
        existing_regressor = subject_output / f'{base_name}_physio-regressors.tsv'
        if existing_regressor.exists():
            print(f'  Skipping {subject} {run_id}: regressor file already exists')
            try:
                existing_df = pd.read_csv(existing_regressor, sep='\t')
                tr = get_tr_from_fmri(bids_folder, subject, run_id)
                regressor_summary.append({
                    'subject_id': subject,
                    'run_id': run_id,
                    'tr': tr,
                    'hrv_calculated': 'hrv' in existing_df.columns,
                    'rvt_calculated': 'rvt' in existing_df.columns,
                    'n_volumes': len(existing_df)
                })
            except Exception as e:
                print(f'    Warning: Could not read existing regressor file: {e}')
            continue
        
        print(f'  Processing {subject} {run_id}...')
        
        # Load physio data
        physio_path = func_folder / physio_file
        phys_data, sampling_rate = load_physio_data(physio_path)
        
        if phys_data is None:
            continue
        
        # Get TR and n_volumes
        tr = get_tr_from_fmri(bids_folder, subject, run_id)
        n_volumes = get_n_volumes_from_fmri(bids_folder, subject, run_id)
        
        if n_volumes is None:
            # Estimate from physio duration
            physio_duration = len(phys_data) / sampling_rate
            n_volumes = int(physio_duration / tr)
            print(f'    Estimated {n_volumes} volumes from physio duration')
        
        print(f'    TR: {tr:.3f}s')
        
        regressors = {}
        
        # Calculate HRV regressor
        if ppg_is_useable and 'cardiac' in phys_data.columns:
            try:
                hrv_regressor = calculate_hrv_regressor(
                    phys_data['cardiac'].values, 
                    sampling_rate, 
                    tr, 
                    n_volumes
                )
                regressors['hrv'] = hrv_regressor
                print(f'    ✓ HRV regressor calculated')
            except Exception as e:
                print(f'    Error calculating HRV: {e}')
        
        # Calculate RVT regressor
        if rsp_is_useable and 'respiratory' in phys_data.columns:
            try:
                rvt_regressor = calculate_rvt_regressor(
                    phys_data['respiratory'].values,
                    sampling_rate,
                    tr,
                    n_volumes
                )
                regressors['rvt'] = rvt_regressor
                print(f'    ✓ RVT regressor calculated')
            except Exception as e:
                print(f'    Error calculating RVT: {e}')
        
        # Save regressors
        if regressors:
            regressor_df = pd.DataFrame(regressors)
            output_file = subject_output / f'{base_name}_physio-regressors.tsv'
            regressor_df.to_csv(output_file, sep='\t', index=False, na_rep='n/a')
            
            # Also save as 1D files for AFNI compatibility
            for reg_name, reg_data in regressors.items():
                output_1d = subject_output / f'{base_name}_{reg_name}.1D'
                np.savetxt(output_1d, reg_data, fmt='%.6f')
            
            regressor_summary.append({
                'subject_id': subject,
                'run_id': run_id,
                'tr': tr,
                'hrv_calculated': 'hrv' in regressors,
                'rvt_calculated': 'rvt' in regressors,
                'n_volumes': n_volumes
            })
    
    return regressor_summary


def main():
    """Main processing function."""
    
    project_folder = Path('/home/ASDPrecision/')
    bids_folder = project_folder / 'data' / 'bids'
    qc_folder = project_folder / 'quality_metrics' / 'physio_qc'
    output_folder = project_folder / 'derivatives' / 'physio_regressors'
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load participants data
    participants_file = bids_folder / 'participants.tsv'
    if not participants_file.exists():
        print(f'Error: participants.tsv not found at {participants_file}')
        return
    
    participants_df = pd.read_csv(participants_file, sep='\t')
    print(f'Loaded {len(participants_df)} participants from participants.tsv')
    
    # Check for autism_diagnosis column
    if 'autism_diagnosis' not in participants_df.columns:
        print(f'Warning: autism_diagnosis column not found. Available columns: {list(participants_df.columns)}')
        for col in ['diagnosis', 'group', 'Diagnosis', 'Group']:
            if col in participants_df.columns:
                participants_df['autism_diagnosis'] = participants_df[col]
                print(f'Using column "{col}" as autism_diagnosis')
                break
    
    # Load review file
    review_file = qc_folder / 'physio_qc_review.tsv'
    useable_ppg, useable_rsp = load_useable_status(review_file)
    if useable_ppg:
        print(f'Loaded {len(useable_ppg)} flagged recordings from review file')
    else:
        print('No review file found - assuming all recordings are useable')
    
    # =========================================================================
    # 1. Calculate percentage of useable data by group
    # =========================================================================
    print('\n' + '='*60)
    print('USEABLE DATA BY GROUP')
    print('='*60)
    
    stats = calculate_useable_percentages(bids_folder, participants_df, useable_ppg, useable_rsp)
    
    # Print participant counts
    print('\nPARTICIPANT PHYSIO DATA AVAILABILITY:')
    for group in ['autism', 'comparison']:
        with_physio = stats['participants']['with_physio'][group]
        without_physio = stats['participants']['without_physio'][group]
        total = with_physio + without_physio
        print(f'  {group.upper()}: {with_physio}/{total} participants have physio data ({without_physio} without)')
    
    # Print session breakdown
    print('\nfMRI SESSION BREAKDOWN:')
    for group in ['autism', 'comparison']:
        total_fmri = stats['sessions']['total_fmri'][group]
        with_physio = stats['sessions']['fmri_with_physio'][group]
        missing = stats['sessions']['fmri_missing_physio'][group]
        extra_physio = stats['sessions']['physio_without_fmri'][group]
        
        print(f'\n  {group.upper()}:')
        print(f'    Total fMRI sessions: {total_fmri}')
        print(f'    With physio data: {with_physio} ({100*with_physio/total_fmri:.1f}%)' if total_fmri > 0 else '    With physio data: 0')
        print(f'    Missing physio: {missing} ({100*missing/total_fmri:.1f}%)' if total_fmri > 0 else '    Missing physio: 0')
        if extra_physio > 0:
            print(f'    Physio without fMRI: {extra_physio} (orphaned physio files)')
    
    print('\nUSEABLE RECORDINGS (of fMRI sessions with physio data):')
    for group in ['autism', 'comparison']:
        ppg_pct = 100 * stats['ppg'][group]['useable'] / stats['ppg'][group]['total'] if stats['ppg'][group]['total'] > 0 else 0
        rsp_pct = 100 * stats['rsp'][group]['useable'] / stats['rsp'][group]['total'] if stats['rsp'][group]['total'] > 0 else 0
        
        print(f'\n{group.upper()} GROUP:')
        print(f'  PPG: {stats["ppg"][group]["useable"]}/{stats["ppg"][group]["total"]} useable ({ppg_pct:.1f}%)')
        print(f'  RSP: {stats["rsp"][group]["useable"]}/{stats["rsp"][group]["total"]} useable ({rsp_pct:.1f}%)')
    
    # =========================================================================
    # 2. Calculate average heart rate and breathing rate by group
    # =========================================================================
    print('\n' + '='*60)
    print('PHYSIOLOGICAL MEASURES BY GROUP (USEABLE RECORDINGS ONLY)')
    print('='*60)
    
    physio_means = calculate_physio_means_by_group(
        bids_folder, qc_folder, participants_df, useable_ppg, useable_rsp
    )
    
    for group in ['autism', 'comparison']:
        print(f'\n{group.upper()} GROUP:')
        
        if physio_means['hr'][group]:
            hr_values = np.array(physio_means['hr'][group])
            print(f'  Heart Rate: {hr_values.mean():.1f} ± {hr_values.std():.1f} bpm (n={len(hr_values)} useable recordings)')
        else:
            print(f'  Heart Rate: No useable recordings')
        
        if physio_means['rr'][group]:
            rr_values = np.array(physio_means['rr'][group])
            print(f'  Resp Rate: {rr_values.mean():.1f} ± {rr_values.std():.1f} breaths/min (n={len(rr_values)} useable recordings)')
        else:
            print(f'  Resp Rate: No useable recordings')
    
    # =========================================================================
    # 3. Calculate HRV and RVT regressors for useable recordings
    # =========================================================================
    print('\n' + '='*60)
    print('CALCULATING PHYSIOLOGICAL REGRESSORS')
    print('='*60)
    
    # Get all subjects
    all_subjects = sorted([s for s in os.listdir(bids_folder) if s.startswith('sub-')])
    
    all_regressor_summary = []
    
    # Process subjects in parallel (up to MAX_WORKERS at a time)
    print(f'\n  Processing {len(all_subjects)} subjects with up to {MAX_WORKERS} in parallel...')
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                calculate_regressors_for_subject,
                subject, bids_folder, output_folder, useable_ppg, useable_rsp
            ): subject
            for subject in all_subjects
        }
        
        for future in as_completed(futures):
            subject = futures[future]
            try:
                regressor_summary = future.result()
                all_regressor_summary.extend(regressor_summary)
                if regressor_summary:
                    print(f'  ✓ {subject}: {len(regressor_summary)} regressors calculated')
            except Exception as e:
                print(f'  ✗ {subject} failed: {e}')
    
    # Save summary
    if all_regressor_summary:
        summary_df = pd.DataFrame(all_regressor_summary)
        summary_file = output_folder / 'regressor_summary.tsv'
        summary_df.to_csv(summary_file, sep='\t', index=False)
        print(f'\n✓ Regressor summary saved to: {summary_file}')
    
    # Create dataset description
    dataset_description = {
        "Name": "Physiological Regressors for fMRI",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "calculate_physio_regressors",
                "Version": "1.0",
                "Description": "HRV and RVT regressors calculated using NeuroKit2"
            }
        ],
        "HowToAcknowledge": "HRV calculated from PPG peak intervals. RVT calculated following Birn et al. (2006).",
        "License": "CC0",
        "ReferencesAndLinks": [
            "Birn, R. M., et al. (2006). Separating respiratory-variation-related fluctuations from neuronal-activity-related fluctuations in fMRI. NeuroImage, 31(4), 1536-1548."
        ]
    }
    
    with open(output_folder / 'dataset_description.json', 'w') as f:
        json.dump(dataset_description, f, indent=4)
    
    print(f'\n✓ Regressor calculation complete!')
    print(f'  Regressors saved to: {output_folder}')
    print(f'  Total regressors calculated: {len(all_regressor_summary)}')


if __name__ == '__main__':
    main()
