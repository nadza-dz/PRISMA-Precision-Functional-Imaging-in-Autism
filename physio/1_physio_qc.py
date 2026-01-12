#!/usr/bin/env python3
"""
Physiological Signal Quality Control.

Processes cardiac (PPG) and respiratory signals to generate QC reports.

Usage:
    python physio_qc.py

Author: Joe Bathelt
Date: December 2025
"""

import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from pathlib import Path

from physio_utils import (
    detect_flatlines,
    HR_MIN, HR_MAX,
    RESP_MIN, RESP_MAX,
    QUALITY_THRESHOLD
)


def process_subject_physio(subject: str, bids_folder: Path, output_folder: Path) -> dict:
    """
    Process all physiological recordings for a single subject.
    
    Parameters:
    -----------
    subject : str
        Subject ID (e.g., 'sub-001')
    bids_folder : Path
        Path to BIDS folder
    output_folder : Path
        Path to output folder for QC results
        
    Returns:
    --------
    dict
        Dictionary with summary statistics for this subject.
    """
    print(f'Processing {subject}...')

    subject_stats = {
        'subject_id': subject,
        'n_runs': 0,
        'filenames': [],
        'cardiac_mean_hr': [],
        'cardiac_std_hr': [],
        'cardiac_rmssd': [],
        'cardiac_sdnn': [],
        'cardiac_quality': [],
        'cardiac_max_flatline': [],
        'cardiac_flatline_pct': [],
        'resp_mean_rate': [],
        'resp_std_rate': [],
        'resp_quality': [],
        'resp_max_flatline': [],
        'resp_flatline_pct': [],
    }

    # Create the output folder
    subject_output = output_folder / subject
    os.makedirs(subject_output, exist_ok=True)

    # Load physio files
    subject_folder = bids_folder / subject / 'func'
    
    if not subject_folder.exists():
        print(f'  Warning: No func folder found for {subject}')
        return subject_stats
    
    phys_files = sorted([f for f in os.listdir(subject_folder) if f.endswith('_physio.tsv.gz')])
    
    if not phys_files:
        print(f'  No physio files found for {subject}')
        return subject_stats

    subject_stats['n_runs'] = len(phys_files)

    for phys_file in phys_files:
        print(f'  Processing file: {phys_file}')

        phys_path = subject_folder / phys_file
        json_path = phys_path.with_suffix('').with_suffix('.json')
        
        if not json_path.exists():
            print(f'    Warning: Missing JSON sidecar for {phys_file}, skipping')
            continue
        
        # Extract run identifier from filename
        file_identifier = None
        for part in phys_file.replace('_physio.tsv.gz', '').split('_'):
            if part.startswith('run-'):
                file_identifier = part
                break
        if file_identifier is None:
            file_identifier = f'run-{subject_stats["n_runs"] + 1:02d}'
        subject_stats['filenames'].append(file_identifier)

        # Load metadata
        with open(json_path, 'r') as f:
            phys_meta = json.load(f)
        
        if 'SamplingFrequency' not in phys_meta:
            print(f'    Warning: No SamplingFrequency in {json_path.name}, skipping')
            continue
            
        sampling_rates = phys_meta['SamplingFrequency']
        if isinstance(sampling_rates, (int, float)):
            sampling_rates = [sampling_rates] * len(phys_meta.get('Columns', []))
        
        print(f'    Sampling Rate(s): {sampling_rates} Hz')

        # Load physio data
        try:
            phys_data = pd.read_csv(phys_path, sep='\t', compression='gzip')
        except Exception as e:
            print(f'    Error reading {phys_file}: {e}')
            continue
        
        # Rename columns based on metadata
        if 'Columns' in phys_meta:
            column_names = [col.split('_')[-1] for col in phys_meta['Columns']]
            phys_data.columns = column_names
        
        # Process cardiac (PPG) signal if present
        if 'cardiac' in phys_data.columns:
            cardiac_sr = sampling_rates[0] if isinstance(sampling_rates, list) else sampling_rates
            plot_file = subject_output / phys_file.replace('_physio.tsv.gz', '_physio_qc_cardiac.png')
            
            try:
                signals, info = nk.ppg_process(
                    phys_data['cardiac'], 
                    sampling_rate=cardiac_sr
                )
                
                # Generate and save plot
                fig = nk.ppg_plot(signals, info)
                fig = plt.gcf()
                fig.set_size_inches(16, 10)
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f'    ✓ Cardiac plot saved: {plot_file.name}')
                
                # Extract summary statistics
                if 'PPG_Rate' in signals.columns:
                    hr_mean = signals['PPG_Rate'].mean()
                    hr_std = signals['PPG_Rate'].std()
                    subject_stats['cardiac_mean_hr'].append(hr_mean)
                    subject_stats['cardiac_std_hr'].append(hr_std)
                    print(f'      Mean HR: {hr_mean:.1f} ± {hr_std:.1f} bpm')
                
                # Extract HRV metrics if available
                if 'PPG_Peaks' in signals.columns:
                    peaks = np.where(signals['PPG_Peaks'] == 1)[0]
                    if len(peaks) > 1:
                        ibi = np.diff(peaks) / cardiac_sr * 1000  # ms
                        if len(ibi) > 1:
                            rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))
                            sdnn = np.std(ibi)
                            subject_stats['cardiac_rmssd'].append(rmssd)
                            subject_stats['cardiac_sdnn'].append(sdnn)
                            print(f'      HRV: RMSSD={rmssd:.1f} ms, SDNN={sdnn:.1f} ms')
                
                # Extract NeuroKit2 signal quality
                if 'PPG_Clean' in signals.columns:
                    ppg_quality = nk.ppg_quality(signals['PPG_Clean'], sampling_rate=cardiac_sr, method="templatematch")
                    ratio_good = (ppg_quality > 0.5).sum() / len(ppg_quality)
                    subject_stats['cardiac_quality'].append(ratio_good)
                    print(f'      Signal Quality: {ratio_good:.2f} (ratio good)')
                
                # Detect flat lines in cardiac signal
                flatline_stats = detect_flatlines(phys_data['cardiac'].values, cardiac_sr, threshold_seconds=10.0)
                subject_stats['cardiac_max_flatline'].append(flatline_stats['max_flatline_duration'])
                subject_stats['cardiac_flatline_pct'].append(flatline_stats['flatline_percentage'])
                if flatline_stats['n_flatline_segments'] > 0:
                    print(f'      ⚠ Flat lines: {flatline_stats["n_flatline_segments"]} segments, '
                          f'max={flatline_stats["max_flatline_duration"]:.1f}s, '
                          f'total={flatline_stats["flatline_percentage"]:.1f}%')
                
            except Exception as e:
                print(f'    Error processing cardiac signal: {e}')

        # Process respiratory signal if present
        if 'respiratory' in phys_data.columns:
            resp_sr = sampling_rates[1] if isinstance(sampling_rates, list) and len(sampling_rates) > 1 else sampling_rates[0]
            plot_file = subject_output / phys_file.replace('_physio.tsv.gz', '_physio_qc_respiration.png')
            
            try:
                resp_signals, resp_info = nk.rsp_process(
                    phys_data['respiratory'], 
                    sampling_rate=resp_sr
                )
                
                # Generate and save plot
                fig = nk.rsp_plot(resp_signals, resp_info)
                fig = plt.gcf()
                fig.set_size_inches(16, 10)
                fig.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f'    ✓ Respiratory plot saved: {plot_file.name}')
                
                # Extract respiratory rate statistics
                if 'RSP_Rate' in resp_signals.columns:
                    resp_mean = resp_signals['RSP_Rate'].mean()
                    resp_std = resp_signals['RSP_Rate'].std()
                    subject_stats['resp_mean_rate'].append(resp_mean)
                    subject_stats['resp_std_rate'].append(resp_std)
                    print(f'      Mean Resp Rate: {resp_mean:.1f} ± {resp_std:.1f} breaths/min')
                
                # Extract respiratory quality
                if 'RSP_Quality' in resp_signals.columns:
                    ratio_good = (resp_signals['RSP_Quality'] > 0.5).sum() / len(resp_signals['RSP_Quality'])
                    subject_stats['resp_quality'].append(ratio_good)
                    print(f'      Signal Quality: {ratio_good:.2f} (ratio good)')
                
                # Detect flat lines in respiratory signal (45s threshold)
                flatline_stats = detect_flatlines(phys_data['respiratory'].values, resp_sr, threshold_seconds=45.0)
                subject_stats['resp_max_flatline'].append(flatline_stats['max_flatline_duration'])
                subject_stats['resp_flatline_pct'].append(flatline_stats['flatline_percentage'])
                if flatline_stats['n_flatline_segments'] > 0:
                    print(f'      ⚠ Flat lines: {flatline_stats["n_flatline_segments"]} segments, '
                          f'max={flatline_stats["max_flatline_duration"]:.1f}s, '
                          f'total={flatline_stats["flatline_percentage"]:.1f}%')
                
            except Exception as e:
                print(f'    Error processing respiratory signal: {e}')
    
    return subject_stats


def create_summary_dataframes(all_stats: list) -> tuple:
    """
    Create PPG and RSP summary dataframes from collected statistics.
    
    Parameters:
    -----------
    all_stats : list
        List of subject statistics dictionaries
        
    Returns:
    --------
    tuple
        (ppg_df, rsp_df) - DataFrames with PPG and RSP summary statistics
    """
    # Collect all unique file identifiers across subjects
    all_file_ids = set()
    for stats in all_stats:
        all_file_ids.update(stats['filenames'])
    all_file_ids = sorted(list(all_file_ids))
    
    # Create PPG (cardiac) dataframe
    ppg_data = []
    for stats in all_stats:
        row = {'subject_id': stats['subject_id'], 'n_runs': stats['n_runs']}
        file_to_idx = {fname: idx for idx, fname in enumerate(stats['filenames'])}
        
        for file_id in all_file_ids:
            idx = file_to_idx.get(file_id)
            prefix = file_id.replace('-', '')
            
            # Heart rate
            if idx is not None and idx < len(stats['cardiac_mean_hr']):
                row[f'{prefix}_mean_hr'] = stats['cardiac_mean_hr'][idx]
                row[f'{prefix}_std_hr'] = stats['cardiac_std_hr'][idx] if idx < len(stats['cardiac_std_hr']) else np.nan
            else:
                row[f'{prefix}_mean_hr'] = np.nan
                row[f'{prefix}_std_hr'] = np.nan
            
            # HRV metrics
            if idx is not None and idx < len(stats['cardiac_rmssd']):
                row[f'{prefix}_rmssd'] = stats['cardiac_rmssd'][idx]
                row[f'{prefix}_sdnn'] = stats['cardiac_sdnn'][idx] if idx < len(stats['cardiac_sdnn']) else np.nan
            else:
                row[f'{prefix}_rmssd'] = np.nan
                row[f'{prefix}_sdnn'] = np.nan
            
            # Quality
            if idx is not None and idx < len(stats['cardiac_quality']):
                row[f'{prefix}_quality_ratio'] = stats['cardiac_quality'][idx]
            else:
                row[f'{prefix}_quality_ratio'] = np.nan
            
            # Flatline
            if idx is not None and idx < len(stats['cardiac_max_flatline']):
                row[f'{prefix}_max_flatline'] = stats['cardiac_max_flatline'][idx]
                row[f'{prefix}_flatline_pct'] = stats['cardiac_flatline_pct'][idx] if idx < len(stats['cardiac_flatline_pct']) else 0.0
            else:
                row[f'{prefix}_max_flatline'] = 0.0
                row[f'{prefix}_flatline_pct'] = 0.0
        
        # Summary statistics
        row['avg_mean_hr'] = np.mean(stats['cardiac_mean_hr']) if stats['cardiac_mean_hr'] else np.nan
        row['avg_quality_ratio'] = np.mean(stats['cardiac_quality']) if stats['cardiac_quality'] else np.nan
        row['max_flatline'] = np.max(stats['cardiac_max_flatline']) if stats['cardiac_max_flatline'] else 0.0
        
        ppg_data.append(row)
    
    ppg_df = pd.DataFrame(ppg_data)
    
    # Create RSP (respiratory) dataframe
    rsp_data = []
    for stats in all_stats:
        row = {'subject_id': stats['subject_id'], 'n_runs': stats['n_runs']}
        file_to_idx = {fname: idx for idx, fname in enumerate(stats['filenames'])}
        
        for file_id in all_file_ids:
            idx = file_to_idx.get(file_id)
            prefix = file_id.replace('-', '')
            
            # Respiratory rate
            if idx is not None and idx < len(stats['resp_mean_rate']):
                row[f'{prefix}_mean_rate'] = stats['resp_mean_rate'][idx]
                row[f'{prefix}_std_rate'] = stats['resp_std_rate'][idx] if idx < len(stats['resp_std_rate']) else np.nan
            else:
                row[f'{prefix}_mean_rate'] = np.nan
                row[f'{prefix}_std_rate'] = np.nan
            
            # Quality
            if idx is not None and idx < len(stats['resp_quality']):
                row[f'{prefix}_quality_ratio'] = stats['resp_quality'][idx]
            else:
                row[f'{prefix}_quality_ratio'] = np.nan
            
            # Flatline
            if idx is not None and idx < len(stats['resp_max_flatline']):
                row[f'{prefix}_max_flatline'] = stats['resp_max_flatline'][idx]
                row[f'{prefix}_flatline_pct'] = stats['resp_flatline_pct'][idx] if idx < len(stats['resp_flatline_pct']) else 0.0
            else:
                row[f'{prefix}_max_flatline'] = 0.0
                row[f'{prefix}_flatline_pct'] = 0.0
        
        # Summary statistics
        row['avg_mean_rate'] = np.mean(stats['resp_mean_rate']) if stats['resp_mean_rate'] else np.nan
        row['avg_quality_ratio'] = np.mean(stats['resp_quality']) if stats['resp_quality'] else np.nan
        row['max_flatline'] = np.max(stats['resp_max_flatline']) if stats['resp_max_flatline'] else 0.0
        
        rsp_data.append(row)
    
    rsp_df = pd.DataFrame(rsp_data)
    
    return ppg_df, rsp_df


def flag_recordings_for_review(all_stats: list) -> list:
    """
    Identify recordings with quality issues for manual review.
    
    Parameters:
    -----------
    all_stats : list
        List of subject statistics dictionaries
        
    Returns:
    --------
    list
        List of flagged recording dictionaries
    """
    flagged_recordings = []
    
    for stats in all_stats:
        subject = stats['subject_id']
        for i, file_id in enumerate(stats['filenames']):
            issues = []
            
            # Check cardiac quality
            if i < len(stats['cardiac_quality']) and stats['cardiac_quality'][i] < QUALITY_THRESHOLD:
                issues.append(f"PPG quality {stats['cardiac_quality'][i]:.2f}")
            if i < len(stats['cardiac_max_flatline']) and stats['cardiac_max_flatline'][i] > 10.0:
                issues.append(f"PPG flatline {stats['cardiac_max_flatline'][i]:.1f}s")
            
            # Check heart rate range
            if i < len(stats['cardiac_mean_hr']):
                hr = stats['cardiac_mean_hr'][i]
                if hr < HR_MIN:
                    issues.append(f"HR low {hr:.1f} bpm")
                elif hr > HR_MAX:
                    issues.append(f"HR high {hr:.1f} bpm")
            
            # Check respiratory quality
            if i < len(stats['resp_quality']) and stats['resp_quality'][i] < QUALITY_THRESHOLD:
                issues.append(f"RSP quality {stats['resp_quality'][i]:.2f}")
            if i < len(stats['resp_max_flatline']) and stats['resp_max_flatline'][i] > 45.0:
                issues.append(f"RSP flatline {stats['resp_max_flatline'][i]:.1f}s")
            
            # Check respiratory rate range
            if i < len(stats['resp_mean_rate']):
                rr = stats['resp_mean_rate'][i]
                if rr < RESP_MIN:
                    issues.append(f"RR low {rr:.1f} br/min")
                elif rr > RESP_MAX:
                    issues.append(f"RR high {rr:.1f} br/min")
            
            if issues:
                flagged_recordings.append({
                    'subject_id': subject,
                    'file_id': file_id,
                    'issues': '; '.join(issues),
                    'ppg_useable': '',
                    'rsp_useable': '',
                    'notes': '',
                    'reviewed_by': '',
                    'review_date': ''
                })
    
    return flagged_recordings


def main():
    """Main processing loop for all subjects."""
    
    project_folder = Path('/home/ASDPrecision/')
    bids_folder = project_folder / 'data' / 'bids'
    output_folder = project_folder / 'quality_metrics' / 'physio_qc'
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Create BIDS dataset_description.json
    dataset_description = {
        "Name": "Physiological Signal Quality Control",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "physio_qc pipeline",
                "Version": "1.0",
                "CodeURL": "https://github.com/neuropsychology/NeuroKit",
                "Container": {
                    "Type": "python",
                    "Tag": f"python-{sys.version.split()[0]}"
                }
            }
        ],
        "SourceDatasets": [
            {
                "URL": str(bids_folder),
                "Version": "1.0"
            }
        ],
        "HowToAcknowledge": "Please cite NeuroKit2: Makowski, D., et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. Behavior Research Methods, 53(4), 1689-1696.",
        "License": "CC0",
        "ProcessingMethod": {
            "Cardiac": {
                "Algorithm": "NeuroKit2 ppg_process",
                "Description": "PPG signal processing with peak detection and HRV analysis",
                "Metrics": ["Heart Rate (mean, std)", "HRV (RMSSD, SDNN)"]
            },
            "Respiratory": {
                "Algorithm": "NeuroKit2 rsp_process",
                "Description": "Respiratory signal processing with rate estimation",
                "Metrics": ["Respiratory Rate (mean, std)"]
            }
        },
        "DateProcessed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    description_file = output_folder / 'dataset_description.json'
    with open(description_file, 'w') as f:
        json.dump(dataset_description, f, indent=4)
    print(f'Created BIDS dataset_description.json: {description_file}\n')
    
    # Get subject list
    subject_list = sorted([
        sub for sub in os.listdir(bids_folder) 
        if sub.startswith('sub-') and (bids_folder / sub).is_dir()
    ])
    print(f'Found {len(subject_list)} subjects to process\n')
    
    if not subject_list:
        print('No subjects found. Exiting.')
        return
    
    # Process each subject
    all_stats = []
    for subject in subject_list:
        try:
            subject_stats = process_subject_physio(subject, bids_folder, output_folder)
            all_stats.append(subject_stats)
        except Exception as e:
            print(f'Error processing {subject}: {e}')
            continue
    
    # Create summary dataframes
    if all_stats:
        ppg_df, rsp_df = create_summary_dataframes(all_stats)
        
        # Save summaries
        ppg_file = output_folder / 'physio_qc_ppg.tsv'
        ppg_df.to_csv(ppg_file, sep='\t', index=False, na_rep='n/a')
        print(f'\n✓ PPG (cardiac) statistics saved to: {ppg_file}')
        
        rsp_file = output_folder / 'physio_qc_rsp.tsv'
        rsp_df.to_csv(rsp_file, sep='\t', index=False, na_rep='n/a')
        print(f'✓ RSP (respiratory) statistics saved to: {rsp_file}')
        
        # Print group statistics
        print('\nGroup-level summary:')
        print(f"  Mean HR across all subjects: {ppg_df['avg_mean_hr'].mean():.1f} ± {ppg_df['avg_mean_hr'].std():.1f} bpm")
        print(f"  Mean Resp Rate across all subjects: {rsp_df['avg_mean_rate'].mean():.1f} ± {rsp_df['avg_mean_rate'].std():.1f} breaths/min")
        print(f"  Mean PPG Quality: {ppg_df['avg_quality_ratio'].mean():.2f} (ratio good)")
        print(f"  Mean RSP Quality: {rsp_df['avg_quality_ratio'].mean():.2f} (ratio good)")
        
        # Report subjects with flat lines
        cardiac_flatline_subjects = ppg_df[ppg_df['max_flatline'] > 10.0]['subject_id'].tolist()
        resp_flatline_subjects = rsp_df[rsp_df['max_flatline'] > 10.0]['subject_id'].tolist()
        if cardiac_flatline_subjects:
            print(f"\n  ⚠ Subjects with cardiac flat lines >10s: {', '.join(cardiac_flatline_subjects)}")
        if resp_flatline_subjects:
            print(f"  ⚠ Subjects with respiratory flat lines >10s: {', '.join(resp_flatline_subjects)}")
        
        # Flag recordings for review
        flagged_recordings = flag_recordings_for_review(all_stats)
        
        print('\n--- Recordings with quality issues ---')
        if flagged_recordings:
            for rec in flagged_recordings:
                print(f"  ⚠ {rec['subject_id']}_{rec['file_id']}: {rec['issues']}")
            
            # Save review file
            review_df = pd.DataFrame(flagged_recordings)
            review_file = output_folder / 'physio_qc_review.tsv'
            if not os.path.exists(review_file):
                review_df.to_csv(review_file, sep='\t', index=False)
            print(f'\n✓ Review file saved to: {review_file}')
            print(f'  {len(flagged_recordings)} recordings flagged for manual review')
            print('  Fill in ppg_useable/rsp_useable columns with: yes, no, or partial')
        else:
            print("  ✓ No recordings with quality issues found")
    
    print(f'\n✓ Physio QC processing complete')
    print(f'Reports saved to: {output_folder}')


if __name__ == '__main__':
    main()
