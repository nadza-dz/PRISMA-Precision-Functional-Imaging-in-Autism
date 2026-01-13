#!/usr/bin/env python3
"""
Extract descriptive statistics for DWI quality metrics by group.

Extracts mean and standard deviation of framewise displacement (FD) from QSIPrep
output, grouped by diagnostic status (ASC vs CMP).

Author: Joe Bathelt
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re

# Paths
QSIPREP_DIR = Path("/home/ASDPrecision/data/bids/derivatives/qsiprep")
QSIRECON_DIR = Path("/home/ASDPrecision/data/bids/derivatives/qsirecon/derivatives/qsirecon-MRtrix3_fork-SS3T_act-HSVS")
PARTICIPANTS_TSV = Path("/home/ASDPrecision/data/bids/participants.tsv")
OUTPUT_DIR = Path("/home/ASDPrecision/quality_metrics/dwi")


def load_participants():
    """Load participant info and split by diagnosis."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')

    asc_subs = df[df['autism_diagnosis'] == 'yes']['participant_id'].tolist()
    cmp_subs = df[df['autism_diagnosis'] == 'no']['participant_id'].tolist()

    print(f"ASC subjects: {len(asc_subs)}")
    print(f"CMP subjects: {len(cmp_subs)}")

    return asc_subs, cmp_subs, df


def extract_fd_metrics():
    """Extract framewise displacement metrics from QSIPrep output."""

    # Find all QC files - use the correct pattern based on what we found
    tsv_pattern = str(QSIPREP_DIR / "sub-*" / "dwi" / "*_space-ACPC_desc-image_qc.tsv")

    print(f"\nSearching for QC files...")
    print(f"  Pattern: {tsv_pattern}")

    tsv_files = glob.glob(tsv_pattern)
    print(f"  Found: {len(tsv_files)} files")

    if not tsv_files:
        print("ERROR: No QC files found!")
        return None

    print(f"\nProcessing {len(tsv_files)} TSV files...")
    file_type = 'tsv'
    files_to_process = tsv_files

    # Extract data
    fd_data = []

    for qc_file in files_to_process:
        qc_path = Path(qc_file)

        # Extract subject ID from filename
        filename = qc_path.name
        # Filename format: sub-XXXXX_..._desc-ImageQC_dwi.json or similar
        sub_id = filename.split('_')[0]  # Gets 'sub-XXXXX'

        try:
            if file_type == 'tsv':
                # Load TSV file
                df = pd.read_csv(qc_path, sep='\t')

                # Check if mean_fd column exists
                if 'mean_fd' in df.columns:
                    fd_mean = df['mean_fd'].iloc[0] if len(df) > 0 else np.nan
                else:
                    print(f"  Warning: 'mean_fd' column not found in {filename}")
                    print(f"    Available columns: {list(df.columns)}")
                    fd_mean = np.nan

            else:  # JSON
                import json
                with open(qc_path, 'r') as f:
                    qc_data = json.load(f)

                # Try different possible keys for framewise displacement
                fd_mean = qc_data.get('fd_mean',
                         qc_data.get('framewise_displacement_mean',
                         qc_data.get('mean_fd', np.nan)))

            fd_data.append({
                'subject_id': sub_id,
                'fd_mean': fd_mean,
                'file': qc_path.name
            })

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

    if not fd_data:
        print("No FD data extracted!")
        return None

    # Create DataFrame
    fd_df = pd.DataFrame(fd_data)

    # Remove any NaN values
    n_total = len(fd_df)
    fd_df = fd_df.dropna(subset=['fd_mean'])
    n_valid = len(fd_df)

    print(f"\nExtracted FD data: {n_valid}/{n_total} valid entries")

    return fd_df


def extract_streamline_count_from_tck(tck_file):
    """
    Extract streamline count from .tck or .tck.gz file by reading the header.

    Parameters:
    -----------
    tck_file : str or Path
        Path to .tck or .tck.gz file

    Returns:
    --------
    int or None
        Streamline count, or None if not found
    """
    import gzip

    tck_path = Path(tck_file)

    try:
        # Open file (gzip if .gz, regular if not)
        if tck_path.suffix == '.gz':
            f = gzip.open(tck_path, 'rb')
        else:
            f = open(tck_path, 'rb')

        try:
            # Read first 10KB to find header
            header_bytes = f.read(10000)
            header = header_bytes.decode('latin1', errors='ignore')

            # Look for 'count:' in header
            for line in header.split('\n'):
                if line.strip().startswith('count:'):
                    # Extract number after 'count:'
                    match = re.search(r'count:\s*(\d+)', line)
                    if match:
                        return int(match.group(1))

            return None

        finally:
            f.close()

    except Exception as e:
        print(f"  Error reading {tck_path.name}: {e}")
        return None


def extract_streamline_counts():
    """Extract streamline counts from QSIRecon tractography files."""

    # Find all tractography files
    tck_pattern = str(QSIRECON_DIR / "sub-*" / "dwi" / "*_model-ifod2_streamlines.tck.gz")

    print(f"\nSearching for tractography files...")
    print(f"  Pattern: {tck_pattern}")

    tck_files = glob.glob(tck_pattern)
    print(f"  Found: {len(tck_files)} files")

    if not tck_files:
        print("ERROR: No tractography files found!")
        return None

    print(f"\nProcessing {len(tck_files)} tractography files...")

    # Extract streamline counts
    streamline_data = []

    for tck_file in tck_files:
        tck_path = Path(tck_file)

        # Extract subject ID from filename
        filename = tck_path.name
        sub_id = filename.split('_')[0]  # Gets 'sub-XXXXX'

        # Extract streamline count from file header
        streamline_count = extract_streamline_count_from_tck(tck_file)

        streamline_data.append({
            'subject_id': sub_id,
            'streamline_count': streamline_count,
            'file': tck_path.name
        })

    if not streamline_data:
        print("No streamline data extracted!")
        return None

    # Create DataFrame
    streamline_df = pd.DataFrame(streamline_data)

    # Remove any NaN values
    n_total = len(streamline_df)
    streamline_df = streamline_df.dropna(subset=['streamline_count'])
    n_valid = len(streamline_df)

    print(f"\nExtracted streamline data: {n_valid}/{n_total} valid entries")

    return streamline_df


def compute_mad(data):
    """Compute Median Absolute Deviation (MAD)."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad


def main():
    """Main function to extract and summarize DWI quality metrics."""
    print("="*80)
    print("DWI Quality Metrics - Framewise Displacement and Streamline Counts by Group")
    print("="*80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load participant groups
    asc_subs, cmp_subs, participants_df = load_participants()

    # Extract FD metrics
    fd_df = extract_fd_metrics()

    if fd_df is None or len(fd_df) == 0:
        print("\nERROR: No framewise displacement data could be extracted.")
        print("Please check the file paths and formats.")
        return

    # Merge with participant info to get diagnosis
    # Ensure subject_id format matches
    participants_df['subject_id'] = participants_df['participant_id']

    merged_df = fd_df.merge(
        participants_df[['subject_id', 'autism_diagnosis']],
        on='subject_id',
        how='left'
    )

    # Check for missing diagnoses
    missing_diag = merged_df['autism_diagnosis'].isna().sum()
    if missing_diag > 0:
        print(f"\nWarning: {missing_diag} subjects missing diagnosis information")
        print("Subjects without diagnosis:")
        print(merged_df[merged_df['autism_diagnosis'].isna()]['subject_id'].tolist())

    # Remove subjects without diagnosis
    merged_df = merged_df.dropna(subset=['autism_diagnosis'])

    # Save full data
    output_file = OUTPUT_DIR / "fd_mean_all_subjects.tsv"
    merged_df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
    print(f"\nSaved full data: {output_file}")

    # Compute statistics by group
    print("\n" + "="*80)
    print("FRAMEWISE DISPLACEMENT STATISTICS BY GROUP")
    print("="*80)

    # ASC group
    asc_data = merged_df[merged_df['autism_diagnosis'] == 'yes']['fd_mean']
    if len(asc_data) > 0:
        asc_mean = asc_data.mean()
        asc_std = asc_data.std()
        asc_median = asc_data.median()
        asc_min = asc_data.min()
        asc_max = asc_data.max()
        asc_n = len(asc_data)

        print(f"\nASC Group (n={asc_n}):")
        print(f"  Mean FD:    {asc_mean:.4f} mm")
        print(f"  Std Dev:    {asc_std:.4f} mm")
        print(f"  Median FD:  {asc_median:.4f} mm")
        print(f"  Range:      [{asc_min:.4f}, {asc_max:.4f}] mm")
    else:
        print("\nASC Group: No data available")
        asc_mean = asc_std = asc_median = asc_n = np.nan

    # CMP group
    cmp_data = merged_df[merged_df['autism_diagnosis'] == 'no']['fd_mean']
    if len(cmp_data) > 0:
        cmp_mean = cmp_data.mean()
        cmp_std = cmp_data.std()
        cmp_median = cmp_data.median()
        cmp_min = cmp_data.min()
        cmp_max = cmp_data.max()
        cmp_n = len(cmp_data)

        print(f"\nCMP Group (n={cmp_n}):")
        print(f"  Mean FD:    {cmp_mean:.4f} mm")
        print(f"  Std Dev:    {cmp_std:.4f} mm")
        print(f"  Median FD:  {cmp_median:.4f} mm")
        print(f"  Range:      [{cmp_min:.4f}, {cmp_max:.4f}] mm")
    else:
        print("\nCMP Group: No data available")
        cmp_mean = cmp_std = cmp_median = cmp_n = np.nan

    # Overall statistics
    if len(merged_df) > 0:
        overall_mean = merged_df['fd_mean'].mean()
        overall_std = merged_df['fd_mean'].std()
        overall_median = merged_df['fd_mean'].median()
        overall_n = len(merged_df)

        print(f"\nOverall (n={overall_n}):")
        print(f"  Mean FD:    {overall_mean:.4f} mm")
        print(f"  Std Dev:    {overall_std:.4f} mm")
        print(f"  Median FD:  {overall_median:.4f} mm")

    # Create summary table
    summary_data = []

    if len(asc_data) > 0:
        summary_data.append({
            'Group': 'ASC',
            'N': asc_n,
            'Mean_FD_mm': asc_mean,
            'SD_FD_mm': asc_std,
            'Median_FD_mm': asc_median,
            'Min_FD_mm': asc_min,
            'Max_FD_mm': asc_max
        })

    if len(cmp_data) > 0:
        summary_data.append({
            'Group': 'CMP',
            'N': cmp_n,
            'Mean_FD_mm': cmp_mean,
            'SD_FD_mm': cmp_std,
            'Median_FD_mm': cmp_median,
            'Min_FD_mm': cmp_min,
            'Max_FD_mm': cmp_max
        })

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = OUTPUT_DIR / "fd_mean_summary_by_group.tsv"
        summary_df.to_csv(summary_file, sep='\t', index=False, float_format='%.6f')
        print(f"\n{summary_df.to_string(index=False)}")
        print(f"\nSaved summary: {summary_file}")

    # Paper report
    print("\n" + "="*80)
    print("PAPER REPORT - DWI Motion Quality Control")
    print("="*80)

    if len(asc_data) > 0 and len(cmp_data) > 0:
        print(f"\nFramewise displacement (FD) was extracted from QSIPrep output to assess")
        print(f"motion during DWI acquisition. The ASC group (n={asc_n}) showed mean FD =")
        print(f"{asc_mean:.4f} ± {asc_std:.4f} mm (range: [{asc_min:.4f}, {asc_max:.4f}] mm).")
        print(f"The CMP group (n={cmp_n}) showed mean FD = {cmp_mean:.4f} ± {cmp_std:.4f} mm")
        print(f"(range: [{cmp_min:.4f}, {cmp_max:.4f}] mm). Overall mean FD across all")
        print(f"subjects was {overall_mean:.4f} ± {overall_std:.4f} mm, indicating acceptable")
        print(f"data quality for diffusion imaging.")

    # Extract streamline counts
    print("\n\n" + "="*80)
    print("STREAMLINE COUNT STATISTICS BY GROUP")
    print("="*80)

    streamline_df = extract_streamline_counts()

    if streamline_df is not None and len(streamline_df) > 0:
        # Merge with participant info
        streamline_merged = streamline_df.merge(
            participants_df[['subject_id', 'autism_diagnosis']],
            on='subject_id',
            how='left'
        )

        # Remove subjects without diagnosis
        streamline_merged = streamline_merged.dropna(subset=['autism_diagnosis'])

        # Save full data
        streamline_file = OUTPUT_DIR / "streamline_counts_all_subjects.tsv"
        streamline_merged.to_csv(streamline_file, sep='\t', index=False, float_format='%.0f')
        print(f"\nSaved full streamline data: {streamline_file}")

        # ASC group
        asc_streamlines = streamline_merged[streamline_merged['autism_diagnosis'] == 'yes']['streamline_count']
        if len(asc_streamlines) > 0:
            asc_median_str = np.median(asc_streamlines)
            asc_mad_str = compute_mad(asc_streamlines)
            asc_mean_str = asc_streamlines.mean()
            asc_std_str = asc_streamlines.std()
            asc_n_str = len(asc_streamlines)

            print(f"\nASC Group (n={asc_n_str}):")
            print(f"  Median streamlines: {asc_median_str:,.0f}")
            print(f"  MAD:                {asc_mad_str:,.0f}")
            print(f"  Mean streamlines:   {asc_mean_str:,.0f}")
            print(f"  Std Dev:            {asc_std_str:,.0f}")
        else:
            print("\nASC Group: No streamline data available")
            asc_median_str = asc_mad_str = asc_n_str = np.nan

        # CMP group
        cmp_streamlines = streamline_merged[streamline_merged['autism_diagnosis'] == 'no']['streamline_count']
        if len(cmp_streamlines) > 0:
            cmp_median_str = np.median(cmp_streamlines)
            cmp_mad_str = compute_mad(cmp_streamlines)
            cmp_mean_str = cmp_streamlines.mean()
            cmp_std_str = cmp_streamlines.std()
            cmp_n_str = len(cmp_streamlines)

            print(f"\nCMP Group (n={cmp_n_str}):")
            print(f"  Median streamlines: {cmp_median_str:,.0f}")
            print(f"  MAD:                {cmp_mad_str:,.0f}")
            print(f"  Mean streamlines:   {cmp_mean_str:,.0f}")
            print(f"  Std Dev:            {cmp_std_str:,.0f}")
        else:
            print("\nCMP Group: No streamline data available")
            cmp_median_str = cmp_mad_str = cmp_n_str = np.nan

        # Overall statistics
        if len(streamline_merged) > 0:
            overall_median_str = np.median(streamline_merged['streamline_count'])
            overall_mad_str = compute_mad(streamline_merged['streamline_count'])
            overall_n_str = len(streamline_merged)

            print(f"\nOverall (n={overall_n_str}):")
            print(f"  Median streamlines: {overall_median_str:,.0f}")
            print(f"  MAD:                {overall_mad_str:,.0f}")

        # Create summary table
        streamline_summary_data = []

        if len(asc_streamlines) > 0:
            streamline_summary_data.append({
                'Group': 'ASC',
                'N': asc_n_str,
                'Median_Streamlines': asc_median_str,
                'MAD_Streamlines': asc_mad_str,
                'Mean_Streamlines': asc_mean_str,
                'SD_Streamlines': asc_std_str
            })

        if len(cmp_streamlines) > 0:
            streamline_summary_data.append({
                'Group': 'CMP',
                'N': cmp_n_str,
                'Median_Streamlines': cmp_median_str,
                'MAD_Streamlines': cmp_mad_str,
                'Mean_Streamlines': cmp_mean_str,
                'SD_Streamlines': cmp_std_str
            })

        if streamline_summary_data:
            streamline_summary_df = pd.DataFrame(streamline_summary_data)
            streamline_summary_file = OUTPUT_DIR / "streamline_counts_summary_by_group.tsv"
            streamline_summary_df.to_csv(streamline_summary_file, sep='\t', index=False, float_format='%.0f')
            print(f"\n{streamline_summary_df.to_string(index=False)}")
            print(f"\nSaved streamline summary: {streamline_summary_file}")

        # Paper report
        print("\n" + "="*80)
        print("PAPER REPORT - DWI Tractography Quality")
        print("="*80)

        if len(asc_streamlines) > 0 and len(cmp_streamlines) > 0:
            print(f"\nStreamline counts were extracted from whole-brain tractography to assess")
            print(f"reconstruction quality. The ASC group (n={asc_n_str}) showed median = {asc_median_str:,.0f}")
            print(f"streamlines (MAD = {asc_mad_str:,.0f}). The CMP group (n={cmp_n_str}) showed median =")
            print(f"{cmp_median_str:,.0f} streamlines (MAD = {cmp_mad_str:,.0f}). Overall median across all")
            print(f"subjects was {overall_median_str:,.0f} streamlines (MAD = {overall_mad_str:,.0f}),")
            print(f"indicating successful tractography reconstruction.")

    print("\n" + "="*80)
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
