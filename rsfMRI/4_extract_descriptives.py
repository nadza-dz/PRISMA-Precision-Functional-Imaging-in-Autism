#!/usr/bin/env python3
"""
Extract and analyze top ISC parcels by condition and group.

Computes mean and standard deviation of ISC values across runs/tasks,
grouped by condition (news vs reality TV) and diagnostic group (ASC vs CMP).

Author: Joe Bathelt
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ISC_DIR = Path("/home/ASDPrecision/quality_metrics/isc_maps")
OUTPUT_DIR = Path("/home/ASDPrecision/quality_metrics/isc_maps")
ATLAS_LABELS = Path("/home/ASDPrecision/data/bids/derivatives/qsirecon/atlases/atlas-4S456Parcels/atlas-4S456Parcels_dseg.tsv")

# Task categories
NEWS_TASKS = ['farmers', 'poland', 'royals']
REALITY_TASKS = ['interviews', 'fomo', 'firstdates']
GROUPS = ['ASC', 'CMP']

# Network hierarchy (from sensory to association cortices)
# Based on the Schaefer 7-network parcellation
NETWORK_HIERARCHY = {
    'Vis': {'name': 'Visual', 'order': 1, 'type': 'sensory'},
    'SomMot': {'name': 'Somatomotor', 'order': 2, 'type': 'sensory'},
    'DorsAttn': {'name': 'Dorsal Attention', 'order': 3, 'type': 'association'},
    'SalVentAttn': {'name': 'Salience/Ventral Attention', 'order': 4, 'type': 'association'},
    'Limbic': {'name': 'Limbic', 'order': 5, 'type': 'association'},
    'Cont': {'name': 'Frontoparietal Control', 'order': 6, 'type': 'association'},
    'Default': {'name': 'Default Mode', 'order': 7, 'type': 'association'},
}


def get_network_from_parcel(parcel_name):
    """Extract network name from parcel label."""
    # Remove hemisphere prefix and extract network
    # Format: LH_Network_XX or RH_Network_XX
    parts = parcel_name.split('_')
    if len(parts) >= 2:
        network = parts[1]
        # Handle special cases
        if network == 'Amygdala' or network == 'Hippocampus':
            return 'Limbic'
        return network
    return 'Unknown'


def load_isc_data(group, tasks):
    """
    Load ISC data for a specific group and list of tasks.

    Parameters:
    -----------
    group : str
        'ASC' or 'CMP'
    tasks : list of str
        List of task names

    Returns:
    --------
    df : pd.DataFrame
        DataFrame with columns: parcel, task1_isc, task2_isc, ..., mean_isc, std_isc
    """
    # Load data for each task
    task_data = {}
    parcels = None

    for task in tasks:
        file_path = ISC_DIR / f"isc_{group}_{task}.tsv"
        if file_path.exists():
            df = pd.read_csv(file_path, sep='\t')
            task_data[task] = df['isc'].values
            if parcels is None:
                parcels = df['parcel'].values

    if not task_data:
        return None

    # Create DataFrame with all tasks
    result_df = pd.DataFrame({'parcel': parcels})

    # Add ISC values for each task
    for task in tasks:
        if task in task_data:
            result_df[f'{task}_isc'] = task_data[task]

    # Calculate mean and std across tasks
    isc_columns = [col for col in result_df.columns if col.endswith('_isc')]
    result_df['mean_isc'] = result_df[isc_columns].mean(axis=1)
    result_df['std_isc'] = result_df[isc_columns].std(axis=1)
    result_df['n_tasks'] = result_df[isc_columns].notna().sum(axis=1)

    # Add network information
    result_df['network'] = result_df['parcel'].apply(get_network_from_parcel)

    return result_df


def compute_network_statistics(df):
    """
    Compute ISC statistics grouped by network.

    Parameters:
    -----------
    df : pd.DataFrame
        ISC data with 'network' and 'mean_isc' columns

    Returns:
    --------
    network_stats : pd.DataFrame
        Statistics per network
    """
    network_stats = df.groupby('network')['mean_isc'].agg(['mean', 'std', 'count']).reset_index()
    network_stats.columns = ['network', 'mean_isc', 'std_isc', 'n_parcels']

    # Add network information
    network_stats['network_full_name'] = network_stats['network'].map(
        lambda x: NETWORK_HIERARCHY.get(x, {}).get('name', x)
    )
    network_stats['network_type'] = network_stats['network'].map(
        lambda x: NETWORK_HIERARCHY.get(x, {}).get('type', 'other')
    )
    network_stats['network_order'] = network_stats['network'].map(
        lambda x: NETWORK_HIERARCHY.get(x, {}).get('order', 99)
    )

    # Sort by hierarchy order
    network_stats = network_stats.sort_values('network_order')

    return network_stats


def print_paper_report(group, condition, tasks, df):
    """
    Print a formatted report suitable for paper publication.

    Parameters:
    -----------
    group : str
        'ASC' or 'CMP'
    condition : str
        'News' or 'Reality TV'
    tasks : list of str
        List of task names
    df : pd.DataFrame
        ISC data with network information
    """
    print(f"\n{'='*80}")
    print(f"PAPER REPORT: {group} Group - {condition} Condition")
    print(f"Tasks included: {', '.join(tasks)}")
    print('='*80)

    # Compute network statistics
    net_stats = compute_network_statistics(df)

    # Overall statistics
    print(f"\n{'Overall ISC Statistics:':<40}")
    print(f"{'  Mean ISC across all parcels:':<40} {df['mean_isc'].mean():.3f} ± {df['mean_isc'].std():.3f}")
    print(f"{'  Median ISC:':<40} {df['mean_isc'].median():.3f}")
    print(f"{'  Range:':<40} [{df['mean_isc'].min():.3f}, {df['mean_isc'].max():.3f}]")

    # Sensory cortices
    print(f"\n{'Sensory Cortices:':<40}")
    sensory_nets = net_stats[net_stats['network_type'] == 'sensory']
    for _, row in sensory_nets.iterrows():
        print(f"  {row['network_full_name']:<38} {row['mean_isc']:.3f} ± {row['std_isc']:.3f} (n={int(row['n_parcels'])} parcels)")

    # Association cortices
    print(f"\n{'Association Cortices:':<40}")
    assoc_nets = net_stats[net_stats['network_type'] == 'association']
    for _, row in assoc_nets.iterrows():
        print(f"  {row['network_full_name']:<38} {row['mean_isc']:.3f} ± {row['std_isc']:.3f} (n={int(row['n_parcels'])} parcels)")

    # Summary by type
    print(f"\n{'Summary by Cortical Type:':<40}")
    sensory_mean = sensory_nets['mean_isc'].mean()
    sensory_std = sensory_nets['mean_isc'].std()
    assoc_mean = assoc_nets['mean_isc'].mean()
    assoc_std = assoc_nets['mean_isc'].std()
    print(f"  {'Sensory cortices (mean):':<38} {sensory_mean:.3f} ± {sensory_std:.3f}")
    print(f"  {'Association cortices (mean):':<38} {assoc_mean:.3f} ± {assoc_std:.3f}")

    # Top parcels
    print(f"\n{'Top 5 Parcels by ISC:':<40}")
    top_parcels = df.nlargest(5, 'mean_isc')
    for _, row in top_parcels.iterrows():
        print(f"  {row['parcel']:<38} {row['mean_isc']:.3f} ± {row['std_isc']:.3f} ({row['network']})")

    print("="*80)

    # Save network statistics
    out_file = OUTPUT_DIR / f"isc_{group}_{condition.lower().replace(' ', '_')}_network_stats.tsv"
    net_stats.to_csv(out_file, sep='\t', index=False, float_format='%.6f')
    print(f"\nNetwork statistics saved: {out_file.name}")

    return net_stats


def main():
    """Main function to extract and analyze top ISC parcels."""
    print("Extracting ISC parcel statistics by condition and group...\n")

    # Store all network stats for final comparison
    all_network_stats = []

    # Process each combination of group and condition
    for group in GROUPS:
        # News condition
        news_df = load_isc_data(group, NEWS_TASKS)
        if news_df is not None:
            # Sort by mean ISC (descending)
            news_df_sorted = news_df.sort_values('mean_isc', ascending=False)

            # Save full results
            out_file = OUTPUT_DIR / f"isc_{group}_news_stats.tsv"
            news_df_sorted.to_csv(out_file, sep='\t', index=False, float_format='%.6f')

            # Generate paper report
            net_stats = print_paper_report(group, 'News', NEWS_TASKS, news_df)
            net_stats['group'] = group
            net_stats['condition'] = 'News'
            all_network_stats.append(net_stats)

        # Reality TV condition
        reality_df = load_isc_data(group, REALITY_TASKS)
        if reality_df is not None:
            # Sort by mean ISC (descending)
            reality_df_sorted = reality_df.sort_values('mean_isc', ascending=False)

            # Save full results
            out_file = OUTPUT_DIR / f"isc_{group}_reality_stats.tsv"
            reality_df_sorted.to_csv(out_file, sep='\t', index=False, float_format='%.6f')

            # Generate paper report
            net_stats = print_paper_report(group, 'Reality TV', REALITY_TASKS, reality_df)
            net_stats['group'] = group
            net_stats['condition'] = 'Reality TV'
            all_network_stats.append(net_stats)

    # Create summary comparison table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Mean ISC by Group and Condition")
    print('='*80)

    summary_data = []
    for group in GROUPS:
        news_df = load_isc_data(group, NEWS_TASKS)
        reality_df = load_isc_data(group, REALITY_TASKS)

        if news_df is not None:
            summary_data.append({
                'Group': group,
                'Condition': 'News',
                'Overall_Mean_ISC': news_df['mean_isc'].mean(),
                'Overall_Std_ISC': news_df['mean_isc'].std(),
                'Max_Parcel_ISC': news_df['mean_isc'].max(),
                'Min_Parcel_ISC': news_df['mean_isc'].min(),
                'N_Parcels': len(news_df)
            })

        if reality_df is not None:
            summary_data.append({
                'Group': group,
                'Condition': 'Reality TV',
                'Overall_Mean_ISC': reality_df['mean_isc'].mean(),
                'Overall_Std_ISC': reality_df['mean_isc'].std(),
                'Max_Parcel_ISC': reality_df['mean_isc'].max(),
                'Min_Parcel_ISC': reality_df['mean_isc'].min(),
                'N_Parcels': len(reality_df)
            })

    summary_df = pd.DataFrame(summary_data)
    summary_file = OUTPUT_DIR / "isc_summary_by_condition_group.tsv"
    summary_df.to_csv(summary_file, sep='\t', index=False, float_format='%.6f')

    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\nSaved: {summary_file.name}")

    # Save combined network statistics across all conditions
    if all_network_stats:
        combined_net_stats = pd.concat(all_network_stats, ignore_index=True)
        combined_file = OUTPUT_DIR / "isc_all_network_stats_combined.tsv"
        combined_net_stats.to_csv(combined_file, sep='\t', index=False, float_format='%.6f')
        print(f"Saved: {combined_file.name}")

    print(f"\n\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
