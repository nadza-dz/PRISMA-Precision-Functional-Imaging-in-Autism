#!/usr/bin/env python3
"""
Visualize structural connectomes grouped by Yeo 7 networks.

Loads connectivity matrices from qsirecon output and creates visualizations
with regions sorted and grouped by their corresponding Yeo network.
Separate visualizations are created for autism and control groups.

Usage:
    python -m dwi.plot_connectome

Author: Joe Bathelt
Date: December 2025
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['CMU Sans Serif', 'DejaVu Sans']

# Default paths
QSIRECON_DIR = "/home/ASDPrecision/data/bids/derivatives/qsirecon/derivatives/qsirecon-MRtrix3_fork-SS3T_act-HSVS"
PARTICIPANTS_TSV = "/home/ASDPrecision/data/bids/participants.tsv"
OUTPUT_DIR = "/home/ASDPrecision/quality_metrics/structural_connectome_visualizations"

# Yeo 7 network colors - muted/desaturated versions
NETWORK_COLORS = {
    'Vis': '#7B5A90',
    'SomMot': '#5A7FA8',
    'DorsAttn': '#4A8C5C',
    'SalVentAttn': '#B87DB8',
    'Limbic': '#A8C99A',
    'Cont': '#C9956A',
    'Default': '#B56B6B',
    'Subcortical': '#8A8A8A',
    'Cerebellum': '#6B6B6B'
}

KNOWN_NETWORKS = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
NETWORK_ORDER = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', 'Subcortical', 'Cerebellum']


def parse_label(label: str) -> tuple:
    """
    Parse a region label to extract hemisphere and network.
    
    Parameters:
    -----------
    label : str
        Region label from atlas
        
    Returns:
    --------
    tuple
        (hemisphere, network)
    """
    if label.startswith('Cerebellar'):
        return 'Both', 'Cerebellum'
    
    if '-' in label:
        hemi = label.split('-')[0]
        return hemi, 'Subcortical'
    
    parts = label.split('_')
    
    if len(parts) < 2:
        return 'Unknown', 'Unknown'
    
    hemi = parts[0]
    
    if parts[1] in ['Hippocampus', 'Amygdala']:
        return hemi, 'Subcortical'
    
    network = parts[1]
    if network in KNOWN_NETWORKS:
        return hemi, network
    
    return hemi, 'Unknown'


def load_participants_info() -> tuple:
    """Load participant diagnosis information from participants.tsv."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep='\t')
    
    autism_subjects = []
    control_subjects = []
    
    for _, row in df.iterrows():
        sub_id = row['participant_id']
        if not sub_id.startswith('sub-'):
            sub_id = f"sub-{sub_id}"
        
        diagnosis = str(row.get('autism_diagnosis', '')).lower().strip()
        
        if diagnosis == 'yes':
            autism_subjects.append(sub_id)
        elif diagnosis == 'no':
            control_subjects.append(sub_id)
    
    return autism_subjects, control_subjects


def load_connectivity_matrix(sub_id: str) -> tuple:
    """Load connectivity matrix from qsirecon output."""
    mat_file = os.path.join(QSIRECON_DIR, sub_id, 'dwi', f'{sub_id}_space-ACPC_connectivity.mat')
    
    if not os.path.exists(mat_file):
        print(f"  Warning: No connectivity file for {sub_id}")
        return None, None
    
    mat = sio.loadmat(mat_file)
    
    conn_key = 'atlas_4S256Parcels_sift2_invnodevol_radius2_count_connectivity'
    if conn_key not in mat:
        for key in mat.keys():
            if 'connectivity' in key.lower() and not key.startswith('_'):
                conn_key = key
                break
    
    conn_matrix = mat[conn_key]
    labels = [l[0] for l in mat['atlas_4S256Parcels_region_labels'][0]]
    
    return conn_matrix, labels


def sort_by_network(conn_matrix: np.ndarray, labels: list) -> tuple:
    """Sort connectivity matrix so regions are grouped by network."""
    region_info = []
    for i, label in enumerate(labels):
        hemi, network = parse_label(label)
        region_info.append({
            'index': i,
            'label': label,
            'hemi': hemi,
            'network': network
        })
    
    def sort_key(r):
        network_idx = NETWORK_ORDER.index(r['network']) if r['network'] in NETWORK_ORDER else 999
        hemi_idx = 0 if r['hemi'] == 'LH' else (1 if r['hemi'] == 'RH' else 2)
        return (network_idx, hemi_idx, r['label'])
    
    sorted_info = sorted(region_info, key=sort_key)
    sorted_indices = [r['index'] for r in sorted_info]
    
    sorted_matrix = conn_matrix[np.ix_(sorted_indices, sorted_indices)]
    sorted_labels = [r['label'] for r in sorted_info]
    sorted_networks = [r['network'] for r in sorted_info]
    
    return sorted_matrix, sorted_labels, sorted_networks


def get_network_boundaries(networks: list) -> list:
    """Get the indices where network changes occur."""
    boundaries = [0]
    current_network = networks[0]
    
    for i, network in enumerate(networks):
        if network != current_network:
            boundaries.append(i)
            current_network = network
    
    boundaries.append(len(networks))
    return boundaries


def plot_connectome(conn_matrix: np.ndarray, labels: list, networks: list, 
                    title: str, output_path: str):
    """Create a connectome visualization with network groupings."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    conn_log = np.log10(conn_matrix + 1)
    im = ax.imshow(conn_log, cmap=plt.cm.magma, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('log10(Streamline count + 1)', fontsize=12)
    
    boundaries = get_network_boundaries(networks)
    
    # Add boxes around diagonal
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        size = end - start
        rect = mpatches.Rectangle((start - 0.5, start - 0.5), size, size,
                                   fill=False, edgecolor='white', linewidth=2.0, alpha=1.0)
        ax.add_patch(rect)
    
    # Draw network dividing lines
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='white', linewidth=0.75, alpha=1.0, zorder=10)
        ax.axvline(b - 0.5, color='white', linewidth=0.75, alpha=1.0, zorder=10)
    
    # Add colored bars for networks
    bar_width = 5
    network_centers = {}
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        network = networks[start_idx]
        network_centers[network] = (start_idx + end_idx) / 2
        color = NETWORK_COLORS.get(network, '#808080')
        
        rect = mpatches.Rectangle((-bar_width - 2, start_idx - 0.5), bar_width, end_idx - start_idx,
                                   color=color, clip_on=False)
        ax.add_patch(rect)
        
        rect = mpatches.Rectangle((start_idx - 0.5, -bar_width - 2), end_idx - start_idx, bar_width,
                                   color=color, clip_on=False)
        ax.add_patch(rect)
    
    # Create legend
    legend_patches = []
    for network in NETWORK_ORDER:
        if network in network_centers:
            color = NETWORK_COLORS.get(network, '#808080')
            legend_patches.append(mpatches.Patch(color=color, label=network))
    
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.15, 1), fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def compute_group_mean(subjects: list) -> tuple:
    """Compute mean connectivity matrix for a group of subjects."""
    matrices = []
    labels = None
    
    for sub_id in subjects:
        conn, sub_labels = load_connectivity_matrix(sub_id)
        if conn is not None:
            matrices.append(conn)
            if labels is None:
                labels = sub_labels
    
    if len(matrices) == 0:
        return None, None
    
    stacked = np.stack(matrices, axis=0)
    mean_matrix = np.mean(stacked, axis=0)
    
    return mean_matrix, labels


def plot_connectome_sidebyside(autism_matrix: np.ndarray, control_matrix: np.ndarray, 
                               networks: list, output_path: str, n_autism: int, n_control: int):
    """Create a side-by-side connectome visualization for ASC and CMP groups."""
    fig_width_inches = 100 / 25.4
    fig_height_inches = 56 / 25.4
    
    fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
    
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.08], wspace=0.15, 
                          left=0.05, right=0.92, top=0.85, bottom=0.18)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    cax_pos = gs[0, 2].get_position(fig)
    cax_height = cax_pos.height * 0.67
    cax_bottom = cax_pos.y0 + (cax_pos.height - cax_height) / 2
    cax = fig.add_axes([cax_pos.x0, cax_bottom, cax_pos.width, cax_height])
    
    axes = [ax1, ax2]
    boundaries = get_network_boundaries(networks)
    
    autism_log = np.log10(autism_matrix + 1)
    control_log = np.log10(control_matrix + 1)
    vmin = min(autism_log.min(), control_log.min())
    vmax = max(autism_log.max(), control_log.max())
    
    matrices = [autism_log, control_log]
    titles = [f'ASC (n=31)', f'CMP (n=32)']
    
    for idx, (ax, matrix, title) in enumerate(zip(axes, matrices, titles)):
        im = ax.imshow(matrix, cmap=plt.cm.magma, aspect='equal', vmin=vmin, vmax=vmax)
        
        n_regions = matrix.shape[0]
        ax.set_xlim(-0.5, n_regions - 0.5)
        ax.set_ylim(n_regions - 0.5, -0.5)
        
        for b in boundaries[1:-1]:
            ax.axhline(b - 0.5, color='white', linewidth=0.25, alpha=1.0)
            ax.axvline(b - 0.5, color='white', linewidth=0.25, alpha=1.0)
        
        bar_width = 3
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = min(boundaries[i + 1], n_regions)
            network = networks[start_idx]
            color = NETWORK_COLORS.get(network, '#808080')
            
            bar_start = start_idx - 0.5
            bar_size = end_idx - start_idx
            
            rect = mpatches.Rectangle((-bar_width - 1, bar_start), bar_width, bar_size,
                                       color=color, clip_on=False, zorder=1)
            ax.add_patch(rect)
            
            rect = mpatches.Rectangle((bar_start, -bar_width - 1), bar_size, bar_width,
                                       color=color, clip_on=False, zorder=1)
            ax.add_patch(rect)
        
        ax.set_title(title, fontsize=9, fontweight='normal', pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('log10(Streamlines)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    legend_patches = []
    network_centers = {}
    for i in range(len(boundaries) - 1):
        network = networks[boundaries[i]]
        if network not in network_centers:
            network_centers[network] = True
    
    for network in NETWORK_ORDER:
        if network in network_centers:
            color = NETWORK_COLORS.get(network, '#808080')
            legend_patches.append(mpatches.Patch(color=color, label=network))
    
    n_cols = 5
    fig.legend(handles=legend_patches, loc='lower center', ncol=n_cols, 
               fontsize=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.0, handleheight=0.7, columnspacing=0.8)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    """Main function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    autism_subjects, control_subjects = load_participants_info()
    print(f"Found {len(autism_subjects)} autism subjects and {len(control_subjects)} control subjects")
    
    print("\nComputing mean connectivity for autism group...")
    autism_mean, labels = compute_group_mean(autism_subjects)
    
    print("\nComputing mean connectivity for control group...")
    control_mean, _ = compute_group_mean(control_subjects)
    
    if autism_mean is None or control_mean is None:
        print("Error: Could not compute group means")
        return
    
    print("\nSorting by Yeo networks...")
    autism_sorted, sorted_labels, networks = sort_by_network(autism_mean, labels)
    control_sorted, _, _ = sort_by_network(control_mean, labels)
    
    print("\nNetwork distribution:")
    network_counts = Counter(networks)
    for network in NETWORK_ORDER:
        if network in network_counts:
            print(f"  {network}: {network_counts[network]} regions")
    
    print("\nCreating visualizations...")
    
    plot_connectome(
        autism_sorted, sorted_labels, networks,
        f'Structural Connectome - Autism Group (n={len(autism_subjects)})',
        os.path.join(OUTPUT_DIR, 'connectome_autism_yeo7.png')
    )
    
    plot_connectome(
        control_sorted, sorted_labels, networks,
        f'Structural Connectome - Control Group (n={len(control_subjects)})',
        os.path.join(OUTPUT_DIR, 'connectome_control_yeo7.png')
    )
    
    plot_connectome_sidebyside(
        autism_sorted, control_sorted, networks,
        os.path.join(OUTPUT_DIR, 'connectome_comparison_yeo7.png'),
        len(autism_subjects), len(control_subjects)
    )
    
    # Compute and plot difference
    diff_matrix = autism_sorted - control_sorted
    
    fig, ax = plt.subplots(figsize=(14, 12))
    vmax = np.percentile(np.abs(diff_matrix), 95)
    im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Difference (Autism - Control)', fontsize=12)
    
    boundaries = get_network_boundaries(networks)
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='grey', linewidth=0.5, alpha=0.7)
        ax.axvline(b - 0.5, color='grey', linewidth=0.5, alpha=0.7)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'connectome_difference_yeo7.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'connectome_difference_yeo7.png')}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
