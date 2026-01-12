# N, Dzinalija, Dec 2025

# Script to create functional connectome plots per condition split by ASD and HC group.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cmasher as cmr
import numpy as np
import pandas as pd 
import os  
import glob
import sys
sys.path.append("/home/ASDPrecision/code/rs_fMRI")
from plot_connectome import (
    NETWORK_COLORS,
    NETWORK_ORDER,
    get_network_boundaries,
    sort_by_network,
    get_network_boundaries,
)

# Paths
xcpd_dir = "/home/ASDPrecision/data/bids/derivatives/xcpd"
demographics_file = "/home/ASDPrecision/Demographics_Final.csv"
output_dir = "/home/ASDPrecision/quality_metrics/functional_connectome_visualizations"
new_subIDs = pd.read_csv("/home/ASDPrecision/data/participant_renaming.tsv", sep="\t")

# Load demographics
df_demo = pd.read_csv(demographics_file)

# Make sure subject IDs match, add 'sub-' prefix if needed
if not df_demo['ParticipantID'].str.startswith('sub-').all():
    df_demo['ParticipantID'] = 'sub-' + df_demo['ParticipantID'].astype(str)

df = df_demo.merge(new_subIDs,left_on='ParticipantID', right_on='participant_id', how='inner').drop(columns=['participant_id'])
df = df[["new_participant_id","AutismDiagnosis"]]

# Functions
def load_matrix(path):
    '''
    Loads matrix, then
    1) Fisher z-transforms it
    2) Normalizes it to values between -1 and +1
    3) Converts diagonal values to 0

    Outputs matrix and labels
    '''
    df = pd.read_csv(path, sep="\t", index_col=0)
    np.fill_diagonal(df.values, 0)
    #df_z = df.apply(np.arctanh)
    #z_min, z_max = df_z.min().min(), df_z.max().max()
    #df_norm = 2 * (df_z - z_min) / (z_max - z_min) - 1
    #return df_norm.to_numpy(), df_norm.index.tolist()
    return df.to_numpy(), df.index.tolist()

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

    vmin = -1
    vmax = 1
    
    matrices = [autism_matrix, control_matrix]
    titles = [f'ASC (n={n_autism})', f'CMP (n={n_control})']
    
    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, cmap=cmr.fusion_r, aspect='equal', vmin=vmin, vmax=vmax)
        
        n_regions = matrix.shape[0]
        ax.set_xlim(-0.5, n_regions - 0.5)
        ax.set_ylim(n_regions - 0.5, -0.5)
        
        for b in boundaries[1:-1]:
            ax.axhline(b - 0.5, color='white', linewidth=0.25, alpha=1.0)
            ax.axvline(b - 0.5, color='white', linewidth=0.25, alpha=1.0)
        
        bar_width = 9
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
    cbar.set_label('Correlation (r)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    
    legend_patches = [] 
    for network in NETWORK_ORDER:
        if any(n == network for n in networks):
            color = NETWORK_COLORS.get(network, '#808080')
            legend_patches.append(mpatches.Patch(color=color, label=network))
    
    fig.legend(handles=legend_patches, loc='lower center', ncol=5, 
               fontsize=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.0, handleheight=0.7, columnspacing=0.8)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"  Saved: {output_path}")

################################################################################
################     Average Functional connectomes        #####################
################################################################################

conditions = ['rest', 'reality', 'news']
groups = ['ASD', 'HC']
group_matrices = {g: {cond: [] for cond in conditions} for g in groups}

for sub in df["new_participant_id"]:
    print(f"processing {sub}'s matrices")
    ASDdiagnosis = df.loc[df["new_participant_id"]==sub, "AutismDiagnosis"].values[0]

    func_dir = os.path.join(xcpd_dir, sub, "func")

    # Extract files
    rest_files=glob.glob(os.path.join(func_dir,f"{sub}_task-rest_*_space-fsLR_seg-4S456Parcels_stat-pearsoncorrelation_relmat.tsv"))
    news_files = [f for kw in ["farmers", "royals", "poland"] for f in glob.glob(os.path.join(func_dir, f"{sub}_task-{kw}_*_space-fsLR_seg-4S456Parcels_stat-pearsoncorrelation_relmat.tsv"))]
    reality_files = [f for kw in ["fomo", "interviews", "firstdates"] for f in glob.glob(os.path.join(func_dir, f"{sub}_task-{kw}_*_space-fsLR_seg-4S456Parcels_stat-pearsoncorrelation_relmat.tsv"))]

    # Load matrices
    rest_matrices = [load_matrix(f)[0] for f in rest_files]
    news_matrices = [load_matrix(f)[0] for f in news_files]
    reality_matrices = [load_matrix(f)[0] for f in reality_files]

    # Load labels
    _, labels = load_matrix(rest_files[0])

    # Compute subject average per condition
    rest_avg = np.mean(np.stack(rest_matrices, axis=2), axis=2)
    news_avg = np.mean(np.stack(news_matrices, axis=2), axis=2)
    reality_avg = np.mean(np.stack(reality_matrices, axis=2), axis=2)

    # Append to group matrices
    if ASDdiagnosis == "Yes":
        group_matrices["ASD"]["rest"].append(rest_avg)
        group_matrices["ASD"]["news"].append(news_avg)
        group_matrices["ASD"]["reality"].append(reality_avg)
    else:   
        group_matrices["HC"]["rest"].append(rest_avg)
        group_matrices["HC"]["news"].append(news_avg)
        group_matrices["HC"]["reality"].append(reality_avg)

# Calculate the average for each condition per group
for group in groups:
    for cond in conditions:
        matrices = group_matrices[group][cond]
        matrices_avg = np.mean(np.stack(matrices, axis=2), axis=2)

        # Add labels back
        matrices_avg_df = pd.DataFrame(matrices_avg, index=labels, columns=labels)

        # Write matrix to tsv file
        matrices_avg_df.to_csv(os.path.join(output_dir, f"{cond}_{group}_average_mat.tsv"),sep="\t")


################################################################################
################            Plot connectomes               #####################
################################################################################

for cond in conditions:
    asd_avg = np.mean(np.stack(group_matrices["ASD"][cond], axis=2), axis=2)
    hc_avg  = np.mean(np.stack(group_matrices["HC"][cond], axis=2), axis=2)

    # Plot matrices
    asd_sorted, sorted_labels, networks = sort_by_network(asd_avg, labels)
    hc_sorted, _, _ = sort_by_network(hc_avg, labels)

    plot_connectome_sidebyside(
        asd_sorted,
        hc_sorted, 
        networks,
        output_path=os.path.join(output_dir, f"{cond}_ASD_HC.png"),
        n_autism=len(group_matrices["ASD"][cond]),
        n_control=len(group_matrices["HC"][cond])
    )
    
    # Calculate between-group correlations per condition
    # Flatten upper triangle of matrix
    asd_upper = asd_avg[np.triu_indices_from(asd_avg, k=1)] 
    hc_upper  = hc_avg[np.triu_indices_from(hc_avg, k=1)] 

    # Correlate flattened matrix
    r = np.corrcoef(asd_upper, hc_upper)[0, 1]
    print(f"{cond} ASD vs HC correlation: {r:.3f}")





#______________________
# Averaging over more than 3-4 participants already produces correlations of r=0.8+:

# n=1 subjects per group, correlation between group averages: r=0.368
# n=2 subjects per group, correlation between group averages: r=0.645
# n=3 subjects per group, correlation between group averages: r=0.740
# n=4 subjects per group, correlation between group averages: r=0.818
# n=5 subjects per group, correlation between group averages: r=0.845
# n=6 subjects per group, correlation between group averages: r=0.847
# n=7 subjects per group, correlation between group averages: r=0.868
# n=8 subjects per group, correlation between group averages: r=0.878
# n=9 subjects per group, correlation between group averages: r=0.896
# n=10 subjects per group, correlation between group averages: r=0.909
# n=11 subjects per group, correlation between group averages: r=0.918
# n=12 subjects per group, correlation between group averages: r=0.924
# n=13 subjects per group, correlation between group averages: r=0.930
# n=14 subjects per group, correlation between group averages: r=0.939
# n=15 subjects per group, correlation between group averages: r=0.941
# n=16 subjects per group, correlation between group averages: r=0.941 
#______________________

cond = "rest"

asd_mats = group_matrices["ASD"][cond]
hc_mats  = group_matrices["HC"][cond]

n_asd = len(asd_mats)
n_hc  = len(hc_mats)

# Function to compute upper triangle of a matrix
def upper_tri(mat):
    return mat[np.triu_indices_from(mat, k=1)]

# Iterate over split sizes
for n in range(1, 17):
    # Make sure we have enough subjects
    if n > n_asd or n > n_hc:
        continue
    
    # Randomly select n subjects for ASD and n for HC
    asd_split = np.mean(np.stack(asd_mats[:n], axis=2), axis=2)
    hc_split  = np.mean(np.stack(hc_mats[:n], axis=2), axis=2)
    
    # Flatten upper triangles
    asd_u = upper_tri(asd_split)
    hc_u  = upper_tri(hc_split)
    
    # Compute correlation
    r = np.corrcoef(asd_u, hc_u)[0,1]
    
    print(f"n={n} subjects per group, correlation between group averages: r={r:.3f}")
