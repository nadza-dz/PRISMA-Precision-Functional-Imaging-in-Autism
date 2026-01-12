# N, Dzinalija, Nov 2025

# Script to extract mriqc quality metrics from T1, T2 and functional data into tabulated form
# and create strip plots of main quality metrics split by ASC and CMP group.
# Metrics chosen for plotting based on AOMIC dataset of L. Snoek (https://doi.org/10.1038/s41597-021-00870-6)

# JB: Added code to extract qsiprep DWI QC metrics and save to TSV.

import json 
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd 
from pathlib import Path
import ptitprince as pt
import re
import seaborn as sns 
from scipy import stats
import csv

# Paths
base_dir = "/home/ASDPrecision"
mriqc_dir = "/home/ASDPrecision/data/bids/derivatives/mriqc"
demographics_file = "/home/ASDPrecision/data/behavioral/Merged_Behavioural_Data_Final.csv"
output_dir = "/home/ASDPrecision/quality_metrics/mriqc"
mriqc_struct_file = os.path.join(output_dir, "mriqc_struct_summary.tsv")
mriqc_func_file = os.path.join(output_dir,"mriqc_func_summary.tsv")
mriqc_dwi_file = os.path.join(output_dir,"mriqc_dwi_summary.tsv")
qsiprep_file = os.path.join(output_dir, "qsiprep_dwi_qc_metrics.tsv")
figures_dir = os.path.join(output_dir, "figures")
bids_dir = "/home/ASDPrecision/data/original_bids"
log_file = os.path.join(output_dir, "mriqc_qc_log.txt")

os.remove(log_file)

# Function to log messages
def log(message, print_terminal=True):
    """Write a message to log file and optionally print to terminal."""
    with open(log_file, "a") as f:
        f.write(message + "\n")
    if print_terminal:
        print(message)

# Load demographics
df_demo = pd.read_csv(demographics_file)
df_demo['AutismDiagnosis'] = df_demo['AutismDiagnosis'].replace({
    "Yes": "ASC",
    "No": "CMP"
})

########################################################################
################      Extract metrics  T1/T2       #####################
########################################################################
# Extract metrics if TSV doesn't exist 
if not os.path.exists(mriqc_struct_file):
    log("MRIQC structural summary file not found. Extracting metrics...")

    # List of metrics to extract
    metrics_to_extract = [
        "cjv","cnr","efc","fber","fwhm_avg","fwhm_x","fwhm_y","fwhm_z",
        "icvs_csf","icvs_gm","icvs_wm","inu_med","inu_range","qi_1","qi_2",
        "rpve_csf","rpve_gm","rpve_wm","size_x","size_y","size_z",
        "snr_csf","snr_gm","snr_total","snr_wm","snrd_csf","snrd_gm","snrd_total","snrd_wm",
        "spacing_x","spacing_y","spacing_z",
        "summary_bg_k","summary_bg_mad","summary_bg_mean","summary_bg_median","summary_bg_n","summary_bg_p05","summary_bg_p95","summary_bg_stdv",
        "summary_csf_k","summary_csf_mad","summary_csf_mean","summary_csf_median","summary_csf_n","summary_csf_p05","summary_csf_p95","summary_csf_stdv",
        "summary_gm_k","summary_gm_mad","summary_gm_mean","summary_gm_median","summary_gm_n","summary_gm_p05","summary_gm_p95","summary_gm_stdv",
        "summary_wm_k","summary_wm_mad","summary_wm_mean","summary_wm_median","summary_wm_n","summary_wm_p05","summary_wm_p95","summary_wm_stdv",
        "tpm_overlap_csf","tpm_overlap_gm","tpm_overlap_wm","wm2max"
    ]

    # Initialize a list to store all results
    all_metrics = []

    # Loop over all folders in mriqc_dir that start with 'sub-'
    for sub in os.listdir(mriqc_dir):
        sub_path = os.path.join(mriqc_dir, sub)
        
        if os.path.isdir(sub_path) and sub.startswith("sub-"):
            anat_path = os.path.join(sub_path, "anat")
            
            if os.path.exists(anat_path):
                log(f"Processing {sub}...")

                # Look for JSON files inside anat
                for fname in os.listdir(anat_path):
                    if fname.endswith(".json"):

                        # Determine modality from filename
                        if "T1w" in fname:
                            modality = "T1w"
                        elif "T2w" in fname:
                            modality = "T2w"
                        else:
                            continue  # skip non-T1/T2 files

                        # Determine run from filename
                        if "run-01" in fname:
                            run = "run-01"
                        elif "run-02" in fname:
                            run = "run-02"
                        else:
                            continue  # skip non-T1/T2 files

                        json_path = os.path.join(anat_path, fname)
                        
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        
                        # Extract the variables defined above
                        metrics = {k: data.get(k, None) for k in metrics_to_extract}
                        
                        # Add subject and modality info
                        metrics = {"SubID": sub, "modality": modality, "run": run, **metrics}
                        
                        all_metrics.append(metrics)
            else:
                log(f"No anat folder for {sub}")


    # Save to CSV
    df_mriqc = pd.DataFrame(all_metrics)
    df_mriqc.to_csv(mriqc_struct_file, sep="\t", index=False)
    log(f"Saved summary to {mriqc_struct_file}")
else:
    log("MRIQC summary already exists. Loading...")
    df_mriqc = pd.read_csv(mriqc_struct_file, sep="\t")


# Check if multiple scans exist for one ppt
scan_counts = df_mriqc.groupby(['SubID', 'modality']).size().reset_index(name='n_scans') 
duplicates = scan_counts[scan_counts['n_scans'] > 1]
if duplicates.empty:
    log("No subjects with multiple scans found.")
else:
    log(f"Subjects with multiple scans ({len(duplicates)} duplicates):")
    for _, row in duplicates.iterrows():
        log(f"{row['SubID']} - {row['modality']} : {row['n_scans']} scans")

# Select best structural scan for each ppt
if not duplicates.empty:
    log("\nSelecting best scan for subjects with multiple entries...")

    metrics = ['cnr', 'cjv', 'efc', 'wm2max']
    best_scan_indices = []
    duplicate_table = []
    best_selection_log = []  

    for _, row in duplicates.iterrows():
        sub = row['SubID']
        modality = row['modality']

        # All scans for this subject+modality
        df_sub = df_mriqc[(df_mriqc['SubID'] == sub) & (df_mriqc['modality'] == modality)].copy()

        # Invert metrics where lower = better
        df_sub['cjv_adj'] = -df_sub['cjv']
        df_sub['efc_adj'] = -df_sub['efc']

        # Compute quality score as a sum of the adjusted metrics
        df_sub['quality_score'] = df_sub['cnr'] + df_sub['cjv_adj'] + df_sub['efc_adj'] + df_sub['wm2max']

        # Identify the best scan (highest quality score)
        best_idx = df_sub['quality_score'].idxmax()
        best_run = df_sub.loc[best_idx, 'run']
        best_score = df_sub.loc[best_idx, 'quality_score']

        # Build table for all duplicate scans of this subject/modality
        for idx, r in df_sub.iterrows():
            duplicate_table.append({
                'SubID': sub,
                'Modality': modality,
                'Run': r['run'],
                'cnr': round(r['cnr'], 3),
                'cjv (inverted)': round(r['cjv_adj'], 3),
                'efc (inverted)': round(r['efc_adj'], 3),
                'wm2max': round(r['wm2max'], 3),
                'Quality_score': round(r['quality_score'], 3),
                'Kept': 'YES' if idx == best_idx else ''
            })

        # Store the best scan index
        best_scan_indices.append(best_idx)
        best_selection_log.append(f"→ Keeping {sub}, {modality}, {best_run} (Quality score = {best_score:.3f})")

    # Convert the duplicate info into a DataFrame
    df_dup_log = pd.DataFrame(duplicate_table)
    log("\nDuplicate scan metrics and quality scores:")
    log(df_dup_log.to_string(index=False))

    # Log which scans were selected
    log("\nSelected scans for each subject:")
    for line in best_selection_log:
        log(line)

    # Remove duplicate scans 
    log("\nRemoving duplicate scans from df_mriqc...")

    dup_pairs = set(tuple(x) for x in duplicates[['SubID', 'modality']].values)

    # Indices of all non-duplicate rows (keep all of them)
    non_dup_indices = [
        idx for idx, r in df_mriqc.iterrows()
        if (r['SubID'], r['modality']) not in dup_pairs
    ]

    # Combine both and keep only unique indices
    all_keep_indices = sorted(set(non_dup_indices + best_scan_indices))

    old_len = len(df_mriqc)
    df_mriqc = df_mriqc.loc[all_keep_indices].reset_index(drop=True)
    new_len = len(df_mriqc)

    log(f"Reduced df_mriqc from {old_len} → {new_len} rows after removing duplicates.")
    df_mriqc.to_csv(os.path.join(output_dir, "mriqc_struct_summary_clean.tsv"), sep="\t", index=False)

    # Verification step 
    scan_counts_after = df_mriqc.groupby(['SubID', 'modality']).size().reset_index(name='n_scans')
    still_dup = scan_counts_after[scan_counts_after['n_scans'] > 1]

    if still_dup.empty:
        log("No remaining duplicate SubID+modality entries in MRIQC summary.")
    else:
        log("WARNING: duplicated entries still remain:")
        log(still_dup.to_string(index=False))

else:
    log("No duplicate scans, keeping all scans.")

df_struct_mriqc = df_mriqc.merge(df_demo[['ParticipantID', 'AutismDiagnosis']],
              left_on='SubID', right_on='ParticipantID', how='inner')

########################################################################
###############   Extract qsiprep metrics for DWI data     #############
########################################################################

project_folder = Path('/home/ASDPrecision/')
bids_folder = project_folder / 'data' / 'bids'
qsiprep_folder = bids_folder / 'derivatives' / 'qsiprep'
subject_list = sorted([sub for sub in bids_folder.glob('sub-*')])

metrics_to_extract = ['mean_fd', 'max_fd', 'max_translation', 'max_rotation', 
           't1_neighbor_corr', 't1_num_bad_slices', 't1_dice_distance',
            'CNR0_mean', 'CNR1_mean']

qsi_metrics_df = []

for subject in subject_list:
    subject_id = subject.name
    print(f'Processing {subject_id}...')

    qsiprep_subject_folder = qsiprep_folder / subject_id / 'dwi'
    metrics_file = qsiprep_subject_folder / f'{subject_id}_space-ACPC_desc-image_qc.tsv'

    if not metrics_file.exists():
        print(f'  Metrics file not found for {subject_id}, skipping.')
        continue
    
    # Load the metrics TSV file
    subject_metrics_df = pd.read_csv(metrics_file, sep='\t')
    subject_metrics_df = subject_metrics_df[metrics_to_extract]
    subject_metrics_df['subject_id'] = subject_id
    qsi_metrics_df.append(subject_metrics_df)

qsi_metrics_df = pd.concat(qsi_metrics_df, ignore_index=True)
qsi_metrics_df.to_csv(qsiprep_file, sep='\t', index=False)

log("DWI quality metrics from qsiprep extracted and saved.")

########################################################################
#############    Extract mriqc metrics for DWI data    #################
########################################################################

# Extract metrics if TSV doesn't exist 
if not os.path.exists(mriqc_dwi_file):
    log("MRIQC diffusion summary file not found. Extracting metrics...")
    
    metrics_to_extract = [
        "NumberOfShells", "bdiffs_max", "bdiffs_mean", 
        "bdiffs_median", "bdiffs_min", "efc_shell01", "efc_shell02", "fa_degenerate",
        "fa_nans","fber_shell01", "fber_shell02", "fd_mean", "fd_num", "fd_perc", "ndc", 
        "sigma_cc", "sigma_pca", "sigma_piesno", "snr_cc_shell0", 
        "snr_cc_shell1_best", "snr_cc_shell1_worst", "spikes_global", "spikes_slice_i", 
        "spikes_slice_j", "spikes_slice_k", "summary_bg_k", "summary_bg_mad", "summary_bg_mean", 
        "summary_bg_median", "summary_bg_n", "summary_bg_p05", "summary_bg_p95", "summary_bg_stdv", 
        "summary_fg_k", "summary_fg_mad", "summary_fg_mean", "summary_fg_median", "summary_fg_n", 
        "summary_fg_p05", "summary_fg_p95", "summary_fg_stdv", "summary_wm_k", "summary_wm_mad", 
        "summary_wm_mean", "summary_wm_median", "summary_wm_n", "summary_wm_p05", "summary_wm_p95", 
        "summary_wm_stdv"]


    all_metrics = []

    # Loop over all folders in mriqc directory that start with 'sub-'
    for sub in os.listdir(mriqc_dir):
        sub_path = os.path.join(mriqc_dir, sub)
        
        if os.path.isdir(sub_path) and sub.startswith("sub-"):
            func_path = os.path.join(sub_path, "dwi")
            
            if os.path.exists(func_path):
                log(f"Processing {sub}...")

                # Look for JSON files inside func
                for fname in os.listdir(func_path):
                    if fname.endswith("_dwi.json"):

                        json_path = os.path.join(func_path, fname)
                        
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        
                        # Infer task
                        acq = data.get("bids_meta", {}).get("acquisition", None)

                        # Extract the variables defined above
                        metrics = {k: data.get(k, None) for k in metrics_to_extract}
                        
                        # Add subject and modality info
                        metrics = {"SubID": sub, "direction": acq, **metrics}
                        
                        all_metrics.append(metrics)
            else:
                log(f"No dwi folder for {sub}")

    # Save to CSV
    dwi_mriqc = pd.DataFrame(all_metrics)
    dwi_mriqc.to_csv(mriqc_dwi_file, sep="\t", index=False)
    log(f"Saved diffusion MRIQC summary stats to {mriqc_dwi_file}")
else:
    log("MRIQC summary already exists. Loading...")
    dwi_mriqc = pd.read_csv(mriqc_dwi_file, sep="\t")

df_dwi_mriqc = dwi_mriqc.merge(df_demo[['ParticipantID', 'AutismDiagnosis']],
              left_on='SubID', right_on='ParticipantID', how='inner')

########################################################################
################    Extract metrics for func data   ####################
########################################################################

# Extract metrics if TSV doesn't exist 
if not os.path.exists(mriqc_func_file):
    log("MRIQC functional summary file not found. Extracting metrics...")

    reality_videos = ["fomo", "interviews", "firstdates"]
    newsclip_videos = ["poland", "farmers", "royals"]

    ses_runs = {
    "ses1": [1, 2, 3],
    "ses2": [4, 5, 6],
    "ses3": [7, 8, 9]
}

    def map_video_category(video,run):

        run_int = int(run)

        # Determine session for rest runs
        session_label = None
        if run_int is not None:
            for ses, runs in ses_runs.items():
                if run_int in runs:
                    session_label = ses
                    break

        if video == "rest":
            return f"Rest_{session_label}"
    
        if video in reality_videos:
            category = "Reality show"
        elif video in newsclip_videos:
            category = "News cast"

        return f"{category}_{video}"
    
    metrics_to_extract = [
        "aor", "aqi", "dummy_trs", "dvars_nstd", "dvars_std", "dvars_vstd", "efc",
        "fber", "fd_mean", "fd_num", "fd_perc", "fwhm_avg", "fwhm_x",
        "fwhm_y", "fwhm_z", "gcor", "gsr_x", "gsr_y",     "size_t",
        "size_x", "size_y", "size_z","snr", "spacing_tr", "spacing_x",
        "spacing_y", "spacing_z", "summary_bg_k", "summary_bg_mad",
        "summary_bg_mean", "summary_bg_median", "summary_bg_n", "summary_bg_p05",
        "summary_bg_p95", "summary_bg_stdv", "summary_fg_k", "summary_fg_mad",
        "summary_fg_mean", "summary_fg_median", "summary_fg_n", "summary_fg_p05",
        "summary_fg_p95", "summary_fg_stdv", "tsnr",
    ]

    all_metrics = []

    # Loop over all folders in mriqc directory that start with 'sub-'
    for sub in os.listdir(mriqc_dir):
        sub_path = os.path.join(mriqc_dir, sub)
        
        if os.path.isdir(sub_path) and sub.startswith("sub-"):
            func_path = os.path.join(sub_path, "func")
            
            if os.path.exists(func_path):
                log(f"Processing {sub}...")

                # Look for JSON files inside func
                for fname in os.listdir(func_path):
                    if fname.endswith("_bold.json"):

                        json_path = os.path.join(func_path, fname)
                        
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        
                        # Infer task
                        video = data.get("bids_meta", {}).get("task", None)
                        run = data.get("bids_meta", {}).get("run", None)
                        condition = map_video_category(video,run)

                        # Extract the variables defined above
                        metrics = {k: data.get(k, None) for k in metrics_to_extract}
                        
                        # Add subject and modality info
                        metrics = {"SubID": sub, "condition": condition, **metrics}
                        
                        all_metrics.append(metrics)
            else:
                log(f"No func folder for {sub}")

    # Save to CSV
    func_mriqc = pd.DataFrame(all_metrics)
    func_mriqc.to_csv(mriqc_func_file, sep="\t", index=False)
    log(f"Saved functional QC summary stats to {mriqc_func_file}")
else:
    log("MRIQC summary already exists. Loading...")
    func_mriqc = pd.read_csv(mriqc_func_file, sep="\t")

df_func_mriqc = func_mriqc.merge(df_demo[['ParticipantID', 'AutismDiagnosis']],
              left_on='SubID', right_on='ParticipantID', how='inner')

df_func_mriqc['condition_group'] = df_func_mriqc['condition'].str.split('_').str[0]

########################################################################
################         Raincloud plots T1/T2     #####################
########################################################################

df = df_struct_mriqc.copy()
sns.set(style="whitegrid", palette="Set2")

metrics_of_interest = ['cnr', 'cjv', 'efc', 'inu', 'wm2max']
df.rename(columns={'inu_med':'inu'}, inplace=True) 

# Labels and interpretive notes 
metric_titles = { 
    'cnr': 'Contrast-to-Noise Ratio', 
    'cjv': 'Coefficient of Joint Variation', 
    'efc': 'Entropy Focus Criterion', 
    'inu': 'Intensity Non-Uniformity (median)', 
    'wm2max': 'White Matter to Max Intensity Ratio' 
} 

metric_notes = { 
    'cnr': 'Higher = better contrast', 
    'cjv': 'Lower = better uniformity', 
    'efc': 'Lower = less noise/artifact', 
    'inu': 'Ideal = 1.0 (uniform intensity)', 
    'wm2max': 'Higher = better normalization' 
}

for modality in ['T1w', 'T2w']:
    df_mod = df[df['modality'] == modality]

    fig, axes = plt.subplots(1, len(metrics_of_interest), figsize=(25, 6), sharey=False)
    palette = sns.color_palette("Set2")
    group_colors = dict(zip(sorted(df_mod['AutismDiagnosis'].dropna().unique()), palette))

    for i, metric in enumerate(metrics_of_interest):
        ax = axes[i]

        # Raincloud for this metric
        pt.RainCloud(
            x="AutismDiagnosis",
            y=metric,
            data=df_mod,
            palette=palette,
            width_viol=0.55,
            orient='v',
            alpha=0.6,
            move=0.2,
            point_size=8,
            pointplot=False,
            ax=ax
        )

        # Titles and axis labels
        ax.set_title(metric_titles[metric], fontsize=15, pad=20)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper(), fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
        ax.get_xaxis().set_visible(False)

        # Interpretive note
        ax.text(0.5, -0.08, metric_notes[metric],
                ha='center', va='top', transform=ax.transAxes, fontsize=14, color='dimgray')

    # Shared legend
    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=8, label=label)
               for label, color in group_colors.items()]
    fig.legend(handles=handles, title="Group", loc="upper right",
           bbox_to_anchor=(0.93, 1.1), frameon = False, fontsize=13, title_fontsize=14) 

    fig.suptitle(f"{modality} image quality metrics", fontsize=18, y=1.05)
    plt.tight_layout(rect=[0, 0, 0.93, 0.95])

    # Save figure
    os.makedirs(figures_dir, exist_ok=True)
    plt.savefig(os.path.join(figures_dir, f"mriqc_metrics_raincloud_{modality}.png"), dpi=300, bbox_inches='tight')
    
log("Raincloud plots saved to /figures directory.")

########################################################################
#############      Raincloud plots diffusion         ###################
########################################################################

df = qsi_metrics_df.copy()
df = df.merge(df_demo[['ParticipantID','AutismDiagnosis']],
              left_on="subject_id", right_on="ParticipantID")
df = df.rename(columns={"mean_fd": "FD mean", "max_fd":"FD max", "CNR0_mean":"CNR0", "t1_neighbor_corr":"T1cor"})

dwi_metrics = ['FD mean', 'FD max', 'CNR0', 'T1cor']

metric_titles = {
    'FD mean': 'Mean Framewise Displacement',
    'FD max': 'Maximum Framewise Displacement',
    'CNR0': 'Contrast-to-Noise (b=0 Volumes)',
    'T1cor': 'T1-DWI Neighbour Correlation'
}   

metric_notes = {
    'FD mean': 'Lower = less motion',
    'FD max': 'Lower = fewer motion spikes',
    'CNR0': 'Higher = better signal and contrast in b=0 image',
    'T1cor': 'Higher = better alignment between DWI and T1 anatomy'
}

sns.set(style="whitegrid", palette="Set2")
fig, axes = plt.subplots(1, len(dwi_metrics), figsize=(25, 6), sharey=False)
palette = sns.color_palette("Set2")

for i, metric in enumerate(dwi_metrics):
    ax = axes[i]

    # Raincloud for this metric, enforcing left/right order via 'order'
    pt.RainCloud(
        x="AutismDiagnosis",
        y=metric,
        data=df,
        palette=palette,
        width_viol=0.55,
        orient='v',
        alpha=0.6,
        move=0.2,
        point_size=8,
        pointplot=False,
        ax=ax
    )

    # Titles and axis labels
    ax.set_title(metric_titles[metric], fontsize=15, pad=20)
    ax.set_xlabel("")
    ax.set_ylabel(metric, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.get_xaxis().set_visible(False)

    # Interpretive note
    ax.text(0.5, -0.08, metric_notes[metric],
            ha='center', va='top', transform=ax.transAxes, fontsize=14, color='dimgray')

# Shared legend 
handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=8, label=label)
            for label, color in group_colors.items()]
fig.legend(handles=handles, title="Group",
           loc="upper right", bbox_to_anchor=(0.93, 1.1),
           fontsize=14, title_fontsize=15, frameon=False)

fig.suptitle(f"Diffusion-weighted image quality metrics", fontsize=18, y=1.05)
plt.tight_layout(rect=[0, 0, 0.93, 0.95])

# Save figure
plt.savefig(os.path.join(figures_dir, f"mriqc_metrics_raincloud_dwi.png"), 
            dpi=300, bbox_inches='tight')

log("Raincloud plots saved to /figures directory.")


########################################################################
################      Raincloud plots func         #####################
########################################################################

df = df_func_mriqc.copy()
df = df.rename(columns={"fd_mean": "FD mean","tsnr":"tSNR","gcor":"GCOR","dvars_std":"DVARS"})
df["Category"] = df["condition"].str.extract(r"(Rest|News cast|Reality show)", expand=False)

# Define plotting orders
category_order = ["Rest", "News cast", "Reality show"]
func_metrics = ['FD mean', 'tSNR', 'GCOR', 'DVARS']

metric_titles = {
    'FD mean': 'Framewise Displacement',
    'tSNR': 'Temporal Signal-to-Noise Ratio',
    'GCOR': 'Global Correlation',
    'DVARS': 'Delta Variation of Signal (standardized)'
}   

metric_notes = {
    'FD mean': 'Lower = less motion',
    'tSNR': 'Higher = more stable signal',
    'GCOR': 'Lower = less artifact',
    'DVARS': 'Lower = fewer signal jumps'
}

sns.set(style="whitegrid",palette="Set2")
palette = sns.color_palette("Set2", n_colors=2)

for metric in func_metrics:

    fig, ax = plt.subplots(figsize=(25, 6))

    # Raincloud plot
    pt.RainCloud(
        x="Category",
        y=metric,
        hue="AutismDiagnosis",
        hue_order=["CMP", "ASC"],  
        data=df,
        order = category_order,
        palette=palette,
        width_viol=0.4,
        width_box=0.1,
        point_dodge=0.9,
        move=0.2,
        orient="v",
        dodge=True,
        pointplot=False,
        jitter=True,
        alpha=0.85,
        point_size=6,
        ax=ax
)
        
    # Titles and labels
    ax.get_legend().remove()
    ax.set_ylabel(metric, fontsize=13)
    ax.set_xlabel("")
    ax.set_title(metric_titles[metric], fontsize=15, loc='center', pad=20)
    ax.text(0.5, 1.01, metric_notes[metric], ha='center', va='bottom', transform=ax.transAxes, fontsize=13, color='dimgray')
    ax.tick_params(axis='x', labelsize=13)

    # Shared legend (once)
    handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=8, label=label)
               for label, color in group_colors.items()]
    
    fig.legend(handles=handles, title="Group", 
           loc="upper right", bbox_to_anchor=(0.93, 1.1),
           fontsize=14, title_fontsize=15, frameon=False)
    
    fig.subplots_adjust(right=0.85)          
    fig.tight_layout(rect=[0, 0, 0.87, 0.92]) 
    fig.savefig(os.path.join(figures_dir, f"mriqc_metrics_raincloud_func_{metric}.png"), dpi=300, bbox_inches="tight", pad_inches=0.2)

plt.close("all") 

########################################################################
################       MRIQC Statistics            #####################
########################################################################

df_stats_struct = df_struct_mriqc.copy()
df_stats_func = df_func_mriqc.copy()
df_stats_dwi = df_dwi_mriqc.copy()
df_stats_dwi_qsiprep = qsi_metrics_df.copy()
df_stats_dwi_qsiprep = df_stats_dwi_qsiprep.merge(df_stats_struct[["SubID", "AutismDiagnosis"]],left_on="subject_id",right_on="SubID",how="left")

# Define function to calculate t-statistics per group (per condition in func data)
def compute_group_stats(df, group_col=None, output_prefix="stats", output_dir="."):
    if group_col is None or group_col not in df.columns:
        groups = [None]
    else:
        groups = df[group_col].unique()

    for group in groups:
        if group is None:
            df_group = df
            suffix = "" 
        else:
            df_group = df[df[group_col] == group]
            suffix = f"_{group.replace(' ', '')}"

    numeric_cols = [
        c for c in df_group.columns
        if c not in ['SubID', 'ParticipantID', 'AutismDiagnosis', group_col, 'condition']
        and pd.api.types.is_numeric_dtype(df_group[c])
    ]

    results = []

    for metric in numeric_cols:
        asc = df_group.loc[df_group['AutismDiagnosis'] == 'ASC', metric]
        cmp  = df_group.loc[df_group['AutismDiagnosis'] == 'CMP',  metric]

        mean_asc, sd_asc = asc.mean(), asc.std()
        mean_cmp,  sd_cmp  = cmp.mean(), cmp.std()

        if asc.nunique() > 1 and cmp.nunique() > 1:
            t_stat, p_val = stats.ttest_ind(asc, cmp, equal_var=False)
        else:
            t_stat, p_val = None, None

        results.append({
            'Metric': metric,
            f'Mean_ASC(n={len(asc)})': f'{mean_asc:.3f}',
            'SD_ASC': f'{sd_asc:.3f}',
            f'Mean_CMP(n={len(cmp)})': f'{mean_cmp:.3f}',
            'SD_CMP': f'{sd_cmp:.3f}',
            't-statistic': f'{t_stat:.3f}' if t_stat is not None else '',
            'p-value': f'{p_val:.3f}' if p_val is not None else ''
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(
        os.path.join(output_dir, f"{output_prefix}{suffix}.tsv"),
        sep="\t",
        index=False
    )

# Compute structural stats
compute_group_stats(df_stats_struct, group_col='modality',
                    output_prefix='mriqc_group_stats', 
                    output_dir=output_dir)
log("Statistics output files created for structural MRIQC data.")

# Compute functional stats
compute_group_stats(df_stats_func, group_col='condition_group',
                    output_prefix='mriqc_group_stats_func',
                    output_dir=output_dir)
log("Statistics output files created for functional MRIQC data.")

# Compute diffusion stats from MRIQC
compute_group_stats(df_stats_dwi, group_col='direction',
                    output_prefix='mriqc_group_stats_dwi',
                    output_dir=output_dir)
log("Statistics output files created for diffusion MRIQC data.")

# Compute diffusion stats from QSIPrep
compute_group_stats(df_stats_dwi_qsiprep,
                    output_prefix='qsiprep_group_stats_dwi',
                    output_dir=output_dir)
log("Statistics output files created for diffusion QSIPrep data.")
