# N, Dzinalija, Nov 2025

# Script to calculate summary statistics for demographic and behavioral data 
# and compare ASC and CMP groups where appropriate.

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
mriqc_dir = "/home/ASDPrecision/data/original_bids/derivatives/mriqc"
xcpd_dir = "/home/ASDPrecision/data/bids/derivatives/xcpd"
demographics_file = "/home/ASDPrecision/data/behavioral/Merged_Behavioural_Data_Final.csv"
output_dir = "/home/ASDPrecision/quality_metrics/mriqc"
mriqc_struct_file = os.path.join(output_dir, "mriqc_struct_summary.tsv")
mriqc_func_file = os.path.join(output_dir,"mriqc_func_summary.tsv")
qsiprep_file = os.path.join(output_dir, "qsiprep_dwi_qc_metrics.tsv")
figures_dir = os.path.join(output_dir, "figures")
bids_dir = "/home/ASDPrecision/data/bids"
arsq_file = "/home/ASDPrecision/data/behavioral/ARSQ_scores_Dec25.csv"   
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

########################################################################
################                ARSQ               #####################
########################################################################

arsq_output_file = "/home/ASDPrecision/data/behavioral/ARSQ_scores_Final.csv"
arsq = pd.read_csv(arsq_file, sep=None, engine="python")
arsq["Participant"] = 'sub-' + arsq["Participant"].astype(str)
participant_renaming_file = '/home/ASDPrecision/data/participant_renaming.tsv'
participant_renaming = pd.read_csv(participant_renaming_file, sep = "\t")
arsq = pd.merge(arsq, participant_renaming, left_on="Participant", right_on="participant_id", how='left').drop_duplicates()

reality_videos = ["FOMO", "interview", "first_dates"]
newsclip_videos = ["polen", "farmers", "royals"]

# Create long-fomat dataset of ARSQ scores per video category per session
def map_video_category(video):
    if video in reality_videos:
        return "Reality"
    elif video in newsclip_videos:
        return "News"
    else:
        return "Rest"

arsq["Video_category"] = arsq["Stimulus"].apply(map_video_category)

v1 = arsq[["new_participant_id", "Session", "Video_category", "Dimension_V1", "Sum_dimension_rating_V1"]].drop_duplicates()
v2 = arsq[["new_participant_id", "Session", "Video_category", "Dimension_V2", "Sum_dimension_rating_V2"]].drop_duplicates()

v1 = v1.rename(columns={
    "Dimension_V1": "Dimension",
    "Sum_dimension_rating_V1": "Score",
})
v2 = v2.rename(columns={
    "Dimension_V2": "Dimension",
    "Sum_dimension_rating_V2": "Score",
})

# 3. Add dimension indicator
v1["Version"] = "V1"
v2["Version"] = "V2"

# 4. Combine
arsq_long = pd.concat([v1, v2], ignore_index=True)

# Clean and save data to output file 
arsq_long.to_csv(arsq_output_file, index=False)
log("ARSQ data saved to file: {arsq_output_file}")

# Pivot to wide format, averaging over session
arsq_wide = arsq_long.pivot_table(
    index="new_participant_id",  
    columns=["Video_category", "Dimension", "Version"],      
    values="Score",
    aggfunc="mean" 
    )

arsq_wide.columns = [f"{Video_category}_{Dimension}_{Version}" for Video_category, Dimension, Version in arsq_wide.columns]
df_arsq = arsq_wide.reset_index()

########################################################################
################            Demographics           #####################
########################################################################

# Merge with ARSQ data 
df_demo_filtered = df_demo.merge(df_arsq, left_on='ParticipantID', right_on='new_participant_id', how='left').drop(columns=['new_participant_id'])

# Define variable types
demo_continuous_vars = ["Age","Handedness","YearsASD","MatrixReasoning_raw_score","MatrixReasoning_scaled_score","Vocabulary_raw_score","Vocabulary_scaled_score"]
demo_categorical_vars = ["YearsEducation","Sex","DutchFirstLang","MatchedEthnicities","ADHDDiag","OtherDiagnosis","AQ_above_diagnostic_threshold"]

ASD_questionnaire_vars = [
    "AQ_total", "AQ_social", "AQ_attention_switching", "AQ_attention_to_detail",
    "AQ_communication", "AQ_imagination", 
    "SDQ_total", "SDQ_emotional_problems", "SDQ_conduct_problems",
    "SDQ_hyperactivity", "SDQ_peer_problems", "SDQ_prosocial"
]

arsq_vars = [c for c in df_demo_filtered.columns if c.endswith("_V1") or c.endswith("_V2")]

# Define duration of ASD diagnosis
df_demo_filtered["YearsASD"]=(df_demo_filtered["session_1_date"].str[0:4].astype(int)-df_demo_filtered["ASDDiagYear"])
df_demo_filtered.loc[df_demo_filtered["YearsASD"] <= 0, "YearsASD"] = 1

# Function to generate table
def generate_table(df, cont_vars, cat_vars):
    """
    Generate table of continuous and categorical variables comparing ASC vs CMP.
    """
    rows = []
    
    # Continuous variables
    for var in cont_vars:
        group_asc = df.loc[df["AutismDiagnosis"]=="Yes", var].dropna()
        group_cmp = df.loc[df["AutismDiagnosis"]=="No", var].dropna()
        t, p = stats.ttest_ind(group_asc, group_cmp, equal_var=False, nan_policy='omit')
        rows.append({
            "Variable": var,
            f"ASC": f"{group_asc.mean():.2f} ± {group_asc.std():.2f}",
            f"CMP": f"{group_cmp.mean():.2f} ± {group_cmp.std():.2f}",
            "Test statistic": f"t({len(group_asc)+len(group_cmp)-2}) = {t:.2f}",
            "p-value": f"{p:.3f}"
        })
    
    # Categorical variables
    for var in cat_vars:
        contingency = pd.crosstab(df[var], df["AutismDiagnosis"])
        min_count = contingency.values.min()
        is_2x2 = contingency.shape == (2, 2)
        if is_2x2 and min_count < 10:
            oddsratio, p = stats.fisher_exact(contingency.values)
            test_stat = f"OR = {oddsratio:.2f}"
            dof = ""
        else:
            chi2, p, dof, exp = stats.chi2_contingency(contingency, correction=False)
            test_stat = f"χ²({dof}) = {chi2:.2f}"
        n_asc = contingency["Yes"].sum()
        n_cmp = contingency["No"].sum()
        for lvl in contingency.index:
            rows.append({
                "Variable": f"{var} - {lvl}" if len(contingency.index) > 1 else var,
                f"ASC": f"{contingency.loc[lvl,'Yes']} ({100*contingency.loc[lvl,'Yes']/n_asc:.1f}%)" if 'Yes' in contingency.columns else "0 (0.0%)",
                f"CMP": f"{contingency.loc[lvl,'No']} ({100*contingency.loc[lvl,'No']/n_cmp:.1f}%)" if 'No' in contingency.columns else "0 (0.0%)",
                "Test statistic": test_stat if lvl == contingency.index[0] else "",
                "p-value": f"{p:.3f}" if lvl == contingency.index[0] else ""
            })
    
    table = pd.DataFrame(rows)
    table["p-value"] = table["p-value"].replace("0.000", "<0.001")
    return table

# Generate tables
demo_table = generate_table(df_demo_filtered, demo_continuous_vars, demo_categorical_vars)
ASD_questionnaire_table = generate_table(df_demo_filtered, ASD_questionnaire_vars, [])
arsq_table = generate_table(df_demo_filtered, arsq_vars, [])

# Clean up ARSQ table
arsq_table[['Video', 'Dimension', 'Version']] = arsq_table['Variable'].str.split('_', n=2, expand=True)
arsq_table = arsq_table.drop(columns=['Variable'])
cols = ['Video', 'Dimension', 'Version'] + [c for c in arsq_table.columns if c not in ['Video', 'Dimension', 'Version']]
arsq_table = arsq_table[cols]
arsq_table = arsq_table.sort_values(['Version', 'Video', 'Dimension']).reset_index(drop=True)

# Export tables
demo_table.to_csv(os.path.join(base_dir,"data","behavioral","Demographics_Table.csv"), index=False)
ASD_questionnaire_table.to_csv(os.path.join(base_dir,"data","behavioral","Autism_Questionnaires_Table.csv"), index=False)
arsq_table.to_csv(os.path.join(base_dir,"data","behavioral","ARSQ_Table.csv"), index=False)

log("Stats files created for demographics, AQ, SDQ, and ARSQ data")

########################################################################
################            Data missing           #####################
########################################################################

subjects = df_demo_filtered.iloc[:, 0].tolist()

# Define categories
categories = {
    "rest": ["rest"],
    "reality": ["firstdates", "fomo", "interviews"],
    "newsclip": ["poland", "farmers", "royals"]
}

# Day runs
day_runs = {
    "Day1": [1, 2, 3],
    "Day2": [4, 5, 6],
    "Day3": [7, 8, 9]
}

# Initialize summary dictionary
summary = []

for sub in subjects:
    sub_path = os.path.join(bids_dir, sub)   

    subj_summary = {"subject": sub}

    # -----------------------------
    # Structural scans
    # -----------------------------
    anat_path = os.path.join(sub_path, "anat")
    if os.path.exists(anat_path):
        anat_files = os.listdir(anat_path)
        subj_summary["T1w"] = int(any("T1w" in f for f in anat_files))
        subj_summary["T2w"] = int(any("T2w" in f for f in anat_files))
    else:
        subj_summary["T1w"] = 0
        subj_summary["T2w"] = 0

    # -----------------------------
    # DWI scans
    # -----------------------------
    dwi_path = os.path.join(sub_path, "dwi")
    if os.path.exists(dwi_path):
        dwi_files = os.listdir(dwi_path)
        subj_summary["DWI_AP"] = int(any("acq-AP" in f and f.endswith(".nii.gz") for f in dwi_files))
        subj_summary["DWI_PA"] = int(any("acq-PA" in f and f.endswith(".nii.gz") for f in dwi_files))
    else:
        subj_summary["DWI_AP"] = 0
        subj_summary["DWI_PA"] = 0

    # -----------------------------
    # Functional + fmap (EPI) scans
    # -----------------------------
    func_path = os.path.join(sub_path, "func")
    fmap_path = os.path.join(sub_path, "fmap")

    func_files = os.listdir(func_path) if os.path.exists(func_path) else []
    fmap_files = os.listdir(fmap_path) if os.path.exists(fmap_path) else []

    # -----------------------------
    # Loop over days and categories
    # -----------------------------
    for day, runs in day_runs.items():
        for cat, tasknames in categories.items():

            # -----------------------------
            # Detect if subject did a category task on that day
            # -----------------------------
            did_task = False
            matching_runs = []  # the run numbers corresponding to this category/day

            for f in func_files:
                if not f.endswith("_bold.nii.gz"):
                    continue

                # find run number
                match = re.search(r"run-(\d+)", f)
                if not match:
                    continue
                run = int(match.group(1))

                # find task name
                task_match = re.search(r"task-([A-Za-z0-9]+)", f)
                if not task_match:
                    continue
                task = task_match.group(1)

                # Check category and day
                if task in tasknames and run in runs:
                    did_task = True
                    matching_runs.append(run)

            subj_summary[f"{day}_{cat}"] = int(did_task)

            # -----------------------------
            # EPI exists for matching run(s) in fmap
            # -----------------------------
            epi_exists = any(
                (f.endswith(".nii.gz") and 
                 any(f"run-{r:02d}" in f or f"run-{r}" in f for r in matching_runs))
                for f in fmap_files
            )

            subj_summary[f"{day}_{cat}_EPI"] = int(epi_exists)

            # -----------------------------
            # Physio exists for matching run(s) in func
            # -----------------------------
            physio_exists = any(
                (f.endswith("_physio.tsv.gz") and
                 any(f"run-{r:02d}" in f or f"run-{r}" in f for r in matching_runs))
                for f in func_files
            )

            subj_summary[f"{day}_{cat}_physio"] = int(physio_exists)

    summary.append(subj_summary)

df_summary = pd.DataFrame(summary)
df_summary.to_csv(os.path.join(base_dir,"data","data_availability.csv"), index=False)


########################################################################
################       Inter-scan intervals        #####################
########################################################################

df_demo_filtered = df_demo_filtered.sort_values("AutismDiagnosis")

int_scan_intervals = (
    df_demo_filtered
    .groupby("AutismDiagnosis")
    .agg(
        mean_days_1_2=("days_between_1_2", "mean"),
        sd_days_1_2=("days_between_1_2", "std"),
        mean_days_2_3=("days_between_2_3", "mean"),
        sd_days_2_3=("days_between_2_3", "std")
    )
)

print(int_scan_intervals)
