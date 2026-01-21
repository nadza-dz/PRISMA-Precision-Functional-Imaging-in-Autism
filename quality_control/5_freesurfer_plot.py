#!/usr/bin/env python3
"""
Script to extract Freesurfer surface metrics (coritcal thickness and surface area) and subcortical
volume and plot them on an inflated brain surface

For the subcortical mapping, the ENIGMA environment needs to be loaded with 'conda activate enigma'
and the enigma toolbox needs to be installed (see https://enigma-toolbox.readthedocs.io/en/latest/pages/01.install/index.html)

Author: Nadza Dzinalija, Jan 2026
"""

import os
import pandas as pd
import numpy as np
from nibabel.freesurfer.io import read_annot
import matplotlib.pyplot as plt
import nibabel as nib
from surfplot import Plot

#from enigmatoolbox.utils.parcellation import parcel_to_surface
#from enigmatoolbox.plotting import plot_cortical
from enigmatoolbox.plotting import plot_subcortical

###########################################################
##################      Cortical            ###############
###########################################################

# Paths
freesurfer_output = "/home/ASDPrecision/quality_metrics/freesurfer"
subjects_dir = "/home/ASDPrecision/data/bids/derivatives/fmriprep/sourcedata/freesurfer"
output_dir = "/home/ASDPrecision/quality_metrics/freesurfer/surface_plots"

lh_thick_file = os.path.join(freesurfer_output,"lh_aparc_thickness.txt")
rh_thick_file = os.path.join(freesurfer_output,"rh_aparc_thickness.txt")
lh_area_file = os.path.join(freesurfer_output,"lh_aparc_surfacearea.txt")
rh_area_file = os.path.join(freesurfer_output,"rh_aparc_surfacearea.txt")
subcortical_file = os.path.join(freesurfer_output,'aseg_volumes.txt')
demographics_file = "/home/ASDPrecision/data/behavioral/Merged_Behavioural_Data_Final.csv"

# Load surfaces (50% inflated, meaning .gii versions of the pial_semi_inflated)
lh_surf_file = '/home/ASDPrecision/data/bids/derivatives/fmriprep/sourcedata/freesurfer/fsaverage/surf/lh.midinflated.gii'
rh_surf_file = '/home/ASDPrecision/data/bids/derivatives/fmriprep/sourcedata/freesurfer/fsaverage/surf/rh.midinflated.gii'

# Load annotations
lh_labels, lh_ctab, lh_names = read_annot(f"{subjects_dir}/fsaverage5/label/lh.aparc.annot")
rh_labels, rh_ctab, rh_names = read_annot(f"{subjects_dir}/fsaverage5/label/rh.aparc.annot")

# Functions
def create_surface_plot_fsaverage(vertex_values_lh, vertex_values_rh, measure, group,
                                 lh_surf, rh_surf, lh_parc, rh_parc,
                                 output_filename, cmap='YlOrRd', color_range=None, title=None):
    """
    Create a surface plot visualization of thickness/area values.

    Parameters
    ----------
    vertex_values_lh, vertex_values_rh : np.ndarray
        Vertexwise values for left and right hemispheres
    measure : str
        "thickness" or "area"
    group : str
        "ASC" or "CMP"
    lh_surf, rh_surf : nibabel gifti objects
        Midinflated surfaces
    lh_parc, rh_parc : np.ndarray
        Parcel labels (from read_annot)
    output_filename : str
        Path to save figure
    cmap : str
        Colormap
    color_range : tuple
        Min/max values for colorbar
    title : str
        Figure title
    """

    # Use vertex values directly
    lh_val = vertex_values_lh
    rh_val = vertex_values_rh

    # Create Plot object (assuming same API as your ISC function)
    width = int(90 * 300 / 25.4)
    height = int(90 * 300 / 25.4)
    p = Plot(lh_surf, rh_surf, zoom=1.2, size=(width, height))

    # Determine color range if not provided
    if color_range is None:
        color_range = (min(lh_val.min(), rh_val.min()), max(lh_val.max(), rh_val.max()))

    # Add thickness/area layer
    p.add_layer({'left': lh_val, 'right': rh_val},
                cmap=cmap,
                color_range=color_range,
                cbar=True)

    # Optional: add parcel outlines
    p.add_layer({'left': lh_parc, 'right': rh_parc},
                cmap='gray', as_outline=True, cbar=False)

    # Build and save
    fig = p.build()
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved: {output_filename}")

    return fig

# Load FreeSurfer metrics extracted with aparcstats2table
lh_thick = pd.read_csv(lh_thick_file, sep="\t", index_col=0)
rh_thick = pd.read_csv(rh_thick_file, sep="\t", index_col=0)
lh_area = pd.read_csv(lh_area_file, sep="\t", index_col=0)
rh_area = pd.read_csv(rh_area_file, sep="\t", index_col=0)

# Load demographics
demo = pd.read_csv(demographics_file).set_index('ParticipantID')
demo["AutismDiagnosis"] = demo['AutismDiagnosis'].map({'Yes': 'ASC', 'No': 'CMP'})

# Merge tables with demographics
lh_thick = lh_thick.join(demo['AutismDiagnosis'])
rh_thick = rh_thick.join(demo['AutismDiagnosis'])
lh_area = lh_area.join(demo['AutismDiagnosis'])
rh_area = rh_area.join(demo['AutismDiagnosis'])

# Save group averages
thick_group = pd.concat([lh_thick, rh_thick], axis=1)
area_group = pd.concat([lh_area, rh_area], axis=1)
thick_group.to_csv(os.path.join(freesurfer_output,"avg_thickness.csv"))
area_group.to_csv(os.path.join(freesurfer_output,"avg_area.csv"))

# Compute group averages
lh_thick_group = lh_thick.groupby('AutismDiagnosis').mean(numeric_only=True)
rh_thick_group = rh_thick.groupby('AutismDiagnosis').mean(numeric_only=True)
lh_area_group = lh_area.groupby('AutismDiagnosis').mean(numeric_only=True)
rh_area_group = rh_area.groupby('AutismDiagnosis').mean(numeric_only=True)

# Combine hemispheres (first left then right)
thick_group = pd.concat([lh_thick_group, rh_thick_group], axis=1)
area_group = pd.concat([lh_area_group, rh_area_group], axis=1)

# Mapping of regions to vertices
measures = ["thickness", "area"]
groups = ["ASC", "CMP"]

group_tables = {
    "thickness": thick_group,
    "area": area_group
}

for measure in measures:
    df = group_tables[measure]

    # Remove columns containing "MeanThickness" or "WhiteSurfArea" (summary columns)
    df_filtered = df.loc[:, ~df.columns.str.contains("MeanThickness|WhiteSurfArea")]

    vmin = df_filtered.values.min()
    vmax = df_filtered.values.max()
    
    
    for group in groups:  
        fig = plt.figure(figsize=(10, 5))
        
        # LEFT hemisphere
        lh_vals = np.zeros_like(lh_labels, dtype=float)
        for i, region_name in enumerate(lh_names):
            fs_name = region_name.decode('utf-8') 
            col_name = f"lh_{fs_name}_{measure}"   
            if col_name in df_filtered.columns:
                lh_vals[lh_labels == i] = df_filtered.loc[group, col_name]

        # RIGHT hemisphere
        rh_vals = np.zeros_like(rh_labels, dtype=float)
        for i, region_name in enumerate(rh_names):
            fs_name = region_name.decode('utf-8')
            col_name = f"rh_{fs_name}_{measure}"   
            if col_name in df_filtered.columns:
                rh_vals[rh_labels == i] = df_filtered.loc[group, col_name]

        # Save figure
        output_file = os.path.join(output_dir, f"{measure}_{group}_midinflated.jpg")
        create_surface_plot_fsaverage(
            lh_vals, rh_vals, measure, group,
            lh_surf_file, rh_surf_file,
            lh_labels, rh_labels,
            output_file,
            cmap='YlOrRd',
            color_range=(vmin, vmax),
            title=f"{measure} {group}"
        )

###########################################################
################     Sub - cortical         ###############
###########################################################

# Load FreeSurfer metrics extracted with asegstats2table
subcort_volume = pd.read_csv(subcortical_file, sep='\t',index_col=0)

# Merge with demographics
subcort_volume = subcort_volume.join(demo['AutismDiagnosis'])

# Select regions for plotting
regions = [
    'Left-Accumbens-area', 'Left-Amygdala', 'Left-Caudate', 'Left-Hippocampus',
    'Left-Pallidum', 'Left-Putamen', 'Left-Thalamus', 'Left-Lateral-Ventricle',
    'Right-Accumbens-area', 'Right-Amygdala', 'Right-Caudate', 'Right-Hippocampus',
    'Right-Pallidum', 'Right-Putamen', 'Right-Thalamus', 'Right-Lateral-Ventricle'
]

subcort_volume = subcort_volume[['AutismDiagnosis'] + regions]

# Compute group averages
subcort_volume_group = subcort_volume.groupby('AutismDiagnosis').mean(numeric_only=True)

# Plotting
groups = ["ASC", "CMP"]

vmin = subcort_volume_group.values.min()
vmax = subcort_volume_group.values.max()
    
for group in groups:  
    
    subcortical_df = subcort_volume_group.loc[group].values

    plot_subcortical(array_name=subcortical_df,
                    size=(800, 400),
                    cmap='YlOrRd',
                    color_bar=True,
                    screenshot=True,
                    transparent_bg=False,
                    filename=os.path.join(output_dir,f'{group}_volume.jpg'),
                    color_range=(vmin, vmax))

