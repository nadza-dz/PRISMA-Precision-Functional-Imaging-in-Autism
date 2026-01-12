#!/usr/bin/env python3
"""
Plot ISC results on brain surface using surfplot.

Run with the surfplot conda environment:
    conda activate surfplot
    python plot_isc_surface.py

Author: Joe Bathelt
Date: December 2025
"""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from surfplot import Plot
from neuromaps.datasets import fetch_fslr

# Paths
OUTPUT_DIR = Path("/home/ASDPrecision/quality_metrics/isc_maps")
SCHAEFER_DLABEL = Path("/home/ASDPrecision/data/atlases/SchaeferAtlas/HCP/fslr32k/cifti/Schaefer2018_400Parcels_7Networks_order.dlabel.nii")


def load_schaefer_parcellation():
    """Load Schaefer 400 parcellation in 32k fsLR space."""
    dlabel = nib.load(SCHAEFER_DLABEL)
    parc_data = dlabel.get_fdata().squeeze()
    
    # Get label mapping from CIFTI
    header = dlabel.header
    label_axis = header.get_axis(0)
    label_dict = label_axis.label[0]
    brain_axis = header.get_axis(1)
    
    # Get cortex vertex counts (32k surfaces: ~32492 per hemisphere)
    total_vertices = parc_data.shape[0]
    n_left = 0
    left_start = 0
    right_start = 0
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX_LEFT' in str(name):
            left_start = slc.start
            n_left = (slc.stop if slc.stop else total_vertices) - slc.start
        elif 'CORTEX_RIGHT' in str(name):
            right_start = slc.start
    n_right = total_vertices - right_start
    
    print(f"Left cortex: {n_left} vertices, Right cortex: {n_right} vertices")
    
    # Create mapping: Schaefer parcel name -> CIFTI index
    # Schaefer labels are like "7Networks_LH_Vis_1"
    # ISC labels are like "LH_Vis_1"
    schaefer_name_to_idx = {}
    for idx, (name, color) in label_dict.items():
        if idx == 0:
            continue
        # Remove "7Networks_" prefix to match ISC parcel names
        isc_name = name.replace('7Networks_', '')
        schaefer_name_to_idx[isc_name] = idx
    
    # Extract cortical parcellation data
    left_parc = parc_data[:n_left]
    right_parc = parc_data[n_left:n_left + n_right]
    
    return left_parc, right_parc, schaefer_name_to_idx, n_left, n_right


def create_surface_plot(isc_file, output_filename, cmap='YlOrRd', 
                        color_range=None, cbar_label='ISC', title=None):
    """
    Create a surface plot visualization of ISC values.
    
    Parameters:
    -----------
    isc_file : Path
        TSV file with parcel names and ISC values
    output_filename : Path
        Output PNG filename
    cmap : str
        Colormap name
    color_range : tuple, optional
        Color range (min, max)
    cbar_label : str
        Colorbar label
    title : str, optional
        Plot title
    """
    # Load parcellation
    left_parc, right_parc, schaefer_name_to_idx, n_left, n_right = load_schaefer_parcellation()
    
    # Load ISC values
    isc_df = pd.read_csv(isc_file, sep='\t')
    isc_dict = dict(zip(isc_df['parcel'], isc_df['isc']))
    
    # Map ISC values to vertices (only cortical parcels)
    lh_val = np.zeros(n_left)
    rh_val = np.zeros(n_right)
    
    mapped_count = 0
    for parcel_name, isc_val in isc_dict.items():
        if parcel_name in schaefer_name_to_idx:
            parcel_idx = schaefer_name_to_idx[parcel_name]
            lh_val[left_parc == parcel_idx] = isc_val
            rh_val[right_parc == parcel_idx] = isc_val
            mapped_count += 1
    
    print(f"  Mapped {mapped_count} cortical parcels")
    
    # Load fsLR 32k surfaces
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    
    # Figure size: 90mm x 90mm at 300 DPI
    width = int(90 * 300 / 25.4)
    height = int(90 * 300 / 25.4)
    
    # Create plot
    p = Plot(lh, rh, zoom=1.2, size=(width, height))
    
    # Set color range
    if color_range is None:
        color_range = (0, np.max([lh_val.max(), rh_val.max()]))
    
    # Add ISC layer
    p.add_layer({'left': lh_val, 'right': rh_val},
                cmap=cmap,
                color_range=color_range,
                cbar=False)
    
    # Add parcel outlines
    p.add_layer({'left': left_parc, 'right': right_parc},
                cmap='gray', as_outline=True, cbar=False)
    
    # Build and save
    fig = p.build()
    
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"  Saved: {output_filename.name}")
    
    return fig


def create_difference_plot(asc_file, cmp_file, output_filename, 
                           cmap='RdBu_r', color_range=None):
    """Create a difference plot (ASC - CMP)."""
    # Load parcellation
    left_parc, right_parc, schaefer_name_to_idx, n_left, n_right = load_schaefer_parcellation()
    
    # Load ISC values
    asc_df = pd.read_csv(asc_file, sep='\t')
    cmp_df = pd.read_csv(cmp_file, sep='\t')
    
    asc_dict = dict(zip(asc_df['parcel'], asc_df['isc']))
    cmp_dict = dict(zip(cmp_df['parcel'], cmp_df['isc']))
    
    # Map difference values to vertices (only cortical parcels)
    lh_val = np.zeros(n_left)
    rh_val = np.zeros(n_right)
    
    for parcel_name in asc_dict:
        if parcel_name in schaefer_name_to_idx and parcel_name in cmp_dict:
            parcel_idx = schaefer_name_to_idx[parcel_name]
            diff_val = asc_dict[parcel_name] - cmp_dict[parcel_name]
            lh_val[left_parc == parcel_idx] = diff_val
            rh_val[right_parc == parcel_idx] = diff_val
    
    # Load fsLR 32k surfaces
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    
    # Figure size
    width = int(90 * 300 / 25.4)
    height = int(90 * 300 / 25.4)
    
    # Create plot
    p = Plot(lh, rh, zoom=1.2, size=(width, height))
    
    # Set symmetric color range
    if color_range is None:
        max_abs = np.max([np.abs(lh_val).max(), np.abs(rh_val).max()])
        color_range = (-max_abs, max_abs)
    
    # Add difference layer
    p.add_layer({'left': lh_val, 'right': rh_val},
                cmap=cmap,
                color_range=color_range,
                cbar=False)
    
    # Add parcel outlines
    p.add_layer({'left': left_parc, 'right': right_parc},
                cmap='gray', as_outline=True, cbar=False)
    
    # Build and save
    fig = p.build()
    fig.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"  Saved: {output_filename.name}")
    
    return fig


def main():
    """Generate all surface plots."""
    print("Creating surface visualizations of ISC maps...")
    
    # Create output directory for surface plots
    surf_dir = OUTPUT_DIR / "surface_plots"
    surf_dir.mkdir(exist_ok=True)
    
    # Common color range for ISC
    isc_range = (0, 0.6)
    
    # Plot all average ISC maps
    configs = [
        ('ASC', 'all_average', 'ASC - All Videos'),
        ('CMP', 'all_average', 'CMP - All Videos'),
        ('ASC', 'news_average', 'ASC - News'),
        ('CMP', 'news_average', 'CMP - News'),
        ('ASC', 'reality_average', 'ASC - Reality TV'),
        ('CMP', 'reality_average', 'CMP - Reality TV'),
    ]
    
    for group, category, title in configs:
        isc_file = OUTPUT_DIR / f"isc_{group}_{category}.tsv"
        if isc_file.exists():
            out_file = surf_dir / f"isc_{group}_{category}_surface.png"
            create_surface_plot(isc_file, out_file, 
                              cmap='inferno',
                              color_range=isc_range,
                              cbar_label='ISC',
                              title=title)
    
    # Create difference plots
    for category in ['all_average', 'news_average', 'reality_average']:
        asc_file = OUTPUT_DIR / f"isc_ASC_{category}.tsv"
        cmp_file = OUTPUT_DIR / f"isc_CMP_{category}.tsv"
        if asc_file.exists() and cmp_file.exists():
            out_file = surf_dir / f"isc_difference_{category}_surface.png"
            create_difference_plot(asc_file, cmp_file, out_file,
                                 color_range=(-0.15, 0.15))
    
    print(f"\nDone! Surface plots saved to: {surf_dir}")


if __name__ == "__main__":
    main()
