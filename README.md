# PRISMA: A precision functional imaging dataset of autistic and non-autistic adults

This repository accompanies the PRISMA dataset published on OpenNeuro under accession number [ds007182](https://openneuro.org/datasets/ds007182/versions/1.0.0) and described in a [preprint](https://doi.org/10.64898/2026.01.12.698952) on BioRxiv. It contains all scripts used for multimodal data preprocessing and quality control.

# Structure

preprocessing/                        # Data preparation pipeline 
  1_calculate_behavioural_scores.py   # AQ-50, SDQ, WAIS-IV and ARSQ questionnaires
  2_parrec_to_bids.py                 # DICOM→BIDS with NORDIC denoising
  3_anonymise_bids.py                 # Remove identifying information
  4_deface_anatomical.py              # Deface T1w images
  5_run_mriqc.py                      # MRI quality control
  6_run_fmriprep.py                   # fMRI preprocessing
  7_run_xcpd.py                       # XCP-D post-processing
  8_run_qsiprep.py                    # DWI preprocessing
  9_run_qsirecon.py                   # DWI reconstruction

quality_control/                      # QC visualization and metrics
  1_mriqc_stats                       # T1w, T2w, DWI and fMRI image quality metrics
  2_behavior_stats.py                 # Behavioral and demographic data metrics
  3_calculate_tsnr.py                 # fMRI quality metrics

dwi/                                  # DWI
  1_qc_maps.py                        # Group-level DWI quality maps
  2_plot_connectome.py                # Visualize DWI connectomes

isc/                                  # Inter-subject correlation
  1_isc_calculate_averages.py         # Inter-subject correlation analysis
  2_isc_plot_surface.py               # Visualize ISC results
  3_

run_physio_qc.py                      # Physiological signal QC (PPG, respiration)
calculate_physio_regressors.py        # HRV/RVT regressors and GLM analysis
```

# Dependencies

- Python 3.10+
- fMRIPrep, XCP-D, QSIPrep, MRIQC (via Docker)
- nilearn, neurokit2, nibabel, pandas, numpy

# Acknowledgements 

This work was supported by the Dutch Research Council (NWO) grant No. 406.XS.24.01.007.

# Authors
Joe Bathelt and Nadza Dzinalija

To get in touch, please email j.m.c.bathelt@uva.nl or n.dzinalija@uva.nl.
