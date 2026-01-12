#!/usr/bin/env python3
"""
Defacing Anatomical MRI Scans for ASD Precision Study using MiDeFace2 from FreeSurfer

Author: Joe Bathelt
Date: October 2025
Version: 1.0
"""
import os
from pathlib import Path
from string import Template
import subprocess
from subprocess import PIPE
from concurrent.futures import ThreadPoolExecutor
import logging

def setup_logger(subject, logs_folder):
    logger = logging.getLogger(subject)
    logger.setLevel(logging.INFO)

    if not os.path.isdir(logs_folder):
        os.makedirs(logs_folder, exist_ok=True)

    handler = logging.FileHandler(logs_folder / f'{subject}_defacing.log')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

def run_docker_command(cmd, bids_folder, logger):
    """
    Run a FreeSurfer command inside Docker container.
    
    Parameters:
    -----------
    cmd : str
        The FreeSurfer command to run
    bids_folder : Path
        Path to BIDS folder for volume mounting
    logger : logging.Logger
        Logger instance
    """
    try:
        # Check if FreeSurfer license exists
        if not freesurfer_license.exists():
            logger.error(f"FreeSurfer license not found at {freesurfer_license}")
            logger.error("Please obtain a license from https://surfer.nmr.mgh.harvard.edu/registration.html")
            return False
        
        # Construct Docker command with OpenMP threads for parallelization
        docker_cmd = f"""docker run --rm \
            -v {bids_folder}:/data \
            -v {freesurfer_license}:/usr/local/freesurfer/.license:ro \
            -e FS_LICENSE=/usr/local/freesurfer/.license \
            -e OMP_NUM_THREADS={num_threads} \
            {freesurfer_image} \
            {cmd}"""
        
        logger.info(f"Running Docker command: {docker_cmd}")
        
        process = subprocess.run(docker_cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        
        if process.stdout:
            logger.info(process.stdout)
        if process.stderr:
            logger.error(process.stderr)

        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
        
        return process.returncode == 0
    except Exception as e:
        logger.error(f"Error running Docker command: {cmd}\n{e}")
        return False

def deface_anatomicals(project_folder, subject, logs_folder):
    logger = setup_logger(subject, logs_folder)
    try:
        # Move the original files to the sourcedata folder
        bids_folder = project_folder / 'data' / 'original_bids'
        if not os.path.isdir(bids_folder / 'sourcedata'):
            os.mkdir(bids_folder / 'sourcedata')
        if not os.path.isdir(bids_folder / 'sourcedata' / 'mideface'):
            os.mkdir(bids_folder / 'sourcedata' / 'mideface')
        if not os.path.isdir(bids_folder / 'sourcedata' / 'mideface' / subject):
            os.mkdir(bids_folder / 'sourcedata' / 'mideface' / subject)
        if not os.path.isdir(bids_folder / 'sourcedata' / 'mideface' / subject / 'anat'):
            os.mkdir(bids_folder / 'sourcedata' / 'mideface' / subject / 'anat')

        # Go through the anat folder and find T1w and T2w files (there may be several runs)
        t1_files = list(bids_folder.glob(f'{subject}/anat/{subject}*_T1w.nii.gz'))
        t2_files = list(bids_folder.glob(f'{subject}/anat/{subject}*_T2w.nii.gz'))

        if not t1_files:
            logger.error(f"No T1w files found for {subject}")
        if not t2_files:
            logger.error(f"No T2w files found for {subject}")

        for t1_file in t1_files:
            os.rename(t1_file, bids_folder / f'sourcedata/mideface/{subject}/anat/{t1_file.name}')

        for t2_file in t2_files:
            os.rename(t2_file, bids_folder / f'sourcedata/mideface/{subject}/anat/{t2_file.name}')

        # Defining the file names and paths
        logger.info(f'Defacing {subject} T1w...')
        defacing_folder = bids_folder / f'sourcedata/mideface/{subject}/anat'
        defaced_t1_files = []
        defaced_t2_files = []
        
        for t1_file in t1_files:
            defaced_t1_file = defacing_folder / t1_file.name.replace('.nii.gz', '_defaced.nii.gz')
            mask_file = defacing_folder / t1_file.name.replace('.nii.gz', '_mask.mgz')
            t1_file_name = t1_file.name
            
            # Convert host paths to Docker container paths
            docker_in_file = f"/data/sourcedata/mideface/{subject}/anat/{t1_file_name}"
            docker_out_file = f"/data/sourcedata/mideface/{subject}/anat/{defaced_t1_file.name}"
            docker_mask_file = f"/data/sourcedata/mideface/{subject}/anat/{mask_file.name}"
            docker_qa_folder = f"/data/sourcedata/mideface/{subject}/anat"

            # Deface T1w
            cmd = Template("""mideface --i $in_file \
            --o $out_file \
            --facemask $mask_file \
            --odir $qa_folder \
            --code '$id'""").substitute({
                'in_file': docker_in_file, 
                'out_file': docker_out_file, 
                'mask_file': docker_mask_file, 
                'qa_folder': docker_qa_folder, 
                'id': subject
                })
            logger.info(f"Running mideface for T1w: {t1_file_name}")
            run_docker_command(cmd, bids_folder, logger)
            defaced_t1_files.append(defaced_t1_file)

        for t2_file in t2_files:
            t1_file = t1_files[-1] # Use the last T1w file for co-registration (if multiple)
            defaced_t2_file = defacing_folder / t2_file.name.replace('.nii.gz', '_defaced.nii.gz')
            reg_file = defacing_folder / f'{subject}_T1w-T2w_registration.lta'
            t1_file_name = t1_file.name
            t2_file_name = t2_file.name
            # Use the MGZ mask file created during T1w defacing
            mask_file_mgz = t1_file.name.replace('.nii.gz', '_mask.mgz')
            # Define MGZ intermediates for robust MiDeFace apply (positional args expect MGZ)
            t1_mgz = t1_file.name.replace('.nii.gz', '.mgz')
            t2_mgz = t2_file.name.replace('.nii.gz', '.mgz')
            t2_defaced_mgz = t2_file.name.replace('.nii.gz', '.defaced.mgz')
            
            # Convert host paths to Docker container paths
            docker_t1_nii = f"/data/sourcedata/mideface/{subject}/anat/{t1_file_name}"
            docker_t2_nii = f"/data/sourcedata/mideface/{subject}/anat/{t2_file_name}"
            docker_t1_mgz = f"/data/sourcedata/mideface/{subject}/anat/{t1_mgz}"
            docker_t2_mgz = f"/data/sourcedata/mideface/{subject}/anat/{t2_mgz}"
            docker_t2_defaced_mgz = f"/data/sourcedata/mideface/{subject}/anat/{t2_defaced_mgz}"
            docker_reg = f"/data/sourcedata/mideface/{subject}/anat/{reg_file.name}"
            docker_out_file = f"/data/sourcedata/mideface/{subject}/anat/{defaced_t2_file.name}"
            docker_mask_mgz = f"/data/sourcedata/mideface/{subject}/anat/{mask_file_mgz}"

            # Convert NIfTI to MGZ inside the container (both T1 and T2), to match MiDeFace docs
            logger.info(f'Converting {subject} T1w/T2w to MGZ for MiDeFace apply...')
            cmd = Template("""bash -lc 'mri_convert $t1nii $t1mgz && mri_convert $t2nii $t2mgz'""").substitute({
                't1nii': docker_t1_nii,
                't1mgz': docker_t1_mgz,
                't2nii': docker_t2_nii,
                't2mgz': docker_t2_mgz
                })
            run_docker_command(cmd, bids_folder, logger)

            # Co-register T2w to T1w using MGZ volumes
            logger.info(f'Co-registering {subject} T2w to T1w (MGZ)...')
            cmd = Template("""mri_coreg --mov $t2mgz --targ $t1mgz --reg $outreg""").substitute({
                't1mgz': docker_t1_mgz,
                't2mgz': docker_t2_mgz,
                'outreg': docker_reg
                })
            if not run_docker_command(cmd, bids_folder, logger):
                logger.error(f"mri_coreg failed for {subject} {t2_file_name}")
                continue

            # Apply defacing using MiDeFace (handles transform + mask semantics)
            # Use positional-argument syntax per MiDeFace documentation
            logger.info(f'Applying MiDeFace mask to {subject} T2w with LTA (positional args)...')
            cmd = Template("""mideface --apply $t2mgz $maskmgz $reg $t2defacedmgz""").substitute({
                't2mgz': docker_t2_mgz,
                'maskmgz': docker_mask_mgz,
                'reg': docker_reg,
                't2defacedmgz': docker_t2_defaced_mgz
                })
            if not run_docker_command(cmd, bids_folder, logger):
                logger.error(f"mideface --apply failed for {subject} {t2_file_name}")
                continue

            # Convert defaced MGZ back to NIfTI
            logger.info(f'Converting defaced MGZ back to NIfTI for {subject} T2w...')
            cmd = Template("""mri_convert $t2defacedmgz $outnii""").substitute({
                't2defacedmgz': docker_t2_defaced_mgz,
                'outnii': docker_out_file
                })
            if not run_docker_command(cmd, bids_folder, logger):
                logger.error(f"mri_convert (defaced MGZ -> NIfTI) failed for {subject} {t2_file_name}")
                continue
            defaced_t2_files.append(defaced_t2_file)
        
        # Move the defaced files to the BIDS folder
        for defaced_t1_file in defaced_t1_files:
            target = bids_folder / subject / 'anat' / defaced_t1_file.name.replace('_defaced', '')
            if defaced_t1_file.exists():
                os.rename(defaced_t1_file, target)
            else:
                logger.error(f"Expected defaced T1 not found: {defaced_t1_file}")
        for defaced_t2_file in defaced_t2_files:
            target = bids_folder / subject / 'anat' / defaced_t2_file.name.replace('_defaced', '')
            if defaced_t2_file.exists():
                os.rename(defaced_t2_file, target)
            else:
                logger.error(f"Expected defaced T2 not found: {defaced_t2_file}")
        logger.info(f"Defacing completed for {subject}")

    except Exception as e:
        logger.error(f"Error processing {subject}: {e}")

# %%
# FreeSurfer Docker configuration
project_folder = Path('/home/jmbathe/Documents/1_Projects/ASDPrecision/')
freesurfer_image = "freesurfer/freesurfer:7.4.1"
freesurfer_license = project_folder / 'data/license.txt'
logs_folder = Path('/home/ASDPrecision/logs/mideface/')
subject_list = sorted([f for f in os.listdir(project_folder / 'data' / 'original_bids') if f.startswith('sub-')])
subject_list = ['sub-MV270225']

# Process sequentially but use all cores within each FreeSurfer command
num_threads = 64  # Use all available cores for OpenMP parallelization

# Process subjects sequentially - parallelization happens inside FreeSurfer tools
for subject in subject_list:
    deface_anatomicals(project_folder, subject, logs_folder)

# %%
