#!/usr/bin/env python3
"""
Robust DICOM to BIDS Converter with NORDIC Processing
Converts neuroimaging data to BIDS format with comprehensive logging and error handling

Author: Joe Bathelt
Date: October 2025
Version: 1.0
"""

import json
import nibabel as nib
import os
from pathlib import Path
import pandas as pd
import re
from shutil import copyfile
import subprocess
import logging
from datetime import datetime
from typing import Dict, Optional
import sys
from collections import defaultdict
import gzip

class BIDSConverter:
    """Handles conversion of neuroimaging data to BIDS format with NORDIC processing."""
    
    def __init__(self, project_folder: Path):
        """Initialize the converter with project paths and logging."""
        self.project_folder = Path(project_folder)
        self.raw_folder = self.project_folder / 'data/raw_data'
        self.nifti_folder = self.project_folder / 'data/raw_nifti'
        self.dicm2nii_path = self.project_folder / 'code/Xiangruili Dicm2nii'
        self.nordic_path = self.project_folder / 'code/NORDIC_Raw'
        self.bids_folder = self.project_folder / 'data/original_bids'
        self.nordic_folder = self.project_folder / 'data/raw_nordic'
        
        # Setup logging
        self.setup_logging()
        
        # Load task condition mapping
        self.task_condition_mapping = self.load_condition_mapping()
        
        # Statistics tracking
        self.stats = defaultdict(lambda: defaultdict(int))
        
    def setup_logging(self):
        """Setup comprehensive logging with file and console handlers."""
        # Create logs directory in the project folder
        self.log_dir = self.project_folder / 'logs' / 'parrce2bids'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.timestamp = timestamp
        
        # Master summary log file
        summary_log_file = self.log_dir / f'summary_{timestamp}.log'
        self.summary_log_file = summary_log_file
        
        # Create logger
        self.logger = logging.getLogger('BIDSConverter')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Summary file handler - logs overall progress
        summary_handler = logging.FileHandler(summary_log_file, mode='w', encoding='utf-8')
        summary_handler.setLevel(logging.INFO)
        summary_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        summary_handler.setFormatter(summary_formatter)
        
        # Console handler - important messages only
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(summary_handler)
        self.logger.addHandler(console_handler)
        
        # Store handlers for later modification
        self.summary_handler = summary_handler
        self.console_handler = console_handler
        
        # Log initialization info
        self.logger.info("=" * 80)
        self.logger.info(f"BIDS Conversion Pipeline Started")
        self.logger.info(f"Timestamp: {timestamp}")
        self.logger.info(f"Summary log file: {summary_log_file}")
        self.logger.info(f"Project folder: {self.project_folder}")
        self.logger.info("=" * 80)
        
    def setup_subject_logging(self, subject_name: str, session: int):
        """Setup logging for a specific subject-session."""
        # Create subject-specific log file
        subject_log_file = self.log_dir / f'{subject_name}_ses-0{session}_{self.timestamp}.log'
        
        # Remove any existing subject handler
        if hasattr(self, 'subject_handler'):
            self.logger.removeHandler(self.subject_handler)
            self.subject_handler.close()
        
        # Create new subject-specific file handler
        subject_handler = logging.FileHandler(subject_log_file, mode='w', encoding='utf-8')
        subject_handler.setLevel(logging.DEBUG)
        subject_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        subject_handler.setFormatter(subject_formatter)
        
        self.logger.addHandler(subject_handler)
        self.subject_handler = subject_handler
        
        self.logger.info(f"Individual log file created: {subject_log_file}")
        
    def cleanup_subject_logging(self):
        """Remove subject-specific logging handler."""
        if hasattr(self, 'subject_handler'):
            self.logger.removeHandler(self.subject_handler)
            self.subject_handler.close()
            delattr(self, 'subject_handler')
        
    def load_condition_mapping(self) -> Dict:
        """Load the task condition mapping from JSON file."""
        mapping_file = self.project_folder / 'code/ConditionSequences.json'
        
        try:
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            self.logger.info(f"Loaded condition mapping for {len(mapping)} subjects")
            return mapping
        except FileNotFoundError:
            self.logger.error(f"Condition mapping file not found: {mapping_file}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in condition mapping: {e}")
            raise
            
    def validate_paths(self) -> bool:
        """Validate that all required paths exist."""
        required_paths = {
            'Raw data': self.raw_folder,
            'dicm2nii': self.dicm2nii_path,
            'NORDIC': self.nordic_path,
        }
        
        all_valid = True
        for name, path in required_paths.items():
            if not path.exists():
                self.logger.error(f"{name} path does not exist: {path}")
                all_valid = False
            else:
                self.logger.debug(f"✓ {name} path validated: {path}")
                
        return all_valid
        
    def run_matlab_script(self, script_content: str, script_name: str = 'myscript.m') -> bool:
        """Run MATLAB script with error handling."""
        # Clean script name to remove special characters that MATLAB doesn't like
        # Keep only alphanumeric characters, underscores, and the .m extension
        base_name = script_name.replace('.m', '')
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name) + '.m'
        
        error_handling_wrapper = f"""
try
    {script_content}
    disp('MATLAB Script completed successfully.');
catch ME
    disp(['Error encountered: ', ME.message]);
    for i = 1:length(ME.stack)
        disp(['  In ', ME.stack(i).name, ' at line ', num2str(ME.stack(i).line)]);
    end
    exit(1);
end
exit(0);
"""
        script_path = self.project_folder / 'scripts' / clean_name
        
        try:
            with open(script_path, 'w') as f:
                f.write(error_handling_wrapper)
            
            self.logger.debug(f"Running MATLAB script: {clean_name}")
            result = subprocess.run(
                f'matlab -nodesktop -nosplash -r "run(\'{script_path}\');exit;"',
                shell=True,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"MATLAB script failed with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
            self.logger.debug("MATLAB script completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"MATLAB script timed out after 30 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Error running MATLAB script: {e}")
            return False
        finally:
            if script_path.exists():
                script_path.unlink()
                
    def extract_run_number(self, path: Path) -> Optional[int]:
        """Extract run number from file path."""
        match = re.search(r'run-(\d+)', str(path))
        return int(match.group(1)) if match else None
        
    def dicom_to_nifti(self, subject: Path, session: int) -> bool:
        """Convert DICOM to NIfTI format."""
        session_folder = self.raw_folder / f'ses-0{session}'
        nifti_session_folder = self.nifti_folder / f'ses-0{session}' / subject.name
        nifti_session_folder.mkdir(parents=True, exist_ok=True)
        
        # Check if conversion already done
        existing_files = list(nifti_session_folder.glob('*.nii.gz'))
        if existing_files:
            self.logger.info(f"  NIfTI files already exist ({len(existing_files)} files), skipping conversion")
            self.stats[f'{subject.name}_ses-0{session}']['dicom2nifti'] = 'skipped'
            return True
            
        self.logger.info(f"  Converting DICOM to NIfTI...")
        
        dicom_folder = session_folder / subject.name
        if not dicom_folder.exists():
            self.logger.error(f"  DICOM folder not found: {dicom_folder}")
            self.stats[f'{subject.name}_ses-0{session}']['dicom2nifti'] = 'failed'
            return False
            
        script_content = f"""
addpath('{self.dicm2nii_path}')
infolder='{dicom_folder}/';
outfolder='{nifti_session_folder}/';
dicm2nii(infolder, outfolder, 1)
"""
        
        success = self.run_matlab_script(script_content)
        
        # Verify output
        output_files = list(nifti_session_folder.glob('*.nii.gz'))
        if success and output_files:
            self.logger.info(f"  ✓ Created {len(output_files)} NIfTI files")
            self.stats[f'{subject.name}_ses-0{session}']['dicom2nifti'] = 'success'
            self.stats[f'{subject.name}_ses-0{session}']['nifti_files'] = len(output_files)
            return True
        else:
            self.logger.error(f"  ✗ DICOM conversion failed or no output files created")
            self.stats[f'{subject.name}_ses-0{session}']['dicom2nifti'] = 'failed'
            return False
            
    def nordic_processing(self, subject: Path, session: int) -> bool:
        """Apply NORDIC denoising to functional images."""
        nifti_session_folder = self.nifti_folder / f'ses-0{session}' / subject.name
        nordic_session_folder = self.nordic_folder / f'ses-0{session}' / subject.name
        nordic_session_folder.mkdir(parents=True, exist_ok=True)
        
        # Check if NORDIC already processed
        existing_nordic = list(nordic_session_folder.glob('*_magnitude.nii.gz'))
        if existing_nordic:
            self.logger.info(f"  NORDIC files already exist ({len(existing_nordic)} files), skipping")
            self.stats[f'{subject.name}_ses-0{session}']['nordic'] = 'skipped'
            return True
            
        # Find magnitude images to process
        magnitude_images = list(nifti_session_folder.glob('*_magnitude.nii.gz'))
        
        if not magnitude_images:
            self.logger.warning(f"  No magnitude images found for NORDIC processing")
            self.stats[f'{subject.name}_ses-0{session}']['nordic'] = 'no_input'
            return True
            
        self.logger.info(f"  Running NORDIC on {len(magnitude_images)} images...")
        
        processed_count = 0
        for image in magnitude_images:
            phase_image = image.parent / image.name.replace('magnitude', 'phase')
            
            if not phase_image.exists():
                self.logger.warning(f"  Phase image not found for {image.name}, skipping")
                continue
                
            script_content = f"""
addpath('{self.nordic_path}')
cd('{nifti_session_folder}')
fn_magn_in='{image.name}';
fn_phase_in='{phase_image.name}';
fn_out=['NORDIC_' fn_magn_in(1:end-7)];
ARG.temporal_phase=1;
ARG.phase_filter_width=3;
NIFTI_NORDIC(fn_magn_in, fn_phase_in, fn_out, ARG)
"""
            
            if self.run_matlab_script(script_content, f'nordic_{image.stem}.m'):
                processed_count += 1
                self.logger.debug(f"    ✓ Processed {image.name}")
            else:
                self.logger.error(f"    ✗ Failed to process {image.name}")
                
        # Compress and move NORDIC outputs
        nordic_files = list(nifti_session_folder.glob('NORDIC_*.nii'))
        compressed_count = 0
        
        for nordic_file in nordic_files:
            try:
                # Compress
                subprocess.run(['gzip', '-f', str(nordic_file)], check=True)
                compressed_file = nordic_file.with_suffix('.nii.gz')
                
                # Move to NORDIC folder
                target_path = nordic_session_folder / compressed_file.name
                compressed_file.rename(target_path)
                compressed_count += 1
                self.logger.debug(f"    ✓ Compressed and moved {compressed_file.name}")
                
            except Exception as e:
                self.logger.error(f"    Error processing {nordic_file.name}: {e}")
                
        if compressed_count > 0:
            self.logger.info(f"  ✓ Compressed and moved {compressed_count} NORDIC files")
            self.stats[f'{subject.name}_ses-0{session}']['nordic'] = 'success'
            self.stats[f'{subject.name}_ses-0{session}']['nordic_files'] = compressed_count
            return True
        else:
            self.logger.error(f"  ✗ NORDIC processing failed")
            self.stats[f'{subject.name}_ses-0{session}']['nordic'] = 'failed'
            return False
            
    def copy_to_bids(self, subject: Path, session: int) -> bool:
        """Copy and organize files into BIDS structure."""
        nifti_session_folder = self.nifti_folder / f'ses-0{session}' / subject.name
        nordic_session_folder = self.nordic_folder / f'ses-0{session}' / subject.name
        
        copied_files = []
        skipped_files = []
        
        # Copy anatomical images with sequential run numbering
        # Anatomical files can be in ANY session, so collect from all sessions
        self.logger.info(f"  Copying anatomical images to BIDS...")
        
        # Collect T1w and T2w files from ALL sessions (not session-specific in BIDS)
        t1w_files = []
        t2w_files = []
        
        for ses in range(1, 4):
            session_path = self.nifti_folder / f'ses-0{ses}' / subject.name
            if session_path.exists():
                t1w_files.extend([f for f in session_path.glob('*.nii.gz') if 'T1w' in f.name])
                t2w_files.extend([f for f in session_path.glob('*.nii.gz') if 'T2w' in f.name])
        
        # Sort to ensure consistent ordering
        t1w_files = sorted(t1w_files)
        t2w_files = sorted(t2w_files)
        
        # Copy T1w files with sequential run numbers starting at 1
        for run_idx, nifti_file in enumerate(t1w_files, start=1):
            destination = self.bids_folder / subject.name / 'anat' / f'{subject.name}_run-{run_idx:02d}_T1w.nii.gz'
            if self._safe_copy(nifti_file, destination):
                copied_files.append(f'T1w-run{run_idx:02d}')
            else:
                skipped_files.append(f'T1w-run{run_idx:02d}')
        
        # Copy T2w files with sequential run numbers starting at 1
        for run_idx, nifti_file in enumerate(t2w_files, start=1):
            destination = self.bids_folder / subject.name / 'anat' / f'{subject.name}_run-{run_idx:02d}_T2w.nii.gz'
            if self._safe_copy(nifti_file, destination):
                copied_files.append(f'T2w-run{run_idx:02d}')
            else:
                skipped_files.append(f'T2w-run{run_idx:02d}')
        
        if len(t1w_files) > 1:
            self.logger.info(f"  Found {len(t1w_files)} T1w files across sessions, numbered sequentially")
        if len(t2w_files) > 1:
            self.logger.info(f"  Found {len(t2w_files)} T2w files across sessions, numbered sequentially")
                    
        # Copy DWI files
        for nifti_file in nifti_session_folder.glob('*.nii.gz'):    
            if 'sDTI128a' in nifti_file.name:
                destination = self.bids_folder / subject.name / 'dwi' / f'{subject.name}_acq-AP_dwi.nii.gz'
                if self._safe_copy(nifti_file, destination):
                    copied_files.append('DWI-AP')
                else:
                    skipped_files.append('DWI-AP')
                    
            elif 'sDTI128p' in nifti_file.name:
                destination = self.bids_folder / subject.name / 'dwi' / f'{subject.name}_acq-PA_dwi.nii.gz'
                if self._safe_copy(nifti_file, destination):
                    copied_files.append('DWI-PA')
                else:
                    skipped_files.append('DWI-PA')
        
        # Copy bval and bvec files
        self.logger.info(f"  Copying diffusion parameter files...")
        for file in os.listdir(nifti_session_folder):
            if file.endswith('.bval') and 'sDTI128a' in file:
                destination = self.bids_folder / subject.name / 'dwi' / f'{subject.name}_acq-AP_dwi.bval'
                if self._safe_copy(nifti_session_folder / file, destination):
                    copied_files.append('bval-AP')
                else:
                    skipped_files.append('bval-AP')
                    
            elif file.endswith('.bvec') and 'sDTI128a' in file:
                destination = self.bids_folder / subject.name / 'dwi' / f'{subject.name}_acq-AP_dwi.bvec'
                if self._safe_copy(nifti_session_folder / file, destination):
                    copied_files.append('bvec-AP')
                else:
                    skipped_files.append('bvec-AP')
                    
            elif file.endswith('.bval') and 'sDTI128p' in file:
                destination = self.bids_folder / subject.name / 'dwi' / f'{subject.name}_acq-PA_dwi.bval'
                if self._safe_copy(nifti_session_folder / file, destination):
                    copied_files.append('bval-PA')
                else:
                    skipped_files.append('bval-PA')
                    
            elif file.endswith('.bvec') and 'sDTI128p' in file:
                destination = self.bids_folder / subject.name / 'dwi' / f'{subject.name}_acq-PA_dwi.bvec'
                if self._safe_copy(nifti_session_folder / file, destination):
                    copied_files.append('bvec-PA')
                else:
                    skipped_files.append('bvec-PA')
                    
        # Copy functional images (NORDIC processed)
        self.logger.info(f"  Copying functional images to BIDS...")
        
        if not nordic_session_folder.exists():
            self.logger.warning(f"  NORDIC folder not found: {nordic_session_folder}")
            return len(copied_files) > 0
            
        for nifti_file in nordic_session_folder.glob('*.nii.gz'):
            # Resting state scans
            for run in range(1, 4):
                if f'WIP_RS{run}' in nifti_file.name:
                    condition = self._get_condition(subject.name, session, run)
                    if condition:
                        # Calculate sequential run number across all sessions
                        sequential_run = (session - 1) * 3 + run
                        destination = (self.bids_folder / subject.name / 
                                     'func' / f'{subject.name}_task-{condition}_run-{sequential_run:02d}_bold.nii.gz')
                        if self._safe_copy(nifti_file, destination):
                            copied_files.append(f'func-{condition}-run{sequential_run:02d}')
                        else:
                            skipped_files.append(f'func-{condition}-run{sequential_run:02d}')
                            
                # Fieldmaps
                if f'WIP_RStopup{run}' in nifti_file.name:
                    # Calculate sequential run number for fieldmaps
                    sequential_run = (session - 1) * 3 + run
                    destination = (self.bids_folder / subject.name / 
                                 'fmap' / f'{subject.name}_dir-PA_run-{sequential_run:02d}_epi.nii.gz')
                    if self._safe_copy(nifti_file, destination):
                        copied_files.append(f'fmap-run{sequential_run:02d}')
                    else:
                        skipped_files.append(f'fmap-run{sequential_run:02d}')
                        
        self.logger.info(f"  ✓ Copied {len(copied_files)} files, skipped {len(skipped_files)} existing files")
        self.stats[f'{subject.name}_ses-0{session}']['copied_files'] = len(copied_files)
        self.stats[f'{subject.name}_ses-0{session}']['skipped_files'] = len(skipped_files)
        
        return True
        
    def _safe_copy(self, source: Path, destination: Path) -> bool:
        """Safely copy file with error handling."""
        if destination.exists():
            self.logger.debug(f"    File already exists, skipping: {destination.name}")
            return False
            
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source, destination)
            self.logger.debug(f"    ✓ Copied: {destination.name}")
            return True
        except Exception as e:
            self.logger.error(f"    ✗ Failed to copy {source.name}: {e}")
            return False
            
    def _get_condition(self, subject_name: str, session: int, run: int) -> Optional[str]:
        """Get task condition from mapping."""
        try:
            return self.task_condition_mapping[subject_name][f"ses-0{session}"][f"rest{run}"]
        except KeyError:
            self.logger.error(f"  Condition not found for {subject_name}, ses-0{session}, rest{run}")
            return None
    
    def extract_acquisition_datetime(self, subject_name: str, session: int) -> Optional[Dict[str, str]]:
        """
        Extract acquisition date and time from PAR files for a specific subject and session.
        
        Parameters:
        -----------
        subject_name : str
            Subject identifier (e.g., 'sub-01')
        session : int
            Session number (1, 2, or 3)
            
        Returns:
        --------
        dict or None
            Dictionary with 'date' and 'time' keys, or None if not found
        """
        session_dir = self.raw_folder / f'ses-0{session}'
        subject_dir = session_dir / subject_name
        
        if not subject_dir.exists():
            self.logger.warning(f"  Subject directory not found: {subject_dir}")
            return None
        
        # Search for RS1, RS2, or RS3 PAR files
        par_file = None
        for rs_type in ['RS1', 'RS2', 'RS3']:
            rs_files = [f for f in os.listdir(subject_dir) if rs_type in f and f.endswith('.PAR')]
            if rs_files:
                par_file = rs_files[0]
                break
        
        if par_file is None:
            self.logger.warning(f"  No RS PAR file found for {subject_name} in session {session}")
            return None
        
        try:
            with open(subject_dir / par_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Examination date' in line:
                        # Parse: "# Examination date / time: 2024.03.09 / 14:30:15"
                        parts = line.strip().split('/')
                        if len(parts) >= 3:
                            acquisition_date = parts[1].split(':')[1].strip()
                            acquisition_time = parts[2].strip()
                            return {
                                'date': acquisition_date,
                                'time': acquisition_time
                            }
                        break
        except Exception as e:
            self.logger.error(f"  Error reading {par_file}: {e}")
            return None
        
        return None
    
    def create_scans_tsv(self, subject: Path) -> bool:
        """
        Create sub-X_scans.tsv file with acquisition dates/times for all sessions.
        
        Parameters:
        -----------
        subject : Path
            Subject folder path
            
        Returns:
        --------
        bool
            Success status
        """
        self.logger.info(f"  Creating scans.tsv file...")
        
        scans_data = []
        
        # Collect data for all sessions
        for session in range(1, 4):
            # Get acquisition datetime
            datetime_info = self.extract_acquisition_datetime(subject.name, session)
            
            if datetime_info is None:
                self.logger.warning(f"    Could not extract datetime for session {session}")
                continue
            
            # Format date and time for BIDS (YYYY-MM-DD and HH:MM:SS)
            try:
                date_obj = pd.to_datetime(datetime_info['date'], format='%Y.%m.%d')
                time_obj = pd.to_datetime(datetime_info['time'], format='%H:%M:%S')
                
                bids_date = date_obj.strftime('%Y-%m-%d')
                bids_time = time_obj.strftime('%H:%M:%S')
                acq_time = f"{bids_date}T{bids_time}"
            except Exception as e:
                self.logger.error(f"    Error formatting datetime for session {session}: {e}")
                continue
            
            # Add entries for each functional run in this session
            for run in range(1, 4):
                condition = self._get_condition(subject.name, session, run)
                if condition:
                    sequential_run = (session - 1) * 3 + run
                    filename = f"func/{subject.name}_task-{condition}_run-{sequential_run:02d}_bold.nii.gz"
                    
                    scans_data.append({
                        'filename': filename,
                        'acq_time': acq_time,
                        'run_order': sequential_run,
                        'session': session
                    })
        
        if not scans_data:
            self.logger.warning(f"  No scan data collected for scans.tsv")
            return False
        
        # Create DataFrame and save
        scans_df = pd.DataFrame(scans_data)
        scans_df = scans_df.sort_values('run_order')
        
        # Save as TSV
        scans_tsv_path = self.bids_folder / subject.name / f'{subject.name}_scans.tsv'
        scans_df.to_csv(scans_tsv_path, sep='\t', index=False)
        
        self.logger.info(f"  ✓ Created {scans_tsv_path.name} with {len(scans_data)} entries")
        return True
            
    def process_physiology_files(self, subject: Path, session: int) -> bool:
        """Process Philips physiology log files and convert to BIDS format."""
        self.logger.info(f"  Processing physiology files...")
        
        session_folder = self.raw_folder / f'ses-0{session}'
        subject_folder = session_folder / subject.name
        log_files = list(subject_folder.glob('*.log'))
        
        if not log_files:
            self.logger.info(f"  No log files found")
            return True
            
        scanphys_files = []
        # Find the Philips SCANLOG files
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    first_line = f.readline().strip()
                    _ = f.readline().strip()
                    third_line = f.readline().strip()
                    if first_line == '## Physlog file version = 2':
                        time = third_line.split(' ')[3]
                        scanphys_files.append({
                            'filename': log_file,
                            'time': time
                        })
            except Exception as e:
                self.logger.debug(f"  Skipping {log_file.name}: {e}")
                
        if not scanphys_files:
            self.logger.info(f"  No physiology log files found")
            return True
            
        self.logger.info(f"  Found {len(scanphys_files)} physiology log files")
        
        # Extract the number of timepoints from each physio file
        phys_files = []
        for scanphys in scanphys_files:
            try:
                df = pd.read_csv(scanphys['filename'], sep=r'\s+', skiprows=6, header=None,
                               names=['v1raw', 'v2raw', 'v1', 'v2', 'ppu', 'resp', 
                                     'gx', 'gy', 'gz', 'mark', 'mark2'])
                phys_files.append({
                    'filename': scanphys['filename'],
                    'n_timepoints': df.shape[0],
                    'time': scanphys['time']
                })
            except Exception as e:
                self.logger.error(f"  Error reading {scanphys['filename'].name}: {e}")
                
        if not phys_files:
            self.logger.warning(f"  No valid physiology files to process")
            return True
            
        # Filtering the physio files that belong to the fMRI sequences (longer than the others)
        phys_file_df = pd.DataFrame(phys_files)
        phys_file_df = phys_file_df.sort_values(by='time', ascending=True)
        phys_file_df = phys_file_df.loc[phys_file_df['n_timepoints'] > 300000]
        
        if phys_file_df.empty:
            self.logger.info(f"  No fMRI physiology files found (files > 300000 timepoints)")
            return True
            
        processed_count = 0
        # Reset index to ensure run numbers start from 1
        phys_file_df = phys_file_df.reset_index(drop=True)
        
        # Process each physiology file
        for idx in range(len(phys_file_df)):
            run = idx + 1  # Run numbers start at 1
            try:
                filename = phys_file_df.iloc[idx]['filename']
                n_timepoints = phys_file_df.iloc[idx]['n_timepoints']
                time = phys_file_df.iloc[idx]['time']
                
                self.logger.debug(f"    Processing physio file: {filename.name} "
                                f"(Timepoints: {n_timepoints}, Time: {time})")
                
                phys_df = pd.read_csv(filename, sep=r'\s+', skiprows=6, header=None,
                                     names=['v1raw', 'v2raw', 'v1', 'v2', 'ppu', 'resp', 
                                           'gx', 'gy', 'gz', 'mark', 'mark2'])
                
                # Crop data to scan start/stop markers (10 = 0x10 start, 20 = 0x20 stop)
                # Find the first occurrence of 10 (scan start marker)
                start_indices = phys_df.index[phys_df['mark'] == 10].tolist()
                # Find the last occurrence of 20 (scan stop marker)
                stop_indices = phys_df.index[phys_df['mark'] == 20].tolist()
                
                if start_indices and stop_indices:
                    start_idx = start_indices[0]
                    stop_idx = stop_indices[-1]
                    
                    if start_idx < stop_idx:
                        original_len = len(phys_df)
                        phys_df = phys_df.iloc[start_idx:stop_idx + 1].reset_index(drop=True)
                        cropped_duration = len(phys_df) / 496  # 496 Hz sampling rate
                        self.logger.debug(f"    Cropped physio data: {original_len} -> {len(phys_df)} samples "
                                        f"({cropped_duration:.1f}s)")
                    else:
                        self.logger.warning(f"    Invalid marker positions (start={start_idx}, stop={stop_idx}), "
                                          "using full data")
                else:
                    self.logger.warning(f"    Scan markers not found (0010: {len(start_indices)}, "
                                      f"0020: {len(stop_indices)}), using full data")
                
                phys_df = phys_df[['ppu', 'resp', 'mark']]
                phys_df.rename(columns={'ppu': 'cardiac', 'resp': 'respiratory', 'mark': 'trigger'}, 
                             inplace=True)
                
                # Get condition for this run
                condition = self._get_condition(subject.name, session, run)
                if not condition:
                    self.logger.warning(f"  Could not determine condition for run {run}, skipping")
                    continue
                    
                # Calculate sequential run number across all sessions
                sequential_run = (session - 1) * 3 + run
                
                # Construct the filename for BIDS
                bids_folder_func = self.bids_folder / subject.name / 'func'
                bids_folder_func.mkdir(parents=True, exist_ok=True)
                
                bids_filename = (bids_folder_func /
                               f"{subject.name}_task-{condition}_run-{sequential_run:02d}_physio.tsv.gz")

                # Save the physiology data with clean gzip headers (no timestamp, no filename)
                # to avoid BIDS validator warnings
                tsv_content = phys_df.to_csv(sep='\t', index=False, header=False)
                with gzip.GzipFile(filename='', fileobj=open(bids_filename, 'wb'), mode='wb', mtime=0) as f:
                    f.write(tsv_content.encode('utf-8'))
                
                # Save the JSON sidecar (remove both .gz and .tsv, then add .json)
                json_filename = bids_filename.with_suffix('').with_suffix('.json')
                json_content = {
                    "TaskName": condition,
                    "SamplingFrequency": 496,
                    "Columns": ["cardiac", "respiratory", "trigger"],
                    "Manufacturer": "Philips",
                    "StartTime": 0,
                    "Session": f"ses-0{session}"
                }
                with open(json_filename, 'w') as jf:
                    json.dump(json_content, jf, indent=4)
                    
                processed_count += 1
                self.logger.debug(f"    ✓ Created physio files: {bids_filename.name}")
                
            except Exception as e:
                self.logger.error(f"    ✗ Error processing physio file: {e}")
                
        self.logger.info(f"  ✓ Processed {processed_count} physiology files")
        self.stats[f'{subject.name}_ses-0{session}']['physio_files'] = processed_count
        
        return True
    
    def create_sidecars(self, subject: Path, session: int) -> bool:
        """Create JSON sidecar files for BIDS compliance."""
        self.logger.info(f"  Creating JSON sidecars...")
        
        created_count = 0
        
        # Fieldmap sidecars
        fmap_files = list(self.bids_folder.glob(f'{subject.name}/fmap/*.nii.gz'))
        for fmap_file in fmap_files:
            json_path = fmap_file.with_suffix('').with_suffix('.json')
            
            if json_path.exists():
                continue
                
            # Extract run number from filename
            run = self.extract_run_number(json_path)
            if run:
                # Calculate which session and within-session run this corresponds to
                session_num = ((run - 1) // 3) + 1
                within_session_run = ((run - 1) % 3) + 1
                condition = self._get_condition(subject.name, session_num, within_session_run)
                if condition:
                    json_data = {
                        "IntendedFor": f"func/{subject.name}_task-{condition}_run-{run:02d}_bold.nii.gz",
                        "SkullStripped": False,
                        "PhaseEncodingDirection": "j",
                        "EffectiveEchoSpacing": 0.00054,
                        "TotalReadoutTime": 0.05994,
                        "Session": f"ses-0{session_num}"
                    }
                    
                    try:
                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=4)
                        created_count += 1
                        self.logger.debug(f"    ✓ Created fieldmap sidecar: {json_path.name}")
                    except Exception as e:
                        self.logger.error(f"    ✗ Failed to create {json_path.name}: {e}")
                        
        # Functional sidecars
        func_files = list(self.bids_folder.glob(f'{subject.name}/func/*_bold.nii.gz'))
        for func_file in func_files:
            json_path = func_file.with_suffix('').with_suffix('.json')
            
            if json_path.exists():
                continue
                
            try:
                img = nib.load(func_file)
                n_volumes = img.shape[-1]
                
                # Extract run number to determine session
                run = self.extract_run_number(func_file)
                if run:
                    session_num = ((run - 1) // 3) + 1
                    json_data = {
                        "NumberOfTemporalPositions": n_volumes,
                        "Session": f"ses-0{session_num}",
                        "SkullStripped": False
                    }
                else:
                    json_data = {
                        "NumberOfTemporalPositions": n_volumes
                    }
                
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                created_count += 1
                self.logger.debug(f"    ✓ Created functional sidecar: {json_path.name}")
                
            except Exception as e:
                self.logger.error(f"    ✗ Failed to create sidecar for {func_file.name}: {e}")
        
        # DWI sidecars
        dwi_files = list(self.bids_folder.glob(f'{subject.name}/dwi/*.nii.gz'))
        for dwi_file in dwi_files:
            json_path = dwi_file.with_suffix('').with_suffix('.json')
            
            if json_path.exists():
                continue
                
            try:
                if 'AP' in dwi_file.name:
                    json_data = {
                        "PhaseEncodingDirection": "j",
                        "MultibandAccelerationFactor": 3,
                        "TotalReadoutTime": 0.100737,
                        "SkullStripped": False
                    }
                elif 'PA' in dwi_file.name:
                    json_data = {
                        "PhaseEncodingDirection": "j-",
                        "MultibandAccelerationFactor": 3,
                        "TotalReadoutTime": 0.100737,
                        "SkullStripped": False
                    }
                else:
                    continue
                
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                created_count += 1
                self.logger.debug(f"    ✓ Created DWI sidecar: {json_path.name}")
                
            except Exception as e:
                self.logger.error(f"    ✗ Failed to create sidecar for {dwi_file.name}: {e}")
                
        self.logger.info(f"  ✓ Created {created_count} JSON sidecars")
        self.stats[f'{subject.name}_ses-0{session}']['sidecars'] = created_count
        
        return True
        
    def process_subject(self, subject: Path, session: int) -> bool:
        """Process a single subject through the entire pipeline."""
        # Setup subject-specific logging
        self.setup_subject_logging(subject.name, session)
        
        self.logger.info("=" * 80)
        self.logger.info(f"Processing {subject.name} - Session {session}")
        self.logger.info("=" * 80)
        
        # Check if subject already processed for this session
        # Calculate sequential run numbers for this session
        sequential_runs = [(session - 1) * 3 + i for i in range(1, 4)]
        
        # Check if functional files exist for these runs
        func_folder = self.bids_folder / subject.name / 'func'
        existing_runs = []
        if func_folder.exists():
            for run_num in sequential_runs:
                run_files = list(func_folder.glob(f'*_run-{run_num:02d}_bold.nii.gz'))
                if run_files:
                    existing_runs.append(run_num)
        
        # Check if anatomical files exist - check for BOTH T1w and T2w
        anat_folder = self.bids_folder / subject.name / 'anat'
        t1w_exists = False
        t2w_exists = False
        if anat_folder.exists():
            # Check for any T1w file with run number (run-01, run-02, etc.)
            t1w_files = list(anat_folder.glob(f'{subject.name}_run-*_T1w.nii.gz'))
            t1w_exists = len(t1w_files) > 0
            # Check for any T2w file with run number
            t2w_files = list(anat_folder.glob(f'{subject.name}_run-*_T2w.nii.gz'))
            t2w_exists = len(t2w_files) > 0
        
        # Anatomicals are considered complete if at least T1w exists 
        # (T2w is optional as not all subjects have it)
        anat_exists = t1w_exists
        
        if len(existing_runs) == 3 and anat_exists:
            self.logger.info(f"All functional and anatomical files for session {session} already exist in BIDS folder")
            # Check if physiology files exist for these runs
            existing_physio = []
            for run_num in sequential_runs:
                physio_files = list(func_folder.glob(f'*_run-{run_num:02d}_physio.tsv.gz'))
                if physio_files:
                    existing_physio.append(run_num)
            
            if len(existing_physio) == 3:
                self.logger.info(f"  Physiology files also exist, skipping entirely")
                self.stats[f'{subject.name}_ses-0{session}']['status'] = 'already_processed'
                self.cleanup_subject_logging()
                return True
            else:
                self.logger.info(f"  Physiology files missing, processing only physiology files")
                # Process only physiology files
                try:
                    if not self.process_physiology_files(subject, session):
                        self.logger.warning(f"Physiology file processing had issues")
                    self.stats[f'{subject.name}_ses-0{session}']['status'] = 'physio_only'
                    self.cleanup_subject_logging()
                    return True
                except Exception as e:
                    self.logger.error(f"✗ Error processing physiology files: {e}", exc_info=True)
                    self.stats[f'{subject.name}_ses-0{session}']['status'] = 'physio_error'
                    self.cleanup_subject_logging()
                    return False
            
        try:
            # Step 1: DICOM to NIfTI
            if not self.dicom_to_nifti(subject, session):
                self.stats[f'{subject.name}_ses-0{session}']['status'] = 'failed_dicom2nifti'
                self.cleanup_subject_logging()
                return False
                
            # Step 2: NORDIC processing
            if not self.nordic_processing(subject, session):
                self.logger.warning(f"NORDIC processing had issues, continuing...")
                
            # Step 3: Copy to BIDS
            if not self.copy_to_bids(subject, session):
                self.stats[f'{subject.name}_ses-0{session}']['status'] = 'failed_bids_copy'
                self.cleanup_subject_logging()
                return False
                
            # Step 4: Create sidecars
            if not self.create_sidecars(subject, session):
                self.logger.warning(f"Sidecar creation had issues, continuing...")
                
            # Step 5: Process physiology files
            if not self.process_physiology_files(subject, session):
                self.logger.warning(f"Physiology file processing had issues, continuing...")
                
            self.stats[f'{subject.name}_ses-0{session}']['status'] = 'success'
            self.logger.info(f"✓ Successfully processed {subject.name} session {session}")
            
            # Cleanup subject logging
            self.cleanup_subject_logging()
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Unexpected error processing {subject.name}: {e}", exc_info=True)
            self.stats[f'{subject.name}_ses-0{session}']['status'] = 'error'
            self.stats[f'{subject.name}_ses-0{session}']['error'] = str(e)
            self.cleanup_subject_logging()
            return False
            
    def print_summary(self):
        """Print processing summary statistics."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 80)
        
        status_counts = defaultdict(int)
        for subject_session, data in self.stats.items():
            status_counts[data.get('status', 'unknown')] += 1
            
        self.logger.info(f"\nTotal subject-sessions processed: {len(self.stats)}")
        for status, count in sorted(status_counts.items()):
            self.logger.info(f"  {status}: {count}")
            
        # Detailed failures
        failed = [s for s, d in self.stats.items() if 'failed' in d.get('status', '') or d.get('status') == 'error']
        if failed:
            self.logger.info(f"\nFailed subject-sessions ({len(failed)}):")
            for subject_session in failed:
                self.logger.info(f"  - {subject_session}: {self.stats[subject_session].get('status')}")
                
        # Export detailed statistics to CSV
        self.export_statistics()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"Summary log file: {self.summary_log_file}")
        self.logger.info(f"Individual participant logs: {self.log_dir}")
        self.logger.info("=" * 80)
        
    def export_statistics(self):
        """Export detailed processing statistics to CSV file in logs folder."""
        if not self.stats:
            return
            
        try:
            # Prepare data for export
            rows = []
            for subject_session, data in sorted(self.stats.items()):
                row = {
                    'subject_session': subject_session,
                    'status': data.get('status', 'unknown'),
                    'dicom2nifti': data.get('dicom2nifti', 'N/A'),
                    'nifti_files': data.get('nifti_files', 0),
                    'nordic': data.get('nordic', 'N/A'),
                    'nordic_files': data.get('nordic_files', 0),
                    'copied_files': data.get('copied_files', 0),
                    'skipped_files': data.get('skipped_files', 0),
                    'sidecars': data.get('sidecars', 0),
                    'error': data.get('error', '')
                }
                rows.append(row)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(rows)
            stats_file = self.log_dir / f'processing_stats_{self.timestamp}.csv'
            df.to_csv(stats_file, index=False)
            
            self.logger.info(f"\nDetailed statistics exported to: {stats_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to export statistics: {e}")


def main():
    """Main execution function."""
    # Define project folder
    project_folder = Path('/home/ASDPrecision/')
    
    # Initialize converter
    converter = BIDSConverter(project_folder)
    
    # Validate paths
    if not converter.validate_paths():
        converter.logger.error("Path validation failed. Please check your directory structure.")
        return 1
        
    # Change to working directory
    os.chdir(converter.project_folder)
    
    # Track all unique subjects across sessions
    all_subjects = set()
    
    # Process all sessions
    for session in range(1, 4):
        session_folder = converter.raw_folder / f'ses-0{session}'
        
        if not session_folder.exists():
            converter.logger.warning(f"Session folder not found: {session_folder}")
            continue
            
        # Get subject list from the session folder
        subject_list = sorted([
            subject for subject in session_folder.iterdir() 
            if subject.is_dir() and subject.name.startswith('sub-')
        ])

        # Check if the subject is in the participants.tsv file
        participants = pd.read_csv(project_folder / 'data' / 'original_bids' / 'participants.tsv', sep='\t')
        subject_list = sorted([subject for subject in subject_list if subject.name in participants['participant_id'].values])
        converter.logger.info(f"\nFound {len(subject_list)} subjects to process in session {session}")
        
        # Process each subject (process_subject will handle skipping logic internally)
        for idx, subject in enumerate(subject_list, 1):
            converter.logger.info(f"\n[{idx}/{len(subject_list)}] Starting {subject.name}")
            converter.process_subject(subject, session)
            all_subjects.add(subject.name)
    
    # Create scans.tsv files for all subjects after all sessions are processed
    converter.logger.info("\n" + "=" * 80)
    converter.logger.info("Creating scans.tsv files for all subjects...")
    converter.logger.info("=" * 80)
    
    for subject_name in sorted(all_subjects):
        converter.logger.info(f"\nCreating scans.tsv for {subject_name}")
        subject_path = Path(subject_name)
        try:
            converter.create_scans_tsv(subject_path)
        except Exception as e:
            converter.logger.error(f"Failed to create scans.tsv for {subject_name}: {e}")
            
    # Print final summary
    converter.print_summary()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())