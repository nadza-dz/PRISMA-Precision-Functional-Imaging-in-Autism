#!/usr/bin/env python3
"""
Run qsirecon for all participants in BIDS dataset using Docker.

Author: Joe Bathelt
Date: November 2025
Version: 1.1
"""
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

def _get_total_mem_mb() -> int:
    """Return total system memory in MB (Linux /proc/meminfo)."""
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith('MemTotal:'):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return max(1, int(int(parts[1]) / 1024))  # kB -> MB
    # Fallback conservative default (64 GB)
    return 64 * 1024

def create_bash_script(
    project_folder: Path,
    participant_label: str,
    *,
    bids_folder: Path,
    qsiprep_folder: Path,
    output_folder: Path,
    freesurfer_folder: Path,
    logs_dir: Path,
    work_dir: Path,
    fs_license_file: Path,
):
    """Create a subject-specific bash script to run qsirecon via Docker."""
    log_file = logs_dir / f'qsirecon_{participant_label}.log'

    script_content = Template(
"""#!/usr/bin/env bash
set -euo pipefail
echo "[$$(date -Is)] Starting qsirecon for ${participant_label}" | tee -a "${log_file}"

docker run --rm \
    -v "${qsiprep_folder}":/data:ro \
    -v "${freesurfer_folder}":/freesurfer:ro \
    -v "${output_folder}":/out \
    -v "${work_folder}":/work \
    -v "${fs_license_file}":/opt/freesurfer/license.txt:ro \
    pennlinc/qsirecon:1.1.1 \
    /data /out participant \
    --participant-label ${participant_label} \
    --recon-spec mrtrix_singleshell_ss3t_ACT-hsvs \
    --fs-subjects-dir /freesurfer \
    --work-dir /work \
    --atlases 4S156Parcels 4S256Parcels 4S356Parcels 4S456Parcels 4S556Parcels Brainnetome246Ext Gordon333Ext \
    --output-resolution 1.75 \
    --n-cpus 16 \
    --omp-nthreads 8 \
    --low-mem \
    --fs-license-file /opt/freesurfer/license.txt >> "${log_file}" 2>&1

status=$$?
if [[ $$status -eq 0 ]]; then
    echo "[$$(date -Is)] Completed qsirecon for ${participant_label}" | tee -a "${log_file}"
else
    echo "[$$(date -Is)] FAILED qsirecon for ${participant_label} with exit code $$status" | tee -a "${log_file}"
fi
exit $$status
"""
    ).substitute(
        {
            'bids_folder': bids_folder.as_posix(),
            'qsiprep_folder': qsiprep_folder.as_posix(),
            'freesurfer_folder': freesurfer_folder.as_posix(),
            'output_folder': output_folder.as_posix(),
            'log_file': log_file.as_posix(),
            'fs_license_file': fs_license_file.as_posix(),
            'participant_label': participant_label,
            'work_folder': (work_dir / participant_label).as_posix(),
        }
    )

    script_path = project_folder / f'scripts/run_qsirecon_{participant_label}.sh'
    with open(script_path, 'w') as script_file:
        script_file.write(script_content)

    # Make the script executable
    os.chmod(script_path, 0o755)
    return script_path.as_posix()

def run_script(script_path):
    subprocess.run(script_path, shell=True, check=True)

def _is_subject_processed(out_dir: Path, subject: str) -> bool:
    """Heuristic: consider subject processed if report html exists or subject dir has files."""
    report = out_dir / f"{subject}.html"
    if report.exists():
        return True
    subj_dir = out_dir / subject
    if not subj_dir.exists():
        return False
    for root, _, files in os.walk(subj_dir):
        if files:
            return True
    return False


def main():
    # Define paths
    project_folder = Path('/home/ASDPrecision/')
    bids_folder = project_folder / 'data' / 'bids'
    scripts_folder = project_folder / 'scripts'
    logs_folder = project_folder / 'logs' / 'qsirecon'
    qsiprep_folder = project_folder / 'data' / 'bids' / 'derivatives' / 'qsiprep'
    qsirecon_folder = project_folder / 'data' / 'bids' / 'derivatives' / 'qsirecon'
    freesurfer_folder = project_folder / 'data' / 'bids' / 'derivatives' / 'fmriprep' / 'sourcedata' / 'freesurfer'
    work_folder = project_folder / 'work' / 'qsirecon'
    fs_license_file = project_folder / 'license.txt'

    # Check prerequisites
    if shutil.which('docker') is None:
        raise SystemExit('docker not found in PATH. Please install it.')
    if not fs_license_file.exists():
        raise SystemExit(f'FreeSurfer license not found at: {fs_license_file}')

    # Create the scripts and logs folders if they don't exist
    os.makedirs(scripts_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(qsirecon_folder, exist_ok=True)
    os.makedirs(work_folder, exist_ok=True)

    # Number of participants to run in parallel (configurable)
    max_parallel = int(os.getenv('QSIRECON_MAX_PARALLEL', '4'))  # Default: 4 participants

    # List subjects
    subject_list = sorted([
        d for d in os.listdir(bids_folder)
        if os.path.isdir(bids_folder / f'{d}') and d.startswith('sub-')
    ])

    # Generate bash scripts for each subject that needs processing
    script_paths = []
    for subject in subject_list:
        if _is_subject_processed(qsirecon_folder, subject):
            print(f"Skipping {subject}: appears already processed.")
            continue
        print(f"Creating script for {subject}")
        script_path = create_bash_script(
            project_folder,
            subject,
            bids_folder=bids_folder,
            qsiprep_folder=qsiprep_folder,
            output_folder=qsirecon_folder,
            freesurfer_folder=freesurfer_folder,
            logs_dir=logs_folder,
            work_dir=work_folder,
            fs_license_file=fs_license_file,
        )
        script_paths.append(script_path)

    if not script_paths:
        print('Nothing to run: all subjects processed or no subjects found.')
        return

    # Run scripts in parallel
    print(f"Running up to {max_parallel} participant(s) in parallel (16 CPUs each)")
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_script, script_path): script_path for script_path in script_paths}

        for future in as_completed(futures):
            script_path = futures[future]
            try:
                future.result()
                print(f"Completed: {Path(script_path).name}")
            except subprocess.CalledProcessError as e:
                subj = Path(script_path).stem.replace('run_qsirecon_', '')
                print(f"Processing of {script_path} failed (exit {e.returncode}). See log: {logs_folder / f'qsirecon_{subj}.log'}")
                print("Continuing with remaining subjects...")
                
if __name__ == "__main__":
    main()