#!/usr/bin/env python3
"""
Run QSIPrep for all participants in BIDS dataset using Docker.

Author: Joe Bathelt
Date: November 2025
Version: 1.0
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
    derivative_folder: Path,
    logs_dir: Path,
    work_dir: Path,
    fs_license_file: Path,
):
    """Create a subject-specific bash script to run QSIPrep via Docker."""
    log_file = logs_dir / f'qsiprep_{participant_label}.log'

    script_content = Template(
"""#!/usr/bin/env bash
set -euo pipefail
echo "[$$(date -Is)] Starting QSIPrep for ${participant_label}" | tee -a "${log_file}"

docker run --rm \\
    -v "${bids_folder}":/data:ro \\
    -v "${derivative_folder}":/out \\
    -v "${work_folder}":/work \\
    -v "${fs_license_file}":/opt/freesurfer/license.txt:ro \\
    pennlinc/qsiprep:1.0.1 \\
    /data /out participant \\
    --participant-label ${participant_label} \\
    --work-dir /work \\
    --output-resolution 1.75 \\
    --hmc-model eddy \\
    --denoise-method dwidenoise \\
    --unringing-method mrdegibbs \\
    --pepolar-method TOPUP \\
    --n-cpus 32 \\
    --omp-nthreads 8 \\
    --b0-threshold 50 \\
    --low-mem \\
    --fs-license-file /opt/freesurfer/license.txt >> "${log_file}" 2>&1

status=$$?
if [[ $$status -eq 0 ]]; then
    echo "[$$(date -Is)] Completed QSIPrep for ${participant_label}" | tee -a "${log_file}"
else
    echo "[$$(date -Is)] FAILED QSIPrep for ${participant_label} with exit code $$status" | tee -a "${log_file}"
fi
exit $$status
"""
    ).substitute(
        {
            'bids_folder': bids_folder.as_posix(),
            'derivative_folder': derivative_folder.as_posix(),
            'log_file': log_file.as_posix(),
            'fs_license_file': fs_license_file.as_posix(),
            'participant_label': participant_label,
            'work_folder': (work_dir / participant_label).as_posix(),
        }
    )

    script_path = project_folder / f'scripts/run_qsiprep_{participant_label}.sh'
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
    logs_folder = project_folder / 'logs' / 'qsiprep'
    derivative_folder = project_folder / 'data' / 'bids' / 'derivatives' / 'qsiprep'
    work_folder = project_folder / 'work' / 'qsiprep'
    fs_license_file = project_folder / 'license.txt'

    # Check prerequisites
    if shutil.which('docker') is None:
        raise SystemExit('docker not found in PATH. Please install it.')
    if not fs_license_file.exists():
        raise SystemExit(f'FreeSurfer license not found at: {fs_license_file}')

    # Create the scripts and logs folders if they don't exist
    os.makedirs(scripts_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(derivative_folder, exist_ok=True)
    os.makedirs(work_folder, exist_ok=True)

    # Number of participants to run in parallel
    max_parallel = 3

    # List subjects
    subject_list = sorted([
        d for d in os.listdir(bids_folder)
        if os.path.isdir(bids_folder / f'{d}') and d.startswith('sub-')
    ])

    # Generate bash scripts for each subject that needs processing
    script_paths = []
    for subject in subject_list:
        if _is_subject_processed(derivative_folder, subject):
            print(f"Skipping {subject}: appears already processed.")
            continue
        print(f"Creating script for {subject}")
        script_path = create_bash_script(
            project_folder,
            subject,
            bids_folder=bids_folder,
            derivative_folder=derivative_folder,
            logs_dir=logs_folder,
            work_dir=work_folder,
            fs_license_file=fs_license_file,
        )
        script_paths.append(script_path)

    if not script_paths:
        print('Nothing to run: all subjects processed or no subjects found.')
        return

    # Run scripts in parallel
    print(f"Running up to {max_parallel} participant(s) in parallel")
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_script, script_path): script_path for script_path in script_paths}

        for future in as_completed(futures):
            script_path = futures[future]
            try:
                future.result()
                print(f"Completed: {Path(script_path).name}")
            except subprocess.CalledProcessError as e:
                subj = Path(script_path).stem.replace('run_qsiprep_', '')
                print(f"Processing of {script_path} failed (exit {e.returncode}). See log: {logs_folder / f'qsiprep_{subj}.log'}")
                print("Continuing with remaining subjects...")
                
if __name__ == "__main__":
    main()