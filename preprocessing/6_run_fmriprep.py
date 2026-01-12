#!/usr/bin/env python3
"""
Run fMRIprep for all participants in BIDS dataset using Docker.

Author: Joe Bathelt
Date: October 2025
Version: 1.0
"""
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

def create_bash_script(
    project_folder: Path,
    participant_label: str,
    *,
    bids_folder: Path,
    fmriprep_folder: Path,
    logs_dir: Path,
    work_dir: Path,
    fs_license_file: Path,
    n_cpus: int,
    omp_threads: int,
):
    """Create a subject-specific bash script to run fMRIPrep via fmriprep-docker."""
    log_file = logs_dir / f'fmriprep_{participant_label}.log'

    script_content = Template(
                """#!/usr/bin/env bash
set -euo pipefail
echo "[$$(date -Is)] Starting fMRIPrep for ${participant_label}" | tee -a "${log_file}"

fmriprep-docker \
        "${bids_folder}" \
        "${fmriprep_folder}" \
        participant \
        --participant-label ${participant_label} \
        --fs-license-file "${fs_license_file}" \
        --cifti-output \
        --work-dir "${work_dir}" \
        --n_cpus ${n_cpus} \
        --omp-nthreads ${omp_threads} >> "${log_file}" 2>&1

status=$$?
if [[ $$status -eq 0 ]]; then
    echo "[$$(date -Is)] Completed fMRIPrep for ${participant_label}" | tee -a "${log_file}"
else
    echo "[$$(date -Is)] FAILED fMRIPrep for ${participant_label} with exit code $$status" | tee -a "${log_file}"
fi
exit $$status
"""
        ).substitute(
        {
            'bids_folder': bids_folder.as_posix(),
            'fmriprep_folder': fmriprep_folder.as_posix(),
            'log_file': log_file.as_posix(),
            'fs_license_file': fs_license_file.as_posix(),
            'participant_label': participant_label,
            'work_dir': (work_dir / participant_label).as_posix(),
            'n_cpus': n_cpus,
            'omp_threads': omp_threads,
        }
    )

    script_path = project_folder / f'scripts/run_fmriprep_{participant_label}.sh'
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
    logs_folder = project_folder / 'logs' / 'fmriprep'
    fmriprep_folder = project_folder / 'data' / 'bids' / 'derivatives' / 'fmriprep'
    work_folder = Path('/home/jmbathe/work/')
    fs_license_file = project_folder / 'data' / 'license.txt'

    # Check prerequisites
    if shutil.which('fmriprep-docker') is None:
        raise SystemExit('fmriprep-docker not found in PATH. Please install it.')
    if not fs_license_file.exists():
        raise SystemExit(f'FreeSurfer license not found at: {fs_license_file}')

    # Create the scripts and logs folders if they don't exist
    os.makedirs(scripts_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(fmriprep_folder, exist_ok=True)
    os.makedirs(work_folder, exist_ok=True)

    # Resource configuration (no argparse): compute sensible defaults
    total_cpus = os.cpu_count() or 1
    # Per-job CPUs: use up to 16 cores per job, but don't exceed total available
    n_cpus_per_job = min(16, total_cpus)
    # OMP threads should be less than n_cpus and reasonable for parallelization
    omp_threads = max(1, n_cpus_per_job // 2)
    # Max parallel jobs: ensure we don't oversubscribe (total_cpus / n_cpus_per_job)
    max_parallel = max(1, total_cpus // n_cpus_per_job)

    # List subjects
    subject_list = sorted([
        d for d in os.listdir(bids_folder)
        if os.path.isdir(bids_folder / f'{d}') and d.startswith('sub-')
    ])

    # Generate bash scripts for each subject that needs processing
    script_paths = []
    for subject in subject_list[1:]:
        if _is_subject_processed(fmriprep_folder, subject):
            print(f"Skipping {subject}: appears already processed.")
            continue
        print(f"Creating script for {subject}")
        script_path = create_bash_script(
            project_folder,
            subject,
            bids_folder=bids_folder,
            fmriprep_folder=fmriprep_folder,
            logs_dir=logs_folder,
            work_dir=work_folder,
            fs_license_file=fs_license_file,
            n_cpus=n_cpus_per_job,
            omp_threads=omp_threads,
        )
        script_paths.append(script_path)

    if not script_paths:
        print('Nothing to run: all subjects processed or no subjects found.')
        return

    # Run scripts in parallel without oversubscribing CPUs
    print(
        f"Launching up to {max_parallel} job(s) in parallel; per-job CPUs={n_cpus_per_job}, OMP={omp_threads}"
    )
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_script, script_path): script_path for script_path in script_paths}

        for future in as_completed(futures):
            script_path = futures[future]
            try:
                future.result()
                print(f"Completed: {Path(script_path).name}")
            except subprocess.CalledProcessError as e:
                subj = Path(script_path).stem.replace('run_fmriprep_', '')
                print(f"Processing of {script_path} failed (exit {e.returncode}). See log: {logs_folder / f'fmriprep_{subj}.log'}")
                
if __name__ == "__main__":
    main()