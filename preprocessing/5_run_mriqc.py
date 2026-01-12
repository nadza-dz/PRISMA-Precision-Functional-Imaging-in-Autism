#!/usr/bin/env python3
"""
Run MRIQC for all participants in BIDS dataset using Docker.

Author: Joe Bathelt
Date: October 2025
Version: 1.0
"""
import os
from pathlib import Path
from subprocess import call
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%
project_folder = Path('/home/ASDPrecision/')
bids_folder = project_folder / 'data' / 'bids'
out_folder = project_folder / 'data' / 'bids/derivatives/mriqc'
log_folder = project_folder / 'logs' / 'mriqc'

out_folder.mkdir(parents=True, exist_ok=True)
log_folder.mkdir(parents=True, exist_ok=True)

participant_list = sorted([f.name for f in bids_folder.glob('sub-*')])
participant_list = [p for p in participant_list if not (out_folder / p).is_dir()]

print(f"Found {len(participant_list)} participants to process")

def run_mriqc(bids_folder, out_folder, log_folder, participant_id):
    from pathlib import Path
    
    # Extract just the ID without 'sub-' prefix
    participant_label = participant_id.replace('sub-', '')
    
    print(f"Starting MRIQC for {participant_id}")
    
    cmd = f"""docker run --rm \
        -v {bids_folder}:/data:ro \
        -v {out_folder}:/out \
        nipreps/mriqc:25.0.0rc0 /data /out \
        participant --participant-label {participant_label} -vv > {log_folder}/mriqc_{participant_id}.log 2>&1"""
    
    print(f"Running command: {cmd}")
    call(cmd, shell=True)

# Run commands in parallel
max_workers = 15  # Number of parallel processes
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_participant = {executor.submit(run_mriqc, bids_folder, out_folder, log_folder, p): p for p in participant_list}

    for future in as_completed(future_to_participant):
        participant = future_to_participant[future]
        try:
            future.result()  # This will raise any exception caught during execution
            print(f"Finished processing: {participant}")
        except Exception as exc:
            print(f"Participant {participant} generated an exception: {exc}")
# %%
