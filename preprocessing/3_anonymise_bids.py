#!/usr/bin/env python3
"""
Helper script to anonymise BIDS dataset by replacing participant IDs with consecutive integers.

Author: Joe Bathelt
Date: January 2026
Version: 1.2

Version History:
1.0 - Initial version
1.1 - Improved JSON and TSV handling for participant ID replacement
1.2 - Added rounding to the acquisition time
"""

import json
import pandas as pd
from pathlib import Path
from shutil import copytree
import sys


# Configuration - Edit these paths as needed
PROJECT_FOLDER = Path('/home/ASDPrecision/')
ORIGINAL_BIDS = Path('/home/ASDPrecision/data/original_bids')
EXPORT_BIDS = PROJECT_FOLDER / 'data/bids'


def replace_participant_id_in_json(content, old_id, new_id):
    """Recursively replace participant ID in JSON content."""
    if isinstance(content, dict):
        return {k: replace_participant_id_in_json(v, old_id, new_id) for k, v in content.items()}
    elif isinstance(content, list):
        return [replace_participant_id_in_json(i, old_id, new_id) for i in content]
    elif isinstance(content, str):
        return content.replace(old_id, new_id)
    else:
        return content


def main():
    """Main function to anonymise BIDS dataset."""
    # Create export directory
    EXPORT_BIDS.mkdir(parents=True, exist_ok=True)
    
    # Load participants
    participants_df = pd.read_csv(ORIGINAL_BIDS / 'participants.tsv', sep='\t')
    participants_df.sort_values(by='participant_id', ascending=True, inplace=True)

    # Replace the participant IDs with consecutive integers
    participants_df['new_participant_id'] = [f'sub-{str(i).zfill(3)}' for i in range(1, len(participants_df) + 1)]
    
    print(f"Anonymising {len(participants_df)} participants...")

    # Process each participant
    for i, row in participants_df.iterrows():
        old_id = str(row['participant_id'])
        new_id = str(row['new_participant_id'])
        
        participant_folder = ORIGINAL_BIDS / old_id
        export_folder = EXPORT_BIDS / new_id

        print(f'Exporting participant {old_id} to {new_id}...')

        if export_folder.exists():
            print(f'  Skipping copy - {new_id} already exists')
        else:
            # Copy the participant's data
            copytree(participant_folder, export_folder)

            # Rename all files and JSON content containing the old participant ID
            for file_path in export_folder.rglob('*'):
                if file_path.is_file():
                    # Rename file if it contains old ID
                    if old_id in file_path.name:
                        new_name = file_path.name.replace(old_id, new_id)
                        new_path = file_path.parent / new_name
                        file_path.rename(new_path)
                        file_path = new_path
                    
                    # Update JSON file contents
                    if file_path.suffix == '.json':
                        try:
                            with open(file_path, 'r') as f:
                                content = json.load(f)
                            
                            # Replace old ID with new ID in JSON content
                            updated_content = replace_participant_id_in_json(content, old_id, new_id)
                            
                            with open(file_path, 'w') as f:
                                json.dump(updated_content, f, indent=4, ensure_ascii=False)
                        except Exception as e:
                            print(f'  Warning: Could not update {file_path}: {e}')
        
        # Update TSV file contents (e.g., scans.tsv files) - always run this
        print(f'  Updating TSV files for {new_id}...')
        for file_path in export_folder.rglob('*.tsv'):
            if file_path.is_file():
                try:
                    # Read TSV file
                    df = pd.read_csv(file_path, sep='\t')
                    
                    # Replace old ID with new ID in all string columns
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).str.replace(old_id, new_id)
                    
                    # Remove acquisition time column for anonymization
                    if 'acq_time' in df.columns:
                        df.drop(columns=['acq_time'], inplace=True)
                    
                    # Save updated TSV
                    df.to_csv(file_path, sep='\t', index=False)
                except Exception as e:
                    print(f'  Warning: Could not update {file_path}: {e}')

    # Save participant renaming mapping
    participants_df[['participant_id', 'new_participant_id']].to_csv(
        PROJECT_FOLDER / 'data/participant_renaming.tsv', sep='\t', index=False
    )

    # Save the new participants.tsv file
    participants_df.drop(columns='participant_id', inplace=True)
    participants_df.rename(columns={'new_participant_id': 'participant_id'}, inplace=True)
    participants_df.to_csv(EXPORT_BIDS / 'participants.tsv', sep='\t', index=False)
    
    print(f"\nAnonymisation complete!")
    print(f"Exported to: {EXPORT_BIDS}")
    print(f"Renaming mapping saved to: {PROJECT_FOLDER / 'data/participant_renaming.tsv'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())