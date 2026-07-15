import pathlib
import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from scipy.ndimage import center_of_mass
import sys
import os
import pandas as pd
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from utils.dataset_handler import Tractoinferno_handler

def compute_com(mri: nib.Nifti1Image) -> np.ndarray:
    """Compute center of mass of a Nifti image"""
    mri = nib.as_closest_canonical(mri) # Ensure the image is in canonical orientation
    mri_data = mri.get_fdata(dtype=np.float32)
    
    affine = mri.affine
    if affine is None:
        raise ValueError("Image has no affine transformation.")
    affine = np.asarray(affine, dtype=np.float64)

    # Compute center of mass in voxel space (only consider non-zero voxels)
    com_vox = center_of_mass(mri_data>0)
    if com_vox is None:
        raise ValueError("Could not compute center of mass: empty or invalid mask.")

    # Convert to RAS space using the affine transformation
    com_ras = np.dot(affine, np.append(com_vox, 1))[:3] 
        
    return com_ras

def process_subject(subject, dataset_handler, verbose):
    subject_data = dataset_handler.get_data_from_subject(subject)
    mri_path = subject_data["T1w"]
    mri = nib.load(mri_path)
    if not isinstance(mri, nib.Nifti1Image):
        raise TypeError(f"Expected Nifti1Image, got {type(mri)}")
    # Compute the center of mass of the brain mask for this subject
    com = compute_com(mri) 

    r_min, r_max = np.inf, -np.inf
    for tract_path in subject_data["tracts"]:
        sft = load_tractogram(str(tract_path), str(mri_path))
        if not isinstance(sft, StatefulTractogram):
            if verbose:
                print(f"Warning: failed to load tractogram '{tract_path}' for subject {subject}, skipping.")
            continue

        all_points = sft.streamlines.get_data()
        r = np.linalg.norm(all_points - com, axis=1)
        r_min = min(r_min, r.min())
        r_max = max(r_max, r.max())
        del sft

    return {
        'subject': subject_data["subject"],
        'com_x': com[0],
        'com_y': com[1],
        'com_z': com[2],
        'r_min': r_min,
        'r_max': r_max
    }

def main(scope: str, verbose: bool = False, 
        output_file: str = 'preprocessing/csvs/normalization_parameters.csv', 
        dataset_path: str = "/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives", 
        max_workers: int = 16):
    start_time = time.time() # This can be deleted
    
    dataset_handler = Tractoinferno_handler(dataset_path, scope=scope)
    subjects = dataset_handler.subjects
    if not subjects:
        raise ValueError(f"No subjects found for scope '{scope}'.")
    
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject, s, dataset_handler, verbose): s for s in subjects}
        for future in as_completed(futures):
            results.append(future.result())

    # Save to CSV
    df = pd.DataFrame(results)
    pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    df.to_csv(output_file, index=False)    

    end_time = time.time() # This can be deleted
    
    if verbose:
        print(f"\nSaved results for {len(results)} subjects to {output_file}")
        print(f"\nTotal time: {end_time - start_time:.2f} seconds") # This can be deleted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute normalization parameters for streamlines')
    parser.add_argument('--scope', type=str, default='testset', help='Scope of the dataset')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--dataset_path', type=str, default="/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives", help='Path to the dataset')
    parser.add_argument('--max_workers', type=int, default=16, help='Maximum number of worker processes')
    args = parser.parse_args()

    output_file = args.output or f'preprocessing/csvs/normalization_parameters_{args.scope}.csv'
    main(args.scope, args.verbose, output_file=output_file, dataset_path=args.dataset_path, max_workers=args.max_workers)
