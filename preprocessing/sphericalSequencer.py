from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from typing import Optional, Tuple, Dict, List, Generator, Union
import pathlib
import pandas as pd
import sys
import os
import h5py
import argparse
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from utils.dataset_handler import Tractoinferno_handler

# White matter bundle labels from the TractInferno dataset (32 tracts)
ENCODED_TRACTS: Dict[str, int] = {'AF_L': 0, 'AF_R': 1, 'CC_Fr_1': 2, 'CC_Fr_2': 3, 'CC_Oc': 4, 'CC_Pa': 5, 'CC_Pr_Po': 6, 'CG_L': 7, 'CG_R': 8, 'FAT_L': 9, 'FAT_R': 10, 'FPT_L': 11, 'FPT_R': 12, 'FX_L': 13, 'FX_R': 14, 'IFOF_L': 15, 'IFOF_R': 16, 'ILF_L': 17, 'ILF_R': 18, 'MCP': 19, 'MdLF_L': 20, 'MdLF_R': 21, 'OR_ML_L': 22, 'OR_ML_R': 23, 'POPT_L': 24, 'POPT_R': 25, 'PYT_L': 26, 'PYT_R': 27, 'SLF_L': 28, 'SLF_R': 29, 'UF_L': 30, 'UF_R': 31}

class SphericalSequencer:
    """
    Converts tractogram streamlines from Cartesian to spherical coordinates,
    normalized per-subject using a brain-mask centroid and global radial range.
    Output is saved as padded HDF5 datasets grouped by tract ID.
    """

    def __init__(self, mri_path: str, csv_dataframe: pd.DataFrame, encoded_tracts: Optional[Dict[str, int]] = None):
        self.mri_path = mri_path

        self.encoded_tracts = encoded_tracts or ENCODED_TRACTS

        self.df = csv_dataframe

    def load_subject_data(self, subject: str) -> Tuple[Tuple[float, float, float], float, float]:
        """Load the center of mass and radial normalization parameters for a given subject from the normalization_parameters CSV"""
        matches = self.df[self.df['subject'] == subject]
        if matches.empty:
            raise ValueError(f"Subject '{subject}' not found in normalization CSV.")
        subject_data = matches.iloc[0]
        return (subject_data['com_x'], subject_data['com_y'], subject_data['com_z']), subject_data['r_min'], subject_data['r_max']

    @staticmethod
    def _name_tract(tract: pathlib.Path) -> str:
        """Extracts the tract name from the file path, assuming the format is 'subject__tractname.trk'"""
        return tract.name.split("__")[-1].split(".")[0]
    
    @staticmethod
    def _process_tract_streamlines(streamlines: Union[List[np.ndarray], ArraySequence], com: Tuple[float, float, float], r_min: float, r_max: float, epsilon: float = 1e-6) -> List[np.ndarray]:
        """Process all streamlines in a tract as a batch where possible."""
        results = []
        for streamline in streamlines:  # Variable length prevents full vstack
            centered = streamline - np.array(com)
            r = np.linalg.norm(centered, axis=1)
            r_norm = (r - r_min) / (r_max - r_min + epsilon)
            theta = np.arccos(np.clip(centered[:, 2] / (r + epsilon), -1, 1))
            phi = np.arctan2(centered[:, 1], centered[:, 0])
            results.append(np.stack((r_norm, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)), axis=1))
        return results

    def iter_processed_streamlines(self,
        tractograms: List[pathlib.Path],
        com: Tuple[float, float, float],
        r_min: float,
        r_max: float,
        ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        Lazily yield (processed_streamline, tract_id) pairs for all tractograms.
        Skips tracts not found in encoded_tracts.
        """
        for tract_path in tractograms:
            tract_name = self._name_tract(tract_path)
            tract_id = self.encoded_tracts.get(tract_name, -1)
            if tract_id == -1:
                print(f"Warning: unknown tract '{tract_name}', skipping.")
                continue

            sft = load_tractogram(str(tract_path), str(self.mri_path))
            if not isinstance(sft, StatefulTractogram):
                print(f"Warning: failed to load tractogram '{tract_path}', skipping.")
                continue

            processed = self._process_tract_streamlines(sft.streamlines, com, r_min, r_max)
            del sft  # Free tractogram memory before yielding
            
            for p in processed:
                yield p, tract_id
    
    def process_and_save_subject(
        self,
        tractograms: List[pathlib.Path],
        subject: str,
        output_path: str
    ):
        """
        Process subject and save to HDF5 file organized by tracts
        
        Args:
            tractograms: List of tractogram file paths
            subject: Subject identifier
            output_path: Path to save the output HDF5 file
        """
        # Load subject-specific normalization parameters
        com, r_min, r_max = self.load_subject_data(subject)

        # Organize streamlines by tract ID for saving
        tract_streamlines = defaultdict(list)
        for streamline, tract_id in self.iter_processed_streamlines(tractograms, com, r_min, r_max):
            tract_streamlines[tract_id].append(streamline)
        
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        # Save the processed streamlines to an HDF5 file
        with h5py.File(output_path, 'w') as f:
            f.attrs['subject'] = subject
            f.attrs['com'] = com
            f.attrs['r_min'] = r_min
            f.attrs['r_max'] = r_max
            
            # Save each tract's streamlines in a separate group
            for tract_id, streamlines in tract_streamlines.items():
                tract_group = f.create_group(f'tract_{tract_id}')
                tract_group.attrs['tract_id'] = tract_id
                tract_group.attrs['n_streamlines'] = len(streamlines)
                
                # Find max length in this tract
                max_len = max(s.shape[0] for s in streamlines)
                
                n_features = streamlines[0].shape[1]
                assert n_features == 5, f"Expected 5 spherical features, got {n_features}"

                # Pad variable-length streamlines to a uniform shape for HDF5 storage,
                # storing actual lengths separately to recover them at read time.
                padded = np.zeros((len(streamlines), max_len, n_features), dtype=np.float32)
                lengths = np.zeros(len(streamlines), dtype=np.int32)
                
                for i, s in enumerate(streamlines):
                    length = s.shape[0]
                    padded[i, :length] = s.astype(np.float32)
                    lengths[i] = length
                
                # Save padded streamlines and their actual lengths
                tract_group.create_dataset(
                    'streamlines',
                    data=padded,
                    chunks=(min(100, len(streamlines)), max_len, n_features)
                )
                tract_group.create_dataset('lengths', data=lengths)


def main(scope: str):
    dataset_handler = Tractoinferno_handler("/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives", scope=scope)
    csv_path = f"preprocessing/csvs/normalization_parameters_{scope}.csv"
    normalization_df = pd.read_csv(csv_path)
    
    for subject in dataset_handler.get_data():
        sequencer = SphericalSequencer(mri_path=subject["T1w"], encoded_tracts=ENCODED_TRACTS, csv_dataframe=normalization_df)
        sequencer.process_and_save_subject(subject["tracts"], subject["subject"], f"sequences/{scope}/" + subject["subject"] + ".hdf5")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate spherical coordinates for streamlines')
    parser.add_argument('--scope', type=str, default='testset', help='Scope of the dataset')
    args = parser.parse_args()
    main(args.scope)

# DUMMY_PATH = "/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1006/anat/sub-1006__T1w.nii.gz"
# DUMMY_STREAMLINE = pathlib.Path("/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives/testset/sub-1006/tractography/sub-1006__CC_Pa.trk")