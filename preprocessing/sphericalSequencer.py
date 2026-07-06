from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from typing import Optional, Tuple, Dict, List, Union
import pathlib
import pandas as pd
import sys
import os
import h5py
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

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

    def __init__(self, mri_path: str, encoded_tracts: Optional[Dict[str, int]] = None):
        self.mri_path = mri_path

        self.encoded_tracts = encoded_tracts or ENCODED_TRACTS

    @staticmethod
    def _name_tract(tract: pathlib.Path) -> str:
        """Extracts the tract name from the file path, assuming the format is 'subject__tractname.trk'"""
        return tract.name.split("__")[-1].split(".")[0]
    
    @staticmethod
    def _process_tract_streamlines(
        streamlines: Union[List[np.ndarray], ArraySequence],
        com: Tuple[float, float, float],
        r_min: float,
        r_max: float,
        epsilon: float = 1e-6,
    ) -> List[np.ndarray]:
        """
        Process all streamlines in a tract in one vectorized pass.

        This function flattens the whole tract's points into a single 
        (N, 3) array (using ArraySequence's internal contiguous storage), computes spherical coordinates once for all
        points, then splits the result back into per-streamline arrays
        using the original offsets/lengths.
        """
        if isinstance(streamlines, ArraySequence):
            data = streamlines._data
            offsets = streamlines._offsets
            lengths = streamlines._lengths
        else:
            raise ValueError("Expected streamlines to be an ArraySequence for vectorized processing.")

        com_arr = np.asarray(com, dtype=np.float64)
        centered = data - com_arr

        r = np.linalg.norm(centered, axis=1)
        r_norm = (r - r_min) / (r_max - r_min + epsilon)
        theta = np.arccos(np.clip(centered[:, 2] / (r + epsilon), -1, 1))
        phi = np.arctan2(centered[:, 1], centered[:, 0])

        features = np.stack(
            (r_norm, np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)),
            axis=1,
        ).astype(np.float32)

        return [features[o:o + l] for o, l in zip(offsets, lengths)]

    def process_and_save_subject(
        self,
        tractograms: List[pathlib.Path],
        subject: str,
        output_path: str, 
        norm_params: Optional[dict] = None
    ):
        """
        Process subject and save to HDF5 file organized by tracts
        
        Args:
            tractograms: List of tractogram file paths
            subject: Subject identifier
            output_path: Path to save the output HDF5 file
            norm_params: Normalization parameters for the subject
        """
        # Load subject-specific normalization parameters
        if norm_params is None:
            raise ValueError("Normalization parameters must be provided for processing.")
        com, r_min, r_max = norm_params["com"], norm_params["r_min"], norm_params["r_max"]

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        tmp_path = output_path + ".tmp"

        try:
            # Save the processed streamlines to an HDF5 file
            with h5py.File(tmp_path, 'w') as f:
                f.attrs['subject'] = subject
                f.attrs['com'] = com
                f.attrs['r_min'] = r_min
                f.attrs['r_max'] = r_max

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
                    del sft

                    n = len(processed)
                    if n == 0:
                        print(f"Warning: no streamlines found in tract '{tract_name}', skipping.")
                        continue
                    
                    lengths = np.array([s.shape[0] for s in processed], dtype=np.int32)
                    max_len = int(lengths.max())

                    n_features = processed[0].shape[1]
                    assert n_features == 5, f"Expected 5 spherical features, got {n_features}"

                    padded = np.zeros((n, max_len, n_features), dtype=np.float32)
                    for i, p in enumerate(processed):
                        padded[i, :lengths[i]] = p
                    del processed

                    group_name = f'tract_{tract_id}'
                    if group_name in f:
                        raise ValueError(
                            f"Duplicate tract_id {tract_id} ('{tract_name}') for subject '{subject}': "
                            f"a group '{group_name}' already exists in the output file. "
                            f"Check for duplicate or conflicting tract files among: {tractograms}."
                        )
                        
                    tract_group = f.create_group(f'tract_{tract_id}')
                    tract_group.attrs['tract_id'] = tract_id
                    tract_group.attrs['n_streamlines'] = n
                    
                    tract_group.create_dataset(
                        'streamlines',
                        data=padded,
                        chunks=(min(100, n), max_len, n_features),
                        compression='gzip',
                        compression_opts=4
                    )
                    tract_group.create_dataset('lengths', data=lengths)

                    del padded
            os.replace(tmp_path, output_path)  # Atomic move to final destination
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e


def _extract_subject_norm_params(df: pd.DataFrame, subject: str) -> dict:
        matches = df[df['subject'] == subject]
        if matches.empty:
            raise ValueError(f"Subject '{subject}' not found in normalization CSV.")
        row = matches.iloc[0]
        return {
            "com": (row['com_x'], row['com_y'], row['com_z']),
            "r_min": row['r_min'],
            "r_max": row['r_max'],
        }

def _process_one_subject(
        subject_dict: dict,
        encoded_tracts: Dict[str, int],
        norm_params: dict,
        scope: str,
        output_dir: str = "sequences"
    ) -> Tuple[str, Optional[str]]:
        """
        Worker function executed in a separate process for one subject.

        Returns (subject_id, error_message_or_None).
        """
        try:
            sequencer = SphericalSequencer(mri_path=subject_dict["T1w"], encoded_tracts=encoded_tracts)
            output_path = f"{output_dir}/{scope}/{subject_dict['subject']}.hdf5"
            sequencer.process_and_save_subject(subject_dict["tracts"], subject_dict["subject"], output_path, norm_params)
            return subject_dict["subject"], None
        except Exception as e:
            return subject_dict["subject"], str(e)
        
def main(scope: str, dataset_path: str, csv_path: Optional[str], output_dir: str, max_workers: Optional[int] = 4):
    dataset_handler = Tractoinferno_handler(dataset_path, scope=scope)
    csv_path = csv_path or f"preprocessing/csvs/normalization_parameters_{scope}.csv"
    normalization_df = pd.read_csv(csv_path)

    pathlib.Path(f"{output_dir}/{scope}").mkdir(parents=True, exist_ok=True)
    subjects = list(dataset_handler.get_data())
    max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for subject in subjects:
            norm_params = _extract_subject_norm_params(normalization_df, subject["subject"])
            futures[executor.submit(
                _process_one_subject, subject, ENCODED_TRACTS, norm_params, scope, output_dir
            )] = subject["subject"]

        for future in as_completed(futures):
            subject_id = futures[future]
            _, error = future.result()
            print(f"Error processing subject '{subject_id}': {error}" if error else f"Finished subject '{subject_id}'")

if __name__ == "__main__":
    import time 
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Generate spherical coordinates for streamlines')
    parser.add_argument('--scope', type=str, default='testset', help='Scope of the dataset')
    parser.add_argument('--dataset_path', type=str, default='/home/blancolote/TFM/Tractoinferno/ds003900-download/derivatives', help='Path to the dataset')
    parser.add_argument('--csv_path', type=str, default=None, help='Path to the CSV file with normalization parameters')
    parser.add_argument('--output_dir', type=str, default='sequences', help='Path to the output directory')
    parser.add_argument('--max_workers', type=int, default=27, help='Maximum number of worker processes to use')
    args = parser.parse_args()
    main(args.scope, args.dataset_path, args.csv_path, args.output_dir, args.max_workers)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")