from torch.utils.data import Dataset, Sampler, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional, Iterator, cast
import h5py
import numpy as np  

class StratifiedEpochSampler(Sampler[int]):
    """
    Stratified sampler that samples proportionally from EACH class EACH epoch.
    
    Guarantees that each class gets exactly sampling_percentage of its indexed 
    streamlines each epoch, maintaining class proportions.
    
    Call set_epoch(epoch) at the start of each epoch to get a new random subset.
    """
    
    def __init__(
        self,
        dataset: 'StreamlineDataset',
        sampling_percentage: float = 0.10,
        min_samples_per_class: int = 10,
        seed: int = 42,
        shuffle: bool = True,
        verbose: bool = False,
        expected_effective_percentage: Optional[float] = None,
        effective_percentage_tolerance: float = 1e-6
    ):
        """
        Args:
            dataset: StreamlineDataset with streamline_index containing 'tract_id'
            sampling_percentage: Percentage to sample from EACH class (0-1)
            min_samples_per_class: Minimum samples per class per epoch
            seed: Base random seed for reproducibility
            shuffle: Whether to shuffle the final indices
            verbose: Whether to print additional information
            expected_effective_percentage: Optional expected effective percentage of samples as the combination of sampling_percentage of the Sampler and the Dataset.
            effective_percentage_tolerance: Tolerance for checking effective percentage mismatch
        """
        self.dataset = dataset
        self.sampling_percentage = sampling_percentage
        self.min_samples_per_class = min_samples_per_class
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

        dataset_pct = getattr(dataset, "sampling_percentage", 1.0)
        self.effective_percentage = dataset_pct * sampling_percentage
        if expected_effective_percentage is not None:
            if abs(self.effective_percentage - expected_effective_percentage) > effective_percentage_tolerance:
                raise ValueError(
                    f"Effective per-epoch sampling rate mismatch: dataset.sampling_percentage "
                    f"({dataset_pct}) * sampler.sampling_percentage ({sampling_percentage}) = "
                    f"{self.effective_percentage:.6f}, but expected_effective_percentage="
                    f"{expected_effective_percentage}. Update one of the two percentages, or "
                    f"update expected_effective_percentage if this change is intentional."
                )
        
        # Build class-to-indices mapping using NumPy for efficiency
        # Works with both dict-based and NumPy structured array index
        self.class_indices: Dict[int, np.ndarray] = {}

        if dataset.streamline_index is None:
                raise ValueError("Dataset's streamline_index is None. Ensure the dataset is properly initialized.")
        
        # NumPy structured array version
        tract_ids = dataset.streamline_index['tract_id']
        tmp_indices: Dict[int, list] = {}
        for idx in range(len(dataset)):
            tract_id = int(tract_ids[idx])
            tmp_indices.setdefault(tract_id, []).append(idx)
        # store as numpy arrays for faster rng.choice
        self.class_indices = {tid: np.asarray(v, dtype=np.int64) for tid, v in tmp_indices.items()}
        
        fully_sampled_tracts = getattr(dataset, "fully_sampled_tracts", set())
        
        # Calculate samples per class
        self.samples_per_class = {}
        total_samples = 0
        for tract_id, indices in self.class_indices.items():
            n_available = len(indices)

            if tract_id in fully_sampled_tracts:
                # This class was already fully retained during dataset indexing,
                # so keep taking 100% of it every epoch too.
                n_to_sample = n_available
            else:
                n_percentage = int(n_available * sampling_percentage)
                n_to_sample = max(self.min_samples_per_class, n_percentage)
                n_to_sample = min(n_to_sample, n_available)

            self.samples_per_class[tract_id] = n_to_sample
            total_samples += n_to_sample

        self.n_samples = total_samples
        
        if verbose:
            print(f"StratifiedEpochSampler: {self.n_samples} samples per epoch "
                f"({self.sampling_percentage*100:.1f}% per class; classes already fully "
                f"retained at indexing time keep 100%: {sorted(fully_sampled_tracts)})")

            for tract_id in sorted(self.samples_per_class.keys()):
                print(f"  Class {tract_id}: {self.samples_per_class[tract_id]} / {len(self.class_indices[tract_id])}")
    
    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch number to generate a different random subset.
        
        Call this at the start of each epoch before iterating.
        """
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[int]:
        """Generate stratified random subset of indices for this epoch."""
        rng = np.random.default_rng(self.seed + self.epoch)
        
        all_indices = []
        
        # Sample from each class proportionally
        for tract_id, indices in self.class_indices.items():
            n_to_sample = self.samples_per_class[tract_id]
            sampled = rng.choice(indices, n_to_sample, replace=False)
            all_indices.extend(sampled.tolist())
        
        # Shuffle all indices together
        if self.shuffle:
            rng.shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return self.n_samples


class StreamlineDataset(Dataset):
    """
    Dataset that indexes a subset of streamlines from the HDF5 files.
    
    Each __getitem__ returns a single streamline with its label (tract_id).
    Use with StratifiedEpochSampler to sample different subsets each epoch.
    
    Memory-optimized: Uses NumPy structured arrays instead of Python dicts
    to reduce RAM usage by ~90%.
    """
    
    # Structured array dtype for memory-efficient index
    INDEX_DTYPE = np.dtype([
        ('file_id', np.int16),        # Index into file_paths_lookup
        ('streamline_idx', np.int32), # Index within the group
        ('length', np.int16),         # Streamline length
        ('tract_id', np.int8),        # Class label (0-31)
    ])
    
    def __init__(
        self, 
        hdf5_file_paths: List[str],
        sampling_percentage: float = 0.10,
        min_streamlines: int = 50,
        max_streamlines_per_tract: Optional[int] = None,
        threshold: int = 1000,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Args:
            hdf5_file_paths: List of HDF5 file paths
            sampling_percentage: Percentage of streamlines to index from each tract (0-1)
            min_streamlines: Minimum number of streamlines per tract
            max_streamlines_per_tract: Maximum number of streamlines per tract (None = no limit)
            seed: Random seed for reproducibility of the initial sampling
            cache_dir: Optional directory to cache the index (for faster startup)
        """
        self.file_paths = hdf5_file_paths
        self.sampling_percentage = sampling_percentage
        self.min_streamlines = min_streamlines
        self.max_streamlines_per_tract = max_streamlines_per_tract
        self.seed = seed
        self.cache_dir = cache_dir
        self.threshold = threshold
        self.verbose = verbose
        
        # File path lookup table (maps file_id -> file_path)
        self.file_paths_lookup = list(hdf5_file_paths)
        self.file_path_to_id = {path: i for i, path in enumerate(self.file_paths_lookup)}
        
        # HDF5 file handle cache (opened lazily, per-worker for multiprocessing safety)
        self._file_handles = {}
        self._worker_id = None  # Track which worker owns these handles
        
        # Build index of sampled streamlines across all files
        # Stored as NumPy structured array for memory efficiency
        self.streamline_index = None  # Will be np.ndarray with INDEX_DTYPE
        self._build_index()
    
    @staticmethod
    def _read_int_attr(group: h5py.Group, key: str) -> int:
        val = group.attrs[key]
        if isinstance(val, h5py.Empty):
            raise ValueError(f"Attribute '{key}' is empty in group '{group.name}'.")
        return int(np.asarray(val))
    
    def _build_index(self):
        """Build an index of sampled streamlines in the dataset."""
        if self.verbose:
            print(f"Building streamline index (sampling {self.sampling_percentage*100:.1f}% per tract)...")

        rng = np.random.default_rng(self.seed)

        if self.verbose:
            print(f"  Fixed full-sample threshold: {self.threshold}")

        index_entries = []

        n_available_per_tract: Dict[int, int] = {}
        n_sampled_per_tract: Dict[int, int] = {}

        for file_path in self.file_paths:
            file_id = self.file_path_to_id[file_path]
            with h5py.File(file_path, 'r') as f:
                for group_name in f.keys():
                    if not group_name.startswith('tract_'):
                        continue

                    tract_group = cast(h5py.Group, f[group_name])
                    tract_id = self._read_int_attr(tract_group, 'tract_id')
                    group_id = int(group_name.split('_')[1])
                    assert group_id == tract_id, (
                        f"Invariant violated: group '{group_name}' has group_id={group_id} but "
                        f"tract_id={tract_id}. Cannot safely drop group_id from the index."
                    )

                    n_available = self._read_int_attr(tract_group, 'n_streamlines')
                    lengths_ds = cast(h5py.Dataset, tract_group['lengths'])
                    lengths = np.asarray(lengths_ds[:])

                    # Calculate how many streamlines to sample
                    if n_available < self.threshold:
                        n_to_sample = n_available
                    else:
                        n_percentage = int(n_available * self.sampling_percentage)
                        n_to_sample = max(self.min_streamlines, n_percentage)

                    if self.max_streamlines_per_tract is not None:
                        n_to_sample = min(n_to_sample, self.max_streamlines_per_tract)

                    n_to_sample = min(n_to_sample, n_available)

                    n_available_per_tract[tract_id] = n_available_per_tract.get(tract_id, 0) + n_available
                    n_sampled_per_tract[tract_id] = n_sampled_per_tract.get(tract_id, 0) + n_to_sample

                    # Sample indices using seeded RNG
                    if n_to_sample < n_available:
                        sampled_indices = np.sort(rng.choice(n_available, n_to_sample, replace=False))
                    else:
                        sampled_indices = np.arange(n_available)

                    for idx in sampled_indices:
                        index_entries.append((
                            file_id,
                            int(idx),
                            int(lengths[idx]),
                            int(tract_id)
                        ))

        # Convert to NumPy structured array (huge memory savings)
        self.streamline_index = np.array(index_entries, dtype=self.INDEX_DTYPE)

        self.fully_sampled_tracts = {
            tid for tid in n_available_per_tract
            if n_sampled_per_tract[tid] == n_available_per_tract[tid]
        }

        if self.verbose:
            print(f"Total streamlines indexed: {len(self.streamline_index)}")

            memory_mb = self.streamline_index.nbytes / (1024 * 1024)
            print(f"  Index memory: {memory_mb:.1f} MB")

            unique_classes, counts = np.unique(self.streamline_index['tract_id'], return_counts=True)
            print(f"  Classes: {len(unique_classes)}")
            print(f"  Streamlines per class: min={counts.min()}, max={counts.max()}")
    
    def __len__(self):
        if self.streamline_index is None:
            raise RuntimeError("Index not built yet. Call _build_index() first.")
        return len(self.streamline_index)
    
    def _get_current_worker_id(self) -> int:
        """Get the current DataLoader worker ID, or -1 for main process."""
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info is not None else -1
    
    def _get_file_handle(self, file_id: int):
        """Get a cached file handle, opening it if necessary.
        
        Handles are cached per-worker to be multiprocessing safe.
        """
        current_worker = self._get_current_worker_id()
        
        # If worker changed (shouldn't happen, but be safe), clear cache
        if self._worker_id is not None and self._worker_id != current_worker:
            self._close_file_handles()
        self._worker_id = current_worker
        
        # Open file if not cached
        if file_id not in self._file_handles:
            file_path = self.file_paths_lookup[file_id]
            self._file_handles[file_id] = h5py.File(file_path, 'r')
        
        return self._file_handles[file_id]
    
    def _close_file_handles(self):
        """Close all cached file handles."""
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass
        self._file_handles = {}
    
    def __del__(self):
        """Cleanup file handles on deletion."""
        self._close_file_handles()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single streamline.
        
        Returns:
            streamline: Tensor of shape (seq_len, 5)
            length: Actual length of the streamline
            tract_id: Label for the streamline
        """
        if self.streamline_index is None:
            raise RuntimeError("Index not built yet. Call _build_index() first.")
        item = self.streamline_index[idx]
        
        # Get cached file handle (much faster than opening/closing each time)
        file_handle = self._get_file_handle(int(item['file_id']))
        group_name = f"tract_{item['tract_id']}"
        
        tract_group = file_handle[group_name]
        
        # Get the single streamline
        streamline = tract_group['streamlines'][item['streamline_idx']]  # (max_len, 5)
        length = int(item['length'])
        
        # Trim to actual length (remove padding)
        streamline = streamline[:length]
        
        return (
            torch.from_numpy(streamline.astype(np.float32)),
            length,
            int(item['tract_id'])
        )
    
    def get_tract_id(self, idx: int) -> int:
        """Get tract_id for an index without loading the streamline (fast)."""
        if self.streamline_index is None:
            raise RuntimeError("Index not built yet. Call _build_index() first.")
        return int(self.streamline_index[idx]['tract_id'])


def streamline_collate_fn(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function that pads streamlines to the same length within a batch.
    
    Args:
        batch: List of (streamline, length, tract_id) tuples
    
    Returns:
        streamlines: Padded tensor of shape (batch_size, max_seq_len, 5)
        lengths: Tensor of shape (batch_size,) with actual lengths
        labels: Tensor of shape (batch_size,) with tract_ids
    """
    streamlines = [item[0] for item in batch]
    lengths = torch.tensor([item[1] for item in batch], dtype=torch.long)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    # Pad streamlines to max length in this batch
    # pad_sequence expects (seq_len, features) tensors and pads along dim 0
    padded_streamlines = pad_sequence(streamlines, batch_first=True, padding_value=0.0)
    
    return padded_streamlines, lengths, labels