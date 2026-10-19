
"""
Fast streamline dataset / sampler / batch loader for the spherical-sequence HDF5 files.
 
Design
------
* The indexed subset is read from HDF5 ONCE (sequential slab reads, vectorized un-padding)
  and kept as a flat ``(total_points, C)`` array plus an ``offsets`` array.
* Optionally the flat arrays are cached on disk (``cache_dir``) and opened with ``mmap`` so
  repeated runs start instantly and DataLoader workers share the OS page cache.
* ``__getitem__`` accepts a LIST of indices and returns a finished, padded batch, so the
  DataLoader runs with ``batch_size=None`` and no per-sample Python / collate overhead.
* ``sin_theta`` (exactly sqrt(1 - cos_theta^2) since theta is in [0, pi]) can be dropped from
  storage (``drop_sin_theta=True``) and is rebuilt when a batch is assembled: -20% memory.
 
Usage
-----
    ds = StreamlineDataset(paths, sampling_percentage=0.10, cache_dir="cache_streamlines")
    sampler = StratifiedEpochSampler(ds, sampling_percentage=0.10)
    loader, batch_sampler = make_dataloader(ds, sampler, batch_size=256, num_workers=2,
                                            bucket_by_length=True)
    for epoch in range(n_epochs):
        batch_sampler.set_epoch(epoch)          # also forwards to the stratified sampler
        for x, lengths, labels in loader:       # x: (B, T, 5) float32, lengths/labels: int64
            ...
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union, cast

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

N_FEATURES = 5                       # (r_norm, cos_theta, sin_theta, cos_phi, sin_phi)
_SLAB_BYTES = 64 * 1024 * 1024       # max bytes of padded HDF5 data read at once
_CACHE_VERSION = 1


# --------------------------------------------------------------------------------------
# Sampler: stratified per-epoch sampling
# --------------------------------------------------------------------------------------
class StratifiedEpochSampler(Sampler[int]):
    """
    Stratified sampler that samples proportionally from EACH class EACH epoch.

    Guarantees that each class gets exactly ``sampling_percentage`` of its indexed
    streamlines each epoch (classes fully retained at indexing time keep 100%).
    Call ``set_epoch(epoch)`` at the start of each epoch to get a new random subset.
    """

    def __init__(
        self,
        dataset: "StreamlineDataset",
        sampling_percentage: float = 0.10,
        min_samples_per_class: int = 10,
        seed: int = 42,
        shuffle: bool = True,
        verbose: bool = False,
        expected_effective_percentage: Optional[float] = None,
        effective_percentage_tolerance: float = 1e-6,
    ):
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

        if dataset.streamline_index is None:
            raise ValueError("Dataset's streamline_index is None. Ensure the dataset is properly initialized.")

        # Vectorized class -> indices mapping (no Python loop over streamlines)
        tract_ids = dataset.streamline_index["tract_id"].astype(np.int64)
        order = np.argsort(tract_ids, kind="stable")
        classes, starts = np.unique(tract_ids[order], return_index=True)
        splits = np.split(order.astype(np.int64), starts[1:])
        self.class_indices: Dict[int, np.ndarray] = {int(c): s for c, s in zip(classes, splits)}

        fully_sampled_tracts = getattr(dataset, "fully_sampled_tracts", set())

        self.samples_per_class: Dict[int, int] = {}
        total_samples = 0
        for tract_id, indices in self.class_indices.items():
            n_available = len(indices)
            if tract_id in fully_sampled_tracts:
                n_to_sample = n_available
            else:
                n_to_sample = max(self.min_samples_per_class, int(n_available * sampling_percentage))
                n_to_sample = min(n_to_sample, n_available)
            self.samples_per_class[tract_id] = n_to_sample
            total_samples += n_to_sample
        self.n_samples = total_samples

        if verbose:
            print(f"StratifiedEpochSampler: {self.n_samples} samples per epoch "
                  f"({self.sampling_percentage * 100:.1f}% per class; classes already fully "
                  f"retained at indexing time keep 100%: {sorted(fully_sampled_tracts)})")
            for tract_id in sorted(self.samples_per_class):
                print(f"  Class {tract_id}: {self.samples_per_class[tract_id]} / {len(self.class_indices[tract_id])}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        parts = [
            rng.choice(indices, self.samples_per_class[tid], replace=False)
            for tid, indices in self.class_indices.items()
        ]
        all_indices = np.concatenate(parts)
        if self.shuffle:
            all_indices = rng.permutation(all_indices)
        return iter(all_indices.tolist())

    def __len__(self) -> int:
        return self.n_samples


# --------------------------------------------------------------------------------------
# Batch sampler: batching + optional length bucketing
# --------------------------------------------------------------------------------------
class StreamlineBatchSampler(Sampler[List[int]]):
    """
    Groups the indices produced by ``sampler`` into batches.

    If ``bucket_by_length`` is True, indices are taken in mega-batches of
    ``batch_size * mega_factor``, sorted by streamline length inside each mega-batch,
    split into batches and the batches are shuffled. This greatly reduces padding while
    keeping batches reasonably random.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        lengths: np.ndarray,
        batch_size: int,
        drop_last: bool = False,
        bucket_by_length: bool = False,
        mega_factor: int = 50,
        seed: int = 42,
    ):
        self.sampler = sampler
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.bucket = bucket_by_length
        self.mega_factor = max(1, mega_factor)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

    def __iter__(self) -> Iterator[List[int]]:
        idx = np.fromiter(self.sampler, dtype=np.int64, count=len(self.sampler))  # type: ignore[arg-type]
        bs = self.batch_size

        if not self.bucket:
            batches = [idx[i:i + bs] for i in range(0, len(idx), bs)]
        else:
            mb = bs * self.mega_factor
            batches = []
            for s in range(0, len(idx), mb):
                chunk = idx[s:s + mb]
                chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
                batches.extend(chunk[i:i + bs] for i in range(0, len(chunk), bs))

        if self.drop_last:
            batches = [b for b in batches if len(b) == bs]

        if self.bucket:
            rng = np.random.default_rng(self.seed + self.epoch + 1_000_003)
            batches = [batches[i] for i in rng.permutation(len(batches))]

        for b in batches:
            yield b.tolist()

    def __len__(self) -> int:
        n, bs = len(self.sampler), self.batch_size  # type: ignore[arg-type]
        if not self.bucket:
            return n // bs if self.drop_last else -(-n // bs)
        mb = bs * self.mega_factor
        full, rem = divmod(n, mb)
        tail = rem // bs if self.drop_last else -(-rem // bs)
        return full * self.mega_factor + tail


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class StreamlineDataset(Dataset):
    """
    Dataset holding a seeded subset of streamlines from the HDF5 files, un-padded and
    stored as a flat array + offsets (in RAM, or memory-mapped from ``cache_dir``).

    ``ds[list_of_ids]`` returns a padded batch ``(x, lengths, labels)``;
    ``ds[int]`` returns a single ``(streamline, length, label)`` for compatibility.
    """

    INDEX_DTYPE = np.dtype([
        ("file_id", np.int16),         # index into file_paths_lookup
        ("streamline_idx", np.int32),  # index within the HDF5 group
        ("length", np.int32),          # real streamline length
        ("tract_id", np.int8),         # class label (0-31)
    ])

    def __init__(
        self,
        hdf5_file_paths: List[str],
        sampling_percentage: float = 0.10,
        min_streamlines: int = 50,
        max_streamlines: Optional[int] = None,
        threshold: int = 1000,
        seed: int = 42,
        verbose: bool = False,
        cache_dir: Optional[str] = None,
        store_dtype: Union[type, np.dtype, str] = np.float32,
        drop_sin_theta: bool = True,
    ):
        """
        Args:
            hdf5_file_paths: HDF5 files (one per subject).
            sampling_percentage: fraction of streamlines indexed from each tract (0-1).
            min_streamlines: minimum streamlines per tract (per file) when sampling.
            max_streamlines: maximum streamlines per tract (per file), None = no limit.
            threshold: groups with fewer streamlines than this are kept entirely.
            seed: seed for the initial (fixed) subset selection.
            cache_dir: if given, the flat arrays are cached here and memory-mapped.
            store_dtype: storage dtype (np.float16 halves memory; see note on drop_sin_theta).
            drop_sin_theta: don't store sin_theta, rebuild it per batch. Exact with float32;
                with float16 storage, error near the poles (cos_theta ~ +-1) can reach ~0.03,
                so use float16 only together with drop_sin_theta=False.
        """
        self.file_paths = list(hdf5_file_paths)
        self.sampling_percentage = sampling_percentage
        self.min_streamlines = min_streamlines
        self.max_streamlines = max_streamlines
        self.seed = seed
        self.threshold = threshold
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.store_dtype = np.dtype(store_dtype)
        self.drop_sin_theta = drop_sin_theta

        self.file_paths_lookup = list(self.file_paths)
        self.file_path_to_id = {p: i for i, p in enumerate(self.file_paths_lookup)}
        # columns of the 5-feature HDF5 data that are stored: drop sin_theta (column 2) if asked
        self._cols = [0, 1, 3, 4] if drop_sin_theta else [0, 1, 2, 3, 4]

        self.streamline_index: Optional[np.ndarray] = None
        self.fully_sampled_tracts: set = set()
        self.data: np.ndarray = np.empty((0, len(self._cols)), dtype=self.store_dtype)
        self.offsets: np.ndarray = np.zeros(1, dtype=np.int64)
        self._cache_path: Optional[Path] = None

        if not self._try_load_cache():
            self._build_index()
            self._materialize()
            self._save_cache()

        assert self.streamline_index is not None
        self.lengths = self.streamline_index["length"].astype(np.int64)
        self.labels = self.streamline_index["tract_id"].astype(np.int64)

    # ---------------------------------------------------------------- index construction
    @staticmethod
    def _read_int_attr(group: h5py.Group, key: str) -> int:
        val = group.attrs[key]
        if isinstance(val, h5py.Empty):
            raise ValueError(f"Attribute '{key}' is empty in group '{group.name}'.")
        return int(np.asarray(val))

    def _build_index(self) -> None:
        if self.verbose:
            print(f"Building streamline index (sampling {self.sampling_percentage * 100:.1f}% per tract)...")
            print(f"  Fixed full-sample threshold: {self.threshold}")

        rng = np.random.default_rng(self.seed)
        chunks: List[np.ndarray] = []
        n_available_per_tract: Dict[int, int] = {}
        n_sampled_per_tract: Dict[int, int] = {}

        for file_path in self.file_paths:
            file_id = self.file_path_to_id[file_path]
            with h5py.File(file_path, "r") as f:
                for group_name in f.keys():
                    if not group_name.startswith("tract_"):
                        continue

                    tract_group = cast(h5py.Group, f[group_name])
                    tract_id = self._read_int_attr(tract_group, "tract_id")
                    group_id = int(group_name.split("_")[1])
                    assert group_id == tract_id, (
                        f"Invariant violated: group '{group_name}' has group_id={group_id} but "
                        f"tract_id={tract_id}."
                    )

                    n_available = self._read_int_attr(tract_group, "n_streamlines")
                    lengths = np.asarray(cast(h5py.Dataset, tract_group["lengths"])[:])

                    if n_available < self.threshold:
                        n_to_sample = n_available
                    else:
                        n_to_sample = max(self.min_streamlines, int(n_available * self.sampling_percentage))
                    if self.max_streamlines is not None:
                        n_to_sample = min(n_to_sample, self.max_streamlines)
                    n_to_sample = min(n_to_sample, n_available)

                    n_available_per_tract[tract_id] = n_available_per_tract.get(tract_id, 0) + n_available
                    n_sampled_per_tract[tract_id] = n_sampled_per_tract.get(tract_id, 0) + n_to_sample

                    if n_to_sample < n_available:
                        sel = np.sort(rng.choice(n_available, n_to_sample, replace=False))
                    else:
                        sel = np.arange(n_available)

                    entry = np.empty(len(sel), dtype=self.INDEX_DTYPE)
                    entry["file_id"] = file_id
                    entry["streamline_idx"] = sel
                    entry["length"] = lengths[sel]
                    entry["tract_id"] = tract_id
                    chunks.append(entry)

        if not chunks:
            raise ValueError("No streamlines found in the given HDF5 files.")
        self.streamline_index = np.concatenate(chunks)  # grouped by (file, tract), ascending inside

        self.fully_sampled_tracts = {
            tid for tid in n_available_per_tract if n_sampled_per_tract[tid] == n_available_per_tract[tid]
        }

        if self.verbose:
            print(f"Total streamlines indexed: {len(self.streamline_index)}")
            print(f"  Index memory: {self.streamline_index.nbytes / (1024 * 1024):.1f} MB")
            classes, counts = np.unique(self.streamline_index["tract_id"], return_counts=True)
            print(f"  Classes: {len(classes)}")
            print(f"  Streamlines per class: min={counts.min()}, max={counts.max()}")

    # ---------------------------------------------------------------- materialization
    def _materialize(self) -> None:
        """Read every selected streamline once (slab reads), drop padding, store flat."""
        idx = self.streamline_index
        assert idx is not None
        n = len(idx)
        lengths = idx["length"].astype(np.int64)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        data = np.empty((int(offsets[-1]), len(self._cols)), dtype=self.store_dtype)

        # The index is built group by group, so (file, tract) runs are contiguous
        # and streamline_idx is ascending inside each run.
        keys = idx["file_id"].astype(np.int64) * 256 + idx["tract_id"].astype(np.int64)
        starts = np.concatenate(([0], np.flatnonzero(np.diff(keys)) + 1))
        ends = np.append(starts[1:], n)
        assert len(np.unique(keys[starts])) == len(starts), "index groups are not contiguous"

        f: Optional[h5py.File] = None
        current_fid = -1
        try:
            for a, b in zip(starts, ends):
                fid = int(idx["file_id"][a])
                tid = int(idx["tract_id"][a])
                if fid != current_fid:
                    if f is not None:
                        f.close()
                    f = h5py.File(self.file_paths_lookup[fid], "r")
                    current_fid = fid

                ds = cast(h5py.Dataset, f[f"tract_{tid}/streamlines"])
                sel = idx["streamline_idx"][a:b].astype(np.int64)
                assert np.all(np.diff(sel) > 0), "streamline indices must be strictly ascending in a group"

                max_len, n_feat = ds.shape[1], ds.shape[2]
                if n_feat != N_FEATURES:
                    raise ValueError(f"Expected {N_FEATURES} features in tract_{tid}, got {n_feat}.")
                chunk0 = ds.chunks[0] if ds.chunks else 1
                row_bytes = max_len * n_feat * ds.dtype.itemsize
                slab = max(chunk0, (_SLAB_BYTES // row_bytes) // chunk0 * chunk0)
                col_range = np.arange(max_len)[None, :]

                for s0 in range(0, ds.shape[0], slab):
                    lo, hi = np.searchsorted(sel, [s0, s0 + slab])
                    if lo == hi:
                        continue
                    block = ds[s0:s0 + slab]                   # contiguous read, each chunk decompressed once
                    rows = block[sel[lo:hi] - s0][..., self._cols]
                    mask = col_range < lengths[a + lo:a + hi][:, None]
                    data[offsets[a + lo]:offsets[a + hi]] = rows[mask]   # un-pad, row-major order
        finally:
            if f is not None:
                f.close()

        self.data, self.offsets = data, offsets

    # ---------------------------------------------------------------- disk cache
    def _cache_key(self) -> str:
        sig = []
        for p in self.file_paths:
            st = os.stat(p)
            sig.append((str(p), st.st_size, st.st_mtime_ns))
        payload = json.dumps({
            "v": _CACHE_VERSION, "files": sig,
            "sp": self.sampling_percentage, "minS": self.min_streamlines,
            "maxS": self.max_streamlines, "thr": self.threshold, "seed": self.seed,
            "dtype": self.store_dtype.name, "drop_sin": self.drop_sin_theta,
        }, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _open_cache(self, d: Path) -> None:
        self.streamline_index = np.load(d / "index.npy")
        self.offsets = np.load(d / "offsets.npy")
        self.data = np.load(d / "data.npy", mmap_mode="r")
        with open(d / "meta.json") as fh:
            self.fully_sampled_tracts = set(json.load(fh)["fully_sampled_tracts"])
        self._cache_path = d

    def _try_load_cache(self) -> bool:
        if self.cache_dir is None:
            return False
        d = Path(self.cache_dir) / self._cache_key()
        if not (d / "meta.json").exists():
            return False
        self._open_cache(d)
        if self.verbose:
            print(f"Loaded cached streamlines from {d} ({len(self.streamline_index)} streamlines)")  # type: ignore[arg-type]
        return True
 
    def _save_cache(self) -> None:
        if self.cache_dir is None:
            return
        final = Path(self.cache_dir) / self._cache_key()
        tmp = final.with_name(final.name + f".tmp{os.getpid()}")
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            np.save(tmp / "data.npy", self.data)
            np.save(tmp / "offsets.npy", self.offsets)
            np.save(tmp / "index.npy", self.streamline_index)
            with open(tmp / "meta.json", "w") as fh:
                json.dump({"fully_sampled_tracts": sorted(int(t) for t in self.fully_sampled_tracts)}, fh)
            os.replace(tmp, final)            # atomic directory move
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)   # another process won the race, or disk error
        if (final / "meta.json").exists():
            self._open_cache(final)           # swap the private RAM copy for a shared mmap

    # With 'spawn' workers, don't pickle the (possibly huge) arrays: re-open the mmap instead.
    def __getstate__(self):
        state = self.__dict__.copy()
        if self._cache_path is not None:
            state["data"] = None
            state["offsets"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._cache_path is not None:
            self.data = np.load(self._cache_path / "data.npy", mmap_mode="r")
            self.offsets = np.load(self._cache_path / "offsets.npy")

    # ---------------------------------------------------------------- access
    def __len__(self) -> int:
        if self.streamline_index is None:
            raise RuntimeError("Index not built yet.")
        return len(self.streamline_index)

    def get_batch(self, ids) -> tuple:
        """Vectorized gather of a whole padded batch -> (x[B,T,5] f32, lengths[B], labels[B])."""
        ids = np.asarray(ids, dtype=np.int64)
        L = self.lengths[ids]
        T = int(L.max())
        ar = np.arange(T)
        mask = ar[None, :] < L[:, None]
        pos = self.offsets[ids][:, None] + ar[None, :]
        rows = np.asarray(self.data[pos[mask]], dtype=np.float32)      # (sum(L), C)

        out = np.zeros((len(ids), T, N_FEATURES), dtype=np.float32)
        if self.drop_sin_theta:
            flat = np.empty((rows.shape[0], N_FEATURES), dtype=np.float32)
            flat[:, 0] = rows[:, 0]
            flat[:, 1] = rows[:, 1]
            flat[:, 2] = np.sqrt(np.clip(1.0 - rows[:, 1] ** 2, 0.0, 1.0))
            flat[:, 3] = rows[:, 2]
            flat[:, 4] = rows[:, 3]
            out[mask] = flat
        else:
            out[mask] = rows
        return torch.from_numpy(out), torch.from_numpy(L), torch.from_numpy(self.labels[ids])

    def __getitem__(self, ids):
        if isinstance(ids, (int, np.integer)):          # single sample (compatibility path)
            x, L, y = self.get_batch([int(ids)])
            n = int(L[0])
            return x[0, :n], n, int(y[0])
        return self.get_batch(ids)

    def get_tract_id(self, idx: int) -> int:
        return int(self.labels[idx])


# --------------------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------------------
def make_dataloader(
    dataset: StreamlineDataset,
    sampler: Sampler[int],
    batch_size: int = 64,
    num_workers: int = 2,
    bucket_by_length: bool = False,
    mega_factor: int = 50,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
    seed: int = 42,
):
    """
    Returns ``(loader, batch_sampler)``. Call ``batch_sampler.set_epoch(epoch)`` each epoch.
    The dataset already returns finished batches, so the DataLoader uses ``batch_size=None``.
    """
    batch_sampler = StreamlineBatchSampler(
        sampler, dataset.lengths, batch_size, drop_last=drop_last,
        bucket_by_length=bucket_by_length, mega_factor=mega_factor, seed=seed,
    )
    kwargs = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    loader = DataLoader(
        dataset,
        batch_size=None,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        **kwargs,
    )
    return loader, batch_sampler