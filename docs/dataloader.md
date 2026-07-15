# Streamline Dataset & Stratified Sampling — Documentation

## Overview

This module provides a memory-efficient PyTorch data pipeline for training on
large-scale tractography streamline data stored in HDF5 files. It consists of
two cooperating components:

1. **`StreamlineDataset`** — indexes and lazily loads individual streamlines.
2. **`StratifiedEpochSampler`** — samples a class-balanced subset each epoch.

A `streamline_collate_fn` is also included to pad variable-length streamlines
within a batch for use with a PyTorch `DataLoader`.

---

## `StreamlineDataset`

### Purpose

Rather than loading all streamline data into RAM, this class builds a lightweight
**index** at initialization time, recording *where* each sampled streamline lives
(file, group, position) without loading the actual coordinate data. Real streamline
arrays are only read from disk on-demand inside `__getitem__`, during training.

### Memory-Efficient Index (`INDEX_DTYPE`)

The index is stored as a NumPy **structured array** instead of a list of Python
dicts/tuples:

| Field | Type | Meaning |
|---|---|---|
| `file_id` | `int16` | Index into `file_paths_lookup` |
| `group_id` | `int16` | Tract group number (0–31) |
| `streamline_idx` | `int32` | Position within the HDF5 group |
| `length` | `int16` | Actual (unpadded) streamline length |
| `tract_id` | `int8` | Class label (0–31) |

Packing fields into fixed-width integers reduces memory usage by roughly 90%
compared to Python dicts, which matters when indexing millions of streamlines.

### `__init__`

Stores configuration (`sampling_percentage`, `min_streamlines`,
`max_streamlines_per_tract`, `full_sample_threshold`, `seed`), builds
`file_path_to_id` (a lookup so file paths can be stored as compact integers
rather than strings), initializes an empty `_file_handles` cache for lazy file
opening, and immediately calls `_build_index()`.

### `_build_index()`: Two-Pass Sampling Strategy

**Pass 1 — Metadata scan.** Iterates every HDF5 file's tract groups, reading only
cheap `.attrs` values (`tract_id`, `n_streamlines`) via `_read_int_attr`. No
streamline coordinate data is loaded yet.

**Adaptive threshold.** Computed as 5% of the largest tract's size (capped by
`full_sample_threshold`). Any tract smaller than this threshold is **fully
included** (100% sampled) instead of being subsampled — this protects small or
rare classes from being reduced to just a handful of examples.

**Pass 2 — Sampling.** Revisits each tract, reads the `lengths` dataset, and
decides how many streamlines to sample using:

- `sampling_percentage` — base fraction to sample from large tracts.
- `min_streamlines` — floor, so no tract contributes too few samples.
- `max_streamlines_per_tract` — optional ceiling, to cap dominant classes.

Indices are drawn via a seeded `np.random.default_rng`, without replacement, for
reproducibility. Each sampled streamline becomes one row in the final structured
array `self.streamline_index`.

### `_read_int_attr` (static helper)

Reads a scalar integer attribute from an HDF5 group, raising `ValueError` if the
attribute is `h5py.Empty` (a null/unset attribute) rather than silently failing
or producing a type-checking error.

### File Handle Caching (multiprocessing safety)

HDF5 file handles cannot safely be shared across PyTorch `DataLoader` worker
processes. `StreamlineDataset` solves this with per-worker lazy caching:

- `_get_current_worker_id()` asks PyTorch which worker (if any) is active.
- `_get_file_handle()` opens each file once per worker and reuses the handle,
  avoiding repeated open/close overhead on every `__getitem__` call.
- If the worker ID unexpectedly changes, stale handles are closed first via
  `_close_file_handles()`.
- `__del__` closes all handles when the dataset object is garbage-collected,
  preventing file descriptor leaks.

### `__getitem__`

Given `idx`, looks up the corresponding row in `streamline_index`, fetches the
cached file handle, reads the single padded streamline array, trims it to its
real `length` (removing padding), and returns:

```python
(torch.Tensor,  # shape (seq_len, 5)
 length: int,
 tract_id: int)
```

### Other Methods

- `__len__` — returns the number of indexed streamlines (raises `RuntimeError`
  if the index has not been built yet).
- `get_tract_id(idx)` — returns only the class label for an index, without
  touching the HDF5 file. Used by `StratifiedEpochSampler` to build class
  groupings cheaply.

---

## `StratifiedEpochSampler`

### Purpose

A PyTorch `Sampler[int]` that guarantees **every class contributes
proportionally** to each epoch's samples, preventing rare classes from being
underrepresented or dropped due to random chance. Unlike a plain random subset
sampler, this samples independently *within* each class before combining results.

### `__init__`

**Step 1 — Group indices by class.** Builds `self.class_indices`, an inverted
lookup mapping each `tract_id` to the list of dataset positions belonging to it:

```python
self.class_indices = {
    0: [3, 7, 12, 45, ...],   # dataset indices where tract_id == 0
    1: [1, 4, 9, 22, ...],    # dataset indices where tract_id == 1
    ...
}
```

Built by reading `dataset.streamline_index['tract_id']` (vectorized column
access from the structured array) and bucketing each index by its class. This
lets the sampler retrieve "all indices for class X" instantly without rescanning
the full index every epoch.

**Step 2 — Adaptive threshold.** Computes `adaptive_threshold = 5% of the
largest class size`, capped by `full_sample_threshold`. This mirrors the same
logic used in `StreamlineDataset._build_index()`.

**Step 3 — Per-class sample counts.** For each class:

- If it's smaller than the adaptive threshold → take everything.
- Otherwise → take `max(min_samples_per_class, sampling_percentage * n_available)`,
  capped at what's actually available.

Results are stored in `self.samples_per_class` (dict) and totaled into
`self.n_samples`. If `verbose=True`, prints a per-class breakdown of samples
taken vs. available.

### `set_epoch(epoch)`

Updates `self.epoch`, which seeds the RNG differently each epoch
(`seed + epoch`) — producing a new but reproducible random subset per epoch.
**Must be called at the start of each epoch** before iterating, otherwise the
same subset repeats.

### `__iter__`

For each class:

1. Randomly draws `samples_per_class[tract_id]` indices without replacement
   from `class_indices[tract_id]`.
2. Concatenates all classes' picks into `all_indices`.
3. If `shuffle=True`, shuffles the combined list so batches mix classes
   randomly instead of being ordered class-by-class.
4. Returns an iterator over the final index list — this is what `DataLoader`
   consumes to fetch batches via `dataset[idx]`.

### `__len__`

Returns `self.n_samples`, the precomputed total across all classes — used by
`DataLoader` and progress bars to know how many samples to expect per epoch.

---

## `streamline_collate_fn`

Used as the `collate_fn` argument to `DataLoader`. Since streamlines have
variable lengths, this function:

1. Separates the batch into `streamlines`, `lengths`, and `labels`.
2. Pads all streamlines in the batch to the same length using
   `pad_sequence(..., batch_first=True, padding_value=0.0)`.
3. Returns `(padded_streamlines, lengths, labels)` as three batched tensors,
   ready for model input. The `lengths` tensor lets the model mask or unpack
   padding downstream (e.g., for RNNs or attention masks).

---

## How the Pieces Fit Together

```
StreamlineDataset              -> builds streamline_index (which streamline lives where)
        |
        v
StratifiedEpochSampler         -> groups indices by class, samples proportionally per epoch
        |
        v
DataLoader(dataset, sampler=sampler, collate_fn=streamline_collate_fn)
        |
        v
dataset.__getitem__(idx)       -> lazily loads and trims the actual streamline from HDF5
        |
        v
streamline_collate_fn(batch)   -> pads streamlines to a common length per batch
```

Use `StratifiedEpochSampler` whenever class imbalance across tracts is a
concern (the common case for fiber bundle classification), since it prevents
small bundles from being starved of training examples across epochs.
