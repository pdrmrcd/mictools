# mictools

**Support tools for the APS Microscopy Group** — loading, processing, and
visualizing scan data from the **In-Situ Nanoprobe (ISN, beamline 19-ID)**.

`mictools` turns the asynchronously-triggered data of an FPGA-controlled
*flyscan* into aligned, trigger-indexed arrays and 2D maps, and provides
lightweight helpers for analyzing conventional *step scans*.

> [!IMPORTANT]
> If you are a coding agent or a new contributor, **read [Concepts](#concepts)
> before touching the pipeline.** The flyscan data model encodes beamline
> conventions (three clocks, trigger-indexed alignment, ghost-frame
> corrections) that are not obvious from the code alone.

---

## Table of contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Concepts](#concepts)
  - [Two acquisition schemes](#two-acquisition-schemes)
  - [Glossary](#glossary)
  - [The three FPGA clocks](#the-three-fpga-clocks)
- [How flyscan processing works](#how-flyscan-processing-works)
  - [Position data (SOCKETSERVER)](#1-position-data-socketserver)
  - [Detector data](#2-detector-data)
  - [Producing a map](#3-producing-a-map)
  - [ROIs and reproducibility](#4-rois-and-reproducibility)
  - [Derived arrays and provenance](#5-derived-arrays-and-provenance)
- [Step-scan analysis](#step-scan-analysis)
- [Repository layout](#repository-layout)
- [On-disk data layout](#on-disk-data-layout)
- [Conventions and caveats](#conventions-and-caveats)
- [Roadmap / not yet implemented](#roadmap--not-yet-implemented)

---

## Installation

Requires **Python ≥ 3.11**. Install in editable mode from a clone:

```bash
git clone git@github.com:pdrmrcd/mictools.git
cd mictools
pip install -e .
```

Versioning is handled by `setuptools-scm` (derived from git tags).

> [!WARNING]
> A few imported packages are **not yet declared** in `pyproject.toml`
> (`plotly`, `lmfit`, `ipywidgets`). Until that is fixed you may need
> to install them manually:
> ```bash
> pip install plotly lmfit ipywidgets
> ```

---

## Quick start

Point `mictools` at the directory that holds your scan files, then plot a map.
Every entry point accepts an explicit `path=` argument; setting a default once
with `set_path` saves you from repeating it.

```python
from mictools.config import set_path
from mictools.roi_utils import Roi
from mictools.plot_data import plot_flyscan

# 1. Set the default data root (contains Scan_XXXX.h5, Raw/, Processed/)
set_path("/data/2026-1/my_experiment")

# 2. Define a region of interest on an area detector (pixel bounds + a name)
roi1 = Roi(y_start=100, y_end=200, x_start=150, x_end=250, name="roi1")

# 3. Process + plot a flyscan map in one call.
#    roi_type selects which ROI scalar to map: "Intensity", "COM_X", or "COM_Y".
plot_flyscan(scanno=42, detector="me7", roi=roi1, roi_type="Intensity")
```

Map a **scalar detector** channel (e.g. a Tetramm ion-chamber current, in nA)
and normalize by another channel:

```python
plot_flyscan(scanno=42, detector="tetramm", ch=0, norm_detector="tetramm", norm_ch=1)
```

Inspect a single detector frame nearest a physical position:

```python
from mictools.plot_data import plot_closest_frame
plot_closest_frame(scanno=42, detector="me7", x=12.5, y=8.0, log_scale=True)
```

> [!TIP]
> `plot_flyscan` caches processed results to `Processed/` and links them into
> the master file. Re-running is cheap; pass `replace=True` to force
> reprocessing.

---

## Concepts

### Two acquisition schemes

| Scheme | What it is | mictools' role |
|---|---|---|
| **Step scan** | Motors move, settle, then detectors integrate at each grid point. | Load and analyze. Leans on existing bluesky/databroker standards; peak fitting via [`peak_modelling`](#step-scan-analysis). |
| **Time-based flyscan** | Motors move **continuously** while detectors are triggered on the fly. | **The main purpose of `mictools`.** Reconstruct maps from asynchronously-triggered, trigger-indexed data. |

Unless stated otherwise, the rest of this document concerns **flyscans**.

### Glossary

| Term | Meaning |
|---|---|
| **FPGA** | Field-Programmable Gate Array — the hardware controller that generates the scan clocks and detector triggers. |
| **SOCKETSERVER** | The DAQ process that records the FPGA clocks and interferometer readings into a labeled table (HDF5, split across files). Source of sample-position data. |
| **Interferometry** | Laser interferometers measuring stage position to nanometer precision. Multiple channels (`I7`, `I15`, …) are combined to compute X/Y/Z. |
| **Trigger** | The pulse telling detectors to acquire one frame. **The trigger number is the master index** linking position to every detector datum. |
| **ROI** | *Region of Interest* — a rectangular sub-window of an area-detector frame, `(y_start, y_end, x_start, x_end)`. |
| **CoM** | *Center of Mass* — the intensity-weighted centroid within an ROI, reported per axis (`COM_X`, `COM_Y`). |
| **Azimuthal integration** | Collapsing a 2D diffraction image into a 1D intensity-vs-angle curve. *(Planned — see [Roadmap](#roadmap--not-yet-implemented).)* |
| **2θ (two-theta)** | Scattering angle in diffraction; the x-axis of an azimuthally-integrated pattern. |
| **Master file** | The per-scan NeXus/bluesky file `Scan_XXXX.h5` at the data root, into which processed results are linked. |

### The three FPGA clocks

A flyscan is timed by three clocks recorded by the SOCKETSERVER:

| # | Clock | SOCKETSERVER column | Ticks when… | Purpose |
|---|---|---|---|---|
| 1 | **Master (1 MHz)** | `Counter1` | Always | Timestamps every row of data. |
| 2 | **Interferometry clock** | `Counter2` | An interferometry reading is produced | Marks each position measurement. |
| 3 | **Trigger clock** | `Counter3` | A detector trigger fires | Marks each detector frame; the map index. |

> [!NOTE]
> The position code groups rows by **`Counter3`** (the trigger), averaging all
> readings per trigger, and drops the first group (trigger 0 has no detector
> data). **`Counter2` (interferometry) is intentionally not used yet** — every
> interferometry reading is folded into its enclosing trigger. It is reserved
> for a future mode that gives users finer control over the position
> reconstruction (e.g. weighting or sub-trigger interpolation).

---

## How flyscan processing works

All flyscan data are **associations of arrays indexed by trigger number**, so
any detector value aligns to a sample position through its shared index. The
pipeline reconstructs positions, reduces detector frames to scalars, aligns the
two, and interpolates onto a 2D grid.

```
Raw SOCKETSERVER files ─► process_position_data ─┐
   (group by trigger, average, → µm, set origin)  │
                                                   ▼
Raw detector files ─► process_detector_data ─► mesh_detector_data ─► plot_flyscan
   (ghost-frame drop,      (ROI: I / COM_X / COM_Y   (align lengths,
    frame-mismatch align)   | channel: nA)            griddata interp,
                                                       abs_pos AFTER interp)
```

### 1. Position data (SOCKETSERVER)

The SOCKETSERVER writes a table with a labeled column per signal (three
counters plus interferometer channels), one row per clock tick, split across
`scan_XXXX_*.h5` files. `process_position_data` reconstructs sample X/Y per
trigger:

1. Load and concatenate all SOCKETSERVER files for the scan.
2. **Group rows by trigger** (`Counter3`) and **average** each group — the
   default `"averaging"` method, which increases statistics per trigger. A
   simpler `"basic"` single-channel method is also available.
3. Drop the first trigger (no detector data).
4. Convert interferometer counts to **microns**, zero to the first trigger
   (origin), and flip the X sign.
5. Save to `Processed/Scan_XXXX/position.h5` and link into the
   master file at `entry/data/Position`.

> [!NOTE]
> **Microns (µm) is the canonical position unit** for the pipeline. The legacy
> [`data_proc.py`](#repository-layout) uses nanometers — do not mix them.

### 2. Detector data

Each trigger produces frames from several detectors under one scan number:

- **Area detectors** (e.g. `me7`/xpress3, `xrd`, `ptycho`, `rayspec`): 3D
  stacks `(n_frames, y, x)` stored at `entry/data/data`.
- **Scalar detectors** (e.g. `tetramm` ion-chamber currents): per-channel
  values.

Two data-integrity corrections live in the pipeline — keep them in mind:

- **Ghost-frame correction.** The xpress3 occasionally writes a leading junk
  frame, flagged by `NDArrayUniqueId[0] == -1`. For `me7`/`rayspec` the first
  row is dropped. *(Intentionally scoped to these two detectors.)*
- **Frame mismatch.** Detector and position lengths can differ.
  `mesh_detector_data` computes `frame_mismatch = len(detector) − len(position)`
  and trims one end, controlled by `missed_frame_position`
  (`"Beginning"` trims the tail, otherwise the head).

### 3. Producing a map

The goal is a **2D scalar map** `Z(X, Y)`:

- **ROI mode** reduces each frame to three scalars — `Intensity`, `COM_Y`,
  `COM_X` — chosen via `roi_type`.
- **Channel mode** (`ch=`) maps a scalar-detector value (e.g. Tetramm nA).
- **Normalization** (`norm_detector`, `norm_ch`) divides the signal by a
  reference channel.
- Interpolation uses `scipy.griddata` (`linear`, with `nearest` filling points
  outside the convex hull).
- **`abs_pos=True`** (default) converts the interpolated grid to absolute stage
  coordinates. This transform is applied **after** interpolation — applying it
  before would distort the `griddata` input.

Maps are saved to the master file under
`entry/data/{DETECTOR}/Images/{roi.name | channel_ch}`.

### 4. ROIs and reproducibility

ROIs are created with the [`Roi`](mictools/roi_utils.py) class:

```python
from mictools.roi_utils import Roi
roi = Roi(y_start=100, y_end=200, x_start=150, x_end=250, name="roi1")
```

When a map is produced from an ROI, the ROI geometry is saved as attributes on
the processed file (`roi_name`, `roi_y_start/end`, `roi_x_start/end`) so the map
records how it was made.

#### ROI registry

A per-experiment, writable **ROI registry** lets you define ROIs once and reuse
them across scans. It is a **YAML sidecar** at
`{data_root}/analysis/roi_registry.yaml` (created on first write), managed by
[`RoiRegistry`](mictools/roi_utils.py):

```python
from mictools.roi_utils import Roi, RoiRegistry
from mictools.plot_data import plot_flyscan

# Define an ROI once (names must be unique within the experiment)
reg = RoiRegistry.load()
reg.add(Roi(y_start=100, y_end=200, x_start=150, x_end=250, name="roi1"))

# Reuse it across scans by name — the pipeline resolves it from the registry
plot_flyscan(scanno=42, detector="me7", roi="roi1")
plot_flyscan(scanno=43, detector="me7", roi="roi1")
```

- **Unique names.** Re-adding a name with the *same* geometry is a no-op.
  Re-adding it with a *different* geometry raises unless you pass
  `override=True` (to `RoiRegistry.add`) or `roi_override=True` (to
  `plot_flyscan`/`mesh_detector_data`).
- **Override keeps history.** On override, the previous definition — together
  with the list of scans already processed with it — is archived under
  `history` in the YAML, so prior maps remain traceable.
- **Usage is push-tracked.** Each time a map is produced for an ROI,
  `mesh_detector_data` records the scan number under that ROI's `used_by_scans`.
- Passing a raw `Roi` object still works and is registered on first use; a name
  clash with a different geometry is rejected (this closes a stale-cache hole,
  since processed results are cached by `roi.name`).

> [!NOTE]
> The registry (`analysis/`) holds **human-curated** artifacts and is kept
> separate from the machine-written `Raw/` and `Processed/` trees. `pyyaml` is
> imported lazily, so importing the pipeline never requires it — but it is now a
> declared dependency.

### 5. Derived arrays and provenance

Some detector data need **multiple steps** to reach a scalar. The target chain
for the diffraction detector:

```
2D image ──azimuthal integration──► 1D I(2θ) ──integrate 2θ window──► scalar ──► map
```

Each derived array is meant to carry a **link back to its parent dataset**, so
any map is traceable to its origin.

> [!NOTE]
> **Planned.** Today provenance is a single `parent_dataset` **string
> attribute** on saved maps, and azimuthal integration
> ([`powder_utils.py`](mictools/powder_utils.py)) is a stub. See
> [Roadmap](#roadmap--not-yet-implemented).

---

## Step-scan analysis

Step scans are analyzed with [`peak_modelling`](mictools/peak_modelling.py),
which fits a **PseudoVoigt + Linear** model (via `lmfit`) and visualizes runs:

```python
from mictools.peak_modelling import fit_scan, analyze_run

# Fit a single scan: x column, y column, optional normalization column
result = fit_scan(scanno=42, xcol="motor", ycol="detector", normcol="i0", visualize=True)

# Fit/track a peak across a series of scans
analyze_run(scans=[40, 41, 42], xcol="motor", ycol="detector", zcol="Scan", normcol="i0")
```

---

## Repository layout

```
mictools/
├── config.py          # Module-global default data path: set_path() / get_path()
├── load_data.py       # Loaders: load_scan, file_names, get_scan_info,
│                       #          load_image_from_scan, load_interferometry_data
├── process_data.py    # CORE flyscan pipeline: process_detector_data,
│                       #   process_position_data, mesh_detector_data, ROI/Tetramm procs
├── plot_data.py       # plot_flyscan, plot_closest_frame, plot_sum_detector_image
├── peak_modelling.py  # STEP-scan lmfit fitting (fit_scan, graph_run, analyze_run)
├── roi_utils.py       # Roi(y_start, y_end, x_start, x_end, name)
├── data_proc.py       # ⚠️ DEPRECATED legacy implementation (different units/paths)
├── powder_utils.py    # (stub) azimuthal integration — not yet implemented
└── __init__.py        # (empty) — import from submodules, e.g. mictools.plot_data
```

| Module | Status | Notes |
|---|---|---|
| `process_data.py` | ✅ Core | The most important file; start here. |
| `load_data.py`, `plot_data.py`, `config.py`, `roi_utils.py` | ✅ Core | Loading, plotting, configuration. |
| `peak_modelling.py` | ✅ Core | Step-scan peak fitting. |
| `data_proc.py` | 🚫 Deprecated | Older parallel implementation with different path/unit conventions (`analysis/`, counts→nm). **Do not use for new work.** |
| `powder_utils.py` | 🔜 Stub | Azimuthal integration placeholder. |

---

## On-disk data layout

Relative to the data root (`get_path()`):

```
{root}/
├── Scan_{scanno:04d}.h5                                    # master file (NeXus/bluesky + link target)
├── Raw/
│   └── Scan_{scanno:04d}/
│       ├── {DETECTOR}/scan_{scanno:04d}_*.h5               # raw detector frames (multi-file)
│       └── SOCKETSERVER/scan_{scanno:04d}_*.h5             # clocks + interferometry
├── Processed/
│   └── Scan_{scanno:04d}/
│       ├── position.h5                                     # processed positions
│       └── {detector}.h5                                   # ALL ROIs + channels for one
│                                                             #   detector, as sibling groups
│                                                             #   under entry/data/{roi.name |
│                                                             #   channel_ch}
└── analysis/
    └── roi_registry.yaml                                   # human-curated ROI registry
```

Master-file external links (unchanged target names in `Scan_XXXX.h5`; only the
source file + internal group changed, per above):

| Data | Link path in `Scan_XXXX.h5` | Source |
|---|---|---|
| Positions | `entry/data/Position` | `Processed/Scan_XXXX/position.h5::entry/data` |
| ROI results | `entry/data/{DETECTOR}/Processed Data/{roi.name}` | `Processed/Scan_XXXX/{detector}.h5::entry/data/{roi.name}` |
| Channel current | `entry/data/{DETECTOR}/Current {ch}` | `Processed/Scan_XXXX/{detector}.h5::entry/data/channel_{ch}` |
| Meshed maps | `entry/data/{DETECTOR}/Images/{roi.name \| channel_ch}` | written directly (no source file) |

Key raw-file HDF5 paths: frames `entry/data/data`; timestamps
`entry/instrument/NDAttributes/NDArrayTimeStamp`; ghost-frame flag
`entry/instrument/NDAttributes/NDArrayUniqueId`.

---

## Conventions and caveats

- **Scan numbers** are 4-digit zero-padded: `Scan_0042.h5`.
- **Detector names**: the directory component is upper-cased (`ME7`), while the
  raw-file glob is lowercase (`scan_0042_*.h5`). Pass the detector name in
  lowercase to the API (`"me7"`, `"xrd"`, `"ptycho"`, `"rayspec"`, `"tetramm"`).
- **Units**: sample position in **microns (µm)** throughout the current
  pipeline; Tetramm currents in **nA**.
- **Caching**: processed results persist under `Processed/` and are linked into
  the master file. Pass `replace=True` to recompute.

---

## Roadmap / not yet implemented

The following are described in the design but **not yet built** — treated as
planned throughout this document:

- [x] **ROI registry** — per-experiment YAML sidecar in `analysis/` holding
      reusable ROIs; unique names with an override path that archives which scans
      already used the prior definition. See
      [ROIs and reproducibility](#4-rois-and-reproducibility).
      *(Still open: duplicating the ROI name/geometry inline on the meshed-map
      `Images/{roi.name}` group — today those maps reference the ROI via their
      path + `parent_dataset`, and the per-ROI processed file carries the
      geometry attributes.)*
- [ ] **Interferometry-clock (`Counter2`) position mode** — optional finer
      position reconstruction using per-reading data instead of per-trigger
      averaging.
- [ ] **Azimuthal integration** (`powder_utils.py`) — image → 1D `I(2θ)`.
- [ ] **2θ-window integration** — reduce `I(2θ)` to a scalar for mapping.
- [ ] **Real provenance links** — HDF5 links from derived arrays to parents
      (currently a `parent_dataset` string attribute only).
- [ ] **Declare missing dependencies** in `pyproject.toml`
      (`plotly`, `lmfit`, `ipywidgets`). `pyyaml` is now declared.
