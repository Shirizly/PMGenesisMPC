# Visualization Script Plan — Modular, Type-Agnostic

Archived from repository root on 2026-07-03. This file is historical context and is no longer the active source of interface truth.

Original content:

---

# Visualization Script Plan — Modular, Type-Agnostic

## Goal

Provide a single entry point `visualize.py` that can:

1. Load raw simulation data files (`.pt` + `.yaml`) from any dataset folder.
2. Convert them to occupancy grids using the existing `PileSweepData` class.
3. Visualize individual samples, entire runs, or model predictions.
4. Save images (PNG) and optionally a short video (MP4).

The script must **not** require the caller to specify whether the data is
vector-based or grid-based; it should infer the representation from the file
extension and use the appropriate conversion path automatically.

## Design Principles

- **Single entry point**: `visualize.py` at the workspace root.
- **No type arguments**: The script decides internally which loader to use.
- **Re-use existing classes**: `PileSweepData`, `PhysicsBounds`, and any
drawing helpers already in `Genesis/training/dataset.py`.
- **Minimal dependencies**: Only standard library + `torch`, `numpy`, `cv2`,
  `matplotlib` (already used elsewhere). No new heavy packages.
- **Clear CLI interface**: Accept a few command-line arguments for input path,
  sample index, output image/video paths, etc.

## High-Level Flow

```text
visualize.py
    ↓ parse args
load_raw_data(paths)          # .pt + .yaml → dict of tensors
    ↓ infer representation
if vector-based:
    convert_vector_to_grid(data_dict)   # uses PileSweepData internally
else:
    grid_already = data_dict["grid"]

visualize_sample(grid, physics=None, output_path=None)
visualize_run(run_dir, samples_per_row=4, output_path=None)
save_as_video(frames, output_path)
```

## Required Functions (to be implemented)

### `load_raw_data(paths: list[str] | str)`

- Accept a single folder path or a list of paths.
- Resolve relative paths under `Genesis/data/`.
- For each run folder locate `_data.pt` and `_config.yaml`.
- Load the dict from `.pt` (torch.load) and parse the YAML.
- Return a tuple `(data_dict, config)` where `data_dict` contains:
  - `states`, `states_`, `p_starts`, `p_stops`, `angles` (as numpy arrays).

### `convert_vector_to_grid(data_dict, config)`

- Use `PileSweepData.__init__` with the run folder path to allocate grids.
- Populate `_input_grid`, `_output_grid` by iterating over samples in
  `data_dict["states"]`.
- Return `(input_grid, output_grid, physics_tensor)`.

### `visualize_sample(input_grid, output_grid, physics=None, title="Sample")`

- Stack `input_grid` and a tool-channel map (if present) into a single RGB
  image: particle occupancy → grayscale, tool start/end → colored overlays.
- Use matplotlib to render; save as PNG if an output path is given.
- Return the figure object for further manipulation.

### `visualize_run(run_dir, samples_per_row=4)`

- List all `_data.pt` files in the run folder.
- For each file call `load_raw_data` → `convert_vector_to_grid`.
- Build a grid of subplots (matplotlib) showing input vs output.
- Save as PNG or PDF if requested.

### `save_as_video(frames, fps=5)`

- Accept a list of RGB numpy arrays.
- Use `cv2.VideoWriter` with MJPG codec to write MP4/AVI.

## CLI Interface (argparse)

```bash
python visualize.py \
    --input <path/to/run> \
    --sample <idx> \
    --output-image <png_path>

# or for a whole run:
python visualize.py --input <run_dir> --samples-per-row 4 --output-image <png>

# or video from a list of frames (passed via stdin JSON):
python visualize.py --frames-from-stdin --fps 5 --output-video <mp4>
```

## Where to Place New Code

Create `visualize.py` at the workspace root (`/home/alon/Code/pile_manipulation/`).
All imports will be local modules already present:
`torch`, `numpy`, `cv2`, `matplotlib.pyplot`, plus the existing classes from
`Genesis/training/dataset.py`. No new packages needed.

## Updating Documentation

After implementation, add a short section to `CODEBASE_OVERVIEW.md` describing
the new visualization utilities and their usage. Also update any relevant
Markdown files with links to this plan for future reference.
