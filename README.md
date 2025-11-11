# CFIS UMAP Explorer

Lightweight Flask dashboard to explore CFIS galaxy imagery through a 2‑D UMAP embedding of the original 192‑dimensional feature vectors. It lets you filter by post-merger stages, color points by any column (linear or log scale), inspect hover thumbnails, request nearest neighbours, and visualize high-dimensional MiniBatchKMeans clusters.

PS: You need the package with the DATA and STATIC folder to run this. Provided under reasonable request.

## Features
- Plotly‐GL scatter of every galaxy’s UMAP projection with custom colormaps.
- Toggle between raw column values (with optional log scaling) and precomputed cluster labels for colorizing.
- Hover previews showing the galaxy cutout plus the active color metric.
- Click to fetch nearest neighbours in both CFIS and TNG using KD‑trees.
- Responsive mosaics for CFIS r-band and TNG100 cutouts.
- Utility scripts:
  - `scripts/compute_clusters.py` – run MiniBatchKMeans in the 192‑D space and persist the `cluster_id`.
  - `scripts/cleanup_dr7_columns.py` – drop legacy `*_x`/`*_y` duplicates and standardize column names.

## Prerequisites
- Python 3.10+
- Virtualenv or similar (optional but recommended)
- Local copies of the data pickles/arrays under `data/UMAPS/...`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the app

```bash
make run
```

This sets `FLASK_APP=webapp.py` and launches `flask run --debug`.

If you prefer manual control:

```bash
export FLASK_APP=webapp.py
flask run --debug
```

Browse to http://127.0.0.1:5000/umap.

## Data maintenance

1. **Normalize DR7 columns** (remove `*_x`, rename `*_y`, and copy UMAP coords):

   ```bash
   python scripts/cleanup_dr7_columns.py --keep-umap
   ```

2. **Compute/update cluster IDs** (if you regenerate the feature arrays):

   ```bash
   python scripts/compute_clusters.py --n-clusters 12
   ```

3. Restart the Flask server so it reloads the refreshed dataframe.

## Configuration highlights
- `DEFAULT_COLOR_COLUMN` in `webapp.py` controls which column is used when the page first loads (currently `nFit2D` in log scale).
- `CLUSTER_COUNT` is derived from the persisted `cluster_id` column; adjust `scripts/compute_clusters.py` to change how clusters are computed.
- KDTree neighbour counts (`CFIS_NEIGHBORS`, `MOCK_NEIGHBORS`) are constants near the top of `webapp.py`.

## Contributing
1. Fork/clone the repo.
2. Create a feature branch.
3. Run `make run` and test changes locally.
4. Submit a PR with a clear summary and screenshots/gifs when UI changes are involved.

## License
MIT (or whatever license applies; update this section accordingly).

