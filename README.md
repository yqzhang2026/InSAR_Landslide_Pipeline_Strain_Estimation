# InSAR-Landslide-Pipeline Desktop GUI

PySide6 desktop app that runs DEM aspect, InSAR N/E/Z displacement, and pipeline strain in three steps.

## Run from source

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Put supporting files next to the program under `data/` (see `data/README.md`). Default paths are relative to the program folder, not the current working directory.

## Workflow

1. **Step 1 (DEM Processing)** — load/merge two DEMs, compute slope aspect, optional smoothing. `Run All of Step 1` rebuilds Step 1 and discards later results (you will be asked to confirm).
2. **Step 2 (InSAR)** — match ascending/descending scenes, compute N/E/Z displacement, filter unstable points.
3. **Step 3** — select a landslide area (interactive contour), build a pipeline mesh, then map displacement and strain.

**Run All Steps** runs Step 1 and Step 2, then opens the contour picker. After you save a contour it continues with mesh + strain. Canceling the picker leaves results at Step 2.

The Visualization **Date** and **NEZ** controls are kept in sync with Step 3 **Date Index** and **Contour Direction**, so the map you see is the epoch/component used for contour pick and strain.

## Strain display options

Pipeline strain is a single tab. **Strain type**, **Strain period** (data period / yearly / entire life), and **Consider pipeline diameter field** only change the plot. Exported CSV/SHP keep the Step 3-3 stored data-period strain (default diameter for bending). Titles are marked `display only`.

## Logos

Left-corner logos load from `assets/UCB logo.jpg` and `assets/opensra2_logo.png`.

## Packaged app (macOS)

Place `data/` next to `InSARLandslideGUI.app` (same folder as the `.app`, not inside the bundle).

```bash
cd landslide_pyside6_app
pyinstaller --noconfirm --clean build/pyinstaller.spec
```

DEM browse still accepts two GeoTIFFs in one dialog. Output defaults to `data/results`.
