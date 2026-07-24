# Mars Swarm Sample Collection

A stigmergy-based multi-agent simulation of an autonomous drone-rover swarm for sample collection on Mars, inspired by honeybee foraging strategies. Developed for the course *AE4350: Bio-Inspired Intelligence and Learning for Aerospace Applications* at TU Delft.

## Overview

This project simulates a heterogeneous robotic fleet — fast aerial scouts (drones) and slower ground collectors (rovers) — coordinating indirectly through a shared stigmergic field (a decaying pheromone-like grid overlaid on real Martian terrain), instead of direct communication. The system is validated against real terrain data from **Jezero Crater** and benchmarked against Perseverance's own sampling sites.

Key features:
- DEM-based terrain simulation using real Mars orbital elevation data (`jezero.tif`)
- Drones: Lévy-flight exploration, POI discovery, trail marking, solar recharge cycle
- Rovers: stigmergic gradient-following, sample collection, dead-end signaling
- Random exploration mode (no prior POI knowledge)
- Batch simulation runner with metrics aggregation (mean ± std over multiple seeds)
- Publication-ready PDF/PGF figure generation for LaTeX reports

## Repository Structure
├── SwarmMars.py # Main simulation code (engine, agents, field, plotting)
├── jezero.tif # DEM (Digital Elevation Model) of Jezero Crater **[NOT included, see below]**
└── results/ # Output figures and CSV metrics (generated, not versioned)

## Requirements
numpy
rasterio
matplotlib
pyproj
scipy
pandas

You will also need a working **LaTeX installation** (`pdflatex`) on your system PATH, since figures are exported via the `pgf` matplotlib backend for direct inclusion in LaTeX documents.

## Data

The DEM file `jezero.tif` is **not included** in this repository due to file size. It can be obtained from [https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/ScienceInvestigationMaps_JPL/M20_JezeroCrater_CTXDEM_20m.tif]. Place it in the repository root before running any script.

## Usage

### Run a single simulation (known POI batch)
```bash
python SwarmMars.py
```
By default this runs the main block at the bottom of the script, simulating a fixed POI batch and saving output figures to `results/`.

### Run batch metrics across multiple seeds
```python
from SwarmMars import run_metrics_all_batches, poi_batch_1_km, poi_batch_2_km, poi_batch_3_km

raw_df, agg_df = run_metrics_all_batches(
    poi_batches_km=[poi_batch_1_km, poi_batch_2_km, poi_batch_3_km],
    batch_labels=["Crater Floor", "Delta Front", "Margin"],
    n_seeds=8,
    output_dir="results"
)
```

### Run pure exploration mode (no initial POIs)
```python
raw_df, agg_df = run_metrics_all_batches(
    poi_batches_km=[[]],
    batch_labels=["Exploration"],
    n_seeds=8,
    early_stop=False,
    output_dir="results"
)
```

## Author

Mattia Arioldi — TU Delft, AE4350
