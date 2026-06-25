---
name: apt-ccd-analysis
description: "Run Compositional Community Detection (CCD) on Atom Probe Tomography data. Use when: analyzing APT point clouds, identifying compositional communities in APT neighborhoods, detecting phases or precipitates in APT data, running CCD on .pos/.apt/.csv files with .rrng range files, interpreting KS statistics heatmaps from APT analysis."
argument-hint: "Provide the path to your APT data file (.pos, .apt, or .csv) and the .rrng range file."
---

# APT Compositional Community Detection (CCD) Analysis

Identify compositionally distinct regions in reconstructed APT point clouds using Compositional Community Detection. This skill preprocesses raw APT data into overlapping spherical neighborhoods, clusters them by composition via k-means, computes Kolmogorov-Smirnov statistics, and applies Louvain community detection to reveal dominant compositional domains. CCD is a qualitative method. Comparative analysis can be done by batching samples.

## When to Use

- Analyze a reconstructed APT dataset (.pos, .apt, or raw .csv) for compositional segregation
- Identify phases, precipitates, defects, or chemical heterogeneity in APT data
- Detect and classify compositionally distinct regions without manual thresholding
- Generate KS-statistics heatmaps showing ion enrichment/depletion per community
- Produce scientific claims and literature search queries from APT results

## Prerequisites

The following Python packages must be available in the active environment:

- `numpy`, `pandas` (data manipulation)
- `scipy` (KDTree, KS statistics)
- `scikit-learn` (MiniBatchKMeans)
- `networkx` (graph construction for community detection)
- `python-louvain` (`community` package for Louvain partitioning)
- `matplotlib`, `seaborn` (visualization)
- `apav` (only if converting `.apt` files to `.pos`)

The CCD scripts are located in [scripts/](./scripts/).

## Key Concepts

| Term | Meaning |
|------|---------|
| **Neighborhood** | A spherical region (default 1 nm radius) centered on a grid point, containing the ions within that sphere. Each neighborhood has a compositional profile. |
| **k-Means clustering** | Groups neighborhoods by compositional similarity into k clusters. |
| **KS statistic** | Kolmogorov-Smirnov statistic quantifying how much each cluster's ion distribution deviates from the bulk. Positive = enrichment, negative = depletion. |
| **Community detection** | Louvain algorithm applied to a graph of KS-statistic embeddings. Clusters with similar KS signatures are grouped into communities. |
| **q percentile** | Edge-weight threshold for the community graph. Only edges below the q-th percentile of cosine distances are retained. Lower q = stricter filtering. |
| **Outer layer** | Neighborhoods on the surface of the dataset, excluded from analysis to avoid edge effects. |

### CCD Parameter Guidance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_values` | `[4, 5, 6]` | List of k values for k-means. More values = more robust community assignment via consensus. Increase range for compositionally complex materials. |
| `q` | `25` | Percentile threshold for graph edge filtering. Lower = more aggressive filtering (fewer, more distinct communities). Higher = more inclusive (may merge distinct features). |
| `ignore_ions` | `[]` | Ion species to exclude from the compositional feature space (e.g., `["O1", "O1H1"]` for oxygen contaminants). |
| `n_repeats` | `5` | Number of random seeds per k value. More repeats = more stable consensus. |
| `radius` | `1` | Neighborhood sphere radius in nm. |
| `overlap` | `0.5` | Overlap fraction between neighboring spheres. |

### Interpreting KS Statistics

- **Positive KS** for an ion in a community → that ion is enriched relative to bulk
- **Negative KS** → depleted relative to bulk
- **Near-zero KS** → no significant deviation from bulk composition
- Communities enriched in specific alloying elements may correspond to precipitate phases
- Ga enrichment (when Ga is not a matrix element) may indicate FIB-induced damage
- Depletion of all matrix elements may suggest voids or pores

## Procedure

### Step 0: Validate inputs and set up directories

1. Confirm the user has provided:
   - `data_file`: path to the APT data file (`.pos`, `.apt`, or `.csv` with columns x, y, z, Da)
   - `rrng_file`: path to the `.rrng` range file
   - `output_dir`: directory for results (default: `./ccd_analysis/`)
2. If the data file is `.apt`, convert to `.pos` first using `apav`:
   ```python
   import apav as ap
   roi = ap.load_apt("sample.apt")
   roi.to_pos("sample.pos")
   ```
3. Create the output directory if it doesn't exist.

### Step 1: Generate neighborhoods

Run the preprocessing script to create overlapping spherical neighborhoods:

```bash
python scripts/apt_preprocessing.py --data <data_file> --rrng <rrng_file> --savedir <output_dir>
```

**What it does:**
- Reads the point cloud and range file to extract ion positions and identities
- Removes noise ions (unranged atoms)
- Builds a KDTree and generates overlapping spherical neighborhoods on a regular grid
- For each neighborhood: counts ions per species, computes fractional compositions (`pX`) and densities (`dX`)
- Marks outer-layer and second-layer neighborhoods for edge-effect filtering
- Writes a neighborhoods CSV with columns: ion counts, compositions, densities, midpoint coordinates, radius, id
- Writes a metadata JSON with sample info, neighborhood count, ion types, mean density

**Outputs:**
- `<output_dir>/<sample>_<radius>nm-radius_<overlap>-overlap.csv` — neighborhood data
- `<output_dir>/<sample>_metadata.json` — sample metadata

### Step 2: Estimate CCD parameters

Examine the neighborhood CSV and metadata to determine optimal CCD parameters. Load the neighborhood CSV and extract summary statistics:

```python
import pandas as pd, json

df = pd.read_csv("<neighborhood_csv>")
ion_cols = sorted([c for c in df.columns if c.startswith('p')])
ion_names = [c[1:] for c in ion_cols]

metadata = {
    "neighborhood_count": len(df),
    "ion_types": ion_names,
    "mean_density": float(df["density"].mean()),
    "std_density": float(df["density"].std()),
    "mean_composition": {
        ion: float(df[col].mean())
        for ion, col in zip(ion_names, ion_cols)
    },
}
print(json.dumps(metadata, indent=2))
```

**Decision criteria for parameters:**

- **k_values**: Choose based on compositional complexity. Few distinct phases → `[3, 4, 5]`. Many phases or gradients → `[5, 6, 7, 8]`. Always provide a range of 3+ values for robust consensus.
- **q**: Start at 25. If too many noisy communities appear, lower to 10–15. If distinct features are merged, raise to 40–50.
- **ignore_ions**: Remove ions that are:
  - Not reliably detected by APT (e.g., `O1`, `O1H1`, `O2` for oxygen in metallic samples)
  - Present at trace levels that add noise without meaningful signal
  - Known contaminants (e.g., `Ga1` from FIB preparation)
- **n_repeats**: Default 5 is usually sufficient. Increase to 10 for noisy data.

If the user provides system information (material composition, expected phases), use domain knowledge to refine these choices. For example, a binary alloy with one expected precipitate phase might use `k_values=[3, 4, 5]` and `q=25`, while a complex multi-component alloy might use `k_values=[5, 6, 7, 8]` and `q=15`.

### Step 3: Run CCD analysis

Run the CCD algorithm with the chosen parameters. This can be done via Python directly:

```python
import sys, os, types, importlib.util

# Bootstrap the module for standalone use
script_dir = os.path.abspath("scripts")
pkg_name = "apt_ccd"
pkg = types.ModuleType(pkg_name)
pkg.__path__ = [script_dir]
pkg.__package__ = pkg_name
sys.modules[pkg_name] = pkg

spec = importlib.util.spec_from_file_location(
    f"{pkg_name}.ccd", os.path.join(script_dir, "ccd.py")
)
ccd = importlib.util.module_from_spec(spec)
ccd.__package__ = pkg_name
sys.modules[f"{pkg_name}.ccd"] = ccd
spec.loader.exec_module(ccd)

result = ccd.detect_compositional_communities(
    neighborhood_data="<neighborhood_csv>",
    savedir="<output_dir>",
    k_values=[4, 5, 6],       # from Step 2
    q=25,                       # from Step 2
    ignore_ions=["O1", "O1H1"],# from Step 2
    n_repeats=5,
)

print(f"Communities found: {result['community_count']}")
print(f"Neighborhood counts: {result['community_neighborhood_counts']}")
```

**What it does:**
1. Loads neighborhoods CSV and removes outer-layer neighborhoods
2. Filters out ignored ions and builds a compositional feature matrix
3. For each k value × n_repeats: runs MiniBatchKMeans, computes signed KS statistics per cluster
4. Builds a graph where nodes are clusters and edges are weighted by cosine distance between KS embeddings
5. Removes edges above the q-th percentile threshold
6. Removes isolated nodes (clusters that don't resemble any other)
7. Runs Louvain community detection on the filtered graph
8. Maps community labels back to individual neighborhoods via majority vote across all k/seed combinations
9. Writes community-labeled `.xyz` file and KS-statistics heatmap

**Outputs:**
- `<output_dir>/KS_stats.png` — heatmap of mean KS statistics per community
- `<output_dir>/*_community_clustering.xyz` — community-labeled point cloud
- `<output_dir>/*_community_clustering_<k>_<seed>seed.json` — per-run clustering metadata

### Step 4: Interpret results

After CCD completes, interpret the results using the KS statistics heatmap and community statistics.

**Information to analyze:**

1. **KS statistics heatmap** (`KS_stats.png`): Each row is an ion species, each column is a community. Color indicates enrichment (red/positive) or depletion (blue/negative).

2. **Community statistics**: From the `result` dict:
   - `community_count`: number of distinct communities
   - `community_neighborhood_counts`: how many neighborhoods belong to each community
   - `community_compositions`: mean KS statistics per community (list of arrays)

3. **Spatial distribution** (optional, from the `.xyz` file): Load the community-labeled point cloud to assess whether communities are spatially localized (precipitates) or dispersed (matrix phases).

**Interpretation guidelines:**

- A community with one or two strongly enriched ions likely represents a precipitate or secondary phase enriched in those elements
- The community with the most neighborhoods and KS values near zero is typically the matrix phase
- Communities enriched in Ga (when not a matrix element) suggest FIB damage
- Communities with all matrix elements depleted may indicate voids or low-density regions
- Spatially localized communities with high neighbor purity indicate coherent precipitates or inclusions
- Spatially extended communities indicate widespread phases

**Generate scientific claims** based on the analysis:
- Each claim should be a single, focused, testable observation
- Avoid overly specific numbers from the analysis
- Frame "Has anyone..." questions that are self-contained and understandable without seeing the data
- Include 3–5 keywords per claim for literature search

### Step 5: Generate report

Compile the analysis into a summary. Include:

1. **System information**: Material, preparation method, instrument parameters
2. **Analysis parameters**: k_values, q, ignore_ions, n_repeats used
3. **CCD statistics**: Number of communities, neighborhood counts per community
4. **Visualizations**: KS-statistics heatmap image
5. **Scientific analysis**: Detailed interpretation of each community's compositional signature
6. **Scientific claims**: 2–4 testable claims with keywords for literature search

Save the report as an HTML file in the output directory. The KS_stats.png can be embedded as a base64 image.

## Batch Analysis (Multiple Samples)

For analyzing multiple APT datasets together:

1. Preprocess each sample independently (Step 1)
2. Estimate parameters using the first sample (Step 2)
3. Merge all neighborhood CSVs into one, adding a `sample_source` column:
   ```python
   frames = []
   for csv_path in csv_paths:
       df = pd.read_csv(csv_path)
       df["sample_source"] = Path(csv_path).stem
       frames.append(df)
   merged = pd.concat(frames, ignore_index=True)
   merged.to_csv("<output_dir>/merged_neighborhoods.csv", index=False)
   ```
4. Run CCD on the merged CSV (Step 3)
5. After CCD, split the merged community-labeled XYZ back into per-sample XYZ files (see below)
6. Compute per-source community distributions for trend analysis
7. Interpret and report with per-sample breakdowns (Steps 4–5)

### Splitting a merged XYZ into per-sample files

`detect_compositional_communities` writes community labels only to an XYZ file, not back to the CSV.
To recover per-sample labeled point clouds after a merged CCD run, join the XYZ on coordinates
with the merged CSV (which carries the `sample` column), then write one XYZ per sample:

```python
import pandas as pd

MERGED_CSV = "<output_dir>/merged_neighborhoods.csv"
XYZ_IN     = "<output_dir>/merged_neighborhoods_community_clustering.xyz"
OUT_DIR    = "<output_dir>"

# Load merged CSV — only need sample label and coordinates
df = pd.read_csv(MERGED_CSV, usecols=['sample', 'midpoint_x', 'midpoint_y', 'midpoint_z'])

# Read the XYZ (inner neighborhoods only; format: community x y z)
xyz = pd.read_csv(XYZ_IN, sep=r'\s+', skiprows=2, header=None,
                  names=['community', 'midpoint_x', 'midpoint_y', 'midpoint_z'])
xyz['community'] = xyz['community'].astype(int)

# Join on coordinates to recover sample labels
joined = xyz.merge(df, on=['midpoint_x', 'midpoint_y', 'midpoint_z'], how='left')

def write_xyz(df, path):
    lines = [f"{len(df)}\n \n"]
    for _, row in df.iterrows():
        lines.append(f"{int(row['community'])}   {row['midpoint_x']}   "
                     f"{row['midpoint_y']}   {row['midpoint_z']}   \n")
    with open(path, 'w') as f:
        f.writelines(lines)

for sample_name, group in joined.groupby('sample'):
    out_path = f"{OUT_DIR}/{sample_name}_community_clustering.xyz"
    write_xyz(group, out_path)
    print(f"  {len(group):,} neighborhoods → {out_path}")
```

The resulting per-sample XYZ files share the same community label space as the merged run,
so communities are directly comparable across samples.
