# VeloChem 1.0.0

VeloChem runs the selected seed-11, epoch-20 B100 molecular model on a CPU. It returns 100 predictions in the frozen model order: 68 class-1 probabilities and 32 regression values on their recorded scales. Endpoint names, categories, class labels and display scales are in `velochem/presentation_catalog.json`; trained transforms and source notes are in `velochem/catalog.json`.

## Install

Python 3.11 or 3.12:

```sh
python -m pip install velochem==1.0.0
```

## Terminal

The original commands remain available. With no `--format`, single-SMILES output stays JSON and CSV output stays wide with its input columns preserved.

```sh
velochem --smiles "CCO"
velochem --smiles "CCO" --format table --sort-by category
velochem --smiles "CCO" --format table --category "Metabolism"
velochem --input-csv example_input.csv --output-csv predictions.csv
velochem --input-csv example_input.csv --output-csv predictions_tidy.csv \
  --format tidy-csv --category "Metabolism" --category "Toxicity and bioactivity" \
  --sort-by category
```

Repeat `--category` to select several categories. An empty category selection means all categories. `--sort-by` accepts `category`, `endpoint`, or `model`; search endpoint names or IDs with `--search`. Tidy CSV preserves each input row and duplicate, includes a status row for blank or invalid SMILES, and keeps numeric values and class-1 scores at full precision. The web app accepts up to 16 nonblank SMILES lines per submission.

## Python

```python
from velochem import Predictor

predictor = Predictor()  # one CPU ONNX session
raw = predictor.predict("CCO")
rows = predictor.predict_rows("CCO", category="Metabolism")
batch = predictor.predict_batch_rows(
    ["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
    categories=["Metabolism", "Toxicity and bioactivity"],
    sort_by="category",
)
```

`predict` and `predict_batch` retain the original raw API. The row APIs use the same 0.50 classification display cutoff, category filters and sort rules as the CLI and Gradio. Empty category selections include all categories; blank or unparsable inputs have an explicit status and no endpoint rows. Batch APIs preserve input order and duplicates.

## Scale notes

The Caco-2 TDC result is displayed as `log(Papp [cm/s])`, in agreement with the benchmark dataset description; it is not exponentiated. The released Deep-PK MDCK scale remains labeled `Recorded scale` because the theory page states nm/s while the fitted-target metadata does not document the transform. MDR1–MDCK is displayed on the logged efflux-ratio scale. These labels do not alter the numeric outputs. See the presentation catalogue for the longer notes.

The package uses ONNX Runtime's CPU provider with one intra-op and one inter-op thread. Inference does not require a network connection. Public redistribution of project code, ONNX weights and the project-derived catalog remains uncleared; see `DISTRIBUTION_RIGHTS.md` before any upload.
