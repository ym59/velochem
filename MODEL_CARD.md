# VeloChem B100 CPU model card

- **Release:** VeloChem 1.0.0, CPU inference package
- **Model:** selected B100 seed-11, epoch-20 student
- **Representation:** FP32 ONNX; ONNX Runtime CPU provider
- **Outputs:** 100 in frozen order (68 classification, 32 regression)
- **Training checkpoint SHA-256:** `806c7d220dac89040530e1bf99e4c05e0dedcd903636b05e164b959bcf2c6311`
- **ONNX SHA-256:** `ceeb2ee754181139aac545408fdaecfaeb7b1818833d3b28c9ab719109026f0f`
- **Catalog:** bundled `velochem/catalog.json`; endpoint order SHA-256 `9ef03e69933d41d33b58b5871c2937e7b7014de46b1fcebdae225d6be1d15b85`

## Intended use

Research inference of the admitted molecular ADMET endpoints from SMILES. The package is not a clinical, regulatory, or experimental substitute. Predictions should be interpreted with the endpoint definitions and provenance in the catalog and manuscript supplement.

## Outputs and interpretation

Classification heads return probabilities for numeric source class 1. Biological polarity is unresolved for 14 outputs in the saved metadata and is not guessed. Regression heads return values after the saved inverse transform to their recorded source scales; the catalog marks any unit ambiguity. `blank_smiles` and `invalid_smiles` rows carry no prediction values.

## Validation evidence

The saved CPU PyTorch/ONNX comparison covered 25,600 values over the approved public 256-molecule fixture at `atol=1e-5`, `rtol=1e-4`, with zero failing endpoints. The package smoke tests do not establish generalization beyond the reported development and retrospective evaluations. Official-test and external-validation outcomes are not redistributed in this model package.

## Data and rights

Training/evaluation source citations are in manuscript Table 1's source inventory and Supplementary Information Table S1. Underlying datasets and per-observation records are not bundled. Public redistribution rights for code, weights, and the project-derived catalog remain unresolved; this local package is not cleared for public release.
