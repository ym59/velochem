"""Shared CPU ONNX Runtime inference path for the CLI and local app."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnxruntime as ort
from rdkit import RDLogger

from .features import collate_graphs, tokenize_smiles

RDLogger.DisableLog("rdApp.*")
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PACKAGE_ROOT / "assets" / "b100_epoch20_fp32.onnx"
DEFAULT_CATALOG = PACKAGE_ROOT / "catalog.json"


class B100Inference:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL,
                 catalog_path: str | Path = DEFAULT_CATALOG,
                 batch_size: int = 32) -> None:
        self.catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.endpoints = self.catalog["endpoints"]
        if len(self.endpoints) != 100 or [e["output_index"] for e in self.endpoints] != list(range(100)):
            raise ValueError("catalog must contain the frozen 100-output order")
        self.batch_size = max(1, int(batch_size))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("inference must use CPUExecutionProvider only")
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name
        expected = ["atom_categories", "environment_ids", "relations", "atom_mask"]
        if self.input_names != expected:
            raise ValueError("ONNX input schema differs from the package contract")

    def _native(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64).copy()
        for endpoint in self.endpoints:
            j = endpoint["output_index"]
            if endpoint["task_type"] == "classification":
                values[:, j] = 1.0 / (1.0 + np.exp(-np.clip(values[:, j], -80.0, 80.0)))
            else:
                transform = endpoint["output_transform"]
                values[:, j] = values[:, j] * float(transform["scale"]) + float(transform["mean"])
        return values

    def predict(self, smiles_rows: Iterable[Any]) -> list[dict[str, Any]]:
        inputs = list(smiles_rows)
        output: list[dict[str, Any] | None] = [None] * len(inputs)
        valid_indices: list[int] = []
        graphs = []
        for i, value in enumerate(inputs):
            if value is None or not isinstance(value, str) or not value.strip():
                output[i] = self._status_row(i, value, "blank_smiles")
                continue
            try:
                graph = tokenize_smiles(value)
            except Exception:
                output[i] = self._status_row(i, value, "invalid_smiles")
                continue
            valid_indices.append(i)
            graphs.append(graph)
        for start in range(0, len(graphs), self.batch_size):
            stop = min(start + self.batch_size, len(graphs))
            batch = collate_graphs(graphs[start:stop])
            feed = {name: value for name, value in zip(self.input_names, batch)}
            raw = self.session.run([self.output_name], feed)[0]
            native = self._native(raw)
            for offset, values in enumerate(native):
                original_index = valid_indices[start + offset]
                output[original_index] = {
                    "row_index": original_index,
                    "smiles": inputs[original_index],
                    "status": "ok",
                    "predictions": {e["endpoint_id"]: float(values[e["output_index"]])
                                    for e in self.endpoints},
                }
        return [row for row in output if row is not None]

    @staticmethod
    def _status_row(index: int, value: Any, status: str) -> dict[str, Any]:
        return {"row_index": index, "smiles": value if isinstance(value, str) else "",
                "status": status, "predictions": {}}
