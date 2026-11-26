"""Shared endpoint display metadata and result-row formatting."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

DEFAULT_PRESENTATION_CATALOG = Path(__file__).with_name("presentation_catalog.json")
CLASS_CUTOFF = 0.50


@lru_cache(maxsize=4)
def _read_catalog(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    if catalog.get("schema") != "velochem-presentation-catalog/1":
        raise ValueError("unsupported VeloChem presentation catalog")
    endpoints = catalog.get("endpoints")
    if not isinstance(endpoints, dict) or len(endpoints) != catalog.get("endpoint_count"):
        raise ValueError("presentation catalog endpoint count mismatch")
    indices = [entry.get("output_index") for entry in endpoints.values()]
    if sorted(indices) != list(range(len(indices))):
        raise ValueError("presentation catalog output indices are not unique and contiguous")
    if any(key != entry.get("endpoint_id") for key, entry in endpoints.items()):
        raise ValueError("presentation catalog endpoint key mismatch")
    return catalog


def load_presentation_catalog(path: str | Path | None = None) -> dict:
    """Load bundled display metadata, optionally from a test catalog."""
    return _read_catalog(str(Path(path or DEFAULT_PRESENTATION_CATALOG).resolve()))


def category_counts(catalog: Mapping | None = None) -> dict[str, int]:
    catalog = catalog or load_presentation_catalog()
    counts = {name: 0 for name in catalog["category_order"]}
    for entry in catalog["endpoints"].values():
        counts[entry["category"]] += 1
    return counts


def format_endpoint_result(endpoint_id: str, value: float,
                           catalog: Mapping | None = None,
                           cutoff: float = CLASS_CUTOFF) -> dict:
    """Format one raw native prediction without changing its numeric scale."""
    catalog = catalog or load_presentation_catalog()
    try:
        entry = catalog["endpoints"][endpoint_id]
    except KeyError as exc:
        raise ValueError(f"unknown endpoint ID: {endpoint_id}") from exc
    numeric = float(value)
    result = {
        "endpoint_id": endpoint_id,
        "output_index": int(entry["output_index"]),
        "endpoint": entry["display_name"],
        "display_name": entry["display_name"],
        "category": entry["category"],
        "task_type": entry["task_type"],
        "numeric_prediction": numeric,
        "unit_or_scale": entry["unit_or_scale"],
        "class_label": None,
        "class1_score": None,
    }
    if entry["task_type"] == "classification":
        call = int(numeric >= cutoff)
        labels = entry.get("class_labels") or ["Class 0", "Class 1"]
        result["numeric_prediction"] = call
        result["class_label"] = labels[call]
        result["class1_score"] = numeric
        result["result"] = labels[call]
    else:
        unit = entry.get("unit_or_scale") or "Recorded scale"
        result["result"] = f"{numeric:.5g} {unit}"
        limits = entry.get("physical_range")
        if limits is not None and ((limits.get("minimum") is not None and numeric < limits["minimum"])
                                   or (limits.get("maximum") is not None and numeric > limits["maximum"])):
            note = "Outside documented range; value shown unchanged."
            result["range_note"] = note
            result["result"] += " — outside documented range; shown unchanged"
        else:
            result["range_note"] = ""
    return result


def filter_sort_rows(rows: Iterable[Mapping], *, categories=None,
                     category: str | None = None, search: str = "",
                     sort_by: str = "category", catalog: Mapping | None = None
                     ) -> list[dict]:
    """Filter and sort formatted rows. Empty categories means all categories."""
    catalog = catalog or load_presentation_catalog()
    selected = set(categories or ())
    if isinstance(categories, str):
        selected = {categories} if categories else set()
    if category:
        selected.add(category)
    unknown = selected.difference(catalog["category_order"])
    if unknown:
        raise ValueError(f"unknown category: {sorted(unknown)[0]}")
    term = str(search or "").strip().casefold()
    category_index = {name: i for i, name in enumerate(catalog["category_order"])}
    result = []
    for row in rows:
        if selected and row["category"] not in selected:
            continue
        if term and term not in (row["endpoint"] + " " + row["endpoint_id"]).casefold():
            continue
        result.append(dict(row))
    if sort_by == "model":
        key = lambda row: row["output_index"]
    elif sort_by == "endpoint":
        key = lambda row: (row["endpoint"].casefold(), row["output_index"])
    elif sort_by == "category":
        key = lambda row: (category_index[row["category"]], row["endpoint"].casefold(),
                           row["output_index"])
    else:
        raise ValueError("sort_by must be 'category', 'endpoint', or 'model'")
    result.sort(key=key)
    return result
