"""Single-SMILES and CSV command line interfaces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import Predictor
from .engine import DEFAULT_CATALOG, DEFAULT_MODEL
from .presentation import filter_sort_rows, load_presentation_catalog


def _write_table(rows) -> None:
    columns = ("Category", "Endpoint", "Result")
    matrix = [[row["category"], row["endpoint"], row["result"]] for row in rows]
    widths = [max(len(columns[i]), *(len(str(row[i])) for row in matrix)) for i in range(3)]
    print("  ".join(columns[i].ljust(widths[i]) for i in range(3)))
    print("  ".join("-" * widths[i] for i in range(3)))
    for row in matrix:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(3)))
    print(f"\n{len(rows)} of 100 predictions")
    print("Class 1 is shown at a score of 0.50 or higher.")


def _csv_reader(path: Path, smiles_column: str):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        if not fields or fields.count(smiles_column) != 1:
            raise ValueError("CSV must contain exactly one declared SMILES column")
        rows = list(reader)
    return fields, rows


def _write_wide_csv(path: Path, fields, source_rows, results, endpoint_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        used = set(fields)

        def output_name(base: str) -> str:
            name, index = base, 2
            while name in used:
                name, index = f"{base}_{index}", index + 1
            used.add(name)
            return name

        status_column = output_name("status")
        source_ids = {entry["endpoint_id"] for entry in endpoint_rows}
        ordered_ids = [entry["endpoint_id"] for entry in endpoint_rows]
        prediction_columns = {endpoint_id: output_name(endpoint_id) for endpoint_id in ordered_ids}
        writer = csv.DictWriter(stream, fieldnames=fields + [status_column] + list(prediction_columns.values()))
        writer.writeheader()
        for source, result in zip(source_rows, results):
            row = dict(source)
            row[status_column] = result["status"]
            row.update({prediction_columns[key]: value for key, value in result["predictions"].items()
                        if key in source_ids})
            writer.writerow(row)


def _write_tidy_csv(path: Path, source_rows, smiles_column: str, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["input_row_id", "input_line", "smiles", "status", "category",
                  "endpoint_id", "endpoint", "task_type", "numeric_prediction",
                  "unit_or_scale", "class_label", "class1_score"]
    id_column = next((name for name in ("record_id", "row_id", "id")
                      if source_rows and name in source_rows[0]), None)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, (source, record) in enumerate(zip(source_rows, records), start=1):
            common = {"input_row_id": source.get(id_column, "") if id_column else index,
                      "input_line": index + 1, "smiles": source.get(smiles_column, ""),
                      "status": record["status"]}
            if record["status"] != "ok" or not record["rows"]:
                writer.writerow(common)
                continue
            for item in record["rows"]:
                writer.writerow({**common, "category": item["category"],
                                 "endpoint_id": item["endpoint_id"],
                                 "endpoint": item["endpoint"],
                                 "task_type": item["task_type"],
                                 "numeric_prediction": item["numeric_prediction"],
                                 "unit_or_scale": item["unit_or_scale"],
                                 "class_label": item["class_label"] or "",
                                 "class1_score": "" if item["class1_score"] is None else item["class1_score"]})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VeloChem B100 epoch-20 CPU predictor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", help="predict one SMILES")
    group.add_argument("--input-csv", type=Path, help="CSV file containing a SMILES column")
    parser.add_argument("--output-csv", type=Path, help="required with --input-csv")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--format", choices=("json", "table", "wide-csv", "tidy-csv"),
                        help="default preserves the original JSON or wide CSV output")
    parser.add_argument("--category", action="append", default=None,
                        help="category filter; repeat to include several categories")
    parser.add_argument("--sort-by", choices=("category", "endpoint", "model"), default="category")
    parser.add_argument("--search", default="", help="filter endpoint names or IDs")
    args = parser.parse_args(argv)
    predictor = Predictor(args.model, args.catalog, args.batch_size)

    if args.smiles is not None:
        if args.format in (None, "json") and args.format is None and not (args.category or args.search):
            row = predictor.predict(args.smiles)
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            return 0 if row["status"] == "ok" else 2
        record = predictor.predict_rows(args.smiles, categories=args.category,
                                        sort_by=args.sort_by, search=args.search)
        if args.format == "table":
            if record["status"] != "ok":
                print(f"Input status: {record['status']}")
                return 2
            _write_table(record["rows"])
        elif args.format in (None, "json"):
            print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        else:
            parser.error("--format wide-csv or tidy-csv requires --input-csv")
        return 0 if record["status"] == "ok" else 2

    if args.output_csv is None:
        parser.error("--output-csv is required with --input-csv")
    if args.format == "table" or args.format == "json":
        parser.error("--format table/json requires --smiles")
    try:
        fields, source_rows = _csv_reader(args.input_csv, args.smiles_column)
    except ValueError as exc:
        parser.error(str(exc))
    smiles_rows = [row.get(args.smiles_column, "") for row in source_rows]
    if args.format == "tidy-csv":
        records = predictor.predict_batch_rows(smiles_rows, categories=args.category,
                                               sort_by=args.sort_by, search=args.search)
        _write_tidy_csv(args.output_csv, source_rows, args.smiles_column, records)
        print(json.dumps({"status":"complete", "format":"tidy-csv", "rows":len(source_rows),
                          "endpoint_rows":sum(len(row["rows"]) for row in records),
                          "invalid_rows":sum(row["status"] != "ok" for row in records),
                          "output":str(args.output_csv)}, ensure_ascii=False))
        return 0

    results = predictor.predict_batch(smiles_rows)
    presentation = load_presentation_catalog()
    endpoint_rows = [{"endpoint_id":endpoint_id, "output_index":entry["output_index"],
                      "endpoint":entry["display_name"], "category":entry["category"]}
                     for endpoint_id, entry in presentation["endpoints"].items()]
    # Original invocations keep the original model order and all output columns.
    if args.format is None and args.category is None and not args.search:
        endpoint_rows = sorted(endpoint_rows, key=lambda row: row["output_index"])
    else:
        endpoint_rows = filter_sort_rows(endpoint_rows, categories=args.category,
                                         sort_by=args.sort_by, search=args.search)
    _write_wide_csv(args.output_csv, fields, source_rows, results, endpoint_rows)
    print(json.dumps({"status":"complete", "format":"wide-csv", "rows":len(source_rows),
                      "valid_rows":sum(row["status"] == "ok" for row in results),
                      "invalid_rows":sum(row["status"] != "ok" for row in results),
                      "endpoint_columns":len(endpoint_rows), "output":str(args.output_csv)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
