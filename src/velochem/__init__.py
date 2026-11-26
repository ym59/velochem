"""VeloChem CPU prediction API."""

from .engine import B100Inference, DEFAULT_CATALOG, DEFAULT_MODEL
from .presentation import filter_sort_rows, format_endpoint_result, load_presentation_catalog

__version__ = "1.0.0"


class Predictor:
    """One-session CPU predictor for a single SMILES or a sequence of SMILES."""

    def __init__(self, model_path=DEFAULT_MODEL, catalog_path=DEFAULT_CATALOG,
                 batch_size: int = 32):
        self._engine = B100Inference(model_path, catalog_path, batch_size)
        self.catalog = self._engine.catalog
        self.endpoints = self._engine.endpoints

    def predict(self, smiles):
        """Predict one SMILES; blank and invalid values return an explicit status."""
        return self._engine.predict([smiles])[0]

    def predict_batch(self, smiles_rows):
        """Predict a sequence while preserving order, duplicates, and invalid rows."""
        return self._engine.predict(smiles_rows)

    def _format_record(self, result, row_index, **options):
        presentation = load_presentation_catalog()
        rows = []
        if result["status"] == "ok":
            rows = [format_endpoint_result(endpoint_id, value, presentation)
                    for endpoint_id, value in result["predictions"].items()]
            rows = filter_sort_rows(rows, catalog=presentation, **options)
        return {"input_row_id": row_index + 1, "row_index": row_index,
                "smiles": result["smiles"],
                "status": result["status"], "rows": rows}

    def predict_rows(self, smiles, *, category=None, categories=None,
                     sort_by="category", search=""):
        """Predict one input and return display-ready, filterable endpoint rows.

        Empty category selections include all categories. Blank or unparsable SMILES
        return an explicit status and an empty rows list.
        """
        result = self.predict(smiles)
        return self._format_record(result, 0, category=category, categories=categories,
                                   sort_by=sort_by, search=search)

    def predict_batch_rows(self, smiles_rows, *, category=None, categories=None,
                           sort_by="category", search=""):
        """Predict a sequence once, preserving duplicates, order, and invalid rows."""
        results = self.predict_batch(smiles_rows)
        options = {"category": category, "categories": categories,
                   "sort_by": sort_by, "search": search}
        return [self._format_record(result, index, **options)
                for index, result in enumerate(results)]
