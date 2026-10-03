"""Assignment 16: data drift and model performance reports with Evidently.

Reads the prediction log written by the API and compares every batch
against the reference batch.

Example:
    python monitor.py --log logs/predictions.csv --reference reference --out reports

For each batch it writes reports/monitoring_<batch>.html with:
  - Data drift: did the input images change? (brightness, contrast, ...)
  - Prediction drift: did the mix of predicted classes or the confidence change?
  - Model performance: accuracy, precision, recall, confusion matrix,
    only when the batch has true labels.
"""
import argparse
import os
import warnings

import pandas as pd
from evidently import DataDefinition, Dataset, MulticlassClassification, Report
from evidently.presets import ClassificationPreset, DataDriftPreset
from sklearn.exceptions import UndefinedMetricWarning

# Precision for a class the model never predicted is 0/0; sklearn warns
# about it on every metric. The report still shows it as 0, so hide the noise.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

FEATURES = ["brightness", "contrast", "sharpness", "mean_red", "mean_green", "mean_blue"]
DRIFT_COLUMNS = FEATURES + ["confidence", "prediction"]


def build_dataset(df: pd.DataFrame, with_labels: bool) -> Dataset:
    definition = DataDefinition(
        numerical_columns=FEATURES + ["confidence"],
        categorical_columns=["prediction"] + (["label"] if with_labels else []),
        classification=(
            [MulticlassClassification(target="label", prediction_labels="prediction")] if with_labels else None
        ),
    )
    columns = DRIFT_COLUMNS + (["label"] if with_labels else [])
    return Dataset.from_pandas(df[columns], data_definition=definition)


def summarize(snapshot) -> dict:
    """Pulls the headline numbers out of an Evidently result."""
    summary = {"drifted_columns": [], "accuracy": None}
    for metric in snapshot.dict()["metrics"]:
        name, value = metric["metric_name"], metric["value"]
        if name.startswith("DriftedColumnsCount"):
            summary["drift_share"] = value["share"]
        elif name.startswith("ValueDrift") and value < 0.05:
            # value is the p-value of the drift test; below 0.05 = drift detected
            summary["drifted_columns"].append(name.split("column=")[1].split(",")[0])
        elif name.startswith("Accuracy"):
            summary["accuracy"] = value
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/predictions.csv")
    parser.add_argument("--reference", default="reference", help="Batch name to use as the reference")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    log = pd.read_csv(args.log, keep_default_na=False)
    log["label"] = log["label"].replace("", None)
    reference = log[log["batch"] == args.reference]
    if reference.empty:
        raise SystemExit(f"No rows with batch '{args.reference}' in {args.log}")
    os.makedirs(args.out, exist_ok=True)

    print(f"Reference: '{args.reference}' ({len(reference)} predictions)\n")
    for batch, current in log[log["batch"] != args.reference].groupby("batch"):
        # Performance needs labels in BOTH batches; drift does not need labels at all
        with_labels = current["label"].notna().all() and reference["label"].notna().all()
        metrics = [DataDriftPreset(columns=DRIFT_COLUMNS)]
        if with_labels:
            metrics.append(ClassificationPreset())

        snapshot = Report(metrics).run(
            build_dataset(current, with_labels), build_dataset(reference, with_labels)
        )
        path = os.path.join(args.out, f"monitoring_{batch or 'untagged'}.html")
        snapshot.save_html(path)

        s = summarize(snapshot)
        print(f"Batch '{batch or 'untagged'}' ({len(current)} predictions) -> {path}")
        # Evidently's default rule: the whole dataset counts as drifted when at
        # least half of the monitored columns drifted. One column on its own can
        # be a false alarm, since each test has a 5% chance of a false positive.
        print(f"  dataset drift   : {'YES' if s['drift_share'] >= 0.5 else 'no'}")
        print(f"  drifted columns : {', '.join(s['drifted_columns']) or 'none'} "
              f"({s['drift_share']:.0%} of {len(DRIFT_COLUMNS)} monitored)")
        print(f"  mean confidence : {current['confidence'].mean():.3f} (reference {reference['confidence'].mean():.3f})")
        if s["accuracy"] is not None:
            ref_acc = (reference["label"] == reference["prediction"]).mean()
            print(f"  accuracy        : {s['accuracy']:.3f} (reference {ref_acc:.3f})")
        else:
            print("  accuracy        : not available (no labels in this batch)")
        print()


if __name__ == "__main__":
    main()
