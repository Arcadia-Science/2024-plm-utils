import os
import pathlib
import subprocess

import click
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics


def calc_metrics(y_true, y_pred_proba):
    """
    Calculate performance metrics for the given true and predicted labels.

    NOTE: this function is copied from the `smallesm` package.
    It cannot be imported from that package because we cannot assume the package
    is installed in the rnasamba env.
    """
    y_pred = (y_pred_proba > 0.5).astype(bool)
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)

    # `roc_auc_score` raises a ValueError if only one class is present in `y_true`.
    try:
        auc_roc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba)
    except ValueError as e:
        print("ValueError in `roc_auc_score`", e)
        auc_roc = np.nan

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "f1": f1,
        "auc_roc": auc_roc,
    }


def train(coding_train_filepath, noncoding_train_filepath, model_filepath, max_length=None):
    """
    Train an RNSamba model.

    TODO: implement filtering with the `max_length` parameter.
    """
    subprocess.run(
        [
            "rnasamba",
            "train",
            # the number of epochs after the lowest validation loss before stopping;
            # we use 1 to stop training quickly, as soon as the validation loss starts increasing.
            "--early_stopping",
            "1",
            "--verbose",
            "2",
            model_filepath,
            coding_train_filepath,
            noncoding_train_filepath,
        ]
    )


def evaluate(coding_test_filepath, noncoding_test_filepath, model_filepath, predictions_filepath):
    """
    Evaluate an RNSamba model.
    """

    test_filepaths = {
        "coding": coding_test_filepath,
        "noncoding": noncoding_test_filepath,
    }

    # temporary filepaths for the per-class predictions.
    prediction_filepaths = {
        "coding": "tmp-coding-predictions.tsv",
        "noncoding": "tmp-noncoding-predictions.tsv",
    }

    for kind in test_filepaths.keys():
        subprocess.run(
            [
                "rnasamba",
                "classify",
                prediction_filepaths[kind],
                test_filepaths[kind],
                model_filepath,
            ]
        )

    coding_predictions = pd.read_csv(prediction_filepaths["coding"], sep="\t")
    noncoding_predictions = pd.read_csv(prediction_filepaths["noncoding"], sep="\t")

    coding_predictions["true_label"] = "coding"
    noncoding_predictions["true_label"] = "noncoding"

    predictions = pd.concat([coding_predictions, noncoding_predictions])
    predictions.to_csv(predictions_filepath, index=False)

    for filepath in prediction_filepaths.values():
        os.remove(filepath)


@click.command()
@click.option("--dirpath", type=click.Path(exists=True, path_type=pathlib.Path), required=True)
@click.option("--dirname-suffix", type=str, required=True)
def command(dirpath, dirname_suffix):
    """
    Train and test an RNASamba model using all pairs of transcript dataset in the given directory.
    """
    timestamp = "2024-02-23"  # pd.Timestamp.now().strftime("%Y-%m-%d")

    models_dirpath = dirpath / f"{timestamp}-rnasamba-models-{dirname_suffix}"
    models_dirpath.mkdir(exist_ok=True)

    coding_dirpath = dirpath / f"cdna-coding-{dirname_suffix}"
    noncoding_dirpath = dirpath / f"ncrna-{dirname_suffix}"

    coding_filenames = sorted([path.stem for path in coding_dirpath.glob("*.fa")])
    noncoding_filenames = sorted([path.stem for path in noncoding_dirpath.glob("*.fa")])

    # sanity-check that the same files are present in both directories.
    assert set(coding_filenames) == set(noncoding_filenames)

    filenames = coding_filenames[:]
    for filename_train in filenames:
        model_filepath = models_dirpath / f"trained-on-{filename_train}.hdf5"

        if model_filepath.exists():
            print(f"Model '{model_filepath}' already exists; skipping training.")
        else:
            print(f"Training on '{filename_train}'")
            train(
                coding_train_filepath=coding_dirpath / f"{filename_train}.fa",
                noncoding_train_filepath=noncoding_dirpath / f"{filename_train}.fa",
                model_filepath=model_filepath,
            )

        for filename_test in filenames:
            print(f"Testing on '{filename_test}'")

            predictions_filepath = (
                models_dirpath / model_filepath.stem / f"{filename_test}-preds.tsv"
            )
            predictions_filepath.parent.mkdir(exist_ok=True, parents=True)
            if predictions_filepath.exists():
                print(f"Predictions '{predictions_filepath}' already exist; skipping evaluation.")
                continue

            evaluate(
                coding_test_filepath=coding_dirpath / f"{filename_test}.fa",
                noncoding_test_filepath=noncoding_dirpath / f"{filename_test}.fa",
                model_filepath=model_filepath,
                predictions_filepath=predictions_filepath,
            )


if __name__ == "__main__":
    command()
