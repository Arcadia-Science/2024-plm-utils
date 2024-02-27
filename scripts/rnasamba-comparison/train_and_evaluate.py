import os
import pathlib
import subprocess

import click
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
from Bio import SeqIO


def filter_transcripts_by_length(input_filepath, output_filepath, max_length):
    """
    filter the given fasta file to remove any sequences that are longer than the max_length.
    """
    with open(output_filepath, "w") as file_out:
        for record in SeqIO.parse(input_filepath, "fasta"):
            if len(record.seq) <= max_length:
                SeqIO.write(record, file_out, "fasta")


def deleted_filtered_files(dirpaths):
    """
    Delete all files in the given directories that have the suffix '-filtered'
    (these are presumed to be files created by `filter_transcripts_by_length`).
    """
    for dirpath in dirpaths:
        for filepath in dirpath.glob("*-filtered.fa"):
            os.remove(filepath)


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


def train(coding_train_filepath, noncoding_train_filepath, model_filepath):
    """
    Train an RNASamba model.

    Note: we call RNASamba using the command-line interface because there appears to be
    a memory leak when calling the RNASamba API from Python.
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
@click.option(
    "--coding-dirpath", type=click.Path(exists=True, path_type=pathlib.Path), required=True
)
@click.option(
    "--noncoding-dirpath", type=click.Path(exists=True, path_type=pathlib.Path), required=True
)
@click.option(
    "--output-dirpath", type=click.Path(exists=False, path_type=pathlib.Path), required=True
)
@click.option(
    "--max-length", type=int, required=False, help="Maximum length of the peptides to use"
)
def command(coding_dirpath, noncoding_dirpath, output_dirpath, max_length):
    """
    Train and test an RNASamba model using all pairs of transcript dataset in the given directory.
    """
    output_dirpath.mkdir(exist_ok=True, parents=True)

    # delete any filtered files that may have been created in a previous call to this function.
    deleted_filtered_files([coding_dirpath, noncoding_dirpath])

    coding_filenames = sorted([path.stem for path in coding_dirpath.glob("*.fa")])
    noncoding_filenames = sorted([path.stem for path in noncoding_dirpath.glob("*.fa")])

    if max_length is not None:
        for filename in coding_filenames:
            filter_transcripts_by_length(
                input_filepath=coding_dirpath / f"{filename}.fa",
                output_filepath=coding_dirpath / f"{filename}-filtered.fa",
                max_length=max_length,
            )
        for filename in noncoding_filenames:
            filter_transcripts_by_length(
                input_filepath=noncoding_dirpath / f"{filename}.fa",
                output_filepath=noncoding_dirpath / f"{filename}-filtered.fa",
                max_length=max_length,
            )

        coding_filenames = [f"{filename}-filtered" for filename in coding_filenames]
        noncoding_filenames = [f"{filename}-filtered" for filename in noncoding_filenames]

    # sanity-check that the same files are present in both directories.
    assert set(coding_filenames) == set(noncoding_filenames)
    filenames = coding_filenames

    for train_filename in filenames:
        model_filepath = output_dirpath / f"trained-on-{train_filename}.hdf5"

        if model_filepath.exists():
            print(f"Model '{model_filepath}' already exists; skipping training.")
        else:
            print(f"Training on '{train_filename}'")
            train(
                coding_train_filepath=coding_dirpath / f"{train_filename}.fa",
                noncoding_train_filepath=noncoding_dirpath / f"{train_filename}.fa",
                model_filepath=model_filepath,
            )

        for test_filename in filenames:
            print(f"Testing on '{test_filename}'")

            predictions_filepath = (
                output_dirpath / model_filepath.stem / f"{test_filename}-preds.tsv"
            )
            predictions_filepath.parent.mkdir(exist_ok=True, parents=True)
            if predictions_filepath.exists():
                print(f"Predictions '{predictions_filepath}' already exist; skipping evaluation.")
                continue

            evaluate(
                coding_test_filepath=coding_dirpath / f"{test_filename}.fa",
                noncoding_test_filepath=noncoding_dirpath / f"{test_filename}.fa",
                model_filepath=model_filepath,
                predictions_filepath=predictions_filepath,
            )

    deleted_filtered_files([coding_dirpath, noncoding_dirpath])


if __name__ == "__main__":
    command()
