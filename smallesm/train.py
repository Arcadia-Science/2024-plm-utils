import pathlib

import click
import numpy as np
import pandas as pd
from Bio import SeqIO

from smallesm import models

RANDOM_STATE = 42


def load_and_filter_embeddings(embeddings_filepath, fasta_filepath=None, max_length=None):
    """
    Load the embeddings and, if a fasta_filepath and max_length is provided,
    filter the embeddings to remove any that correspond to sequences
    that are longer than the max_length.
    """
    embeddings_filepath = pathlib.Path(embeddings_filepath)
    embeddings = np.load(embeddings_filepath)

    if max_length is None:
        return embeddings

    if fasta_filepath is None:
        raise ValueError("The fasta_filepath must be provided if the max_length is not None.")

    filtered_inds = [
        ind
        for ind, record in enumerate(SeqIO.parse(fasta_filepath, "fasta"))
        if len(record.seq) <= max_length
    ]
    return embeddings[filtered_inds, :]


def load_embeddings_and_create_labels(
    coding_embeddings_filepath,
    noncoding_embeddings_filepath,
    coding_fasta_filepath=None,
    noncoding_fasta_filepath=None,
    max_length=None,
):
    """
    Load embeddings from the given filepaths and create labels for coding/noncoding.
    """
    embeddings_coding = load_and_filter_embeddings(
        coding_embeddings_filepath, coding_fasta_filepath, max_length
    )
    embeddings_noncoding = load_and_filter_embeddings(
        noncoding_embeddings_filepath, noncoding_fasta_filepath, max_length
    )

    # create labels for the embeddings using 1 for 'coding' and 0 for 'noncoding'.
    labels_coding = np.ones(embeddings_coding.shape[0])
    labels_noncoding = np.zeros(embeddings_noncoding.shape[0])

    labels_all = np.concatenate([labels_coding, labels_noncoding])
    embeddings_all = np.concatenate([embeddings_coding, embeddings_noncoding], axis=0)

    return embeddings_all, labels_all


@click.command()
@click.option("--coding-filepath", type=click.Path(exists=True), required=True)
@click.option("--noncoding-filepath", type=click.Path(exists=True), required=True)
@click.option("--model-dirpath", type=click.Path(exists=False), required=False)
def train_command(coding_filepath, noncoding_filepath, model_dirpath):
    """
    Train a classifier using embeddings of peptides from coding and noncoding transcripts
    and print the validation and test metrics.

    Note: we do not allow the user to specify a max_length with this command
    because it requires providing the paths to the corresponding fasta files.
    """
    x_all, y_all = load_embeddings_and_create_labels(
        coding_embeddings_filepath=coding_filepath,
        noncoding_embeddings_filepath=noncoding_filepath,
    )
    model = models.EmbeddingsClassifier.init(verbose=True)
    model.train(x_all, y_all)

    if model_dirpath is not None:
        model.save(model_dirpath)
        print(f"Model saved to '{model_dirpath}'")
    return model


@click.command()
@click.option("--model-dirpath", type=click.Path(exists=True), required=True)
@click.option("--embeddings-filepath", type=click.Path(exists=True), required=True)
@click.option("--fasta-filepath", type=click.Path(exists=True), required=False)
@click.option("--output-filepath", type=click.Path(exists=False), required=True)
def predict_command(model_dirpath, embeddings_filepath, fasta_filepath, output_filepath):
    """
    Predict the labels for an embeddings matrix and a saved model,
    and write the predictions to the output filepath as a CSV.
    If a fasta filepath is provided, the sequence IDs will be included in the CSV file.
    """
    x = np.load(embeddings_filepath)
    model = models.EmbeddingsClassifier.load(model_dirpath)
    predicted_probabilities = model.predict_proba(x)[:, 1]
    predicted_labels = ["coding" if p > 0.5 else "noncoding" for p in predicted_probabilities]

    predictions = pd.DataFrame(
        {"predicted_label": predicted_labels, "predicted_probability": predicted_probabilities}
    )
    predictions["sequence_id"] = None

    if fasta_filepath is not None:
        records = list(SeqIO.parse(fasta_filepath, "fasta"))
        if len(records) != len(predictions):
            raise ValueError(
                f"The number of records in the given fasta file ({len(records)}) "
                f"does not match the number of predictions ({len(predictions)})."
            )
        # we assume that the order of the records in the fasta file matches the order
        # of the predictions (that is, the order of the rows in the embeddings matrix).
        for ind, _ in predictions.iterrows():
            predictions.at[ind, "sequence_id"] = records[ind].id

    # reorder columns just for readability.
    predictions = predictions[["sequence_id", "predicted_probability", "predicted_label"]]
    predictions.to_csv(output_filepath, index=False, float_format="%.2f")
    print(f"Predictions saved to '{output_filepath}'")
