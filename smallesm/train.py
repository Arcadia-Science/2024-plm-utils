import click
import numpy as np
import pandas as pd
from Bio import SeqIO

from smallesm import models

RANDOM_STATE = 42


def filter_embeddings_by_sequence_length(embeddings, fasta_filepath, max_length):
    """
    Filter the embeddings matrix to remove rows that correspond to sequences
    that are longer than `max_length`.
    """
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
    Load embedding matrices from the given filepaths, filter them by sequence length
    if a max_length was provided, create an array of labels for coding/noncoding,
    and return a single concatenated embedding matrix and labels array.
    """
    embeddings_coding = np.load(coding_embeddings_filepath)
    embeddings_noncoding = np.load(noncoding_embeddings_filepath)

    if max_length is not None:
        if coding_fasta_filepath is None or noncoding_fasta_filepath is None:
            raise ValueError("FASTA files must be provided if max_length is not None.")
        embeddings_coding = filter_embeddings_by_sequence_length(
            embeddings_coding, coding_fasta_filepath, max_length
        )
        embeddings_noncoding = filter_embeddings_by_sequence_length(
            embeddings_noncoding, noncoding_fasta_filepath, max_length
        )

    # create labels for the embeddings using 1 for 'coding' and 0 for 'noncoding'.
    labels_coding = np.ones(embeddings_coding.shape[0])
    labels_noncoding = np.zeros(embeddings_noncoding.shape[0])

    labels_all = np.concatenate([labels_coding, labels_noncoding])
    embeddings_all = np.concatenate([embeddings_coding, embeddings_noncoding], axis=0)

    return embeddings_all, labels_all


@click.command()
@click.option(
    "--coding-filepath",
    type=click.Path(exists=True),
    required=True,
    help="Path to a numpy file of embeddings of ORFs from coding transcripts.",
)
@click.option(
    "--noncoding-filepath",
    type=click.Path(exists=True),
    required=True,
    help="Path to a numpy file of embeddings of ORFs from noncoding transcripts.",
)
@click.option(
    "--model-dirpath",
    type=click.Path(exists=False),
    required=False,
    help="Path to the directory to which the trained model will be saved.",
)
def train_command(coding_filepath, noncoding_filepath, model_dirpath):
    """
    Train a classifier using embeddings of peptides from coding and noncoding transcripts,
    print the validation metrics, and save the trained model to a directory if one is provided.

    TODO: currently this command does not allow the user to specify a maximum sequence length
    by which to filter the embeddings. Doing so would require adding CLI options to specify
    both a max length and the fasta filepaths of the coding and noncoding sequences
    corresponding to the embedding matrices, as the sequence length
    is not included in the embeddings files themselves.
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


@click.command()
@click.option(
    "--model-dirpath", type=click.Path(exists=True), required=True, help="Path to a saved model."
)
@click.option(
    "--embeddings-filepath",
    type=click.Path(exists=True),
    required=True,
    help="Path to a numpy file of embeddings.",
)
@click.option(
    "--fasta-filepath",
    type=click.Path(exists=True),
    required=False,
    help="Path to the fasta file of sequences to which the embeddings correspond.",
)
@click.option(
    "--output-filepath",
    type=click.Path(exists=False),
    required=True,
    help="Path to which to save the CSV of coding/noncoding predictions.",
)
def predict_command(model_dirpath, embeddings_filepath, fasta_filepath, output_filepath):
    """
    Predict the labels for the given embeddings matrix and saved model,
    append the sequence IDs to the resulting predictions (if a fasta filepath is provided),
    and write the predictions to the output filepath as a CSV.
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

    # reorder columns, just for the sake of readability.
    predictions = predictions[["sequence_id", "predicted_probability", "predicted_label"]]
    predictions.to_csv(output_filepath, index=False, float_format="%.2f")
    print(f"Predictions saved to '{output_filepath}'")
