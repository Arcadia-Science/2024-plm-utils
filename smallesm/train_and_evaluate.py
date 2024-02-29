import pathlib

import click
import numpy as np
import pandas as pd
from Bio import SeqIO

from smallesm import models
from smallesm.train import load_embeddings_and_create_labels


def fasta_filepath_from_embedding_filepath(embedding_filepath):
    """
    Given the path to an embedding file, return the path to the fasta file
    from which the sequences were taken, assuming the directory structure
    created by `smallesm datasets construct`.
    """
    return embedding_filepath.parent.parent.parent / "peptides" / f"{embedding_filepath.stem}.fa"


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
    Train and test models using all pairs of embedding matrices in the given directories.
    These directories are assumed to have been created via `smallesm datasets construct`.

    Parameters
    ----------
    coding_dirpath, noncoding_dirpath: Paths to directories of embedding matrices
        of coding and noncoding sequences, saved as numpy files.
    output_filepath: Path to the CSV file to save the test metrics to.
    max_length: Maximum length of the peptides to use during training and testing.
    """
    output_dirpath.mkdir(exist_ok=True, parents=True)
    coding_filenames = sorted([path.stem for path in coding_dirpath.glob("*.npy")])
    noncoding_filenames = sorted([path.stem for path in noncoding_dirpath.glob("*.npy")])

    # sanity-check that the same files are present in both directories.
    assert set(coding_filenames) == set(noncoding_filenames)
    filenames = coding_filenames

    for filename_train in filenames:
        print(f"Training on '{filename_train}'")

        coding_embeddings_filepath = coding_dirpath / f"{filename_train}.npy"
        noncoding_embeddings_filepath = noncoding_dirpath / f"{filename_train}.npy"

        x, y = load_embeddings_and_create_labels(
            coding_embeddings_filepath=coding_embeddings_filepath,
            noncoding_embeddings_filepath=noncoding_embeddings_filepath,
            coding_fasta_filepath=fasta_filepath_from_embedding_filepath(
                coding_embeddings_filepath
            ),
            noncoding_fasta_filepath=fasta_filepath_from_embedding_filepath(
                noncoding_embeddings_filepath
            ),
            max_length=max_length,
        )

        model = models.EmbeddingsClassifier.init(verbose=True)
        model.train(x, y)

        for filename_test in filenames:
            coding_embeddings_filepath = coding_dirpath / f"{filename_test}.npy"
            noncoding_embeddings_filepath = noncoding_dirpath / f"{filename_test}.npy"

            x_coding = np.load(coding_embeddings_filepath)
            x_noncoding = np.load(noncoding_embeddings_filepath)

            preds_coding = model.predict_proba(x_coding)[:, 1]
            preds_noncoding = model.predict_proba(x_noncoding)[:, 1]

            records_coding = SeqIO.parse(
                fasta_filepath_from_embedding_filepath(coding_embeddings_filepath), "fasta"
            )
            records_noncoding = SeqIO.parse(
                fasta_filepath_from_embedding_filepath(noncoding_embeddings_filepath), "fasta"
            )

            predictions = pd.DataFrame(
                {
                    "sequence_length": [
                        len(record.seq) for record in list(records_coding) + list(records_noncoding)
                    ],
                    "true_label": (
                        ["coding"] * len(records_coding) + ["noncoding"] * len(records_noncoding)
                    ),
                    "predicted_probability": list(preds_coding) + list(preds_noncoding),
                }
            )

            predictions.to_csv(
                output_dirpath / f"trained-on-{filename_train}-tested-on-{filename_test}-preds.csv",
                index=False,
            )
