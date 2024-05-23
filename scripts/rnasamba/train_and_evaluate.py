import os
import pathlib
import subprocess

import click
import pandas as pd
from Bio import SeqIO


def intersect_fasta_files(input_filepaths, output_filepath):
    """
    Extract the sequences whose ids appear in both of the input FASTA files.

    TODO: this is copied from `plmutils.tasks.classify_orfs` but cannot be imported from that module
    because the plmutils package cannot installed in the env in which this script is run.
    """
    input_filepath_1, input_filepath_2 = input_filepaths

    tmp_ids_filepath = input_filepath_1.with_suffix(".ids")
    command = f"seqkit seq {input_filepath_1} --name --only-id > {tmp_ids_filepath}"
    subprocess.run(command, shell=True)

    command = f"seqkit grep -f {tmp_ids_filepath} {input_filepath_2} -o {output_filepath}"
    subprocess.run(command, shell=True)
    os.remove(tmp_ids_filepath)


def filter_transcripts_by_longest_peptide_length(
    transcripts_fasta_filepath, output_filepath, max_length
):
    """
    filter the given fasta file of transcripts to remove any transcripts whose longest putative ORF
    is longer than the max_length.
    """
    # TODO: this path to the fasta file of longest putative ORFs from each transcript
    # is hard-coded according to the directory structure created by
    # `plmutils classify-orfs construct-data`.
    peptides_fasta_filepath = (
        transcripts_fasta_filepath.parent.parent
        / "peptides"
        / f"{transcripts_fasta_filepath.stem}.fa"
    )

    # we first filter the peptides fasta file by length to generate a fasta file
    # with which to filter the transcripts.
    short_peptides_fasta_filepath = peptides_fasta_filepath.with_suffix(".short.fa")
    filter_sequences_by_length(
        input_filepath=peptides_fasta_filepath,
        output_filepath=short_peptides_fasta_filepath,
        max_length=max_length,
    )
    intersect_fasta_files(
        [transcripts_fasta_filepath, short_peptides_fasta_filepath], output_filepath
    )
    os.remove(short_peptides_fasta_filepath)


def filter_sequences_by_length(input_filepath, output_filepath, max_length):
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


def train_rnasamba(coding_train_filepath, noncoding_train_filepath, model_filepath):
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


def evaluate_rnasamba(
    coding_test_filepath, noncoding_test_filepath, model_filepath, predictions_filepath
):
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
    Train an RNASamba model on each dataset in the given directories,
    then generate predictions for each dataset using each model to generate
    a complete matrix of train-test results for all pairs of datasets.

    Each 'dataset' consists of a pair of fasta files of transcripts, each having the same name,
    one in the `coding_dirpath` directory containing coding transcripts,
    and one in `noncoding_dirpath` containing noncoding transcripts.

    max_length: used to filter the transcripts used for training and evaluation by dropping
    transcripts whose longest putative ORF is longer than this length (in amino acids).
    """
    output_dirpath.mkdir(exist_ok=True, parents=True)

    # delete any filtered files that may have been created in a previous call to this function.
    deleted_filtered_files([coding_dirpath, noncoding_dirpath])

    coding_filenames = sorted([path.stem for path in coding_dirpath.glob("*.fa")])
    noncoding_filenames = sorted([path.stem for path in noncoding_dirpath.glob("*.fa")])

    if max_length is not None:
        for dirpath, filenames in [
            (coding_dirpath, coding_filenames),
            (noncoding_dirpath, noncoding_filenames),
        ]:
            for filename in filenames:
                filter_transcripts_by_longest_peptide_length(
                    transcripts_fasta_filepath=dirpath / f"{filename}.fa",
                    output_filepath=dirpath / f"{filename}-filtered.fa",
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
            train_rnasamba(
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

            evaluate_rnasamba(
                coding_test_filepath=coding_dirpath / f"{test_filename}.fa",
                noncoding_test_filepath=noncoding_dirpath / f"{test_filename}.fa",
                model_filepath=model_filepath,
                predictions_filepath=predictions_filepath,
            )

    deleted_filtered_files([coding_dirpath, noncoding_dirpath])


if __name__ == "__main__":
    command()
