import pathlib

import numpy as np
import pandas as pd
from click.testing import CliRunner

from plmutils import classify


def test_train_and_predict(tmpdir):
    """
    Test that the `train` and `predict` commands work together as expected.
    """
    runner = CliRunner()
    model_dirpath = pathlib.Path(tmpdir) / "model"
    model_dirpath.mkdir()

    positive_class_embeddings_filepath = pathlib.Path(tmpdir) / "embeddings_positive_class.npy"
    negative_class_embeddings_filepath = pathlib.Path(tmpdir) / "embeddings_negative_class.npy"

    num_sequences = 50
    np.save(positive_class_embeddings_filepath, np.random.rand(num_sequences, 320))
    np.save(negative_class_embeddings_filepath, np.random.rand(num_sequences, 320))

    runner.invoke(
        classify.train_command,
        [
            "--positive-class-filepath",
            positive_class_embeddings_filepath,
            "--negative-class-filepath",
            negative_class_embeddings_filepath,
            "--model-dirpath",
            model_dirpath,
            "--n-pcs",
            10,
        ],
    )

    model_filepath = model_dirpath / "model.joblib"
    assert model_filepath.exists()

    fasta_filepath = pathlib.Path(tmpdir) / "sequences.fa"
    sequence_ids = [f"seq-{ind}" for ind in range(num_sequences)]
    with open(fasta_filepath, "w") as file:
        for sequence_id in sequence_ids:
            file.write(f">{sequence_id}\n")
            file.write("ACGT\n")

    output_filepath = pathlib.Path(tmpdir) / "predictions.csv"
    runner.invoke(
        classify.predict_command,
        [
            "--model-dirpath",
            model_dirpath,
            "--embeddings-filepath",
            positive_class_embeddings_filepath,
            "--fasta-filepath",
            fasta_filepath,
            "--output-filepath",
            output_filepath,
        ],
    )

    # Check that the predictions were written to the output filepath.
    assert output_filepath.exists()

    # Check that the predictions have the expected format.
    predictions = pd.read_csv(output_filepath)
    assert predictions.shape[0] == num_sequences
    assert set(predictions["sequence_id"]) == set(sequence_ids)
    assert all(predictions["predicted_label"].isin(["positive", "negative"]))

    # Check that the probabilities are between 0 and 1 and are not all the same.
    assert all(predictions["predicted_probability"].between(0, 1))
    assert predictions["predicted_probability"].sum() > 0
    assert predictions["predicted_probability"].sum() < num_sequences
