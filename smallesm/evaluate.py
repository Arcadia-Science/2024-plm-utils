import pathlib

import click
import pandas as pd

from smallesm.train import train


@click.command()
@click.option(
    "--embeddings-dirpath", type=click.Path(exists=True, path_type=pathlib.Path), required=True
)
@click.option("--embeddings-suffix", type=str, required=True)
def command(embeddings_suffix, embeddings_dirpath):
    """
    Train and test a model using all pairs of embeddings in the given directory.
    """

    coding_dirpath = embeddings_dirpath / f"cdna-coding-{embeddings_suffix}"
    noncoding_dirpath = embeddings_dirpath / f"ncrna-{embeddings_suffix}"

    coding_filenames = sorted([path.stem for path in coding_dirpath.glob("*.npy")])
    noncoding_filenames = sorted([path.stem for path in noncoding_dirpath.glob("*.npy")])

    # sanity-check that the same files are present in both directories.
    assert set(coding_filenames) == set(noncoding_filenames)

    results = []
    for filename_train in coding_filenames:
        for filename_test in coding_filenames:
            print(f"Training on '{filename_train}' and testing on '{filename_test}'")
            test_metrics = train(
                coding_train_filepath=coding_dirpath / f"{filename_train}.npy",
                noncoding_train_filepath=noncoding_dirpath / f"{filename_train}.npy",
                coding_test_filepath=coding_dirpath / f"{filename_test}.npy",
                noncoding_test_filepath=noncoding_dirpath / f"{filename_test}.npy",
            )
            results.append(
                dict(filename_train=filename_train, filename_test=filename_test, **test_metrics)
            )

    results = pd.DataFrame(results)
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    results.to_csv(
        embeddings_dirpath / f"{timestamp}-smallesm-results-{embeddings_suffix}.csv", index=False
    )
