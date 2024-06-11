import numpy as np
import pytest
from Bio import SeqIO

from plmutils import embed


def test_embed(artifacts_dirpath, tmpdir):
    """
    Test that the embed function generates embeddings from a realistic FASTA file
    of peptide sequences.
    """
    model_name = "esm2_t6_8M_UR50D"
    fasta_filepath = artifacts_dirpath / "peptides.fa"
    output_filepath = tmpdir / "embeddings.npy"

    embed.embed(
        fasta_filepath=fasta_filepath,
        output_filepath=output_filepath,
        model_name=model_name,
        layer_ind=-1,
    )

    records = list(SeqIO.parse(fasta_filepath, "fasta"))
    embeddings = np.load(str(output_filepath))
    assert embeddings.shape == (len(records), embed.MODEL_NAMES_TO_DIMS[model_name])


@pytest.mark.parametrize("sequence", ["", "K", "KZ", "K-", "ABC", "ACDEFGHIKLMNPQRSTVWY"])
def test_embed_edge_cases(sequence, tmpdir):
    """
    Test that the embed function generates embeddings for various edge cases.
    """
    model_name = "esm2_t6_8M_UR50D"
    fasta_filepath = tmpdir / "peptides.fa"
    output_filepath = tmpdir / "embeddings.npy"

    with open(fasta_filepath, "w") as file:
        file.write(">some_sequence_id\n")
        file.write(f"{sequence}\n")

    embed.embed(
        fasta_filepath=fasta_filepath,
        output_filepath=output_filepath,
        model_name=model_name,
        layer_ind=-1,
    )

    embeddings = np.load(str(output_filepath))
    assert embeddings.shape == (1, embed.MODEL_NAMES_TO_DIMS[model_name])


def test_embed_long_sequence(tmpdir):
    """
    Test that the embed function generates embeddings for a sequence longer than the maximum
    allowed by the ESM-2 model.
    """
    model_name = "esm2_t6_8M_UR50D"
    fasta_filepath = tmpdir / "peptides.fa"
    output_filepath = tmpdir / "embeddings.npy"

    # The longest sequence that can be embedded by the ESM-2 model is 1024 amino acids.
    long_sequence = "A" * 1024 * 2
    with open(fasta_filepath, "w") as file:
        file.write(">some_sequence_id\n")
        file.write(f"{long_sequence}\n")

    embed.embed(
        fasta_filepath=fasta_filepath,
        output_filepath=output_filepath,
        model_name=model_name,
        layer_ind=-1,
    )

    embeddings = np.load(str(output_filepath))
    assert embeddings.shape == (1, embed.MODEL_NAMES_TO_DIMS[model_name])


def test_embedding_matrix_row_order(artifacts_dirpath, tmpdir):
    """
    Test that the order of the rows of the embeddings matrix matches the order of the sequences
    in the input file.
    To make this test less fragile, we don't hard-code the expected embeddings for each sequence
    but instead check that the order of the embeddings is reversed when we reverse the order
    of the sequences in the input file.
    """
    original_fasta_filepath = artifacts_dirpath / "peptides.fa"
    reordered_fasta_filepath = tmpdir / "reordered.fa"

    original_output_filepath = tmpdir / "embeddings.npy"
    reordered_output_filepath = tmpdir / "reordered_embeddings.npy"

    with open(reordered_fasta_filepath, "w") as file:
        for record in reversed(list(SeqIO.parse(original_fasta_filepath, "fasta"))):
            SeqIO.write(record, file, "fasta")

    for fasta_filepath, output_filepath in [
        (original_fasta_filepath, original_output_filepath),
        (reordered_fasta_filepath, reordered_output_filepath),
    ]:
        embed.embed(
            fasta_filepath=fasta_filepath,
            output_filepath=output_filepath,
            model_name="esm2_t6_8M_UR50D",
            layer_ind=-1,
        )

    original_embeddings = np.load(str(original_output_filepath))
    reordered_embeddings = np.load(str(reordered_output_filepath))

    assert not np.allclose(original_embeddings, reordered_embeddings)
    assert np.allclose(original_embeddings, reordered_embeddings[::-1])
