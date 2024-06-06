import numpy as np
from Bio import SeqIO

from plmutils import embed


def test_embed(artifacts_dirpath, tmpdir):
    """
    Test that the embed function generates the expected embeddings for a file of peptide sequences.
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


def test_embed_order(artifacts_dirpath, tmpdir):
    """
    Test that the order of the rows of the embeddings matrix matches the order of the sequences
    in the input file.
    To make this more robust, we don't hard-code the expected embeddings but instead check that
    the order of the embeddings changes when we change the order of the input sequences.
    """
    fasta_filepath = artifacts_dirpath / "peptides.fa"
    reordered_fasta_filepath = tmpdir / "reordered.fa"

    output_filepath = tmpdir / "embeddings.npy"
    reordered_output_filepath = tmpdir / "reordered_embeddings.npy"

    with open(reordered_fasta_filepath, "w") as file:
        for record in reversed(list(SeqIO.parse(fasta_filepath, "fasta"))):
            SeqIO.write(record, file, "fasta")

    embed.embed(
        fasta_filepath=fasta_filepath,
        output_filepath=output_filepath,
        model_name="esm2_t6_8M_UR50D",
        layer_ind=-1,
    )

    embed.embed(
        fasta_filepath=reordered_fasta_filepath,
        output_filepath=reordered_output_filepath,
        model_name="esm2_t6_8M_UR50D",
        layer_ind=-1,
    )

    embeddings = np.load(str(output_filepath))
    reordered_embeddings = np.load(str(reordered_output_filepath))

    assert not np.allclose(embeddings, reordered_embeddings)
    assert np.allclose(embeddings, reordered_embeddings[::-1])
