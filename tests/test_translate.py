import pathlib

from Bio import SeqIO

from plmutils import translate


def test_translate(artifacts_dirpath, tmpdir):
    """
    Test that the translate function generates the expected number of peptides for a given input.
    """
    input_filepath = artifacts_dirpath / "transcripts.fa"
    output_filepath = pathlib.Path(tmpdir) / "peptides.fa"

    translate.translate(
        input_filepath=input_filepath, output_filepath=output_filepath, longest_only=False
    )

    records = list(SeqIO.parse(output_filepath, "fasta"))
    assert len(records) == 142


def test_translate_longest_only(artifacts_dirpath, tmpdir):
    """
    Test that the translate function returns one sequence for each input sequence
    when longest_only is set to True.
    """
    input_filepath = artifacts_dirpath / "transcripts.fa"
    output_filepath = pathlib.Path(tmpdir) / "peptides.fa"

    translate.translate(
        input_filepath=input_filepath, output_filepath=output_filepath, longest_only=True
    )

    input_records = list(SeqIO.parse(input_filepath, "fasta"))
    output_records = list(SeqIO.parse(output_filepath, "fasta"))

    assert set([r.id for r in input_records]) == set([r.id for r in output_records])
