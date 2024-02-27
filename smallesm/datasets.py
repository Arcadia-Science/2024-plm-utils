import concurrent.futures
import functools
import os
import pathlib
import shutil
import subprocess
import urllib
from concurrent.futures import ThreadPoolExecutor

import click
import pandas as pd
import requests
import tqdm

from smallesm.embed import embed
from smallesm.translate import translate


def add_suffix_to_path(path: pathlib.Path, suffix: str) -> pathlib.Path:
    """
    Add a suffix to the filename or final directory name in the given path,
    assuming the filename or directory name does not begin with a dot.

    Examples for the suffix "modified":
        "/path/to/dir" -> "/path/to/dir-modified"
        "/path/to/file.txt" -> "/path/to/file-modified.txt"
        "/path/to/file.fa.gz" -> "/path/to/file-modified.fa.gz"
    """
    dot = "."
    stem, *exts = path.name.split(dot)
    new_stem = f"{stem}-{suffix}"
    new_name = dot.join((new_stem, *exts))
    return path.with_name(new_name)


def log_calls(func):
    """
    Decorator to print the arguments and keyword arguments of a function call.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function `{func.__name__}` called with args {args} and kwargs {kwargs}")
        return func(*args, **kwargs)

    return wrapper


def with_output_checks(func):
    """
    Decorator intended for functions that generate a single output file.
    It will skip running the wrapped function if the output file already exists
    and an "overwrite" kwarg is not set to `True`,
    and will ensure that the output directory exists before running the function.

    The output filepath is assumed to be the value of an "output_filepath" kwarg
    or else the second positional argument of the wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        overwrite = kwargs.get("overwrite", False)
        output_filepath = kwargs.get("output_filepath")
        if output_filepath is None:
            output_filepath = args[1]
        if output_filepath.exists() and not overwrite:
            print(
                f"Skipping `{func.__name__}` because the output file already exists "
                f"at '{output_filepath}'"
            )
            return

        # assume the wrapped function does not expect an "overwrite" kwarg.
        kwargs.pop("overwrite", None)

        # ensure the output directory exists.
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)

    return wrapper


@log_calls
@with_output_checks
def download_file(url, output_filepath):
    """
    Download a file from `url` to `output_path` with streaming to avoid loading the whole file
    into memory at once.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(output_filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def download_files(urls_filepaths, overwrite=False):
    """
    Download a list of files.

    urls_filepaths: a list of (url, filepath) tuples
    """
    futures = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        for url, filepath in urls_filepaths:
            futures.append(executor.submit(download_file, url, filepath, overwrite=overwrite))

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as exception:
                print(f"Error downloading dataset: {exception}")


def download_transcriptomes(
    dataset_metadata_filepath, output_dirpath, overwrite=False, dry_run=False
):
    """
    Download the coding and non-coding transcriptomes for each species in the metadata file.

    The metadata file should have the following columns:
        - species_id: a short and unique identifier for the species
        - root_url: the base URL for the species
        - cdna_endpoint: the relative URL for the fasta file of coding transcripts
        - ncrna_endpoint: the relative URL for the fasta file of non-coding transcripts

    The coding and non-coding transcripts are downloaded to separate subdirectories
    of the output directory and named with the species_id:
        output_dirpath/
            cdna/
                {species_id}.fa.gz
            ncrna/
                {species_id}.fa.gz
    """

    metadata = pd.read_csv(dataset_metadata_filepath, sep="\t")

    cdna_dirpath = output_dirpath / "cdna"
    ncrna_dirpath = output_dirpath / "ncrna"

    cdna_dirpath.mkdir(parents=True, exist_ok=True)
    ncrna_dirpath.mkdir(parents=True, exist_ok=True)

    urls_filepaths = []
    for _, row in metadata.iterrows():
        for endpoint, dirpath in [
            (row.cdna_endpoint, cdna_dirpath),
            (row.ncrna_endpoint, ncrna_dirpath),
        ]:
            url = urllib.parse.urljoin(row.root_url, endpoint)
            filepath = dirpath / f"{row.species_id}.fa.gz"
            urls_filepaths.append((url, filepath))

    if not dry_run:
        download_files(urls_filepaths, overwrite=overwrite)
    return cdna_dirpath, ncrna_dirpath


@log_calls
@with_output_checks
def gunzip_file(input_filepath, output_filepath):
    subprocess.run(f"gunzip -c {input_filepath} > {output_filepath}", shell=True)


@log_calls
@with_output_checks
def cat_files(input_filepaths, output_filepath):
    subprocess.run(f"cat {' '.join(map(str, input_filepaths))} > {output_filepath}", shell=True)


@log_calls
@with_output_checks
def prepend_filename_to_sequence_ids(input_filepath, output_filepath):
    """
    Prepend the filename of the input fasta file to its sequence IDs and write the result
    to the output fasta file.
    """
    sep = "."
    prefix = input_filepath.stem.split(".")[0].upper()
    command = f"""
        seqkit replace \
            --pattern ^ \
            --replacement {prefix}{sep} \
            -o {output_filepath} \
            {input_filepath}
    """
    subprocess.run(command, shell=True)


@log_calls
@with_output_checks
def extract_protein_coding_transcripts_from_cdna(input_filepath, output_filepath):
    """
    Filter out transcripts that are not of type 'protein_coding'.
    This is necessary because ensembl cDNA files include transcripts for pseudogenes
    and other non-coding RNAs.
    """
    command = f"""
        seqkit grep \
            --use-regexp \
            --by-name \
            --pattern "transcript_biotype:protein_coding" \
            -o {output_filepath} \
            {input_filepath}
    """
    subprocess.run(command, shell=True)


@log_calls
def cluster_with_mmseqs(input_filepath, output_filepath_prefix, overwrite=False):
    """
    Cluster the sequences in the input FASTA file using mmseqs2.
    """
    # this is one of the files that will be created by mmseqs2; we need to hard-code its name
    # so we can check if it exists and skip running mmseqs2 if it does.
    mmseqs_output_filepath = (
        output_filepath_prefix.parent / f"{output_filepath_prefix.name}_rep_seq.fasta"
    )
    if mmseqs_output_filepath.exists() and not overwrite:
        print(
            "mmseqs will not be run because one of its outputs already exists: "
            f"'{mmseqs_output_filepath}'"
        )
        return

    output_filepath_prefix.parent.mkdir(parents=True, exist_ok=True)

    mmseqs_internal_dir = output_filepath_prefix.parent / "mmseqs-internal"
    command = f"""
        mmseqs easy-cluster \
        {input_filepath} \
        {output_filepath_prefix} \
        {mmseqs_internal_dir} \
        --min-seq-id 0.8 \
        --cov-mode 1 \
        --cluster-mode 2
    """
    print(f"Running mmseqs:\n{command}")
    subprocess.run(command, shell=True)
    shutil.rmtree(mmseqs_internal_dir)


@log_calls
@with_output_checks
def intersect_fasta_files(input_filepaths, output_filepath):
    """
    Extract the sequences whose ids appear in both of the input FASTA files.
    """
    input_filepath_1, input_filepath_2 = input_filepaths

    tmp_ids_filepath = input_filepath_1.with_suffix(".ids")
    command = f"seqkit seq {input_filepath_1} --name --only-id > {tmp_ids_filepath}"
    subprocess.run(command, shell=True)

    command = f"seqkit grep -f {tmp_ids_filepath} {input_filepath_2} -o {output_filepath}"
    subprocess.run(command, shell=True)
    os.remove(tmp_ids_filepath)


@log_calls
@with_output_checks
def subsample_fasta_file(input_filepath, output_filepath, subsample_rate):
    """
    Subsample a FASTA file by a given rate.
    """
    command = f"seqkit sample -p {subsample_rate:0.3f} {input_filepath} -o {output_filepath}"
    subprocess.run(command, shell=True)


@log_calls
@with_output_checks
def filter_by_sequence_id_prefix(input_filepath, output_filepath, prefix):
    """
    Filter a FASTA file by the prefix of its sequence IDs.
    """
    prefix = prefix.upper()
    command = f"seqkit grep -r -p ^{prefix} {input_filepath} -o {output_filepath}"
    subprocess.run(command, shell=True)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset_metadata_filepath", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_dirpath", type=click.Path(path_type=pathlib.Path))
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def download(dataset_metadata_filepath, output_dirpath, overwrite):
    """
    Download the coding and non-coding transcriptomes for each species in the metadata file.
    """
    download_transcriptomes(
        dataset_metadata_filepath, output_dirpath / "raw", overwrite=overwrite, dry_run=False
    )


@cli.command()
@click.argument("dataset_metadata_filepath", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_dirpath", type=click.Path(path_type=pathlib.Path))
@click.option(
    "--subsample-factor",
    type=int,
    default=3,
    help="Factor by which to subsample the sequences in each dataset",
)
def construct(dataset_metadata_filepath, output_dirpath, subsample_factor):
    """
    Construct the training datasets from the downloaded transcriptomes.

    This command assumes that the coding and non-coding transcriptomes have been downloaded
    to the output_dirpath directory using the `download` command.

    Note: the de facto pipeline that this command implements
    is based on the snakemake pipeline defined in the `arcadia-science/peptigate` repo
    to construct a training dataset for RNASamba models.

    TODO: there is currently no overwrite option; the user must manually delete
    output directories or files in order to re-run certain steps of this de facto pipeline.
    """
    raw_cdna_dirpath, raw_ncrna_dirpath = download_transcriptomes(
        dataset_metadata_filepath, output_dirpath / "raw", dry_run=True
    )

    output_dirpath = output_dirpath / "processed"

    # prepend the species ID to the sequence IDs in all of the coding and noncoding fasta files.
    # (we use the word "labeled" to refer to this)
    cdna_labeled_dirpath = output_dirpath / "labeled" / "cdna"
    ncrna_labeled_dirpath = output_dirpath / "labeled" / "ncrna"
    for filepath in raw_cdna_dirpath.glob("*.fa.gz"):
        prepend_filename_to_sequence_ids(filepath, (cdna_labeled_dirpath / filepath.name))
    for filepath in raw_ncrna_dirpath.glob("*.fa.gz"):
        prepend_filename_to_sequence_ids(filepath, (ncrna_labeled_dirpath / filepath.name))

    # concatenate all of the labeled coding and noncoding transcripts into single files.
    all_cdna_filepath = output_dirpath / "labeled" / "cdna.fa.gz"
    all_ncrna_filepath = output_dirpath / "labeled" / "ncrna.fa.gz"
    cat_files(cdna_labeled_dirpath.glob("*.fa.gz"), all_cdna_filepath)
    cat_files(ncrna_labeled_dirpath.glob("*.fa.gz"), all_ncrna_filepath)

    # filter out transcripts from the cdna file that are not truly protein coding.
    # note that this coincides with switching nomenclature from 'cdna' to 'coding'.
    all_coding_filepath = output_dirpath / "labeled" / "coding.fa.gz"
    extract_protein_coding_transcripts_from_cdna(all_cdna_filepath, all_coding_filepath)

    # no filtering is necessary for the ncrna transcripts, but we switch nomenclature here
    # to be consistent with the coding transcripts.
    all_noncoding_filepath = output_dirpath / "labeled" / "noncoding.fa.gz"
    shutil.move(all_ncrna_filepath, all_noncoding_filepath)

    # merge the coding and noncoding transcripts into a single file for clustering.
    merged_coding_noncoding_filepath = output_dirpath / "labeled" / "coding-and-noncoding.fa.gz"
    cat_files(
        input_filepaths=(all_coding_filepath, all_noncoding_filepath),
        output_filepath=merged_coding_noncoding_filepath,
    )

    merged_coding_noncoding_unzipped_filepath = merged_coding_noncoding_filepath.with_suffix("")
    gunzip_file(
        input_filepath=merged_coding_noncoding_filepath,
        output_filepath=merged_coding_noncoding_unzipped_filepath,
    )
    merged_coding_noncoding_filepath = merged_coding_noncoding_unzipped_filepath

    # cluster all of the transcripts to reduce redundancy.
    mmseqs_output_filepath_prefix = output_dirpath / "mmseqs" / "mmseqs-output"
    cluster_with_mmseqs(
        input_filepath=merged_coding_noncoding_filepath,
        output_filepath_prefix=mmseqs_output_filepath_prefix,
    )

    # this is the name of the file created by mmseqs2 containing representative sequences
    # for each cluster.
    rep_seqs_filepath = (
        mmseqs_output_filepath_prefix.parent / f"{mmseqs_output_filepath_prefix.stem}_rep_seq.fasta"
    )

    for filepath in (all_coding_filepath, all_noncoding_filepath):
        # filter out the non-representative sequences from each file to "deduplicate" it
        deduped_filepath = add_suffix_to_path(filepath, "dedup")
        intersect_fasta_files(
            input_filepaths=[rep_seqs_filepath, filepath], output_filepath=deduped_filepath
        )

        # subsample the deduped file to reduce the number of sequences and make training faster.
        subsampled_filepath = add_suffix_to_path(deduped_filepath, f"ssx{subsample_factor}")
        subsample_fasta_file(
            input_filepath=deduped_filepath,
            output_filepath=subsampled_filepath,
            subsample_rate=(1 / subsample_factor),
        )

        # split the subsampled file into separate files for each species using the prefixes
        # we added to the sequence IDs above (using `prepend_filename_to_sequence_ids`).
        metadata = pd.read_csv(dataset_metadata_filepath, sep="\t")
        subsampled_filename = subsampled_filepath.with_suffix("").stem
        final_output_dirpath = output_dirpath / "final" / subsampled_filename / "transcripts"
        for species_id in metadata.species_id:
            filter_by_sequence_id_prefix(
                input_filepath=subsampled_filepath,
                output_filepath=(final_output_dirpath / f"{species_id}.fa"),
                prefix=species_id,
            )


@cli.command()
@click.argument("dirpaths", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
def translate_and_embed(dirpaths):
    """
    Translate the fasta files in the specified directories of amino acid sequences
    and embed them using ESM.
    """

    # TODO: don't hard-code the model name.
    model_name = "esm2_t6_8M_UR50D"

    for dirpath in dirpaths:
        peptides_dirpath = dirpath.parent / "peptides"
        embeddings_dirpath = dirpath.parent / "embeddings" / model_name

        peptides_dirpath.mkdir(parents=True, exist_ok=True)
        embeddings_dirpath.mkdir(parents=True, exist_ok=True)

        for input_filepath in dirpath.glob("*.fa"):
            peptides_filepath = peptides_dirpath / input_filepath.name
            translate(
                input_filepath=input_filepath, output_filepath=peptides_filepath, longest_only=True
            )

            embeddings_filepath = embeddings_dirpath / f"{input_filepath.stem}.npy"
            embed(
                fasta_filepath=peptides_filepath,
                model_name=model_name,
                layer_ind=-1,
                output_filepath=embeddings_filepath,
            )
