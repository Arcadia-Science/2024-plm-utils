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


def add_suffix_to_path(path, suffix):
    """
    Add a suffix to the filename or final directory name in the given path.
    For example, if the filepath is "/path/to/file.txt" and the suffix is "modified",
    this function will return "/path/to/file-modified.txt".

    TODO: this function doesn't handle filenames with multiple extensions.
    For example, if the filename is "file.tar.gz" and the suffix is "modified",
    this function will return "file.tar-modified.gz".
    """
    return path.with_name(f"{path.stem}-{suffix}{path.suffix}")


def log_calls(func):
    """
    Decorator to print the arguments and keyword arguments of a function call.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Function `{func.__name__}` called with args {args} and kwargs {kwargs}")
        return func(*args, **kwargs)

    return wrapper


def skip_if_output_exists(func):
    """
    Decorator to skip running a function if the output file already exists
    and an "overwrite" kwarg is not set to `True`.

    The output filepath is assumed to be the second positional argument to the function
    or else the value of the "output_filepath" kwarg.
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

        # TODO: creating the output directory here is a bit of a hack.
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        return func(*args, **kwargs)

    return wrapper


@log_calls
@skip_if_output_exists
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


@log_calls
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
    cdna_dirpath.mkdir(parents=True, exist_ok=True)

    ncrna_dirpath = output_dirpath / "ncrna"
    ncrna_dirpath.mkdir(parents=True, exist_ok=True)

    urls_filepaths = []
    for _, row in metadata.iterrows():
        for endpoint, dirpath in [
            (row.cdna_endpoint, cdna_dirpath),
            (row.ncrna_endpoint, ncrna_dirpath),
        ]:
            urls_filepaths.append(
                (urllib.parse.urljoin(row.root_url, endpoint), dirpath / f"{row.species_id}.fa.gz")
            )

    if not dry_run:
        download_files(urls_filepaths, overwrite=overwrite)
    return cdna_dirpath, ncrna_dirpath


@log_calls
@skip_if_output_exists
def gunzip_file(input_filepath, output_filepath):
    subprocess.run(f"gunzip -c {input_filepath} > {output_filepath}", shell=True)


@log_calls
@skip_if_output_exists
def cat_files(input_filepaths, output_filepath):
    subprocess.run(f"cat {' '.join(map(str, input_filepaths))} > {output_filepath}", shell=True)


@log_calls
@skip_if_output_exists
def prepend_filename_to_sequence_ids(input_filepath, output_filepath):
    """
    Prepend the filename of the input fasta file to its sequence IDs and write the result
    to the output fasta file.
    """
    sep = "."
    prefix = input_filepath.stem.split(".")[0].upper()
    subprocess.run(
        (
            f"seqkit replace --pattern ^ --replacement {prefix}{sep} "
            f"-o {output_filepath} {input_filepath}",
        ),
        shell=True,
    )


@log_calls
@skip_if_output_exists
def extract_protein_coding_transcripts_from_cdna(input_filepath, output_filepath):
    """
    Filter out transcripts that are not of type 'protein_coding'.
    This is necessary because ensembl cDNA files include transcripts for pseudogenes
    and other non-coding RNAs.
    """
    command = (
        'seqkit grep --use-regexp --by-name --pattern "transcript_biotype:protein_coding" '
        f"-o {output_filepath} {input_filepath}"
    )
    subprocess.run(command, shell=True)


@log_calls
def cluster_transcripts_with_mmseqs(input_filepath, output_filepath_prefix, overwrite=False):
    """
    Cluster the transcripts in the input FASTA file using mmseqs2.
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
    command = (
        f"mmseqs easy-cluster {input_filepath} {output_filepath_prefix} "
        f"{mmseqs_internal_dir} --min-seq-id 0.8 --cov-mode 1 --cluster-mode 2"
    )
    print(f"Running mmseqs:\n{command}")
    subprocess.run(command, shell=True)
    shutil.rmtree(mmseqs_internal_dir)


@log_calls
@skip_if_output_exists
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
@skip_if_output_exists
def subsample_fasta_file(input_filepath, output_filepath, subsample_rate):
    """
    Subsample a FASTA file by a given rate.
    """
    command = f"seqkit sample -p {subsample_rate:0.3f} {input_filepath} -o {output_filepath}"
    subprocess.run(command, shell=True)


@log_calls
@skip_if_output_exists
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
        dataset_metadata_filepath, output_dirpath, overwrite=overwrite, dry_run=False
    )


@cli.command()
@click.argument("dataset_metadata_filepath", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_dirpath", type=click.Path(path_type=pathlib.Path))
def construct(dataset_metadata_filepath, output_dirpath):
    """
    Construct the training datasets from the downloaded transcriptomes.

    Assumes that the coding and non-coding transcriptomes have been downloaded
    to the output_dirpath directory using the `download` command.

    TODO: there is currently no overwrite option; the user must manually delete
    output directories or files in order to re-run certain steps of this de facto pipeline.
    """
    cdna_dirpath, ncrna_dirpath = download_transcriptomes(
        dataset_metadata_filepath, output_dirpath, dry_run=True
    )

    # prepend the species ID to the sequence IDs in all of the coding and noncoding fasta files.
    # (we use the word "labeled" for this)
    cdna_labeled_dirpath = add_suffix_to_path(cdna_dirpath, "labeled")
    ncrna_labeled_dirpath = add_suffix_to_path(ncrna_dirpath, "labeled")

    for filepath in cdna_dirpath.glob("*.fa.gz"):
        prepend_filename_to_sequence_ids(filepath, (cdna_labeled_dirpath / filepath.name))

    for filepath in ncrna_dirpath.glob("*.fa.gz"):
        prepend_filename_to_sequence_ids(filepath, (ncrna_labeled_dirpath / filepath.name))

    # concatenate all of the labeled coding and noncoding transcripts into single files.
    all_cdna_filepath = output_dirpath / "all-cdna.fa.gz"
    all_ncrna_filepath = output_dirpath / "all-ncrna.fa.gz"
    cat_files(cdna_labeled_dirpath.glob("*.fa.gz"), all_cdna_filepath)
    cat_files(ncrna_labeled_dirpath.glob("*.fa.gz"), all_ncrna_filepath)

    # filter out transcripts from the cdna file that are not truly protein coding.
    # (by definition, this not necessary for the noncoding transcripts)
    all_cdna_coding_filepath = output_dirpath / "all-cdna-coding.fa.gz"
    extract_protein_coding_transcripts_from_cdna(all_cdna_filepath, all_cdna_coding_filepath)

    # merge the coding and noncoding transcripts into a single file for clustering.
    merged_cdna_and_ncrna_filepath = output_dirpath / "cdna-and-ncrna.fa.gz"
    cat_files(
        input_filepaths=(all_cdna_coding_filepath, all_ncrna_filepath),
        output_filepath=merged_cdna_and_ncrna_filepath,
    )
    merged_cdna_and_ncrna_unzipped_filepath = merged_cdna_and_ncrna_filepath.with_suffix("")
    gunzip_file(
        input_filepath=merged_cdna_and_ncrna_filepath,
        output_filepath=merged_cdna_and_ncrna_unzipped_filepath,
    )

    # cluster all of the transcripts to reduce redundancy.
    mmseqs_output_filepath_prefix = output_dirpath / "mmseqs" / "mmseqs-output"
    cluster_transcripts_with_mmseqs(
        input_filepath=merged_cdna_and_ncrna_unzipped_filepath,
        output_filepath_prefix=mmseqs_output_filepath_prefix,
    )

    # this is the name of the file created by mmseqs2 containing representative sequences
    # for each cluster.
    rep_seqs_filepath = (
        mmseqs_output_filepath_prefix.parent / f"{mmseqs_output_filepath_prefix.stem}_rep_seq.fasta"
    )

    for kind, filepath in [("cdna", all_cdna_coding_filepath), ("ncrna", all_ncrna_filepath)]:
        clustered_filepath = output_dirpath / f"{kind}-clustered.fa"
        intersect_fasta_files(
            input_filepaths=[rep_seqs_filepath, filepath], output_filepath=clustered_filepath
        )

        subsample_period = 1
        subsampled_filepath = add_suffix_to_path(clustered_filepath, f"ssx{subsample_period}")
        subsample_fasta_file(
            input_filepath=clustered_filepath,
            output_filepath=subsampled_filepath,
            subsample_rate=(1 / subsample_period),
        )

        metadata = pd.read_csv(dataset_metadata_filepath, sep="\t")
        for species_id in metadata.species_id:
            output_filepath = output_dirpath / subsampled_filepath.stem / f"{species_id}.fa"
            filter_by_sequence_id_prefix(
                input_filepath=subsampled_filepath,
                output_filepath=output_filepath,
                prefix=species_id,
            )


@cli.command()
@click.argument("dirpaths", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
def translate_and_embed(dirpaths):
    """
    Translate the fasta files in the specified directory to protein sequences
    and embed them using ESM.
    """
    model_name = "esm2_t6_8M_UR50D"

    for dirpath in dirpaths:
        peptides_dirpath = add_suffix_to_path(dirpath, "peptides")
        peptides_dirpath.mkdir(parents=True, exist_ok=True)

        embeddings_dirpath = add_suffix_to_path(dirpath, f"embeddings--{model_name}")
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
