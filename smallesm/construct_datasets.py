import concurrent.futures
import pathlib
import shutil
import subprocess
import urllib
from concurrent.futures import ThreadPoolExecutor

import click
import pandas as pd
import requests
import tqdm


def cat_and_gunzip_files(input_filepaths, output_filepath):
    """
    Concatenate and gunzip a list of files to a single file.
    """
    if output_filepath.exists():
        print(f"Skipping {output_filepath} because it already exists")
        return
    subprocess.run(
        f"cat {' '.join(map(str, input_filepaths))} | gunzip > {output_filepath}", shell=True
    )


def download_file(url, output_path):
    """
    Download a file from `url` to `output_path` with streaming.
    """
    print(f"Downloading {url} to {output_path}")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)


def download_files(urls_filepaths, overwrite=False):
    """
    Download a list of files.

    urls_filepaths: a list of (url, filepath) tuples
    """
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for url, output_filepath in urls_filepaths:
            if overwrite or not output_filepath.exists():
                futures.append(executor.submit(download_file, url, output_filepath))
            if not overwrite and output_filepath.exists():
                print(f"Skipping {output_filepath} because it already exists")

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Downloading datasets",
        ):
            try:
                future.result()
            except Exception as exception:
                print(f"Error downloading dataset: {exception}")


def download_transcriptomes(dataset_metadata_filepath, output_dirpath, overwrite=False):
    """ """

    metadata = pd.read_csv(dataset_metadata_filepath, sep="\t")

    (output_dirpath / "cdna").mkdir(parents=True, exist_ok=True)
    (output_dirpath / "ncrna").mkdir(parents=True, exist_ok=True)

    urls_filepaths = []
    for _, row in metadata.iterrows():
        urls_filepaths += [
            (
                urllib.parse.urljoin(row.root_url, row.cdna_endpoint),
                output_dirpath / "cdna" / f"{row.species_id}.fa.gz",
            ),
            (
                urllib.parse.urljoin(row.root_url, row.ncrna_endpoint),
                output_dirpath / "ncrna" / f"{row.species_id}.fa.gz",
            ),
        ]
    download_files(urls_filepaths, overwrite=overwrite)


def preprocess_transcriptomes(output_dirpath, overwrite=False):
    """
    Preprocess the downloaded transcriptomes by:
    - Prepending the filename (assumed to be a species ID) to the sequence IDs
    - Filtering out non-protein-coding transcripts from the raw cDNA files

    """
    for cdna_filepath in (output_dirpath / "cdna").glob("*.fa.gz"):
        prepend_filename_to_sequence_ids(
            cdna_filepath, (output_dirpath / "cdna-labeled" / cdna_filepath.name)
        )

    for ncrna_filepath in (output_dirpath / "ncrna").glob("*.fa.gz"):
        prepend_filename_to_sequence_ids(
            ncrna_filepath, (output_dirpath / "ncrna-labeled" / ncrna_filepath.name)
        )

    for cdna_filepath in (output_dirpath / "cdna-labeled").glob("*.fa.gz"):
        extract_protein_coding_orfs_from_cdna(
            cdna_filepath, (output_dirpath / "cdna-labeled-coding" / cdna_filepath.name)
        )


def prepend_filename_to_sequence_ids(input_filepath, output_filepath):
    """
    Prepend the sequence IDs in the input FASTA file with the filename of the input file,
    which is assumed to be a species identifier.
    """
    if output_filepath.exists():
        print(f"Skipping {output_filepath} because it already exists")
        return

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    sep = "_"
    prefix = input_filepath.stem.upper()

    # TODO: it appears that seqkit uses a dot instead of an underscore as the separator here.
    subprocess.run(
        (
            f"seqkit replace --pattern ^ --replacement {prefix}{sep} "
            f"-o {output_filepath} {input_filepath}",
        ),
        shell=True,
    )


def extract_protein_coding_orfs_from_cdna(input_filepath, output_filepath):
    """
    Filter out transcripts that are not of type 'protein_coding'.
    This is necessary because ensembl cDNA files include transcripts for pseudogenes
    and other non-coding RNAs.
    """
    if output_filepath.exists():
        print(f"Skipping {output_filepath} because it already exists")
        return

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    command = (
        'seqkit grep --use-regexp --by-name --pattern "transcript_biotype:protein_coding" '
        f"-o {output_filepath} {input_filepath}"
    )
    subprocess.run(command, shell=True)


def cluster_transcripts(input_filepath, output_filepath_prefix, overwrite=False):
    """
    Cluster the transcripts in the input FASTA file using mmseqs2.
    """
    if len(list(output_filepath_prefix.parent.glob("*cluster.tsv"))) and not overwrite:
        print(f"Skipping clustering because {output_filepath_prefix} already exists")
        return

    output_filepath_prefix.parent.mkdir(parents=True, exist_ok=True)

    mmseqs_internal_dir = output_filepath_prefix.parent / "tmp_mmseqs2"
    command = (
        f"mmseqs easy-cluster {input_filepath} {output_filepath_prefix} "
        f"{mmseqs_internal_dir} --min-seq-id 0.8 --cov-mode 1 --cluster-mode 2"
    )
    print(f"Running command: {command}")
    subprocess.run(command, shell=True)
    shutil.rmtree(mmseqs_internal_dir)


@click.command()
@click.argument("dataset_metadata_filepath", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("output_dirpath", type=click.Path(path_type=pathlib.Path))
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
def command(dataset_metadata_filepath, output_dirpath, overwrite):
    """
    Download, preprocess, aggregate, and cluster transcriptomes from the given dataset.
    """
    download_transcriptomes(dataset_metadata_filepath, output_dirpath, overwrite=overwrite)
    preprocess_transcriptomes(output_dirpath, overwrite=overwrite)

    cat_and_gunzip_files(
        (output_dirpath / "cdna-labeled-coding").glob("*.fa.gz"),
        output_dirpath / "cdna-labeled-coding.fa",
    )
    cat_and_gunzip_files(
        (output_dirpath / "ncrna-labeled").glob("*.fa.gz"),
        output_dirpath / "ncrna-labeled.fa",
    )

    # cat cdna and ncrna files together
    subprocess.run(
        " ".join(
            map(
                str,
                [
                    "cat",
                    output_dirpath / "cdna-labeled-protein-coding.fa",
                    output_dirpath / "ncrna-labeled.fa",
                    ">",
                    output_dirpath / "cdna-and-ncrna.fa",
                ],
            )
        ),
        shell=True,
    )

    # cluster all of the transcripts to remove redundancy.
    cluster_transcripts(
        input_filepath=output_dirpath / "cdna-and-ncrna.fa",
        output_filepath_prefix=output_dirpath / "mmseqs-output",
        overwrite=overwrite,
    )

    # TODO: split clustered transcripts into separate files for cDNA and ncRNA
    # "$ seqkit seq tmp/raw/cdna-labeled-protein-coding.fa --name --only-id > cdna-ids.txt
    # "$ seqkit grep -f cdna-ids.txt clustered-seqs.fasta -o cdna-labeled-coding-dedup.fa"

    # check that all of the species are represented in the clustered sequences
    # $ seqkit sample -p 0.001 cdna-labeled-coding-dedup.fa| seqkit seq -n -i | sort > tmp.txt

    # filter by species (here human)
    # $ seqkit grep -r -p ^HSAP cdna-labeled-coding-dedup.fa -o tmp.fa
