# plm-utils
This repo contains a python package called `plmutils` that provides a basic set of tools for generating and analyzing embeddings of protein sequences using pre-trained protein language models (PLMs).

## Installation
Create a virtual env from the `envs/dev.yml` file:
```bash
mamba env create -n plmutils-env -f envs/dev.yml
```

Then activate the environment:
```bash
conda activate plmutils-env
```

Then clone this repo, `cd` into it, and install the `plmutils` package in editable mode:
```bash
pip install -e .
```

Finally, check that pytorch can find the GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Although a GPU is not required to generate embeddings, it provides a significant speedup (of 10x or more) compared to using the CPU.

## Usage
The main package defines several generic (i.e., task-nonspecific) commands:

__`plmutils translate`__<br>
This command uses `orfipy` to find putative ORFs in a fasta file of transcripts and translates them to protein sequences. The translated sequences are saved to a new fasta file. The `--longest-only` option can be used to retain only the longest ORF for each transcript.

__`plmutils embed`__<br>
This command uses a pre-trained protein language model to embed a fasta file of protein sequences. Currently only [ESM-2 models](https://github.com/facebookresearch/esm?tab=readme-ov-file#available) are supported. The resulting matrix of embeddings is saved as a numpy array in a `.npy` file. The order of the rows of this matrix corresponds to the order of the sequences in the input fasta file.

__`plmutils train`__<br>
This command trains a generic binary classifier given two embedding matrices, one for the positive class and one for the negative class. The trained classifier is optionally saved to a user-specified directory.

__`plmutils predict`__<br>
This command generates predictions given an embedding matrix and a pre-trained classifier generated by the `train` command above.

## Specific uses
### ORF prediction
This repo is prospectively organized as a generic tool for working with PLM embeddings but it was motivated by, and also includes code specific to, the concrete task of predicting whether putative open reading frames (ORFs) are coding or noncoding. This code is confined to the `plmutils.tasks.orf_prediction` module. It include both the code to construct training and test datasets and the code to train and evaluate classification  models. It relies on extant annotated transcriptomes (i.e., datasets of coding and noncoding transcripts) to obtain a set of putative ORFs that are likely to be "real" (from a coding transcript) or "not real" (from a noncoding transcript). The ESM embeddings of these ORFs are then used to train a classifier to predict whether a given putative ORF is likely to be "real" or not on the basis of its ESM embedding. Refer to [this jupyter notebook](./notebooks/2024-coding-noncoding-prediction.ipynb) for more details.

## Development
We use `ruff` for formatting and linting; use `make format` and `make lint` to run formatting and linting locally. The CLIs are written using `click`. There is a single virtualenv that defines all direct deps of the `plmutils` package. As a design/engineering principle, we try to avoid multiple virtualenvs or the use of workflow managers like snakemake.

## Testing
There is a test dataset in `/tests/data`. This dataset was constructed manually by heavily downsampling several of the transcriptomes used to develop the ORF prediction methods (see above). It can also be used to manually test the generic `plmutils` CLI commands. There is currently no automated testing.


## Origin and context
This repo was initially developed as a tool to identify sORFs for the [peptigate pipeline](https://github.com/Arcadia-Science/peptigate). This effort was motivated by the fact that existing tools for identifying sORFs are limited, poorly maintained, and generally operate at the transcript rather than ORF level (i.e., they predict whether a given *transcript* is coding or noncoding, rather than whether a given putative ORF represents a real protein/peptide or not). However, given the versatility and generality of embeddings produced by large PLMs like ESM-2, we decided to prospectively organize this repo in a more generic way to facilitate its use for future tasks that can be addressed with embeddings of protein sequences.