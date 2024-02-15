# small-esm
This repo is an attempt to predict sORFs using ESM embeddings. It is part of the larger peptigate project.


## Installation
This repo depends on CUDA 12 and pytorch 2.2.
First create a new python 3.11 virtual env from the dev.yml file using the following commands:
```bash
mamba env create -n small-esm-env -f dev.yml
```
Then activate the environment using the following command:
```bash
conda activate small-esm-env
```

Then install pytorch using the following commands:
```bash
mamba install pytorch=2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Finally, `esm` was installed using pip from a local clone on the `v1.0.3` tag.
