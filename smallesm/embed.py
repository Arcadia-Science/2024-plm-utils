import click
import esm
import esm.data
import esm.pretrained
import numpy as np
import torch
import tqdm

# the allowed ESM-2 model variants and their embeddings dimensions.
# (taken from https://github.com/facebookresearch/esm)
MODEL_NAMES_TO_DIMS = {
    "esm2_t48_15B_UR50D": 5120,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}


@click.command()
@click.argument("fasta_filepath", type=click.Path(exists=True))
@click.option(
    "--model-name",
    type=click.Choice(MODEL_NAMES_TO_DIMS.keys()),
    required=True,
    default="esm2_t6_8M_UR50D",
    help="ESM model name",
)
@click.option(
    "--layer-ind",
    type=int,
    required=True,
    help="Layer index from which to extract the embeddings (use -1 for the last layer)",
)
@click.option("--output-filepath", type=click.Path(exists=False), required=True, help="Output file")
def embed_command(fasta_filepath, model_name, layer_ind, output_filepath):
    embed(fasta_filepath, model_name, layer_ind, output_filepath)


def embed(fasta_filepath, model_name, layer_ind, output_filepath):
    """
    Generate per-sequence embeddings for the sequences in a FASTA file using an ESM model,
    and write the resulting matrix of embeddings to a numpy file.

    This function is loosely based on the `esm/sscripts/extract.py` module
    from the facebookresearch/esm repository.

    TODO (KC): currently there is no explicit association between rows of the embeddings matrix
    and the sequences in the input FASTA file; we rely on the preservation of order,
    which is not a good idea.
    """

    # TODO (KC): understand these numbers (they are copied from esm/scripts/extract.py)
    toks_per_batch = 4096
    truncation_seq_length = 1022

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        print("Warning: no GPU found")

    dataset = esm.FastaBatchedDataset.from_file(fasta_filepath)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )
    print(f"Read {fasta_filepath} with {len(dataset)} sequences")

    if layer_ind == -1:
        layer_ind = model.num_layers

    if layer_ind < 0 or layer_ind > model.num_layers:
        raise ValueError(f"Invalid layer index {layer_ind}")

    with torch.no_grad():
        mean_embeddings = []

        for sequence_ids, sequences, toks in tqdm.tqdm((data_loader), total=len(batches)):  # noqa B007
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            results = model(toks, repr_layers=[layer_ind], return_contacts=False)
            raw_embeddings = results["representations"][layer_ind].to(device="cpu")

            for ind, sequence in enumerate(sequences):
                truncate_len = min(len(sequence), truncation_seq_length)

                # the first token is the BOS token, so we skip it.
                raw_embedding = raw_embeddings[ind, 1 : truncate_len + 1]

                # call clone on the tensor because that's what the ESM script does.
                # TODO (KC): this may not be necessary because we are using numpy,
                # rather than torch, to save the tensors.
                mean_embedding = raw_embedding.mean(0).clone()
                mean_embeddings.append(mean_embedding.numpy())

        mean_embeddings = np.stack(mean_embeddings)
        print(f"Embeddings matrix shape: {mean_embeddings.shape}")

        if mean_embeddings.shape[1] != MODEL_NAMES_TO_DIMS[model_name]:
            print(
                f"Warning: Expected {MODEL_NAMES_TO_DIMS[model_name]}-dimensional embeddings, "
                f"but got {mean_embeddings.shape[1]}"
            )
        if mean_embeddings.shape[0] != len(dataset):
            raise ValueError(
                f"Expected {len(dataset)} embeddings, but only generated {mean_embeddings.shape[0]}"
            )

        np.save(output_filepath, mean_embeddings)
        print(f"Embeddings matrix written to '{output_filepath}'")
