import pathlib

import click
import esm
import esm.data
import esm.pretrained
import numpy as np
import torch
import tqdm
from Bio import SeqIO

# the allowed ESM-2 model variants and their embeddings dimensions.
# (from https://github.com/facebookresearch/esm)
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
def command(fasta_filepath, model_name, layer_ind, output_filepath):
    embed(fasta_filepath, model_name, layer_ind, output_filepath)


def embed(fasta_filepath, model_name, layer_ind, output_filepath):
    """
    Generate per-sequence embeddings for the sequences in a FASTA file using an ESM model,
    and write the resulting matrix of embeddings to a numpy file.

    This function is loosely based on the `esm/scripts/extract.py` module
    from the facebookresearch/esm repository.

    TODO (KC): currently there is no explicit association between rows of the embeddings matrix
    and the sequences in the input FASTA file; we rely on the preservation of order,
    which is not a good idea.
    """
    output_filepath = pathlib.Path(output_filepath)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    # TODO (KC): understand the logic behind these numbers.
    # (they are copied from esm/scripts/extract.py)
    toks_per_batch = 4096
    truncation_seq_length = 1022

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device '{device}' for model inference.")

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model.to(device)

    dataset = esm.FastaBatchedDataset.from_file(fasta_filepath)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    # note: the dataloader yields batches in the form of `(sequence_ids, sequences, tokens)`.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )
    print(f"Read file '{fasta_filepath}' and found {len(dataset)} sequences")

    if layer_ind == -1:
        layer_ind = model.num_layers

    if layer_ind < 0 or layer_ind > model.num_layers:
        raise ValueError(f"Invalid layer index {layer_ind}")

    with torch.no_grad():
        sequence_ids = []
        mean_embeddings = []

        for sequence_headers, sequences, toks in tqdm.tqdm(dataloader, total=len(batches)):
            toks = toks.to(device, non_blocking=True)
            results = model(toks, repr_layers=[layer_ind], return_contacts=False)
            raw_embeddings = results["representations"][layer_ind]

            for ind, (sequence_header, sequence) in enumerate(
                zip(sequence_headers, sequences, strict=True)
            ):
                truncate_len = min(len(sequence), truncation_seq_length)

                # The first token is the BOS token, so we skip it.
                raw_embedding = raw_embeddings[ind, 1 : truncate_len + 1]

                # Call `detach` to create a new tensor that is not part of the computation graph.
                mean_embedding = raw_embedding.mean(dim=0).detach().cpu().numpy()

                # We need to keep track of the sequence IDs because the dataloader
                # does not preserve the order of the sequences.
                sequence_id = sequence_header.split(" ")[0]
                sequence_ids.append(sequence_id)
                mean_embeddings.append(mean_embedding)

        mean_embeddings = np.stack(mean_embeddings)
        print(f"Embeddings matrix shape: {mean_embeddings.shape}")

        if mean_embeddings.shape[1] != MODEL_NAMES_TO_DIMS[model_name]:
            print(
                f"Warning: expected {MODEL_NAMES_TO_DIMS[model_name]}-dimensional embeddings, "
                f"but got {mean_embeddings.shape[1]}."
            )
        if mean_embeddings.shape[0] != len(dataset):
            print(
                f"Warning: the number of sequences in the fasta file was {len(dataset)}, "
                f"but the number of generated embeddings is {mean_embeddings.shape[0]}."
            )

        # Reorder the embeddings matrix to match the order of the sequences in the input FASTA file.
        fasta_file_sequence_ids = [record.id for record in SeqIO.parse(fasta_filepath, "fasta")]
        sequence_id_to_ind = {sequence_id: ind for ind, sequence_id in enumerate(sequence_ids)}
        reordered_inds = [
            sequence_id_to_ind[sequence_id] for sequence_id in fasta_file_sequence_ids
        ]
        mean_embeddings = mean_embeddings[reordered_inds, :]

        np.save(output_filepath, mean_embeddings)
        print(f"Embeddings matrix written to '{output_filepath}'")
