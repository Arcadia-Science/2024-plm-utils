import pathlib

import click

from smallesm import datasets, embed, train, translate


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(embed.command, name="embed")
cli.add_command(translate.command, name="translate")
cli.add_command(train.command, name="train")
cli.add_command(datasets.cli, name="datasets")


@cli.command()
@click.argument("dirname_suffix", type=str, required=True)
@click.option("--train-id", type=str, required=True)
@click.option("--test-id", type=str, required=True)
def train_test(dirname_suffix, train_id, test_id):
    """
    Train and test a model using the given species.
    """
    dirpath = pathlib.Path("tmp/data/")
    coding_dirpath = dirpath / f"cdna-clustered-{dirname_suffix}"
    noncoding_dirpath = dirpath / f"ncrna-clustered-{dirname_suffix}"

    train.train(
        coding_train_filepath=coding_dirpath / f"{train_id}.npy",
        noncoding_train_filepath=noncoding_dirpath / f"{train_id}.npy",
        coding_test_filepath=coding_dirpath / f"{test_id}.npy",
        noncoding_test_filepath=noncoding_dirpath / f"{test_id}.npy",
    )
