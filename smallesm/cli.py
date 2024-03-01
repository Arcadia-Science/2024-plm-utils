import click

from smallesm import datasets, embed, train, train_and_evaluate, translate


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(datasets.cli, name="datasets")
cli.add_command(embed.command, name="embed")
cli.add_command(translate.command, name="translate")
cli.add_command(train.train_command, name="train")
cli.add_command(train.predict_command, name="predict")
cli.add_command(train_and_evaluate.command, name="train-and-evaluate")
