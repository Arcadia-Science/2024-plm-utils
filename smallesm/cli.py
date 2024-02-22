import click

from smallesm import construct_datasets, embed, train, translate


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(embed.command, name="embed")
cli.add_command(translate.command, name="translate")
cli.add_command(train.command, name="train")
cli.add_command(construct_datasets.cli, name="construct-datasets")
