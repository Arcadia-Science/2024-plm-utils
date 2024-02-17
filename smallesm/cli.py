import click

from smallesm import embed, translate


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(embed.embed_command, name="embed")
cli.add_command(translate.translate_command, name="translate")
