import click

from plmutils import classifier, embed, translate
from plmutils.tasks import classify_orfs


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(classify_orfs.cli, name="classify-orfs")
cli.add_command(embed.command, name="embed")
cli.add_command(translate.command, name="translate")
cli.add_command(classifier.train_command, name="train")
cli.add_command(classifier.predict_command, name="predict")
