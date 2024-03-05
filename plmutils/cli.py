import click

from plmutils import classify, embed, translate
from plmutils.tasks import orf_prediction


@click.group(invoke_without_command=True, context_settings=dict(help_option_names=["-h", "--help"]))
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(embed.command, name="embed")
cli.add_command(translate.command, name="translate")
cli.add_command(classify.train_command, name="train")
cli.add_command(classify.predict_command, name="predict")
cli.add_command(orf_prediction.cli, name="orf-prediction")
