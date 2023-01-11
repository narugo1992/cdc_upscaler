import click
from click.core import Context, Option

from cdc_upscaler.config.meta import __TITLE__, __VERSION__
from .base import GLOBAL_CONTEXT_SETTINGS
from .download import _add_download_subcommand
from .list import _add_list_command
from .trans import _add_trans_command


# noinspection PyUnusedLocal
def print_version(ctx: Context, param: Option, value: bool) -> None:
    """
    Print version information of cli
    :param ctx: click context
    :param param: current parameter's metadata
    :param value: value of current parameter
    """
    if not value or ctx.resilient_parsing:
        return  # pragma: no cover
    click.echo(f'Model tools for {__TITLE__}, version {__VERSION__}.')
    ctx.exit()


@_add_trans_command
@_add_list_command
@_add_download_subcommand
@click.group(context_settings=GLOBAL_CONTEXT_SETTINGS)
@click.option('-v', '--version', is_flag=True,
              callback=print_version, expose_value=False, is_eager=True,
              help="Show model tools' version information.")
def cli():
    pass  # pragma: no cover
