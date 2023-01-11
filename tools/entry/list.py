import click

from .base import EXIST_TORCH_MODELS, GLOBAL_CONTEXT_SETTINGS


def _add_list_command(cli: click.Group) -> click.Group:
    @cli.command('list', help='List all the downloadable pth models',
                 context_settings={**GLOBAL_CONTEXT_SETTINGS})
    def list_():
        for model in EXIST_TORCH_MODELS:
            click.echo(model)

    return cli
