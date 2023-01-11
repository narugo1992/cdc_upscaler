import os.path
import shutil

import click
from huggingface_hub import hf_hub_download

from .base import GLOBAL_CONTEXT_SETTINGS, EXIST_TORCH_MODELS


def _add_download_subcommand(cli: click.Group) -> click.Group:
    @cli.command('download', help='Export torch model to onnx format',
                 context_settings={**GLOBAL_CONTEXT_SETTINGS})
    @click.option('--repo_id', 'repo_id', type=str, default='7eu7d7/CDC_anime',
                  help="Repository id on hugging face.", show_default=True)
    @click.option('--filename', 'filename', type=click.Choice(EXIST_TORCH_MODELS), required=True,
                  help='Filename in repository.')
    @click.option('--output', '-o', 'output_filename', type=str, required=True,
                  help='Output model file on local drive.')
    def download(repo_id: str, filename: str, output_filename: str):
        cached_file = hf_hub_download(repo_id=repo_id, filename=filename)
        output_dir, _ = os.path.split(output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        shutil.copyfile(cached_file, output_filename)

    return cli
