import logging
import os
import tempfile
import warnings

import click
import onnx
import onnxoptimizer
import onnxsim
import torch
from torch import nn

from tools.entry.base import GLOBAL_CONTEXT_SETTINGS
from tools.upscale import load_cdc_model, parse_ckpt_name


def get_torch_model(ckpt: str, scala=None, inc=3, n_HG=6, inter_supervis=True, gpus=1) -> nn.Module:
    try:
        _, model_scala, _ = parse_ckpt_name(ckpt)
        if scala and model_scala != scala:
            warnings.warn(f'Given scala {scala!r} not match with model\'s scala {model_scala!r}, '
                          f'value of argument \'scala\' will be ignored.')
        scala = model_scala
    except ValueError:
        if not scala:
            raise ValueError('Scala can not be extracted from ckpt\'s filename, please provide the scala of model.')

    return load_cdc_model(ckpt, scala, inc, n_HG, inter_supervis, gpus)


def _add_trans_command(cli: click.Group) -> click.Group:
    @cli.command('trans', help='Export torch model to onnx format',
                 context_settings={**GLOBAL_CONTEXT_SETTINGS})
    @click.option('--input', '-i', 'input_model_filename',
                  type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
                  help='Input filename for torch model.', required=True)
    @click.option('--output', '-o', 'output_model_filename', type=str,
                  help='Output filename for onnx model.', required=True)
    @click.option('--opset_version', 'opset_version', type=int, default=14,
                  help='Opset version for onnx exporting.', show_default=True)
    @click.option('--verbose', 'verbose', is_flag=True, default=False,
                  help='Enable verbose information when exporting.', show_default=True)
    @click.option('--no_optimize', 'no_optimize', is_flag=True, default=False,
                  help='Do not optimize the model', show_default=True)
    def trans(input_model_filename: str, output_model_filename: str, opset_version: int,
              verbose: bool, no_optimize: bool):
        logging.basicConfig(level=logging.INFO)
        model = get_torch_model(ckpt=input_model_filename)
        example_input = torch.randn((1, 3, 640, 640))

        with torch.no_grad(), tempfile.TemporaryDirectory() as td:
            onnx_model_file = os.path.join(td, 'model.onnx')
            torch.onnx.export(
                model,
                example_input,
                onnx_model_file,
                verbose=verbose,
                input_names=["input"],
                output_names=["output"],

                opset_version=opset_version,
                dynamic_axes={
                    "input": {0: "batch", 2: "height", 3: "width"},
                    "output": {0: "batch", 3: "height", 5: "width"},
                }
            )

            model = onnx.load(onnx_model_file)
            if not no_optimize:
                model = onnxoptimizer.optimize(model)
                model, check = onnxsim.simplify(model)
                assert check, "Simplified ONNX model could not be validated"

            output_model_dir, _ = os.path.split(output_model_filename)
            if output_model_dir:
                os.makedirs(output_model_dir, exist_ok=True)
            onnx.save(model, output_model_filename)

    return cli
