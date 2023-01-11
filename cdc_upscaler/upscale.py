import logging
import os
import re
import warnings
from functools import lru_cache
from typing import Union, Optional, Mapping, Callable, List, Tuple

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

from .device import _ensure_onnxruntime

_ensure_onnxruntime()
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

from .device import get_onnx_provider
from .functional import to_ndarray, array_divide, array_merge, to_pil_image


def _open_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise TypeError(f'Unknown image type - {image!r}.')


CKPT_NAME_PATTERN = re.compile(r'^HGSR-MHR-(?P<type>anime|anime-aug)_X(?P<scale>\d+)_(?P<steps>\d+)\.onnx$')


def parse_ckpt_name(filename: str):
    matching = CKPT_NAME_PATTERN.fullmatch(os.path.basename(filename))
    if matching:
        return matching.group('type'), int(matching.group('scale')), int(matching.group('steps'))
    else:
        raise ValueError(f'Unrecognized filename of ckpt - {filename!r}.')


@lru_cache()
def _open_onnx_model(ckpt: str, provider: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "CPUExecutionProvider":
        options.intra_op_num_threads = os.cpu_count()

    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, [provider])


def _get_onnx_4x_model() -> str:
    return hf_hub_download(
        repo_id='narugo/CDC_anime_onnx',
        # filename='HGSR-MHR-anime_X4_280.onnx',
        filename='HGSR-MHR-anime-aug_X4_320.onnx',
    )


INPUT_UNIT = 16


def _native_image_upscale(image: Union[str, Image.Image], ckpt: str, scala: Optional[int] = None,
                          provider: Optional[str] = None, psize: int = 512, overlap=64, batch: int = 1,
                          rgb_range: float = 1.0) -> Image.Image:
    ckpt = os.path.normcase(os.path.normpath(os.path.abspath(ckpt)))
    model = _open_onnx_model(ckpt, get_onnx_provider(provider))

    try:
        _, model_scala, _ = parse_ckpt_name(ckpt)
        if scala and model_scala != scala:
            warnings.warn(f'Given scala {scala!r} not match with model\'s scala {model_scala!r}, '
                          f'value of argument \'scala\' will be ignored.')
        scala = model_scala
    except ValueError:
        if not scala:
            raise ValueError('Scala can not be extracted from ckpt\'s filename, please provide the scala of model.')

    image: Image.Image = _open_image(image)
    raw_data = np.expand_dims(to_ndarray(image) * rgb_range, axis=0)
    o_batch, o_channels, o_height, o_width = raw_data.shape
    assert o_batch == 1 and o_channels == 3

    divided = array_divide(raw_data, psize, overlap)
    divided = np.concatenate(divided)

    iters = (divided.shape[0] + batch - 1) // batch
    results = []
    for i in range(iters):
        logging.info(f'Inference {i + 1} / {iters} ...')
        input_ = divided[i * batch: (i + 1) * batch]
        b_batch, b_channels, b_height, b_width = input_.shape
        nb_height = b_height + (0 if b_height % INPUT_UNIT == 0 else INPUT_UNIT - b_height % INPUT_UNIT)
        nb_width = b_width + (0 if b_width % INPUT_UNIT == 0 else INPUT_UNIT - b_width % INPUT_UNIT)
        real_input = np.pad(input_, ((0, 0), (0, 0), (0, nb_height - b_height), (0, nb_width - b_width)),
                            mode='reflect')
        output, = model.run(['output'], {'input': real_input})
        results.append(output.reshape(b_batch, b_channels, nb_height * scala, nb_width * scala)
                       [:, :, :b_height * scala, :b_width * scala])

    results = np.concatenate(results)
    final_data = array_merge(results, (o_batch, o_channels, o_height * scala, o_width * scala),
                             psize * scala, overlap * scala)
    f_batch, f_channels, f_height, f_width = final_data.shape
    assert f_batch == 1 and f_channels == 3 and f_height == o_height * scala and f_width == o_width * scala
    return to_pil_image(np.clip(final_data[0] / rgb_range, a_min=0.0, a_max=1.0))


_DEFAULT_CKPTS: Mapping[int, Callable[[], str]] = {
    4: _get_onnx_4x_model,
}


def _parse_custom_ckpt(ckpt: Union[Mapping[int, str], List[Union[str, Tuple[int, str]]]]) -> Mapping[int, str]:
    data = {}
    if isinstance(ckpt, dict):
        ckpt = list(ckpt.items())

    for i, item in enumerate(ckpt):
        if isinstance(item, str):
            scala, path = parse_ckpt_name(item), item
        elif isinstance(item, tuple):
            scala, path = item
        else:
            raise TypeError(f'Unknown custom ckpt type on {i}th - {item!r}.')

        data[scala] = path

    return data


def image_upscale(image: Union[str, Image.Image], scale: float,
                  ckpt: Union[Mapping[int, str], List[Union[str, Tuple[int, str]]], None] = None,
                  provider: Optional[str] = None, psize: int = 512, overlap=64, batch: int = 1,
                  rgb_range: float = 1.0) -> Image.Image:
    image: Image.Image = _open_image(image)
    origin_width, origin_height = image.size
    target_width, target_height = map(lambda x: int(round(x * scale)), (origin_width, origin_height))

    _real_ckpts = {**_DEFAULT_CKPTS, **_parse_custom_ckpt(ckpt or {})}
    _model_scales = sorted(_real_ckpts.keys())
    scales = []
    while scale > 1.0:
        found = None
        for s in _model_scales:
            found = s
            if s >= scale:
                break

        scales.append(found)
        scale /= found
    logging.info(f'Scheduled upscale plan: {[*scales, scale]!r}')

    for i, scale_item in enumerate(scales):
        logging.info(f'Upscaling step {i + 1} / {len(scales)} - {scale_item}x ...')
        image = _native_image_upscale(
            image, _real_ckpts[scale_item](), scale_item, provider,
            psize, overlap, batch, rgb_range
        )

    return image.resize((target_width, target_height))
