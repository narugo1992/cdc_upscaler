import logging
import os
import os.path
import re
import warnings
from functools import lru_cache
from typing import Optional, Union, Mapping, Callable

import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .model import HourGlassNetMultiScaleInt
from .utils import get_default_device, tensor_merge, tensor_divide, to_pil_image, to_tensor, load_weights


@lru_cache()
def load_cdc_model(ckpt: Optional[str], scala=4, inc=3, n_HG=6, inter_supervis=True, gpus=1):
    generator = HourGlassNetMultiScaleInt(
        in_nc=inc, out_nc=inc, upscale=scala,
        nf=64, res_type='res', n_mid=2, n_HG=n_HG, inter_supervis=inter_supervis
    )
    ckpt = os.path.normcase(os.path.normpath(os.path.abspath(ckpt)))
    generator = load_weights(generator, ckpt, gpus, just_weight=False, strict=True)
    generator = generator.to(get_default_device())
    generator.eval()
    return generator


def mod_crop(im, scala):
    w, h = im.size
    return im.crop((0, 0, w - w % scala, h - h % scala))


def image_to_tensor(image: Image.Image, scala: int = 4, transform=None, rgb_range: float = 1.0) -> torch.Tensor:
    lr_img = mod_crop(image.convert('RGB'), scala)
    if transform is not None:
        lr_img = transform(lr_img)

    tensor = to_tensor(lr_img) * rgb_range
    return tensor.reshape((1, *tensor.shape))


def open_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str):
        return Image.open(image)
    else:
        raise TypeError(f'Unknown image type - {image!r}.')


CKPT_NAME_PATTERN = re.compile(r'^HGSR-MHR-(?P<type>anime|anime-aug)_X(?P<scale>\d+)_(?P<steps>\d+)\.pth$')


def parse_ckpt_name(filename: str):
    matching = CKPT_NAME_PATTERN.fullmatch(os.path.basename(filename))
    if matching:
        return matching.group('type'), int(matching.group('scale')), int(matching.group('steps'))
    else:
        raise ValueError(f'Unrecognized filename of ckpt - {filename!r}.')


def _get_4x_ckpt() -> str:
    return hf_hub_download(
        repo_id='7eu7d7/CDC_anime',
        filename='HGSR-MHR-anime_X4_280.pth',
    )


_DEFAULT_CKPTS: Mapping[int, Callable[[], str]] = {
    4: _get_4x_ckpt
}


def _native_image_upscale(input_image: Union[str, Image.Image], ckpt: Optional[str] = None,
                          psize=512, overlap=64, scala=None, inc=3, n_HG=6, rgb_range=1.0,
                          inter_supervis=True, gpus=1) -> Image.Image:
    if not ckpt:
        ckpt = _get_4x_ckpt()

    try:
        _, model_scala, _ = parse_ckpt_name(ckpt)
        if scala and model_scala != scala:
            warnings.warn(f'Given scala {scala!r} not match with model\'s scala {model_scala!r}, '
                          f'value of argument \'scala\' will be ignored.')
        scala = model_scala
    except ValueError:
        if not scala:
            raise ValueError('Scala can not be extracted from ckpt\'s filename, please provide the scala of model.')

    # Init Net
    logging.info('Build Generator Net...')

    generator = load_cdc_model(ckpt, scala, inc, n_HG, inter_supervis, gpus)
    with torch.no_grad():
        tensor = image_to_tensor(open_image(input_image), scala=scala, rgb_range=rgb_range)
        B, C, H, W = tensor.shape
        blocks = tensor_divide(tensor, psize, overlap)
        blocks = torch.cat(blocks, dim=0)
        results = []

        iters = blocks.shape[0] // gpus if blocks.shape[0] % gpus == 0 else blocks.shape[0] // gpus + 1
        for idx in range(iters):
            if idx + 1 == iters:
                input_ = blocks[idx * gpus:]
            else:
                input_ = blocks[idx * gpus: (idx + 1) * gpus]
            hr_var = input_.to(get_default_device())
            sr_var, sr_map = generator(hr_var)

            if isinstance(sr_var, (list, tuple)):
                sr_var = sr_var[-1]

            results.append(sr_var.to('cpu'))
            logging.info(f'Processing Image, Part: {idx + 1} / {iters}')

        results = torch.cat(results, dim=0)
        sr_img = tensor_merge(results, None, psize * scala, overlap * scala,
                              tensor_shape=(B, C, H * scala, W * scala))

    return to_pil_image(torch.clamp(sr_img[0].cpu() / rgb_range, min=0.0, max=1.0))


def image_upscale(input_image: Union[str, Image.Image], scale: float,
                  psize=512, overlap=64, inc=3, n_HG=6, rgb_range=1.0,
                  inter_supervis=True, gpus=1) -> Image.Image:
    image = open_image(input_image)
    origin_width, origin_height = image.size
    target_width, target_height = map(lambda x: int(round(x * scale)), (origin_width, origin_height))

    _model_scales = sorted(_DEFAULT_CKPTS.keys())
    scales = []
    while scale > 1.0:
        found = None
        for s in _model_scales:
            found = s
            if s >= scale:
                break

        scales.append(found)
        scale /= found

    for scale_item in scales:
        image = _native_image_upscale(
            image, _DEFAULT_CKPTS[scale_item](), psize, overlap, scale_item,
            inc, n_HG, rgb_range, inter_supervis, gpus
        )

    return image.resize((target_width, target_height))
