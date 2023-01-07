import os.path
import sys
from functools import lru_cache
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from .model import HourGlassNetMultiScaleInt
from .utils import mod_crop, get_default_device, tensor_merge, tensor_divide, to_pil_image, to_tensor, pil_loader, \
    load_weights


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


def open_image(image_filename: str, scala: int = 4, transform=None, rgb_range: float = 1.0) -> torch.Tensor:
    image = pil_loader(image_filename, mode='RGB')
    lr_img = mod_crop(image, scala)
    if transform is not None:
        lr_img = transform(lr_img)

    tensor = to_tensor(lr_img) * rgb_range
    return tensor.reshape((1, *tensor.shape))


def image_upscale(input_filename: str, output_filename: str, ckpt: Optional[str] = None,
                  psize=512, overlap=64, scala=4, inc=3, n_HG=6, rgb_range=1.0,
                  inter_supervis=True, gpus=1):
    output_dir, _ = os.path.split(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not ckpt:
        ckpt = hf_hub_download(
            repo_id='narugo/cdc_pretrianed_model',
            filename='HGSR-MHR-anime_X4_280.pth',
        )

    # Init Net
    print('Build Generator Net...')

    generator = load_cdc_model(ckpt, scala, inc, n_HG, inter_supervis, gpus)
    with torch.no_grad():
        tensor = open_image(input_filename, scala=scala, rgb_range=rgb_range)
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
            print(f'Processing Image, Part: {idx + 1} / {iters}', end='\r')
            sys.stdout.flush()

        results = torch.cat(results, dim=0)
        sr_img = tensor_merge(results, None, psize * scala, overlap * scala,
                              tensor_shape=(B, C, H * scala, W * scala))

    image = to_pil_image(torch.clamp(sr_img[0].cpu() / rgb_range, min=0.0, max=1.0))
    image.save(output_filename)
    print(f'Saving to: {output_filename}')
    sys.stdout.flush()
