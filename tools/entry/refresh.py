import logging
import os.path

import click
from tqdm import tqdm

from cdc_upscaler import image_upscale
from .base import GLOBAL_CONTEXT_SETTINGS

EXAMPLE_IMAGES = [
    ('angelina.png', 4),
    ('angelina_elite2.png', 4)
]


def _add_refresh_command(cli: click.Group) -> click.Group:
    @cli.command('refresh', help='Refresh the example pictures',
                 context_settings={**GLOBAL_CONTEXT_SETTINGS})
    @click.option('--psize', 'psize', type=int, default=256,
                  help='Psize for image upscale.', show_default=True)
    @click.option('--overlap', 'overlap', type=int, default=32,
                  help='Overlap for image upscale.', show_default=True)
    def refresh(psize: int, overlap: int):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s][v2raycli] %(message)s",
            datefmt='%Y/%m/%d %H:%M:%S',
        )
        iters = tqdm(EXAMPLE_IMAGES)
        for i, (filename, scale) in enumerate(iters, start=1):
            iters.set_description(f'{i}th - {filename} x{scale}')
            srcfile = os.path.join('test', 'testfile', filename)
            f_body, f_ext = os.path.split(filename)
            dstfile = os.path.join('test', 'testfile', f'{f_body}_x{scale}{f_ext}')

            new_image = image_upscale(srcfile, scale=scale, psize=psize, overlap=overlap)
            new_image.save(dstfile)

    return cli
