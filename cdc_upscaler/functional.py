from typing import List, Tuple

import numpy as np
from PIL import Image


def to_ndarray(pic: Image.Image) -> np.ndarray:
    if pic.mode != 'RGB':
        pic = pic.convert('RGB')

    img = np.asarray(pic, dtype=np.uint8)  # H, W, C
    array = np.ascontiguousarray(img.transpose(2, 0, 1)) / 255  # C, H, W
    return array.astype(np.float32)


def to_pil_image(pic: np.ndarray) -> Image.Image:
    channels, height, width = pic.shape
    assert channels == 3
    bytes_pic = (pic * 255).astype(np.uint8)  # C, H, W
    raw_array = bytes_pic.transpose(1, 2, 0)  # H, W, C
    return Image.fromarray(raw_array, mode='RGB')


def array_divide(array, psize, overlap) -> List[np.ndarray]:
    batch, channel, height, width = array.shape
    assert batch == 1, f'Only one sample in batch supported, but {batch!r} found.'

    # Pad to number that can be divisible
    h_pad = psize - height % psize if height % psize != 0 else 0
    w_pad = psize - width % psize if width % psize != 0 else 0
    height += h_pad
    width += w_pad
    if h_pad or w_pad:
        array = np.pad(array, ((0, 0), (0, 0), (0, h_pad), (0, w_pad)), mode='reflect')

    h_block = height // psize
    w_block = width // psize
    if overlap != 0:
        array = np.pad(array, ((0, 0), (0, 0), (overlap, overlap), (overlap, overlap)), mode='reflect')

    blocks = []
    for i in range(h_block):
        for j in range(w_block):
            part = array[:, :, i * psize: (i + 1) * psize + 2 * overlap, j * psize: (j + 1) * psize + 2 * overlap]
            blocks.append(part)

    return blocks


def array_merge(blocks: np.ndarray, shape: Tuple[int, int, int, int], psize, overlap):
    batch, channels, height, width = shape
    assert batch == 1, f'Only one sample in batch supported, but {batch!r} found.'

    h_pad = psize - height % psize if height % psize != 0 else 0
    w_pad = psize - width % psize if width % psize != 0 else 0
    height += h_pad
    width += w_pad

    array_new = np.empty((batch, channels, height, width), dtype=np.float32)
    h_block = height // psize
    w_block = width // psize
    for i in range(h_block):
        for j in range(w_block):
            part = np.expand_dims(blocks[i * w_block + j], axis=0)
            _, _, p_height, p_width = part.shape
            array_new[:, :, i * psize: (i + 1) * psize, j * psize: (j + 1) * psize] = \
                part[:, :, overlap: p_height - overlap, overlap: p_width - overlap]

    _, _, e_height, e_width = shape
    return array_new[:, :, :e_height, :e_width]
