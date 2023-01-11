import pytest
from PIL import Image

from .testings import get_testfile


@pytest.fixture()
def original_image() -> str:
    return get_testfile('eyjafjalla.jpg')


@pytest.fixture()
def original_image_pil(original_image) -> Image.Image:
    return Image.open(original_image).convert('RGB')


@pytest.fixture()
def upscaled_x3_image_pil() -> Image.Image:
    return Image.open(get_testfile('eyjafjalla_x3.png'))
