import pytest
from PIL import Image

from cdc_upscaler import image_upscale
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


@pytest.mark.unittest
class TestCdcUpscale:
    def test_image_upscale(self, original_image_pil, image_diff, upscaled_x3_image_pil):
        upscaled_image = image_upscale(original_image_pil, 3)
        assert upscaled_image.size == (1536, 1536)
        assert image_diff(upscaled_image, upscaled_x3_image_pil, throw_exception=False) < 0.2

    def test_image_upscale_psize_256(self, original_image_pil, image_diff, upscaled_x3_image_pil):
        upscaled_image = image_upscale(original_image_pil, 3, psize=256)
        assert upscaled_image.size == (1536, 1536)
        assert image_diff(upscaled_image, upscaled_x3_image_pil, throw_exception=False) < 0.2
