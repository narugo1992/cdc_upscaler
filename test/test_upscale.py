import pytest

from cdc_upscaler import image_upscale


@pytest.mark.unittest
class TestCdcUpscale:
    def test_image_upscale(self, original_image_pil, image_diff, upscaled_x3_image_pil):
        upscaled_image = image_upscale(original_image_pil, 3)
        assert upscaled_image.size == (1536, 1536)
        assert image_diff(upscaled_image, upscaled_x3_image_pil, throw_exception=False) < 0.03

    def test_image_upscale_psize_256(self, original_image_pil, image_diff, upscaled_x3_image_pil):
        upscaled_image = image_upscale(original_image_pil, 3, psize=267, overlap=17)
        assert upscaled_image.size == (1536, 1536)
        assert image_diff(upscaled_image, upscaled_x3_image_pil, throw_exception=False) < 0.03
