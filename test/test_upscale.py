import pytest

from cdc_upscaler import image_upscale


@pytest.mark.unittest
class TestCdcUpscale:
    def test_image_upscale(self):
        image_upscale(
            'images/2797632045_90956483_p0_master1200_1_y.jpg',
        )

    def test_image_upscale_psize_256(self):
        image_upscale(
            'images/2797632045_90956483_p0_master1200_1_y.jpg',
            psize=256,
        )
