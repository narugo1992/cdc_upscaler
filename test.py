import pytest

from demo import image_upscale


@pytest.mark.unittest
class TestRunTest:
    def test_image_upscale(self):
        image_upscale(
            'images/2797632045_90956483_p0_master1200_1_y.jpg',
            'output/result.png'
        )

    def test_image_upscale_psize_256(self):
        image_upscale(
            'images/2797632045_90956483_p0_master1200_1_y.jpg',
            'output/result.png',
            psize=256,
        )
