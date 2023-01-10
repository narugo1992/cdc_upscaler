import numpy as np
import pytest
import torch

from cdc_upscaler.functional import to_ndarray, to_pil_image, array_divide, array_merge
from tools.utils.functional import to_tensor, tensor_divide, tensor_merge


@pytest.fixture()
def original_image_array(original_image_pil) -> np.ndarray:
    return to_tensor(original_image_pil).numpy()


@pytest.fixture()
def batched_image_array(original_image_array) -> np.ndarray:
    return np.expand_dims(original_image_array, axis=0)


@pytest.mark.unittest
class TestFunctional:
    def test_to_ndarray(self, original_image_pil):
        torch_result = to_tensor(original_image_pil)
        np_result = to_ndarray(original_image_pil)
        assert torch.isclose(torch.from_numpy(np_result), torch_result).all()

    def test_to_pil_image(self, original_image_pil, image_diff, original_image_array):
        image = to_pil_image(original_image_array)
        assert image_diff(original_image_pil, image, throw_exception=False) < 1e-3

    @pytest.mark.parametrize(['psize', 'overlap'], [(512, 64), (256, 64), (250, 37), (73, 0)])
    def test_array_divide(self, batched_image_array, psize, overlap):
        t_result = tensor_divide(torch.from_numpy(batched_image_array), psize=psize, overlap=overlap)
        np_result = array_divide(batched_image_array, psize=psize, overlap=overlap)

        assert len(t_result) == len(np_result), \
            f'Blocks count not match, {len(t_result)} expected but {len(np_result)} found.'
        for i, (t_item, np_item) in enumerate(zip(t_result, np_result)):
            assert np_item.shape == t_item.shape, \
                f'Array shape not match on {i}th item, {t_item.shape!r} expected but {np_item.shape!r} found.'
            assert torch.isclose(torch.from_numpy(np_item), t_item).all(), \
                f'Divide result match on {i}th item.'

    @pytest.mark.parametrize(['psize', 'overlap'], [(512, 64), (256, 64), (250, 37), (73, 0)])
    def test_array_merge(self, batched_image_array, psize, overlap, original_image_array):
        divided = torch.cat(tensor_divide(torch.from_numpy(batched_image_array),
                                          psize=psize, overlap=overlap)).numpy()

        channels, height, width = original_image_array.shape
        t_result = tensor_merge(torch.from_numpy(divided), None, psize, overlap,
                                tensor_shape=(1, channels, height, width))
        np_result = array_merge(divided, (1, channels, height, width), psize, overlap)
        assert np.isclose(t_result, np_result).all()

    @pytest.mark.parametrize(['psize', 'overlap'], [(512, 64), (256, 64), (250, 37), (73, 0)])
    def test_full(self, batched_image_array, psize, overlap, original_image_pil, image_diff):
        divided = array_divide(batched_image_array, psize=psize, overlap=overlap)
        width, height = original_image_pil.size
        array_data = array_merge(np.concatenate(divided, axis=0), (1, 3, height, width), psize, overlap)
        image = to_pil_image(array_data[0])
        assert image_diff(original_image_pil, image, throw_exception=True) < 1e-3
