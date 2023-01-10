import os
from unittest import skipUnless

import pytest

from cdc_upscaler.device import get_onnx_provider


@pytest.mark.unittest
class TestDevice:
    @skipUnless(not os.environ.get('GPU'), 'No GPU required')
    def test_get_onnx_provider_cpu(self):
        assert get_onnx_provider() == 'CPUExecutionProvider'

    @skipUnless(os.environ.get('GPU'), 'GPU required')
    def test_get_onnx_provider_gpu(self):
        assert get_onnx_provider() == 'CUDAExecutionProvider'

    def test_get_onnx_provider(self):
        assert get_onnx_provider('gpu') == 'CUDAExecutionProvider'
        assert get_onnx_provider('trt') == 'TensorrtExecutionProvider'

        assert get_onnx_provider('cuda') == 'CUDAExecutionProvider'
        assert get_onnx_provider('cpu') == 'CPUExecutionProvider'

        with pytest.raises(ValueError):
            _ = get_onnx_provider('wtf')
