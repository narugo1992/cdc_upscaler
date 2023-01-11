# CDC Image Upscaler

[![PyPI](https://img.shields.io/pypi/v/cdc_upscaler)](https://pypi.org/project/cdc_upscaler/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cdc_upscaler)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/narugo1992/cdc_upscaler/blob/main/examples/cdc_upscaler_example.ipynb)
![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/76c126fca51d24785534a1f3c8cac20d/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/narugo1992/76c126fca51d24785534a1f3c8cac20d/raw/comments.json)

[![Code Test](https://github.com/narugo1992/cdc_upscaler/workflows/Code%20Test/badge.svg)](https://github.com/narugo1992/cdc_upscaler/actions?query=workflow%3A%22Code+Test%22)
[![Package Release](https://github.com/narugo1992/cdc_upscaler/workflows/Package%20Release/badge.svg)](https://github.com/narugo1992/cdc_upscaler/actions?query=workflow%3A%22Package+Release%22)
[![codecov](https://codecov.io/gh/narugo1992/cdc_upscaler/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/narugo1992/cdc_upscaler)

![GitHub Org's stars](https://img.shields.io/github/stars/narugo1992)
[![GitHub stars](https://img.shields.io/github/stars/narugo1992/cdc_upscaler)](https://github.com/narugo1992/cdc_upscaler/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/narugo1992/cdc_upscaler)](https://github.com/narugo1992/cdc_upscaler/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/narugo1992/cdc_upscaler)
[![GitHub issues](https://img.shields.io/github/issues/narugo1992/cdc_upscaler)](https://github.com/narugo1992/cdc_upscaler/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/narugo1992/cdc_upscaler)](https://github.com/narugo1992/cdc_upscaler/pulls)
[![Contributors](https://img.shields.io/github/contributors/narugo1992/cdc_upscaler)](https://github.com/narugo1992/cdc_upscaler/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/narugo1992/cdc_upscaler)](https://github.com/narugo1992/cdc_upscaler/blob/master/LICENSE)

Wrapped tools based
on [xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution).

First you need to install this with `pip`:

```shell
pip install cdc_upscaler
```

Here is a simple example:

```python
import logging
import os

from PIL import Image

from cdc_upscaler import image_upscale

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    original_image = Image.open('images/your input image.png')

    # any scale is supported, such as 1.5, 2, even 6 (which may take some more time)
    upscaled_image = image_upscale(original_image, scale=4)
    os.makedirs('output', exist_ok=True)
    upscaled_image.save('output/result.png')

```

| **#** |                                                  **original**                                                  |                                                        **4x**                                                        |
|:-----:|:--------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|
|   1   |        ![angelina.png](https://github.com/narugo1992/cdc_upscaler/blob/main/test/testfile/angelina.png)        |        ![angelina_x4.png](https://github.com/narugo1992/cdc_upscaler/blob/main/test/testfile/angelina_x4.png)        |
|   2   | ![angelina_elite2.png](https://github.com/narugo1992/cdc_upscaler/blob/main/test/testfile/angelina_elite2.png) | ![angelina_elite2_x4.png](https://github.com/narugo1992/cdc_upscaler/blob/main/test/testfile/angelina_elite2_x4.png) |

This pretrained model is hosted on [7eu7d7/CDC_anime](https://huggingface.co/7eu7d7/CDC_anime), which is provided
by [7eu7d7](https://github.com/7eu7d7). The onnx model used is hosted
on [narugo/CDC_anime_onnx](https://huggingface.co/narugo/CDC_anime_onnx/tree/main).
