# CDC Image Upscaler

Wrapped tools based
on [xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution).

First you need to install this repo (PyPI package is coming soon):

```shell
git clone https://github.com/narugo1992/cdc_upscaler.git
cd cdc_upscaler
pip install -r requirements.txt
```

Here is a simple example:

```python
from cdc_upscaler import image_upscale

if __name__ == '__main__':
    image_upscale(
        input_filename='images/your input image',
        output_filename='output/result.png',
    )

```

At present, only 4x up-scaling is supported. More features is coming soon.

This pretrained model is hosted on [hugging face](https://huggingface.co/narugo/cdc_pretrianed_model), which is provided
by [7eu7d7](https://github.com/7eu7d7).

