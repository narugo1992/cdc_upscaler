import numpy as np
import torch
from PIL import Image
from torch.nn.functional import pad as tensor_pad

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        # noinspection PyUnresolvedReferences
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    # noinspection PyUnresolvedReferences
    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def to_pil_image(pic, mode=None):
    """Convert a tensor or an cv.ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not (_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'
            # return Image.fromarray(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB), mode=mode)
    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def tensor_divide(tensor, psize, overlap, pad=True):
    """
    Divide Tensor Into Blocks, Especially for Remainder
    :param tensor:
    :param psize:
    :param overlap:
    :return: List
    """
    B, C, H, W = tensor.shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad
        if h_pad != 0 or w_pad != 0:
            tensor = tensor_pad(tensor, (0, w_pad, 0, h_pad), mode='reflect').data

    h_block = H // psize
    w_block = W // psize
    blocks = []
    if overlap != 0:
        tensor = tensor_pad(tensor, (overlap, overlap, overlap, overlap), mode='reflect').data

    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor.shape[2] if i + 1 == h_block else (i + 1) * psize + 2 * overlap
            end_w = tensor.shape[3] if j + 1 == w_block else (j + 1) * psize + 2 * overlap
            # end_h = (i + 1) * psize + 2 * overlap
            # end_w = (j + 1) * psize + 2 * overlap
            part = tensor[:, :, i * psize: end_h, j * psize: end_w]
            blocks.append(part)
    return blocks


def tensor_merge(blocks, tensor, psize, overlap, pad=True, tensor_shape=None):
    """
    Combine many small patch into one big Image
    :param blocks: List of 4D Tensors or just a 4D Tensor
    :param tensor:  has the same size as the big image
    :param psize:
    :param overlap:
    :return: Tensor
    """
    if tensor_shape is None:
        B, C, H, W = tensor.shape
    else:
        B, C, H, W = tensor_shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad

    tensor_new = torch.FloatTensor(B, C, H, W)
    h_block = H // psize
    w_block = W // psize
    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor_new.shape[2] if i + 1 == h_block else (i + 1) * psize
            end_w = tensor_new.shape[3] if j + 1 == w_block else (j + 1) * psize
            # end_h = (i + 1) * psize
            # end_w = (j + 1) * psize
            part = blocks[i * w_block + j]

            if len(part.shape) < 4:
                part = part.unsqueeze(0)

            tensor_new[:, :, i * psize: end_h, j * psize: end_w] = \
                part[:, :, overlap: part.shape[2] - overlap, overlap: part.shape[3] - overlap]

    # Remove Pad Edges
    if tensor_shape is None:
        B, C, H, W = tensor.shape
    else:
        B, C, H, W = tensor_shape
    tensor_new = tensor_new[:, :, :H, :W]
    return tensor_new
