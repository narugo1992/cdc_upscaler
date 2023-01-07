import torch.utils.data as data

from .filetools import _all_images
from .functional import to_tensor
from .loaders import pil_loader


# import dlib


def _mod_crop(im, scala):
    w, h = im.size
    return im.crop((0, 0, w - w % scala, h - h % scala))


class SRDataListLR(data.Dataset):
    """
    DataSet for Large images, hard to read once
    need buffer
    need to random crop
    all the image are Big size (DIV2K for example)
    load image from name.txt which contains all images` paths
    """

    def __init__(self, data_path, scala=4, mode='RGB', transform=None, rgb_range=1.):
        """
        :param data_path: Path to data root
        :param scala: SR scala
        :param mode: 'RGB' or 'Y'
        """
        self.image_file_list = _all_images(data_path)
        print('Initializing DataSet, image list: %s ...' % data_path)
        print('Found %d Images...' % len(self.image_file_list))
        self.scala = scala
        self.mode = mode
        self.transform = transform
        self.rgb_range = rgb_range

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, index):
        data_ = {}

        im_path = self.image_file_list[index].strip('\n')
        data_['PATH'] = im_path

        image = pil_loader(self.image_file_list[index], mode=self.mode)

        lr_img = _mod_crop(image, self.scala)
        if self.transform is not None:
            lr_img = self.transform(lr_img)

        data_['LR'] = to_tensor(lr_img) * self.rgb_range
        return data_
