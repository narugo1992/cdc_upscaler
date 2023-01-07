import os

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def _is_image_file(filename):
    """
    judge if the file is an image file
    :param filename: path
    :return: bool of judgement
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def _all_images(path, sort=True):
    """
    return all images in the folder
    :param path: path to Data Folder, absolute path
    :return: 1D list of image files absolute path
    """
    # TODO: Tail Call Elimination
    abs_path = os.path.abspath(path)
    image_files = list()
    for subpath in os.listdir(abs_path):
        if os.path.isdir(os.path.join(abs_path, subpath)):
            image_files = image_files + _all_images(os.path.join(abs_path, subpath))
        else:
            if _is_image_file(subpath):
                image_files.append(os.path.join(abs_path, subpath))
    if sort:
        image_files.sort()
    return image_files
