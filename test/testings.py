import os.path

here = os.path.abspath(os.path.dirname(__file__))


def get_testfile(*path: str):
    return os.path.normpath(os.path.join(here, 'testfile', *path))
