import os

from huggingface_hub import HfFileSystem

hf_fs = HfFileSystem()

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)
EXIST_TORCH_MODELS = [
    os.path.basename(file) for file in
    hf_fs.glob('7eu7d7/CDC_anime/*.pth')
]
