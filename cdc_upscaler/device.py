from typing import Optional

from onnxruntime import get_available_providers, get_all_providers

alias = {
    'gpu': "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
}


def get_onnx_provider(provider: Optional[str] = None):
    if not provider:
        if "CUDAExecutionProvider" in get_available_providers():
            return "CUDAExecutionProvider"
        else:
            return "CPUExecutionProvider"
    elif provider.lower() in alias:
        return alias[provider.lower()]
    else:
        for p in get_all_providers():
            if provider.lower() == p.lower() or f'{provider}ExecutionProvider'.lower() == p.lower():
                return p

        raise ValueError(f'One of the {get_all_providers()!r} expected, '
                         f'but unsupported provider {provider!r} found.')
