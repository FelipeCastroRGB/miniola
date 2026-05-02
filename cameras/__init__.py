from .pi import PiCameraAdapter
from .ximea import XimeaAdapter

def get_camera_provider(name: str):
    name = name.lower().strip()
    if name == 'ximea':
        return XimeaAdapter()
    elif name == 'pi':
        return PiCameraAdapter()
    else:
        raise ValueError(f"Câmera não suportada: {name}")
