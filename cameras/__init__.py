def get_camera_provider(name: str, video_path: str = None):
    name = name.lower().strip()
    if name == 'ximea':
        from .ximea import XimeaAdapter
        return XimeaAdapter()
    elif name == 'pi':
        from .pi import PiCameraAdapter
        return PiCameraAdapter()
    elif name == 'mock':
        from .mock import MockCameraProvider
        return MockCameraProvider(video_path=video_path)
    else:
        raise ValueError(f"Câmera não suportada: {name}")
