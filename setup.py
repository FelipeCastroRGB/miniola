from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sys
import subprocess

def get_opencv_flags():
    """Tenta localizar o caminho dos headers e bibliotecas do OpenCV via pkg-config do apt."""
    try:
        import pkgconfig
        return pkgconfig.parse('opencv4')
    except Exception:
        # Tenta os caminhos absolutos padrões do Raspberry Pi OS (Debian)
        return {
            'include_dirs': ['/usr/include/opencv4'], 
            'libraries': ['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs']
        }

cv_flags = get_opencv_flags()

ext_modules = [
    Pybind11Extension("miniola_cv",
        ["src/miniola_cv.cpp"],
        include_dirs=cv_flags.get('include_dirs', ['/usr/include/opencv4']),
        libraries=cv_flags.get('libraries', ['opencv_core', 'opencv_imgproc']),
        library_dirs=cv_flags.get('library_dirs', []),
        cxx_std=14
        ),
]

setup(
    name="miniola_cv",
    version="0.1.0",
    author="Miniola Scanner",
    description="Motor nativo OpenCV C++ para otimização do Miniola",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
