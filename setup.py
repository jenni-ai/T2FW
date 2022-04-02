from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='lfw',
    ext_modules=[
        cpp_extension.CppExtension('lfw', [
            'lfw/cpp/lfw.cpp',
            'lfw/cpp/lfw_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(
            no_python_abi_suffix=True)
    }
)
