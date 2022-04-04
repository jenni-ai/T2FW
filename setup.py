from setuptools import setup, Extension
from torch.utils import cpp_extension

import os
import codecs


def readme():
    """ Return the README text.
    """
    with codecs.open('README.md', encoding='utf-8') as fh:
        return fh.read()


setup(
    # Package information
    name='lfw',
    version='0.0.1',
    description='Fine-Tuning Pre-trained Transformers into Decaying Fast Weights',
    long_description=readme(),
    long_description_content_type="text/markdown",

    # Author information
    url='https://github.com/altum-io/T2FW',
    author='Huanru Henry Mao',
    author_email='henry@altum.io',

    # In the package
    packages=['lfw', 'lfw/csrc'],
    package_data={
        '': ['*.txt', '*.rst', '*.md', '*.cpp', '*.cu']
    },
)
