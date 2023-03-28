from setuptools import find_packages
from distutils.core import setup
 
#setup(name = "MLTools", packages = find_packages())
setup(
    name='MLTools',
    version='1.0',
    author='Bilal EL SAFAH',
    description='Neural network development tools',
    url='https://github.com/bilal02s/Digit-String-Recognition',
    keywords='deep learning, neural network',
    python_requires='>=3.7, <4',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'Pillow'
    ]
)