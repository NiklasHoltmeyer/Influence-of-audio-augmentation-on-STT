# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('LICENSE') as f:
    license = f.read()
    
#with open('README.rst') as f:
    #readme = f.read()
    
setup(
    name='audioengine',
    version='0.1.0',
    description='',
    #long_description=readme,
    author='Niklas Holtmeyer',
    url='https://github.com/NiklasHoltmeyer/stt-audioengine',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        "modin",
        "librosa",
        "SoundFile",
        "pydub",
        "numpy",
        "scikit-learn",
        "tensorflow-io", 
        "swifter",
        "pandas", #==1.2.3
        "tensorflow",
        "tqdm",
        "sklearn",
        "torchaudio",
        "torch",
        "datasets", #hugging_face
        "jiwer>=2.2.0",
        "transformers",
        "wandb"
    ],
)