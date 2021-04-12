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
        #"librosa~=0.8.0",
        #"SoundFile~=0.10.3.post1",
        #"pydub~=0.25.1",
        #"numpy~=1.19.5",
        #"scikit-learn~=0.24.1"
        "librosa",
        "SoundFile",
        "pydub",
        "numpy",
        "scikit-learn",
        "tensorflow-io", 
        "swifter"
    ],
)

