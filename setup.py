from setuptools import setup, find_packages
from plotbitrate import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()    

setup(
    name='rezun-plotbitrate',
    version=__version__,
    packages=find_packages(),
    description='A simple bitrate plotter for media files',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Steve Schmidt',
    author_email='azcane@gmail.com',
    license='BSD',
    url='https://github.com/rezun/plotbitrate',
    py_modules=['plotbitrate'],
    classifiers=[
        'Topic :: Multimedia :: Sound/Audio',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='ffprobe bitrate plot',
    python_requires='>=3.5',
    entry_points={
        'console_scripts': [
            'plotbitrate = plotbitrate:main'
        ]
    },
    install_requires=[
        'matplotlib',
        'dataclasses;python_version=="3.6"',
        'pyqt5'
    ]
)
