from setuptools import setup, find_packages
from plotbitrate import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()    

setup(
    name='plotbitrate',
    version=__version__,
    packages=find_packages(),
    description='A simple bitrate plotter for media files',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Eric Work',
    author_email='work.eric@gmail.com',
    license='BSD',
    url='https://github.com/zeroepoch/plotbitrate',
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
        'pyqt5'
    ]
)
