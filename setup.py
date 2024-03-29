from setuptools import setup, find_packages

setup(
    name="napari_dab_cellcount",
    version="0.1.4",

    python_requires=">=3.9, <3.11",
    author="Jyotirmay Srivastava :: Heavily Inspired from Cellpose-napari",
    author_email="jyotirmaysrivastava.in@gmail.com",
    license="MIT",
    description="A napari plugin for counting cells.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "napari",
        "napari-plugin-engine>=0.1.4",
        "PyQt5",
        "PyQt5.sip",
        "numpy",
        "numba",
        "scipy",
        "torch",
        "opencv-python-headless",
        "natsort",
        "tqdm",
        "imagecodecs",
        "tifffile",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Framework :: napari",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    entry_points={
        "napari.plugin": [
            "napari_dab_cellcount = napari_dab_cellcount",
        ],
    },
)
