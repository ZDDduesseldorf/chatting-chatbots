#!/usr/bin/env python
import os
from pathlib import Path
from setuptools import find_packages, setup


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "chattingchatbots", "__version__.py")) as f:
    exec(f.read(), version)

setup(
    name="chattingchatbots",
    version=version["__version__"],
    description="Student project Chatting Chatbots.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['*tests*']),
    package_data={"chattingchatbots": ["data/*.csv"]},
    license="MIT",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
    ],
    extras_require={"dev": ["isort>=5.1.0",
                            "pylint<2.12",
                            "prospector[with_pyroma]",
                            "pytest",
                            "pytest-cov",
                            "testfixtures",
                            "yapf",]},
)
