import os

import setuptools


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(BASE_DIR, "README.md"), "r", encoding="UTF-8") as readme:
    long_description = readme.read()
with open(
    os.path.join(BASE_DIR, "build", "requirements.txt"), "r", encoding="UTF-8"
) as requirements:
    install_requires = requirements.read().split("\n")


setuptools.setup(
    name="circa-clue",
    version="0.1.1",
    author="limjcst",
    author_email="limjcst@163.com",
    description="Causal Inference-based Root Cause Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NetManAIOps/CIRCA",
    packages=setuptools.find_packages(include=["circa", "circa.*"]),
    package_data={
        "circa.graph.r": ["*.R"],
    },
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: BSD License",
    ],
    license="BSD 3-Clause License",
)
