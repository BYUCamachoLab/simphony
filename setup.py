import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simphony",
    version="0.1.0",
    author="Sequoia Ploeg",
    author_email="sequoia.ploeg@ieee.org",
    description="Simphony: A Simulator for Photonic circuits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sequoiap/simphony",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)