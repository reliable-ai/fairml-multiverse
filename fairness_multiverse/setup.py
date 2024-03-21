from setuptools import find_packages, setup

setup(
    name="fairness_multiverse",
    packages=find_packages(),
    version="0.1.0",
    description=(
        "Helper package to assisst in conducting "
        "a multiverse analysis of algorithmic fairness"
    ),
    author="Jan Simson, Florian Pfisterer, Christoph Kern",
    license="MIT",
)
