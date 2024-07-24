from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

setup(
    name="backpacker-companion",
    version="0.1.0",
    packages=find_packages(),
    install_requires=reqs,
    python_requires=">=3.11",
)
