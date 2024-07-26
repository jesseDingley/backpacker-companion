from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

setup(
    name="backpacker-companion",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'init-vectordb=backend.core:init_vectordb',
            'update-vectordb=backend.core:update_vectordb',
        ],
    },
    install_requires=reqs,
    python_requires=">=3.11",
)
