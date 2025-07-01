from setuptools import setup, find_packages

with open("requirements-dev.txt") as f:
    reqs = f.read().splitlines()

setup(
    name="backpacker-companion",
    version="0.2",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "init-vectordb=backend.core:init_vectordb",
            "update-vectordb=backend.core:update_vectordb",
            "init-docstore=backend.core:init_docstore"
        ],
    },
    install_requires=reqs,
    python_requires=">=3.11",
)
