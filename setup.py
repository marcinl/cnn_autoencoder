from setuptools import setup, find_packages

setup(
    name="cnn_autoencoder",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
    ],
)
