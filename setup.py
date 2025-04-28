from setuptools import setup, find_packages

setup(
    name="ac-experimental",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    ],
    author="soulitzer",
    author_email="soulitzer@example.com",
    description="New API for Activation Checkpointing",
    keywords="pytorch, activation, checkpointing, memory",
    url="https://github.com/soulitzer/ac-experimental",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)