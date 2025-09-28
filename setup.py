from setuptools import setup, find_packages
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Flipkart Recommender",
    version="0.1",
    author="Divyansh",
    packages=find_packages(),
    install_requires=requirements,
)