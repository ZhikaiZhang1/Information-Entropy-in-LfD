from setuptools import setup
import os

requirementPath = os.path.dirname(os.path.realpath(__file__)) + "/requirements.txt"
with open(requirementPath) as f:
    install_requires = f.read().splitlines()
setup(
    name="pypbd",
    version="1.0",
    description="????",
    author="Sir Guyman",
    author_email="foomail@foo.com",
    packages=["pypbd"],  # same as name
    install_requires=install_requires,
)
