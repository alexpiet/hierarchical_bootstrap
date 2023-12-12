from setuptools import setup

setup(name='hierarchical-bootstrap',
    packages=['hierarchical_bootstrap'],
    description='implementation of a hierarchical bootstrap for pandas',
    url='https://github.com/alexpiet/hierarchical_bootstrap',
    author='Alex Piet',
    author_email="alexpiet@gmail.com",
    version='0.1.0',
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "scipy",
        "tqdm",
        ],
    )
