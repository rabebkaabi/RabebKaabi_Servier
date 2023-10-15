from setuptools import setup, find_packages

setup(
    name="myflaskapp",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'servier=myflaskapp.cli:main',
        ],
    },
)
