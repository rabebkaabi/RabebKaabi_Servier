from setuptools import setup, find_packages


setup(
    name='myflaskapp',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'Flask',
        'tensorflow',
        'rdkit',
    ],
    entry_points={
        'console_scripts': [
            'servier=myflaskapp.app:main',
        ],
    },
)
