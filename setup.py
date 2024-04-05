from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='cfr',  # required
    version='2024.4.4',
    description='cfr: a Python package for Climate Field Reconstruction',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Feng Zhu, Julien Emile-Geay',
    author_email='fengzhu@ucar.edu, julieneg@usc.edu',
    url='https://github.com/fzhu2e/cfr',
    packages=find_packages(),
    include_package_data=True,
    license='BSD 3-Clause',
    zip_safe=False,
    keywords='climate field reconstruction',
    scripts=['bin/cfr'],
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=[
        'colorama',
        'seaborn',
        'pandas',
        'tqdm',
        'xarray',
        'netCDF4',
        'nc-time-axis',
        'dask',
        'statsmodels',
        'eofs',
        'plotly',
        'pyresample',
    ],
    extras_require={
        'psm': [
            'pathos',
            'fbm',
            'pyvsl',
        ],
        'ml': [
            'scikit-learn',
            'torch',
            'torchvision',
        ],
        'graphem': [
            'cython',
            'scikit-learn',
            'cfr-graphem',
        ]
    }
)
