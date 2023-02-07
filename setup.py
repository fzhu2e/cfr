from setuptools import setup, find_packages

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(
    name='cfr',  # required
    version='0.2.0',
    description='cfr: the Python package for Climate Field Reconstruction',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Feng Zhu, Julien Emile-Geay',
    author_email='fengzhu@ucar.edu, julieneg@usc.edu',
    url='https://github.com/fzhu2e/cfr',
    packages=find_packages(),
    include_package_data=True,
    license='BSD 3-Clause',
    zip_safe=False,
    keywords='paleocliamte reconstruction',
    scripts=['bin/cfr'],
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    install_requires=[
        'termcolor',
        'seaborn',
        'pandas',
        'tqdm',
        'xarray',
        'netCDF4',
        'nc-time-axis',
        'dask',
        'statsmodels',
        'sklearn',
        'pathos',
        'eofs',
        'plotly',
    ],
    extras_require={
        'psm': [
            'pathos',
            'fbm',
            'pyvsl',
        ],
        'ml': [
            'torch',
            'torchvision',
        ],
        'graphem': [
            'cfr-graphem',
        ]
    }
)
