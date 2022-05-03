from setuptools import setup, find_packages

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(
    name='cfr',  # required
    version='0.0.7',
    description='cfr: the library for climate field reconstruction',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author='Feng Zhu',
    author_email='fzhu@nuist.edu.cn',
    url='https://github.com/fzhu2e/cfr',
    packages=find_packages(),
    include_package_data=True,
    license='BSD 3-Clause',
    zip_safe=False,
    keywords='paleocliamte reconstruction',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
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
    ],
    extras_require={
        'psm': [
            'pathos',
            'fbm',
            'statsmodels',
            'pyvsl',
        ],
        'ml': [
            'torch',
            'torchvision',
            'sklearn',
        ]
    }
)
