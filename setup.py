from setuptools import setup, find_packages
from distutils.core import setup, Extension
from distutils import sysconfig
from Cython.Distutils import build_ext


with open('README.rst', 'r') as fh:
    long_description = fh.read()

quiclib = Extension(
    'quiclib', sources=['./cfr/graphem/QUIC.cpp'],
    extra_link_args=['-llapack', '-lblas', '-lstdc++', '-fPIC'],
)

class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return os.path.join('./cfr/graphem', filename.replace(suffix, "")+ext)

setup(
    name='cfr',  # required
    version='0.1.2',
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
    cmdclass={'build_ext': NoSuffixBuilder},
    ext_modules=[quiclib],
    keywords='paleocliamte reconstruction',
    scripts=['bin/cfr'],
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
        'statsmodels',
        'sklearn',
        'pens',
        'pathos'
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
        ]
    }
)
