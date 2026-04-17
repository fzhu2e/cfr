# Installation

## Install the Conda environment

You may skip this step if your Conda environment has been installed already.

### Step 1: Download the installation script for miniconda3

#### macOS (Intel)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

#### macOS (Apple Silicon)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

#### Linux

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### Step 2: Install Miniconda3

```bash
chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-*.sh
```

During the installation, a path `<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., `~/.bashrc` or `~/.zshrc`) as below to enable the `conda` package manager (remember to change `<base-path>` with your real location):

```bash
export PATH="<base-path>/bin:$PATH"
. <base-path>/etc/profile.d/conda.sh
```

### Step 3: Test your Installation

```bash
source ~/.bashrc  # assume you are using Bash shell
which python  # should return a path under <base-path>
which conda  # should return a path under <base-path>
```

## Install `cfr`

Taking a clean installation as example, first let's create a new environment named `cfr-env` via `conda`

```bash
conda create -n cfr-env python=3.13
conda activate cfr-env
```

Then install some dependencies via `conda`:

```bash
conda install jupyter notebook cartopy statsmodels pykdtree netcdf4
```

Once the above dependencies have been installed, simply

```bash
pip install cfr
```

and you are ready to

```python
import cfr
```

in Python.

If you'd like to also enable the usage of more advanced Proxy System Models in addition to linear regression based models, which requires some extra dependencies, simply

```bash
pip install "cfr[psm]"
```

and, you are ready to

```python
from cfr import psm
```

in Python.

Similarly, if you'd like to enable the usage of the GraphEM algorithm, simply

```bash
pip install cython  # in case it's not installed yet
pip install "cfr[graphem]"
```
