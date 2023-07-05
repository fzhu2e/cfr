Installation
===============


Install the Conda environment
-----------------------------

You may skip this step if your Conda environment has been installed already.

Step 1: Download the installation script for miniconda3
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

macOS (Intel)
'''''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

macOS (Apple Silicon)
''''''''''

.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

Linux
'''''
.. code-block:: bash

  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Step 2: Install Miniconda3
"""""""""""""""""""""""""""

.. code-block:: bash

  chmod +x Miniconda3-latest-*.sh && ./Miniconda3-latest-*.sh

During the installation, a path :code:`<base-path>` needs to be specified as the base location of the python environment.
After the installation is done, we need to add the two lines into your shell environment (e.g., :code:`~/.bashrc` or :code:`~/.zshrc`) as below to enable the :code:`conda` package manager (remember to change :code:`<base-path>` with your real location):

.. code-block:: bash

  export PATH="<base-path>/bin:$PATH"
  . <base-path>/etc/profile.d/conda.sh

Step 3: Test your Installation
"""""""""""""""""""""""""""""""

.. code-block:: bash

  source ~/.bashrc  # assume you are using Bash shell
  which python  # should return a path under <base-path>
  which conda  # should return a path under <base-path>


Install `cfr`
---------------


Taking a clean installation as example, first let's create a new environment named :code:`cfr-env` via :code:`conda`

.. code-block:: bash

    conda create -n cfr-env python=3.10
    conda activate cfr-env

Then install some dependencies via :code:`conda`:

.. code-block:: bash

    conda install jupyter notebook cartopy statsmodels

Once the above dependencies have been installed, simply

.. code-block:: bash

    pip install cfr

and you are ready to

.. code-block:: python

    import cfr

in Python.

If you'd like to also enable the usage of the Proxy System Models, which requires some extra dependencies, simply

.. code-block:: bash

    pip install "cfr[psm]"

and, you are ready to

.. code-block:: python

    from cfr import psm

in Python.

Similarly, if you'd like to enable the usage of the GraphEM algorithm, simply

.. code-block:: bash

    pip install cython  # in case it's not installed yet
    pip install "cfr[graphem]"
