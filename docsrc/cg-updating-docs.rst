Updating the Documentation
==========================

About the `cfr` documentation
"""""""""""""""""""""""""""""""""
`cfr`'s documentation is built automatically from the function and class docstrings, via `Sphinx The Docs <https://www.sphinx-doc.org>`_.
It is therefore especially important for your code to include a docstring, and to modify the docstrings of the functions/classes you modified to make sure the documentation is current.

Updating docstrings
"""""""""""""""""""
You may use existing docstrings as examples. A good docstring explains:

- what the function/class is about
- what it does, with what properties/inputs/outputs

Updating tutorial notebooks
"""""""""""""""""""""""""""

To update an existing or add a new tutorial notebook, you need to edit and execute the notebook locally under the `docsrc/notebooks` directory, and build the documentation following the guide in the next section.

The notebooks are named following a naming convention: `<topic>-<details>.ipynb`
For instance, the `<topic>` can be `climate`, `proxy`, `psm`, `lmr`, `graphem`, and `pp2k`, and the `<details>` can be a keyword of your notebook, something like `ppe-pages2k`.
Note that if `<topic>` is `test`, the notebook will only be for a temporary test and will be ignored for building.

In the notebook, please make sure the below block is executed in the first cell so that the `cfr` version, an important message for the users, will be printed out::

    import cfr
    print(cfr.__version__)

To include a new notebook into the User Guide, you need to edit the corresponding `.rst` files to include the relative path to your notebook.
For instance, if the new notebook is about proxy system modeling, then edit `docsrc/ug-psm.rst` and add the relative path to your notebook under the existing ones.

Building the `cfr` documentation
""""""""""""""""""""""""""""""""

Navigate to the `docsrc` folder and type `./build_publish.sh`.
This may require installing other sphinx-related packages and `pandoc <https://pandoc.org/installing.html>`_.
One may install them via the below command::

    pip install sphinx nbsphinx sphinx-book-theme numpydoc twine sphinx-copybutton sphinxcontrib-napoleon sphinx-design

Previewing the `cfr` documentation
""""""""""""""""""""""""""""""""""

Navigate to the `docs` folder and open the `index.html` file with a browser.

Pushing your changes
""""""""""""""""""""""""""""""""""

This step is same as the step when you push your changes with the codebase, see `here <cg-working-with-codebase.html#pushing-your-changes>`_.