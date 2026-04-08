# Updating the Documentation

## About the cfr documentation

`cfr`'s documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material](https://squidfunk.github.io/mkdocs-material/) theme. API reference pages are auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).
It is therefore especially important for your code to include a docstring, and to modify the docstrings of the functions/classes you modified to make sure the documentation is current.

## Updating docstrings

You may use existing docstrings as examples. A good docstring explains:

- what the function/class is about
- what it does, with what properties/inputs/outputs

## Updating tutorial notebooks

To update an existing or add a new tutorial notebook, you need to edit and execute the notebook locally under the `docs/notebooks` directory.

The notebooks are named following a naming convention: `<topic>-<details>.ipynb`
For instance, the `<topic>` can be `climate`, `proxy`, `psm`, `lmr`, `graphem`, and `pp2k`, and the `<details>` can be a keyword of your notebook, something like `ppe-pages2k`.
Note that if `<topic>` is `test`, the notebook will only be for a temporary test and will not be included in the documentation.

In the notebook, please make sure the below block is executed in the first cell so that the `cfr` version, an important message for the users, will be printed out:

```python
import cfr
print(cfr.__version__)
```

To include a new notebook in the User Guide, you need to:

1. Add the notebook to `docs/notebooks/`
2. Add a link to it in the corresponding `docs/*.md` guide page
3. Add an entry in `mkdocs.yml` under the appropriate section in `nav`

## Building the cfr documentation

Install the documentation dependencies:

```bash
pip install mkdocs mkdocs-material mkdocs-jupyter mkdocstrings[python]
```

Then serve the docs locally:

```bash
mkdocs serve
```

This starts a local development server at `http://127.0.0.1:8000/` with live reload.

## Publishing the documentation

To publish the documentation to GitHub Pages:

```bash
mkdocs gh-deploy
```

This builds the site and pushes it to the `gh-pages` branch.

## Pushing your changes

This step is the same as the step when you push your changes with the codebase, see [Pushing your changes](working-with-codebase.md#pushing-your-changes).
