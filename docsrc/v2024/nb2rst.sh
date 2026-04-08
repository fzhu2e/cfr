#!/usr/bin/env bash

jupyter-nbconvert --to=rst ../notebooks/*.ipynb --output-dir='./tutorial/'

sed -i -e 's/parsed-literal::/code:: Output/g' tutorial/*.rst
rm tutorial/*.rst-e
