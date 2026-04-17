pypi:
	rm -f ./dist/*
	python -m build
	twine upload ./dist/*
