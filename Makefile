
install:
	pip install -r requirements-dev.txt --upgrade
	pip install -e .
	pre-commit install

lint:
	flake8 .

test:
	pytest

test-force:
	pytest --force-regen
