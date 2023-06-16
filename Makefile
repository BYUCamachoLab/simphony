help:
	@echo 'make <TARGET>'
	@echo '     venv: Create a virtual environment'
	@echo '     install: Install the package in editable mode with all dependencies'
	@echo '     doc: Build the documentation'
	@echo '     serve: Start a server to serve the documentation'
	@echo '     format: Run the code formatter'
	@echo '     mypy: Run the code static anlaysis checker'
	@echo '     lint: Run the linter'
	@echo '     test: Run tests'
	@echo '     tox: Run tests with tox'
	@echo '     test-force: Run tests and force regeneration of test data'
	@echo '     clean: Remove all build artifacts'
	@echo '     build: Create Python build artifacts'
	@echo '     major: Bump major version number'
	@echo '     minor: Bump minor version number'
	@echo '     patch: Bump patch version number'
	@echo '     rc: Bump release candidate version number'
	@echo '     release: Release a new version'

install:
	pip install --upgrade pip setuptools wheel
	pip install -e .[dev,doc,test]
	pre-commit install

doc:
	rm docs/api
	jb clean docs && jb build docs

serve:
	cd docs/_build/html && python3 -m http.server

format:
	black simphony

mypy:
	mypy -p simphony

lint:
	flake8 .

test:
	coverage run -m pytest
	coverage report

tox:
	tox -e py

test-force:
	pytest --force-regen

clean:
	rm -rf dist
	rm -r docs/api
	jb clean docs

build:
	rm -rf dist
	python3 -m build

major:
	bumpversion major
	VERSION=$(shell python3 -c "import simphony; print(simphony.__version__)") && \
	python3 scripts/create_changelog_entry.py $$VERSION

minor:
	bumpversion minor
	VERSION=$(shell python3 -c "import simphony; print(simphony.__version__)") && \
	python3 scripts/create_changelog_entry.py $$VERSION

patch:
	bumpversion patch
	VERSION=$(shell python3 -c "import simphony; print(simphony.__version__)") && \
	python3 scripts/create_changelog_entry.py $$VERSION

rc:
	bumpversion build

release:
	pre-commit run --all-files
	python3 scripts/release.py
