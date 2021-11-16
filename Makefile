# Makefile
SHELL := /bin/bash

.PHONY: help
help:
	@echo "Commands:"
	@echo "install	: install required packages"
	@echo "venv		: list of main operations."
	@echo "style	: creates development environment."
	@echo "dvc		: pushes versioned data and models to storage."
	@echo "test		: run non-training tests."

.PHONY: install
install:
	python -m pip install -e . --no-cache-dir

.ONESHELL:
venv:
	python -m venv venv
	source venv/Scripts/activate && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e ".[dev]" --no-cache-dir && \
	pre-commit install && \
	pre-commit autoupdate

# Styling
.PHONY: style
style:
	black .
	flake8
	isort .

# DVC
.PHONY: dvc
dvc:
	dvc add cord19_data/database/articles.sqlite
	dvc add cord19_data/metadata/entry-dates.csv
	dvc add cord19_data/metadata/metadata.csv

	dvc add https://drive.google.com/drive/u/0/folders/112AvV5iSXu0KkItAKhhMPZjwHHmIYyy4 \
			-o t5/checkpoints/sentence_selection --to-remote -r stores
	dvc add https://drive.google.com/drive/u/0/folders/12eBOXaFeQ9ww-hjo7TRRD9UB6QrRJnm7 \
			-o t5/checkpoints/label_prediction --to-remote -r stores
	dvc push
