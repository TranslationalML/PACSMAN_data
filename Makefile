install:
	poetry install --with dev,test --all-extras

lint:
	poetry run black .
	poetry run ruff .
	poetry run mypy .

test: lint
	poetry run pytest
	@echo "[ OK ] All tests passed"
