.PHONY: lint
lint:
	ruff check --exit-zero .

.PHONY: format
format:
	ruff --fix .
	ruff format .

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v .
