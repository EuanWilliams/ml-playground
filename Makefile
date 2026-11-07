black:
	uv run black --line-length=160 --exclude="(.venv|venv)" .

mypy-strict:
	MYPYPATH=src/ uv run mypy --config-file pyproject.toml --strict --show-error-codes .