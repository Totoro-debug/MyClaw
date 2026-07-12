# MyClaw

MyClaw is a local-first Personal Agent runtime for Python 3.12 and newer.

## Development

After the project dependencies are available in the local Python environment, all
verification commands run without network access:

```text
python -m pip install --no-index --no-deps --no-build-isolation -e .
pytest
ruff check .
ruff format --check .
mypy src tests
python -m build --no-isolation
```

The automated tests use scripted boundary fakes and temporary filesystem paths. They do
not call model providers or other external services.
