# Contributing to curvefit-api

This document describes how to set up a development environment and how changes
are made to this project.

## Development setup

Requirements: [git](https://git-scm.com/) and [uv](https://docs.astral.sh/uv/).
uv installs the required Python version itself.

```bash
git clone git@github.com:SwarnoB/curvefit-api.git
cd curvefit-api
uv sync                        # create .venv with all dependencies from uv.lock
uv run pre-commit install      # activate the git hooks (once per clone)
```

Check that everything works:

```bash
uv run pytest
```

### Running the API locally

```bash
uv run uvicorn curvefit_api.main:app --reload
```

The API is then available at http://127.0.0.1:8000, with interactive
documentation at http://127.0.0.1:8000/docs.

## Project structure

```
src/curvefit_api/
├── domain/      # pure numerics: models, fitting, statistics (no web or database code)
└── main.py      # FastAPI application
tests/
├── unit/        # fast tests of individual functions
└── integration/ # tests of the API through FastAPI's test client
```

The domain layer must not import from the web or database layers. This keeps
the numerical code testable in isolation.

## Code quality checks

Every commit is checked by pre-commit hooks, and every pull request by CI.
To run the checks by hand:

```bash
uv run pre-commit run --all-files   # formatting, linting, type checking, hygiene checks
uv run pytest -v                    # tests
```

The project uses:

- **ruff** for formatting and linting (configured in `pyproject.toml`)
- **mypy** in strict mode for type checking
- **pytest** for tests; warnings are treated as errors

## Making a change

1. Start from an up-to-date `main` and create a branch:

   ```bash
   git switch main
   git pull
   git switch -c feat/short-description
   ```

   Branch prefixes follow the commit types below (`feat/`, `fix/`, `docs/`, ...).

2. Make small commits, each containing one logical change, with its tests.

3. Push the branch and open a pull request against `main`.

`main` is protected: changes reach it only through pull requests with passing CI.
Pull requests are squash-merged, so the pull request title becomes the commit
message on `main`.

### Commit messages

Commit messages and pull request titles follow
[Conventional Commits](https://www.conventionalcommits.org/): a lowercase type,
a colon, and a short summary in the imperative mood.

```
feat: add lorentzian model
fix: reject initial parameters on the bounds
```

| Type | Use for |
|---|---|
| `feat` | new behaviour |
| `fix` | bug fixes |
| `test` | adding or changing tests only |
| `docs` | documentation only |
| `refactor` | code changes that do not change behaviour |
| `style` | formatting only |
| `build` | dependencies and tooling |
| `ci` | the CI workflow |
| `chore` | other housekeeping |

### Pull request descriptions

Describe the change under these headings:

- **What** - what the pull request changes
- **Why** - the problem it solves
- **Decisions** - design choices and their reasons
- **Testing** - how the change was verified
- **Not in this PR** - known limitations or deferred work (optional)

## Testing guidelines

- Every change in behaviour comes with tests in the same pull request.
- Each test checks one behaviour and can fail for only one reason.
- Test boundary values (for example, a value exactly on a bound), not only values
  far outside the valid range.
- Test through public functions rather than private ones (names starting with `_`).
- Use fixed random seeds (`np.random.default_rng(42)`) so that tests are deterministic,
  and justify statistical tolerances in a comment.
- Before committing a new test, check that it can fail: break the code on purpose,
  confirm the test fails, then undo the change.

## Adding a model

Models live in `src/curvefit_api/domain/models.py`. To add one:

1. Write the model function and its analytic Jacobian. Both take
   `(x, params)` and return float arrays; the Jacobian has shape
   `(len(x), len(params))`.
2. Create a `Model` with a unique name, a description, parameter names, and
   lower and upper bounds, and add it to the `MODELS` registry.
3. Add example parameters for it to `EXAMPLE_PARAMS` in
   `tests/unit/domain/test_models.py`. The test suite fails until you do, and then
   automatically checks the Jacobian against finite differences.
4. Add at least one test checking known function values.

Parameter names are part of the public API. Use the same name for the same
concept across models (for example `height`, `center`, `baseline`), and name
widths precisely (`hwhm`, `sigma_width`).
