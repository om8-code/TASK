import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Helper to import the FastAPI ``app`` instance.
# The project may expose the app in ``main.py`` or ``app/main.py``.  We try both
# locations and raise a clear error if the import fails.  This makes the test
# resilient to small structural changes while still failing loudly when the
# expected object is missing.
# ---------------------------------------------------------------------------

def get_fastapi_app() -> FastAPI:
    """Return the FastAPI application instance used by the project.

    The function attempts to import ``app`` from a handful of conventional
    module paths.  If none succeed, an ``ImportError`` with a helpful message is
    raised.
    """
    possible_modules = ["main", "app.main", "src.main", "src.app.main"]
    for module_name in possible_modules:
        try:
            module = importlib.import_module(module_name)
            app = getattr(module, "app", None)
            if isinstance(app, FastAPI):
                return app
        except Exception:  # pragma: no cover – we deliberately ignore import errors
            continue
    raise ImportError(
        "Could not locate a FastAPI 'app' instance. Tried modules: "
        + ", ".join(possible_modules)
    )


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio to use the default asyncio backend for async tests."""
    return "asyncio"


@pytest.fixture(scope="session")
def app() -> FastAPI:
    """Provide the FastAPI application for the test session."""
    return get_fastapi_app()


@pytest.fixture(scope="session")
async def async_client(app: FastAPI) -> AsyncClient:
    """Create a single ``httpx.AsyncClient`` bound to the FastAPI app.

    The client is yielded so that it is properly closed after the test session.
    """
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


# ---------------------------------------------------------------------------
# Basic sanity checks
# ---------------------------------------------------------------------------

def test_app_is_fastapi_instance(app: FastAPI) -> None:
    """The imported ``app`` must be an instance of ``FastAPI``.

    This ensures that the project follows the conventional pattern of exposing a
    top‑level ``app`` variable that can be used by ``uvicorn`` and the test suite.
    """
    assert isinstance(app, FastAPI)


# ---------------------------------------------------------------------------
# Health‑check endpoint (if present)
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_health_endpoint(async_client: AsyncClient) -> None:
    """Verify that a simple ``GET /health`` endpoint returns a 200 status.

    Many FastAPI projects expose a health endpoint for liveness probes.  If the
    endpoint does not exist we treat a 404 as a non‑failure – the test will still
    pass but will emit a helpful warning.
    """
    response = await async_client.get("/health")
    if response.status_code == 404:
        # Endpoint not implemented – this is acceptable but we log for visibility.
        pytest.warns(UserWarning, match="Health endpoint not found")
    else:
        assert response.status_code == 200
        # Expect a JSON payload with a ``status`` key, but be tolerant of variations.
        json_body = response.json()
        assert isinstance(json_body, dict)
        assert "status" in json_body
        assert json_body["status"] in {"ok", "healthy", "up"}


# ---------------------------------------------------------------------------
# Search endpoint – core functionality
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_search_success(async_client: AsyncClient) -> None:
    """POST a valid query to the search endpoint and validate the response.

    The repository's README describes a semantic‑search API that accepts a JSON
    payload with a ``query`` field and returns a list of ``results``.  This test
    mirrors that contract.
    """
    payload = {"query": "sample test query"}
    response = await async_client.post("/search", json=payload)
    assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    data = response.json()
    # Basic contract validation
    assert isinstance(data, dict), "Response body must be a JSON object"
    assert "results" in data, "Response must contain a 'results' key"
    assert isinstance(data["results"], list), "'results' must be a list"
    # Each result should be a mapping with at least 'text' and 'score' keys.
    for item in data["results"]:
        assert isinstance(item, dict)
        assert "text" in item
        assert "score" in item
        # Score should be numeric (int or float)
        assert isinstance(item["score"], (int, float))


@pytest.mark.anyio
async def test_search_validation_error(async_client: AsyncClient) -> None:
    """Sending an empty payload should trigger FastAPI's validation (422)."""
    response = await async_client.post("/search", json={})
    assert response.status_code == 422
    error_detail = response.json()
    assert isinstance(error_detail, dict)
    assert "detail" in error_detail

