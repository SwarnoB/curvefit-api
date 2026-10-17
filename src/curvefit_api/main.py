"""FastAPI application entry point."""

from fastapi import FastAPI

from curvefit_api.api.routes import router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="curvefit-api", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report that the service is running."""
        return {"status": "ok"}

    app.include_router(router)

    return app


app = create_app()
