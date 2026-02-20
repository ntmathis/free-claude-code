"""CLI entry point for running the API server."""

import uvicorn

from cli.process_registry import kill_all_best_effort
from config.settings import get_settings


def main() -> None:
    """Run the API server using configured settings."""
    settings = get_settings()
    try:
        uvicorn.run(
            "api.app:app",
            host=settings.host,
            port=settings.port,
            log_level="debug",
            timeout_graceful_shutdown=5,
        )
    finally:
        kill_all_best_effort()
