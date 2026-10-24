"""ASGI entry point: ``uvicorn llm_gateway.asgi:app``."""

from llm_gateway.app import create_app

app = create_app()
