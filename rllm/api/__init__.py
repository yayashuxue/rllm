"""
rLLM API Module

This module provides a web API interface for serving rLLM workflows via HTTP.
It includes FastAPI-based endpoints for synchronous and asynchronous workflow execution,
batch processing, and live trajectory collection.
"""

from .api_server import app
from .workflow_server import serve, serve_workflow

__all__ = ["app", "serve", "serve_workflow"] 