#!/usr/bin/env python3
"""
Simple script to run the rLLM API server.
This can be executed from the project root: python rllm/api/run_server.py
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import and run the server
if __name__ == "__main__":
    from rllm.api.api_server import app
    import uvicorn
    
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"🚀 Starting rLLM API server on {host}:{port}")
    print(f"📖 Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(app, host=host, port=port) 