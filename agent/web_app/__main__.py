#!/usr/bin/env python3
import os
from .app import app
import uvicorn

if __name__ == "__main__":
    # Enable reload in development, disable in production
    reload = os.getenv("ENVIRONMENT", "development") == "development"
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=reload)
