"""server.py — `eda-rl serve`: the FastAPI app + its CLI entry point.

API and frontend share one process/one container: the built React app (if
present, via EDA_RL_FRONTEND_DIST) is mounted as static files alongside the
/api router, so there's no second nginx container and no CORS story in
production. CORS is only opened up for the Vite dev server origin, for local
frontend development against a separately-running `eda-rl serve`.
"""

from __future__ import annotations

import argparse
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router

_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def create_app() -> FastAPI:
    app = FastAPI(
        title="eda-rl dashboard API",
        description="Read-only REST API over eda-rl campaign logs.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_DEV_ORIGINS,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(router)

    dist = os.environ.get("EDA_RL_FRONTEND_DIST")
    if dist and os.path.isdir(dist):
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="eda-rl serve",
        description="Launch the eda-rl REST API + dashboard (needs the [api] extra).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="autoreload for local dev")
    args = parser.parse_args()

    uvicorn.run("eda_rl.api.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
