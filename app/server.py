"""Launch entry point: the Streamlit app plus HTTP security headers (blocks other sites from framing it).

Start with `idea-miner ui`, or `streamlit run app/server.py`. Running `streamlit run app/streamlit_app.py` directly
still works and still binds to this computer only (app/.streamlit/config.toml), but without these headers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402
from starlette.middleware import Middleware  # noqa: E402

from idea_miner.web_security import SecurityHeadersMiddleware  # noqa: E402

app = st.App(Path(__file__).with_name("streamlit_app.py"), middleware=[Middleware(SecurityHeadersMiddleware)])
