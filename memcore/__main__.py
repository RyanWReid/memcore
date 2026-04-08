"""MemCore entry point — run with `python -m memcore`."""

import logging
import uvicorn

from memcore.api.mcp_server import create_app
from memcore.config import MCP_HOST, MCP_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT, log_level="info")
