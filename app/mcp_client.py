"""MCP client manager: spawns the custom MCP server as a stdio subprocess at app
startup, performs the initialize handshake, discovers tools, and keeps the session
open for the app's lifetime."""
from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from datetime import timedelta

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from . import config


class MCPManager:
    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.tools: list = []

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(config.MCP_SERVER_PATH)],
            env={**os.environ, "DATAAI_DATA_DIR": str(config.DATA_DIR)},
        )
        try:
            read, write = await self._stack.enter_async_context(stdio_client(params))
            # read_timeout_seconds: a subprocess that dies mid-request raises instead of
            # hanging the HTTP request forever.
            self.session = await self._stack.enter_async_context(
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=60)))
            await self.session.initialize()
            self.tools = (await self.session.list_tools()).tools
        except Exception:
            await self.stop()  # tear down a partially-started subprocess; reset state
            raise
        print(f"[mcp] connected — tools: {[t.name for t in self.tools]}")

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self.session = None
            self.tools = []

    async def ping(self) -> bool:
        """Liveness probe: mark the session dead if the MCP subprocess stopped responding,
        so chat returns the friendly unavailable message instead of hanging or raising raw."""
        if self.session is None:
            return False
        try:
            with anyio.fail_after(5):
                await self.session.send_ping()
            return True
        except Exception:
            self.session = None
            self.tools = []
            return False

    @property
    def ready(self) -> bool:
        return self.session is not None


manager = MCPManager()
