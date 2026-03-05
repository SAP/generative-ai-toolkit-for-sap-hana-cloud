"""
Stdio MCP server launcher for HANA AI Tools.

Starts an MCP server over stdio transport, exposing all (or selected) HANA ML tools.
Designed for use with MCP clients that manage the process lifecycle, such as Cline,
Claude Desktop, or any tool supporting the MCP stdio transport.

Connection context can be provided via environment variables or CLI args.

Environment variables:
- HANA_ADDRESS:      HANA host (e.g., "your.hana.ondemand.com")
- HANA_PORT:         HANA port (default: 443)
- HANA_USER:         HANA username
- HANA_PASSWORD:     HANA password
- HANA_ENCRYPT:      Use TLS (true/false, optional)
- HANA_SSL_VALIDATE: Validate TLS certificate (true/false, optional)
- BUILD_CODE:        Suppress display() calls outside Jupyter (true/false, default: true)

Usage examples:
  # Using environment variables:
  HANA_ADDRESS=your.host HANA_USER=user HANA_PASSWORD=pass \\
    python examples/mcp_stdio_server.py

  # Using CLI arguments:
  python examples/mcp_stdio_server.py \\
    --address "$HANA_ADDRESS" --port "$HANA_PORT" \\
    --user "$HANA_USER" --password "$HANA_PASSWORD"

  # Expose only graph tools:
  python examples/mcp_stdio_server.py --tools graph

Notes:
- If your environment has an HTTP proxy configured, ensure the HANA host can bypass it:
  export NO_PROXY=127.0.0.1,localhost,<your-hana-host>
"""

import os
import sys
import time
import argparse
import logging
import warnings
from typing import Optional

# Suppress harmless warnings before importing dependencies.
# Critical for stdio transport: any stray output on stdout corrupts the MCP protocol stream.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
logging.getLogger("hana_ml.visualizers.shared").setLevel(logging.CRITICAL)

from hana_ml import ConnectionContext
from hana_ai.tools.toolkit import HANAMLToolkit


def build_connection_context(
    address: str,
    port: int,
    user: str,
    password: str,
    encrypt: Optional[bool] = None,
    ssl_validate_certificate: Optional[bool] = None,
) -> ConnectionContext:
    params = {
        "address": address,
        "port": port,
        "user": user,
        "password": password,
    }
    # Optional TLS params if provided
    if encrypt is not None:
        params["encrypt"] = bool(encrypt)
    if ssl_validate_certificate is not None:
        params["sslValidateCertificate"] = bool(ssl_validate_certificate)
    return ConnectionContext(**params)


def env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def parse_bool(val: str) -> bool:
    return val.strip().lower() in {"1", "true", "yes", "on"}


def main():
    parser = argparse.ArgumentParser(
        description="Start a stdio MCP server exposing HANA AI tools."
    )

    # HANA connection args
    parser.add_argument("--address", default=os.environ.get("HANA_ADDRESS"),
                        help="HANA host/address")
    parser.add_argument("--port", type=int, default=int(os.environ.get("HANA_PORT", "443")),
                        help="HANA port (default: 443)")
    parser.add_argument("--user", default=os.environ.get("HANA_USER"),
                        help="HANA user")
    parser.add_argument("--password", default=os.environ.get("HANA_PASSWORD"),
                        help="HANA password")
    parser.add_argument("--encrypt", type=parse_bool, default=env_bool("HANA_ENCRYPT"),
                        help="Use TLS encrypt (true/false)")
    parser.add_argument("--ssl-validate", dest="ssl_validate",
                        type=parse_bool, default=env_bool("HANA_SSL_VALIDATE"),
                        help="Validate TLS certificate (true/false)")

    # MCP server args
    parser.add_argument("--server-name", dest="server_name", default="HANATools",
                        help="MCP server name (default: HANATools)")
    parser.add_argument("--tools", default="all", choices=["all", "graph"],
                        help="Which tools to expose: 'all' (default) or 'graph'")
    parser.add_argument("--build-code", dest="build_code",
                        type=parse_bool, default=env_bool("BUILD_CODE", default=True),
                        help="Suppress display() calls outside Jupyter/BAS (default: true)")

    args = parser.parse_args()

    # Stdio transport: log to stderr so stdout stays clean for MCP protocol
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )

    # Validate required HANA parameters
    missing = [name for name, val in {
        "address": args.address,
        "user": args.user,
        "password": args.password,
    }.items() if not val]
    if missing:
        raise SystemExit(
            f"Missing required HANA connection parameters: {', '.join(missing)}. "
            f"Provide via env or CLI (see --help)."
        )

    logging.warning("Connecting to HANA at %s:%s", args.address, args.port)
    cc = build_connection_context(
        address=args.address,
        port=args.port,
        user=args.user,
        password=args.password,
        encrypt=args.encrypt,
        ssl_validate_certificate=args.ssl_validate,
    )

    if args.tools == "graph":
        from hana_ai.tools.hana_ml_tools.graph_tools import DiscoveryAgentTool, DataAgentTool
        toolkit = HANAMLToolkit(connection_context=cc)
        toolkit.reset_tools([
            DiscoveryAgentTool(connection_context=cc),
            DataAgentTool(connection_context=cc),
        ])
    else:
        toolkit = HANAMLToolkit(connection_context=cc, used_tools="all")

    # Set BAS mode to suppress display() errors in non-Jupyter environments
    toolkit.set_bas(args.build_code)

    toolkit.launch_mcp_server(
        server_name=args.server_name,
        transport="stdio",
    )

    logging.warning("MCP server is running. Press Ctrl+C to stop.")
    try:
        # Keep main thread alive while server runs in background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.warning("Shutdown requested. Exiting...")


if __name__ == "__main__":
    main()
