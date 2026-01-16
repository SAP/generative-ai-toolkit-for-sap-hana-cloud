"""
Simple HTTP MCP server launcher for HANA Graph Tools.

This script starts an MCP server over HTTP and registers only
DiscoveryAgentTool and DataAgentTool from hana_ai.tools.hana_ml_tools.graph_tools.

Connection context can be provided via environment variables or CLI args.

Environment variables (optional):
- HANA_ADDRESS: HANA host (e.g., "your.hana.host")
- HANA_PORT: HANA port (e.g., 443 or 39017)
- HANA_USER: HANA username
- HANA_PASSWORD: HANA password
- HANA_ENCRYPT: "true"/"false" (optional)
- HANA_SSL_VALIDATE: "true"/"false" (optional)

Usage examples:
  python examples/mcp_http_server.py \
    --address "$HANA_ADDRESS" --port "$HANA_PORT" \
    --user "$HANA_USER" --password "$HANA_PASSWORD" \
    --host 127.0.0.1 --port-http 8001 --server-name HANAGraphTools

Notes:
- If your environment has an HTTP proxy configured, ensure localhost bypass:
  export NO_PROXY=127.0.0.1,localhost
"""

import os
import time
import logging
import argparse
from typing import Optional

from hana_ml import ConnectionContext
from hana_ai.tools.toolkit import HANAMLToolkit
from hana_ai.tools.hana_ml_tools.graph_tools import DiscoveryAgentTool, DataAgentTool


def build_connection_context(
    address: str,
    port: int,
    user: str,
    password: str,
    encrypt: Optional[bool] = None,
    ssl_validate_certificate: Optional[bool] = None,
):
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


def main():
    parser = argparse.ArgumentParser(description="Start HTTP MCP server for HANA Graph tools")
    # HANA connection args
    parser.add_argument("--address", default=os.environ.get("HANA_ADDRESS"), help="HANA host/address")
    parser.add_argument("--port", type=int, default=int(os.environ.get("HANA_PORT", "443")), help="HANA port")
    parser.add_argument("--user", default=os.environ.get("HANA_USER"), help="HANA user")
    parser.add_argument("--password", default=os.environ.get("HANA_PASSWORD"), help="HANA password")
    parser.add_argument("--encrypt", type=env_bool, default=env_bool("HANA_ENCRYPT"), help="Use TLS encrypt (true/false)")
    parser.add_argument(
        "--ssl-validate",
        dest="ssl_validate",
        type=env_bool,
        default=env_bool("HANA_SSL_VALIDATE"),
        help="Validate TLS certificate (true/false)",
    )

    # MCP server args
    parser.add_argument("--host", dest="host", default="127.0.0.1", help="Server host for HTTP MCP")
    parser.add_argument("--port-http", dest="port_http", type=int, default=8001, help="Server port for HTTP MCP")
    parser.add_argument("--server-name", dest="server_name", default="HANAGraphTools", help="MCP server name")
    parser.add_argument("--auth-token", dest="auth_token", default=None, help="Optional auth token (Bearer)")
    parser.add_argument("--max-retries", dest="max_retries", type=int, default=5, help="Port retry count if occupied")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Validate required HANA parameters
    missing = [name for name, val in {
        "address": args.address, "user": args.user, "password": args.password
    }.items() if not val]
    if missing:
        raise SystemExit(f"Missing required HANA connection parameters: {', '.join(missing)}. "
                         f"Provide via env or CLI (see --help).")

    logging.info("Connecting to HANA at %s:%s", args.address, args.port)
    cc = build_connection_context(
        address=args.address,
        port=args.port,
        user=args.user,
        password=args.password,
        encrypt=args.encrypt,
        ssl_validate_certificate=args.ssl_validate,
    )

    # Build toolkit and restrict to graph tools
    toolkit = HANAMLToolkit(connection_context=cc)
    discovery_tool = DiscoveryAgentTool(connection_context=cc)
    data_tool = DataAgentTool(connection_context=cc)
    toolkit.reset_tools([discovery_tool, data_tool])

    logging.info("Starting MCP HTTP server %s on http://%s:%d/mcp",
                 args.server_name, args.host, args.port_http)

    toolkit.launch_mcp_server(
        server_name=args.server_name,
        host=args.host,
        transport="http",
        port=args.port_http,
        auth_token=args.auth_token,
        max_retries=args.max_retries,
    )

    logging.info("MCP server is running. Press Ctrl+C to stop.")
    try:
        # Keep main thread alive while server runs in background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutdown requested. Exiting...")


if __name__ == "__main__":
    main()
