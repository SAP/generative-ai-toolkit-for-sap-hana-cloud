"""
Stdio MCP server launcher for HANA AI Tools.

Starts an MCP server over stdio transport, exposing all (or selected) HANA ML tools.
Designed for use with MCP clients that manage the process lifecycle, such as Cline,
Claude Desktop, or any tool supporting the MCP stdio transport.

Connection context can be provided via environment variables or CLI args.

You can also point the server at an env-style file containing credentials
(including VCAP_SERVICES) using `--env-file`.

Environment variables:
- HANA_ENV_FILE:    Path to a .env file to load into environment (optional)
- ENV_FILE:         Alternative .env path variable (optional)
- HANA_ADDRESS:      HANA host (e.g., "your.hana.ondemand.com")
- HANA_PORT:         HANA port (default: 443)
- HANA_USER:         HANA username
- HANA_PASSWORD:     HANA password
- HANA_ENCRYPT:      Use TLS (true/false, optional)
- HANA_SSL_VALIDATE: Validate TLS certificate (true/false, optional)
- BUILD_CODE:        Suppress display() calls outside Jupyter (true/false, default: true)

Usage examples:
    # Load a .env file first (e.g., examples/sample.env):
    HANA_ENV_FILE=examples/sample.env python examples/mcp_stdio_server.py

  # Using environment variables:
  HANA_ADDRESS=your.host HANA_USER=user HANA_PASSWORD=pass \\
    python examples/mcp_stdio_server.py

  # Using CLI arguments:
  python examples/mcp_stdio_server.py \\
    --address "$HANA_ADDRESS" --port "$HANA_PORT" \\
    --user "$HANA_USER" --password "$HANA_PASSWORD"

    # Using an env-style file containing VCAP_SERVICES:
    python examples/mcp_stdio_server.py --env-file nutest/testscripts/env.example


Notes:
- If your environment has an HTTP proxy configured, ensure the HANA host can bypass it:
  export NO_PROXY=127.0.0.1,localhost,<your-hana-host>

Runtime update (no restart):
- This server exposes two MCP admin tools:
    - `admin_update_connection_context`: pass new connection params explicitly.
    - `admin_reload_connection_context_from_file`: read a VCAP_SERVICES file and reload.
- An MCP client can call either one to switch the live `ConnectionContext` after env/credentials change,
    without restarting the MCP server process.

  Example (Python HTTP transport client):
    from hana_ai.client.mcp_client import HTTPMCPClient
    import asyncio

    async def main():
        c = HTTPMCPClient(base_url="http://127.0.0.1:8001/mcp")
        await c.initialize()
        r = await c.call_tool(
            "admin_update_connection_context",
            {
                "address": "new.host",
                "port": 443,
                "user": "NEW_USER",
                "password": "NEW_PASS",
                "encrypt": True,
                "ssl_validate_certificate": True,
                "test_connection": True,
            },
        )
        print(r)

        # Or: reload from a VCAP_SERVICES file
        r2 = await c.call_tool(
            "admin_reload_connection_context_from_file",
            {"file_path": "path/to/env.file", "test_connection": True},
        )
        print(r2)
        await c.close()

    asyncio.run(main())
"""

import os
import sys
import time
import argparse
import logging
import warnings
import json
from typing import Optional

# Suppress harmless warnings before importing dependencies.
# Critical for stdio transport: any stray output on stdout corrupts the MCP protocol stream.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
logging.getLogger("hana_ml.visualizers.shared").setLevel(logging.CRITICAL)

from hana_ml import ConnectionContext
from hana_ai.tools.toolkit import HANAMLToolkit


_DOTENV_PATH_VARS = ("HANA_ENV_FILE", "ENV_FILE")


def _load_env_file(path: str, *, override: bool = False) -> int:
    """Load KEY=VALUE pairs from a .env-style file into os.environ.

    - Does not print to stdout (safe for stdio MCP transport).
    - By default does not override existing environment variables.

    Returns the count of variables loaded.
    """
    loaded = 0
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if not override and key in os.environ:
                continue
            os.environ[key] = value
            loaded += 1
    return loaded


def _maybe_load_env_file_from_env() -> Optional[str]:
    """Load a .env file into the environment if a configured path exists.

    The path is read from one of: HANA_ENV_FILE, ENV_FILE.
    Relative paths are resolved against CWD first, then the script directory.

    Returns the resolved path if loaded, else None.
    """
    script_dir = os.path.dirname(__file__)
    for var_name in _DOTENV_PATH_VARS:
        candidate = os.environ.get(var_name)
        if not candidate:
            continue
        candidate = os.path.expandvars(os.path.expanduser(candidate))
        candidates = [
            candidate,
            os.path.join(os.getcwd(), candidate),
            os.path.join(script_dir, candidate),
        ]
        for path in candidates:
            if os.path.isfile(path):
                _load_env_file(path, override=False)
                return path
    return None


def _resolve_env_file_path(candidate: str) -> Optional[str]:
    if not candidate:
        return None
    candidate = os.path.expandvars(os.path.expanduser(candidate))
    script_dir = os.path.dirname(__file__)
    candidates = [
        candidate,
        os.path.join(os.getcwd(), candidate),
        os.path.join(script_dir, candidate),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _maybe_apply_vcap_services() -> Optional[dict]:
    """Map VCAP_SERVICES hana credentials into HANA_* env vars if present.

    Returns the credentials dict if applied, else None.
    """
    vcap = os.environ.get("VCAP_SERVICES")
    if not vcap:
        return None
    try:
        payload = json.loads(vcap)
    except json.JSONDecodeError:
        logging.warning("VCAP_SERVICES is not valid JSON; skipping.")
        return None
    hana_services = payload.get("hana") or []
    if not hana_services:
        return None
    credentials = hana_services[0].get("credentials") or {}
    mapping = {
        "HANA_ADDRESS": credentials.get("host"),
        "HANA_PORT": credentials.get("port"),
        "HANA_SCHEMA": credentials.get("schema"),
        "HANA_USER": credentials.get("user"),
        "HANA_PASSWORD": credentials.get("password"),
    }
    for key, value in mapping.items():
        if value is None:
            continue
        if key not in os.environ:
            os.environ[key] = str(value)
    return credentials


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


def build_connection_context_from_userkey(
    userkey: str,
    encrypt: Optional[bool] = None,
    ssl_validate_certificate: Optional[bool] = None,
) -> ConnectionContext:
    params: dict = {
        "userkey": userkey,
    }
    if encrypt is not None:
        params["encrypt"] = bool(encrypt)
    if ssl_validate_certificate is not None:
        params["sslValidateCertificate"] = bool(ssl_validate_certificate)

    # Best-effort: use certifi trust store if available.
    try:
        import certifi
        params["sslKeyStore"] = certifi.where()
    except Exception:
        pass

    return ConnectionContext(**params)


def env_bool(name: str, default: Optional[bool] = None) -> Optional[bool]:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def parse_bool(val: str) -> bool:
    return val.strip().lower() in {"1", "true", "yes", "on"}


def main():
    # Stdio transport: log to stderr so stdout stays clean for MCP protocol
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Start a stdio MCP server exposing HANA AI tools."
    )

    # Optional env file (explicit) - loaded before resolving connection args.
    parser.add_argument(
        "--env-file",
        dest="env_file",
        default=None,
        help="Path to an env-style file to load (supports VCAP_SERVICES='...json...').",
    )
    parser.add_argument(
        "--override-env",
        dest="override_env",
        action="store_true",
        help="If set, values in --env-file override existing environment variables.",
    )

    # HANA connection args
    parser.add_argument("--userkey", default=os.environ.get("HANA_USERKEY"),
                        help="HANA userkey (e.g., RaysKey). If provided, it overrides address/user/password.")
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
                        type=parse_bool, default=env_bool("HANA_SSL_VALIDATE", default=False),
                        help="Validate TLS certificate (true/false)")

    # MCP server args
    parser.add_argument("--server-name", dest="server_name", default="HANATools",
                        help="MCP server name (default: HANATools)")
    parser.add_argument("--build-code", dest="build_code",
                        type=parse_bool, default=env_bool("BUILD_CODE", default=True),
                        help="Suppress display() calls outside Jupyter/BAS (default: true)")

    args = parser.parse_args()

    # 1) Load env file (explicit CLI takes priority), else fallback to HANA_ENV_FILE/ENV_FILE.
    loaded_env_path = None
    if args.env_file:
        resolved = _resolve_env_file_path(args.env_file)
        if not resolved:
            raise SystemExit(f"Env file not found: {args.env_file}")
        _load_env_file(resolved, override=bool(args.override_env))
        loaded_env_path = resolved
    else:
        loaded_env_path = _maybe_load_env_file_from_env()

    if loaded_env_path:
        logging.warning("Loaded environment variables from %s", loaded_env_path)

    # 2) If VCAP_SERVICES exists (possibly from env file), map it into HANA_* env vars.
    vcap_credentials = _maybe_apply_vcap_services()
    if vcap_credentials:
        logging.warning("Applied HANA credentials from VCAP_SERVICES")

    # 3) If connection args were not explicitly provided, re-fill from env AFTER loading.
    if not args.userkey:
        args.userkey = os.environ.get("HANA_USERKEY")
    if args.address is None:
        args.address = os.environ.get("HANA_ADDRESS")
    if args.user is None:
        args.user = os.environ.get("HANA_USER")
    if args.password is None:
        args.password = os.environ.get("HANA_PASSWORD")
    try:
        # argparse already makes this int, but if it came from env after load, normalize.
        if args.port is None:
            args.port = int(os.environ.get("HANA_PORT", "443"))
    except Exception:
        pass
    if args.encrypt is None:
        args.encrypt = env_bool("HANA_ENCRYPT")
    if args.ssl_validate is None:
        args.ssl_validate = env_bool("HANA_SSL_VALIDATE", default=False)

    # Stdio transport: log to stderr so stdout stays clean for MCP protocol
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )

    if args.userkey:
        logging.warning("Connecting to HANA via userkey '%s'", args.userkey)
        cc = build_connection_context_from_userkey(
            userkey=args.userkey,
            encrypt=args.encrypt,
            ssl_validate_certificate=args.ssl_validate,
        )
    else:
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
