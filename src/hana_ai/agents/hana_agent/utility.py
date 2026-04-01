"""
This module contains utility functions used for agents.
"""
import json
import logging
import os


logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT = 30
REQUEST_TIMEOUT_ENV_VAR = "HANA_AI_HTTP_TIMEOUT"

def _get_request_timeout():
    value = os.environ.get(REQUEST_TIMEOUT_ENV_VAR)
    if not value:
        return DEFAULT_REQUEST_TIMEOUT
    try:
        if "," in value:
            parts = [p.strip() for p in value.split(",")]
            if len(parts) != 2:
                raise ValueError("Expect two comma-separated numbers for connect,read")
            return (float(parts[0]), float(parts[1]))
        return float(value)
    except Exception as exc:
        logger.warning(
            "Invalid %s value '%s': %s. Using default %s.",
            REQUEST_TIMEOUT_ENV_VAR,
            value,
            str(exc),
            DEFAULT_REQUEST_TIMEOUT,
        )
        return DEFAULT_REQUEST_TIMEOUT

def _concatenate_ai_core_certificate_string(credentials: dict) -> str:
    """
    Create a properly formatted AI Core certificate string.

    Parameters
    ----------
    credentials : dict
        The certificate string to be formatted.

    Returns
    -------
    str
        The formatted certificate string.
    """
    result = None
    key_certificate = credentials.get("key", "")
    certificate = credentials.get("certificate", "")

    if key_certificate and certificate:
        result = key_certificate + certificate

    return result

def _get_access_token(credentials: dict) -> str:
    """
    Get access token from credentials.

    Parameters
    ----------
    credentials : dict
        The credentials dictionary.

    Returns
    -------
    str
        The access token.
    """
    # Use requests library to send request
    import requests
    import urllib.parse
    import tempfile

    certurl = credentials.get("certurl")
    clientid = credentials.get("clientid")
    certificate = credentials.get("certificate", "")
    key_certificate = credentials.get("key", "")
    # Save certificate and private key to temporary files certificate.pem and private_key.pem

    with tempfile.NamedTemporaryFile(delete=False) as cert_file:
        cert_file.write(certificate.encode())
        cert_file_path = cert_file.name
    with tempfile.NamedTemporaryFile(delete=False) as key_file:
        key_file.write(key_certificate.encode())
        key_file_path = key_file.name
    token_url = urllib.parse.urljoin(certurl, "/oauth/token")
    data = {
        "grant_type": "client_credentials",
        "client_id": clientid
    }
    response = requests.post(token_url, data=data, cert=(cert_file_path, key_file_path), timeout=_get_request_timeout())
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get("access_token", "")
        logger.info("Successfully obtained access token.")
        # If temporary files exist, delete them
        if os.path.exists(cert_file_path):
            os.remove(cert_file_path)
        if os.path.exists(key_file_path):
            os.remove(key_file_path)
        logger.info("Temporary certificate files removed.")
        return access_token
    else:
        # Delete temporary files
        if os.path.exists(cert_file_path):
            os.remove(cert_file_path)
        if os.path.exists(key_file_path):
            os.remove(key_file_path)
        logger.info("Temporary certificate files removed.")
        raise Exception(f"Failed to get access token: {response.status_code} {response.text}")

def _get_deployment_id(credentials: dict) -> str:
    """
    Get deployment ID from credentials.

    Parameters
    ----------
    credentials : dict
        The credentials dictionary.

    Returns
    -------
    str
        The deployment ID.
    """
    import requests
    ai_api_url = credentials.get("serviceurls", {}).get("AI_API_URL")
    access_token = _get_access_token(credentials)
    headers = {
        "Authorization": f"Bearer {access_token}",
        "AI-Resource-Group": "default"
    }
    deployments_url = f"{ai_api_url}/v2/lm/deployments"
    response = requests.get(deployments_url, headers=headers, timeout=_get_request_timeout())

    if response.status_code == 200:
        deployments_data = response.json()
        logger.info("Deployments details: %s", deployments_data)
        resources = deployments_data.get("resources", [])
        if resources:
            for res in resources:
                d_id = res.get("id", None)
                if res.get("scenarioId", None) == "orchestration":
                    logger.info("Successfully obtained deployment ID: %s", d_id)
                    return d_id
        else:
            raise Exception("No deployments found.")
    else:
        raise Exception(f"Failed to get deployments: {response.status_code} {response.text}")

def _sql_string_literal(value) -> str:
    """
    Convert a Python value into an SQL string literal or NULL.

    Parameters
    ----------
    value : Any
        The value to serialize.

    Returns
    -------
    str
        SQL literal representation.
    """
    if value is None:
        return "NULL"
    if not isinstance(value, str):
        value = json.dumps(value)
    return "'%s'" % value.replace("'", "''")

def _call_agent_sql(
    remote_source_schema_name: str | None,
    remote_source_name: str,
    ai_metadata_schema_name: str,
    ai_metadata_object_prefix: str,
    model_and_version: str | None,
    query: str,
    options: dict | str | None,
    schema_name: str,
    procedure_name: str,
) -> str:
    """
    Create SQL string to call an AI retrieval procedure.

    Parameters
    ----------
    remote_source_schema_name : str, optional
        Remote source schema name.
    remote_source_name : str
        Remote source name.
    ai_metadata_schema_name : str
        Schema containing metadata objects.
    ai_metadata_object_prefix : str
        Prefix for metadata objects.
    model_and_version : str, optional
        Model and version identifier.
    query : str
        User query.
    options : dict or str, optional
        Optional NCLOB options payload.
    schema_name : str
        Schema containing the procedure.
    procedure_name : str
        Procedure name to invoke.

    Returns
    -------
    str
        The SQL string that executes the procedure and returns RESPONSE.
    """
    return (
        "DO\n"
        "BEGIN\n"
        "DECLARE output NCLOB;\n"
        "CALL %s.%s(%s, %s, %s, %s, %s, %s, output, %s);\n"
        "SELECT :output FROM DUMMY;\n"
        "END"
        % (
            schema_name,
            procedure_name,
            _sql_string_literal(remote_source_schema_name),
            _sql_string_literal(remote_source_name),
            _sql_string_literal(ai_metadata_schema_name),
            _sql_string_literal(ai_metadata_object_prefix),
            _sql_string_literal(model_and_version),
            _sql_string_literal(query),
            _sql_string_literal(options),
        )
    )
