"""Load the single source of truth for the inference service.

Reads config/config.yaml, resolving ``${VAR}`` / ``${VAR:-default}``
placeholders against the environment, and locates the repo root by
walking up from this file to find config.yaml — the same trick
notebooks 5/6 use with docker-compose.yml, so behavior matches
regardless of the current working directory.
"""

from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path
from typing import Any

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}")


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until config/config.yaml is found."""
    current = (start or Path(__file__)).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "config" / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError("Could not locate config/config.yaml above " + str(current))


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, str):

        def substitute(match: re.Match) -> str:
            var_name, _, default = match.groups()
            return os.environ.get(var_name, default if default is not None else "")

        return _ENV_VAR_PATTERN.sub(substitute, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(v) for v in value]
    return value


@cache
def load_config(config_path: str | None = None) -> dict:
    """Load and return config/config.yaml with env-var placeholders resolved.

    Cached because the config is immutable for the lifetime of a process;
    pass a different `config_path` (as a string, for hashability) in tests
    that need a fresh file.
    """
    path = Path(config_path) if config_path else find_repo_root() / "config" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _resolve_env_vars(raw)
