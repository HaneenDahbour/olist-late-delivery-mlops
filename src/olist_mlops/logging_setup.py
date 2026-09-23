"""Central logging setup: console (plain) + rotating file (JSON).

Called once at process startup (app.main / CLI entry points). Every
other module just does `logging.getLogger(__name__)` — no print().
"""

from __future__ import annotations

import logging
import logging.handlers

from pythonjsonlogger.json import JsonFormatter

from olist_mlops.config import find_repo_root


def configure_logging(config: dict) -> None:
    log_cfg = config["logging"]
    level = getattr(logging, log_cfg["level"].upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    if log_cfg.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_cfg["format"]))
        root.addHandler(console_handler)

    if log_cfg.get("file", True):
        log_dir = find_repo_root() / config["paths"]["service_log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "service.log",
            maxBytes=log_cfg["max_bytes"],
            backupCount=log_cfg["backup_count"],
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(file_handler)
