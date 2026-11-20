"""Startup config logging must not keep credentials that are embedded in URLs.

These tests follow the service startup path that writes the loaded config at
INFO. They fail while that line still contains userinfo or credential query
parameters, and they pass once those values are removed before the line is
written. Named password and token settings must still be masked.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Obvious placeholders. They stand in for credential material and are not real secrets.
REDIS_USER = "placeholder-redis-user"
REDIS_USERINFO = "placeholder-redis-userinfo"
REDIS_USERINFO_ENCODED = "placeholder%2Dredis%2Duserinfo"
REDIS_QUERY = "placeholder-redis-query"
MESH_USER = "placeholder-mesh-user"
MESH_USERINFO = "placeholder-mesh-userinfo"
MESH_QUERY = "placeholder-mesh-query"
BOOK_USER = "placeholder-book-user"
BOOK_USERINFO = "placeholder-book-userinfo"
BOOK_TOKEN = "placeholder-book-token"
METRICS_USER = "placeholder-metrics-user"
METRICS_USERINFO = "placeholder-metrics-userinfo"
METRICS_QUERY = "placeholder-metrics-query"
LIST_USER = "placeholder-list-user"
LIST_USERINFO = "placeholder-list-userinfo"
LIST_QUERY = "placeholder-list-query"
MQTT_PASSWORD = "placeholder-mqtt-password"
API_TOKEN = "placeholder-api-token"
REDIS_PASSWORD = "placeholder-redis-password"

URL_CREDENTIALS = (
    REDIS_USER,
    REDIS_USERINFO,
    REDIS_USERINFO_ENCODED,
    REDIS_QUERY,
    MESH_USER,
    MESH_USERINFO,
    MESH_QUERY,
    BOOK_USER,
    BOOK_USERINFO,
    BOOK_TOKEN,
    METRICS_USER,
    METRICS_USERINFO,
    METRICS_QUERY,
    LIST_USER,
    LIST_USERINFO,
    LIST_QUERY,
)

NAMED_CREDENTIALS = (
    MQTT_PASSWORD,
    API_TOKEN,
    REDIS_PASSWORD,
)

# Hosts, paths, and non-credential query values that the redacted log should keep.
PUBLIC_MARKERS = (
    "redis.example",
    "6379",
    "db=2",
    "mesh.example",
    "timeout=9",
    "book.example",
    "metrics.example",
    "hooks.example",
    "mode=queue",
    "***REDACTED***",
)


def _config_text(log_file: Path) -> str:
    return textwrap.dedent(
        f"""\
        mqtt:
          host: 127.0.0.1
          port: 1
          password: {MQTT_PASSWORD}
        meshmonitor:
          base_url: "http://{MESH_USER}:{MESH_USERINFO}@mesh.example:8080/mm?access_token={MESH_QUERY}&timeout=9"
          api_token: {API_TOKEN}
        pager_phonebook_api:
          enabled: false
          base_url: "https://{BOOK_USER}:{BOOK_USERINFO}@book.example/api"
          api_token: {BOOK_TOKEN}
        dedupe:
          enabled: true
          window_seconds: 60
          backend: redis
          redis_url: "redis://{REDIS_USER}:{REDIS_USERINFO_ENCODED}@redis.example:6379/0?password={REDIS_QUERY}&db=2"
          redis_password: {REDIS_PASSWORD}
        runtime:
          log_level: INFO
          log_file: {log_file}
        metrics_url: "http://{METRICS_USER}:{METRICS_USERINFO}@metrics.example/hook?api_key={METRICS_QUERY}"
        callback_urls:
          - "http://{LIST_USER}:{LIST_USERINFO}@hooks.example/cb?token={LIST_QUERY}&mode=queue"
        """
    )


def _child_env(config_path: Path, break_dump: bool) -> dict:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "LANG": "C",
        "CONFIG_PATH": str(config_path),
        "PYTHONPATH": str(ROOT),
    }
    if break_dump:
        env["BREAK_DUMP"] = "1"
    return env


def _run_startup(log_file: Path, config_path: Path, break_dump: bool) -> subprocess.CompletedProcess:
    code = textwrap.dedent(
        """\
        import os
        import sys

        sys.path.insert(0, os.environ["PYTHONPATH"])
        import pocsag_forwarder

        if os.environ.get("BREAK_DUMP") == "1":
            def _boom(*_args, **_kwargs):
                raise RuntimeError("dump failed")

            pocsag_forwarder.yaml.safe_dump = _boom

        try:
            pocsag_forwarder.main()
        except Exception:
            pass
        """
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=_child_env(config_path, break_dump),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


class StartupConfigRedactionTests(unittest.TestCase):
    def _assert_startup_logs_hide_url_credentials(self, break_dump: bool) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_file = tmp_path / "meshsag.log"
            config_path = tmp_path / "config.yaml"
            config_path.write_text(_config_text(log_file), encoding="utf-8")

            completed = _run_startup(log_file, config_path, break_dump)
            log_text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
            stderr = completed.stderr or ""

        self.assertEqual(
            completed.returncode,
            0,
            "startup logging did not finish",
        )
        for label, text in (("stderr", stderr), ("log file", log_text)):
            self.assertIn("Config (redacted)", text, f"{label} did not record the startup config line")
            self.assertIn(" INFO ", text, f"{label} did not record the config line at INFO")
            for marker in PUBLIC_MARKERS:
                self.assertIn(marker, text, f"{label} dropped public config detail {marker}")
            for secret in URL_CREDENTIALS + NAMED_CREDENTIALS:
                self.assertNotIn(secret, text, f"{label} contains a credential placeholder")

    def test_startup_info_log_omits_url_credentials(self) -> None:
        self._assert_startup_logs_hide_url_credentials(break_dump=False)

    def test_startup_info_log_fallback_omits_url_credentials(self) -> None:
        self._assert_startup_logs_hide_url_credentials(break_dump=True)


if __name__ == "__main__":
    unittest.main()
