#!/usr/bin/env python3
import hashlib
import json
import logging
import logging.handlers
import os
import queue
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import yaml
from paho.mqtt import client as mqtt


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _cfg_get(cfg: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict) and key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


def _get_str(env_name: str, cfg: Optional[Dict[str, Any]], key: str, default: Optional[str]) -> Optional[str]:
    env_val = os.getenv(env_name)
    if env_val is not None:
        return env_val
    val = _cfg_get(cfg, key, default)
    if val is None:
        return default
    return str(val)


def _get_int(env_name: str, cfg: Optional[Dict[str, Any]], key: str, default: int) -> int:
    env_val = os.getenv(env_name)
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            return default
    val = _cfg_get(cfg, key, default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _get_float(env_name: str, cfg: Optional[Dict[str, Any]], key: str, default: float) -> float:
    env_val = os.getenv(env_name)
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            return default
    val = _cfg_get(cfg, key, default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_bool(env_name: str, cfg: Optional[Dict[str, Any]], key: str, default: bool) -> bool:
    env_val = os.getenv(env_name)
    if env_val is not None:
        return _to_bool(env_val, default)
    return _to_bool(_cfg_get(cfg, key, default), default)


def _get_json(env_name: str, cfg: Optional[Dict[str, Any]], key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    env_val = os.getenv(env_name)
    if env_val is not None:
        try:
            parsed = json.loads(env_val)
            return parsed if isinstance(parsed, dict) else default
        except json.JSONDecodeError:
            return default
    val = _cfg_get(cfg, key, default)
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else default
        except json.JSONDecodeError:
            return default
    return default


def _resolve_path(path: str, base_dir: Optional[str]) -> str:
    if not path:
        return path
    if os.path.isabs(path) or not base_dir:
        return path
    return os.path.join(base_dir, path)


def _load_yaml_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config(path: str) -> Dict[str, Any]:
    data = _load_yaml_file(path) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _load_mapping_file(path: str, label: str) -> Dict[str, Any]:
    try:
        data = _load_yaml_file(path) or {}
    except FileNotFoundError:
        logging.warning("%s file not found: %s", label, path)
        return {}
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to load %s file %s: %s", label, path, exc)
        return {}
    if not isinstance(data, dict):
        logging.warning("%s file %s must be a mapping", label, path)
        return {}
    return data


def _merge_ric_phonebook(config: Dict[str, Any], base_dir: Optional[str]) -> Dict[str, Any]:
    phonebook = config.get("ric_phonebook", {})
    if not isinstance(phonebook, dict):
        phonebook = {}

    phonebook_path = config.get("ric_phonebook_file") or config.get("ric_phonebook_path")
    if phonebook_path is None or str(phonebook_path).strip() == "":
        config["ric_phonebook"] = phonebook
        return config

    resolved = _resolve_path(str(phonebook_path), base_dir)
    loaded = _load_mapping_file(resolved, "RIC phonebook")
    merged = {**loaded, **phonebook}
    config["ric_phonebook"] = merged
    return config


def _redact_config(data: Any) -> Any:
    sensitive_keys = {
        "password",
        "api_token",
        "redis_password",
        "token",
        "secret",
        "api_key",
        "apikey",
    }
    if isinstance(data, dict):
        redacted: Dict[Any, Any] = {}
        for k, v in data.items():
            key = str(k).lower()
            if (
                key in sensitive_keys
                or key.endswith("_password")
                or key.endswith("_token")
                or key.endswith("_secret")
                or key.endswith("_apikey")
                or key.endswith("_api_key")
            ):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = _redact_config(v)
        return redacted
    if isinstance(data, list):
        return [_redact_config(v) for v in data]
    return data


def _redact_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.port:
            host = f"{host}:{parsed.port}"
        path = parsed.path or ""
        scheme = parsed.scheme or "redis"
        return f"{scheme}://{host}{path}"
    except Exception:  # noqa: BLE001
        return "<redacted>"


def _normalize_ric(value: Any) -> Tuple[Optional[int], str]:
    s = str(value).strip()
    try:
        return int(s), str(int(s))
    except (ValueError, TypeError):
        return None, s


def _compile_rics(rics: List[Any]) -> Tuple[set, List[Tuple[int, int]]]:
    singles = set()
    ranges: List[Tuple[int, int]] = []
    for item in rics:
        if item is None:
            continue
        if isinstance(item, int):
            singles.add(item)
            continue
        s = str(item).strip()
        if not s:
            continue
        if "-" in s:
            parts = [p.strip() for p in s.split("-", 1)]
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                except ValueError:
                    continue
                if start > end:
                    start, end = end, start
                ranges.append((start, end))
                continue
        try:
            singles.add(int(s))
        except ValueError:
            continue
    return singles, ranges


def _split_rics(rics: List[Any]) -> Tuple[List[Any], List[Any]]:
    includes: List[Any] = []
    excludes: List[Any] = []
    for item in rics:
        if isinstance(item, str):
            s = item.strip()
            if s.startswith("!"):
                val = s[1:].strip()
                if val:
                    excludes.append(val)
                continue
        includes.append(item)
    return includes, excludes


def _ric_matches(ric_int: Optional[int], singles: set, ranges: List[Tuple[int, int]]) -> bool:
    if ric_int is None:
        return False
    if ric_int in singles:
        return True
    for start, end in ranges:
        if start <= ric_int <= end:
            return True
    return False


class RoutingConfig:
    def __init__(
        self,
        channel_filters: List[Dict[str, Any]],
        ric_to_user: Dict[str, str],
        global_exclude_rics: List[Any],
        ric_phonebook: Optional[Dict[str, str]] = None,
    ):
        self.channel_filters = []
        for entry in channel_filters:
            if not isinstance(entry, dict):
                continue
            channel = entry.get("channel")
            rics = entry.get("rics", [])
            if channel is None:
                continue
            if not isinstance(rics, list):
                continue
            include_rics, channel_excludes = _split_rics(rics)
            singles, ranges = _compile_rics(include_rics)
            exclude_singles, exclude_ranges = _compile_rics(channel_excludes)
            self.channel_filters.append(
                {
                    "channel": channel,
                    "singles": singles,
                    "ranges": ranges,
                    "exclude_singles": exclude_singles,
                    "exclude_ranges": exclude_ranges,
                }
            )
        self.ric_to_user = {}
        for k, v in ric_to_user.items():
            _, key = _normalize_ric(k)
            self.ric_to_user[key] = str(v).strip()

        self.ric_phonebook = {}
        if isinstance(ric_phonebook, dict):
            for k, v in ric_phonebook.items():
                _, key = _normalize_ric(k)
                label = str(v).strip() if v is not None else ""
                if not key or not label:
                    continue
                self.ric_phonebook[key] = label

        self.exclude_singles, self.exclude_ranges = _compile_rics(global_exclude_rics or [])

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RoutingConfig":
        channel_filters = data.get("channel_filters", [])
        ric_to_user = data.get("ric_to_user", {})
        exclude_rics = data.get("exclude_rics", [])
        ric_phonebook = data.get("ric_phonebook", {})
        if not isinstance(channel_filters, list):
            channel_filters = []
        if not isinstance(ric_to_user, dict):
            ric_to_user = {}
        if not isinstance(exclude_rics, list):
            exclude_rics = []
        if not isinstance(ric_phonebook, dict):
            ric_phonebook = {}
        return RoutingConfig(channel_filters, ric_to_user, exclude_rics, ric_phonebook)

    @staticmethod
    def load(path: str) -> "RoutingConfig":
        data = _load_config(path)
        base_dir = os.path.dirname(os.path.abspath(path))
        data = _merge_ric_phonebook(data, base_dir)
        return RoutingConfig.from_dict(data)

    def route(self, ric_key: str, ric_int: Optional[int]) -> Optional[Tuple[str, Any]]:
        if _ric_matches(ric_int, self.exclude_singles, self.exclude_ranges):
            logging.debug("RIC %s ignored by global exclude list", ric_key)
            return None
        if ric_key in self.ric_to_user:
            return ("user", self.ric_to_user[ric_key])
        for entry in self.channel_filters:
            if _ric_matches(ric_int, entry["singles"], entry["ranges"]) and not _ric_matches(
                ric_int, entry["exclude_singles"], entry["exclude_ranges"]
            ):
                return ("channel", entry["channel"])
        return None

    def route_all(self, ric_key: str, ric_int: Optional[int]) -> List[Tuple[str, Any]]:
        if _ric_matches(ric_int, self.exclude_singles, self.exclude_ranges):
            logging.debug("RIC %s ignored by global exclude list", ric_key)
            return []
        results: List[Tuple[str, Any]] = []
        seen = set()

        if ric_key in self.ric_to_user:
            dest = self.ric_to_user[ric_key]
            key = ("user", dest)
            results.append(key)
            seen.add(key)

        for entry in self.channel_filters:
            if _ric_matches(ric_int, entry["singles"], entry["ranges"]) and not _ric_matches(
                ric_int, entry["exclude_singles"], entry["exclude_ranges"]
            ):
                dest = entry["channel"]
                key = ("channel", dest)
                if key not in seen:
                    results.append(key)
                    seen.add(key)
            elif _ric_matches(ric_int, entry["exclude_singles"], entry["exclude_ranges"]):
                logging.debug("RIC %s ignored by channel %s exclude list", ric_key, entry["channel"])

        return results


class DedupeCache:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.enabled = _get_bool("DEDUPE_ENABLED", cfg, "enabled", False)
        self.window_seconds = _get_int("DEDUPE_WINDOW_SECONDS", cfg, "window_seconds", 0)
        self.backend = (_get_str("DEDUPE_BACKEND", cfg, "backend", "memory") or "memory").lower()
        self.key_prefix = _get_str("DEDUPE_KEY_PREFIX", cfg, "key_prefix", "pocsag") or "pocsag"

        self._memory: Dict[str, float] = {}
        self._cleanup_every = 250
        self._ops = 0
        self._redis = None

        if self.enabled and self.window_seconds > 0 and self.backend == "redis":
            self._init_redis(cfg)

    def _init_redis(self, cfg: Dict[str, Any]) -> None:
        try:
            import redis  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logging.warning("Redis backend requested but redis module missing: %s", exc)
            self.backend = "memory"
            return

        url = _get_str("REDIS_URL", cfg, "redis_url", None)
        if url:
            logging.debug("Redis dedupe using url=%s", _redact_url(url))
            self._redis = redis.Redis.from_url(url)
            return

        host = _get_str("REDIS_HOST", cfg, "redis_host", "localhost")
        port = _get_int("REDIS_PORT", cfg, "redis_port", 6379)
        db = _get_int("REDIS_DB", cfg, "redis_db", 0)
        password = _get_str("REDIS_PASSWORD", cfg, "redis_password", None)
        tls = _get_bool("REDIS_TLS", cfg, "redis_tls", False)
        logging.debug("Redis dedupe using host=%s port=%s db=%s tls=%s", host, port, db, tls)
        scheme = "rediss" if tls else "redis"
        url = f"{scheme}://{host}:{port}/{db}"
        self._redis = redis.Redis.from_url(url, password=password)

    def _redis_seen(self, key: str) -> Optional[bool]:
        if self._redis is None:
            return None
        try:
            redis_key = f"{self.key_prefix}:{key}"
            exists = bool(self._redis.exists(redis_key))
            logging.debug(
                "Redis dedupe exists key=%s exists=%s",
                redis_key,
                exists,
            )
            return exists
        except Exception as exc:  # noqa: BLE001
            logging.warning("Redis dedupe failed, falling back to memory: %s", exc)
            self._redis = None
            self.backend = "memory"
            return None

    def _redis_mark(self, key: str) -> Optional[bool]:
        if self._redis is None:
            return None
        try:
            redis_key = f"{self.key_prefix}:{key}"
            self._redis.set(redis_key, "1", ex=self.window_seconds)
            logging.debug(
                "Redis dedupe set key=%s ttl=%ss",
                redis_key,
                self.window_seconds,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logging.warning("Redis dedupe failed, falling back to memory: %s", exc)
            self._redis = None
            self.backend = "memory"
            return None

    def _memory_seen(self, key: str) -> bool:
        now = time.time()
        self._ops += 1
        if self._ops % self._cleanup_every == 0:
            self._cleanup(now)
        expires_at = self._memory.get(key)
        return bool(expires_at and expires_at > now)

    def _memory_mark(self, key: str) -> None:
        now = time.time()
        self._memory[key] = now + self.window_seconds

    def _cleanup(self, now: float) -> None:
        expired = [k for k, v in self._memory.items() if v <= now]
        for k in expired:
            self._memory.pop(k, None)

    def is_duplicate(self, key: str) -> bool:
        if not self.enabled or self.window_seconds <= 0:
            return False
        cache_key = f"{self.key_prefix}:{key}"
        if self.backend == "redis":
            exists = self._redis_seen(key)
            if exists is not None:
                return exists
            logging.debug("Redis dedupe unavailable, falling back to memory")
        return self._memory_seen(cache_key)

    def mark_sent(self, key: str) -> None:
        if not self.enabled or self.window_seconds <= 0:
            return
        cache_key = f"{self.key_prefix}:{key}"
        if self.backend == "redis":
            marked = self._redis_mark(key)
            if marked is not None:
                return
            logging.debug("Redis dedupe unavailable, falling back to memory")
        self._memory_mark(cache_key)


class MeshMonitorSender:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.base_url = _get_str("MESHMONITOR_BASE_URL", cfg, "base_url", "http://localhost:8080").rstrip("/")
        self.send_path = _get_str("MESHMONITOR_SEND_PATH", cfg, "send_path", "/api/v1/messages")
        self.api_token = (_get_str("MESHMONITOR_API_TOKEN", cfg, "api_token", "") or "").strip()
        self.timeout = _get_float("MESHMONITOR_TIMEOUT", cfg, "timeout", 10.0)
        self.verify_tls = _get_bool("MESHMONITOR_VERIFY_TLS", cfg, "verify_tls", True)

        self.message_field = _get_str("MM_MESSAGE_FIELD", cfg, "message_field", "text")
        self.channel_field = _get_str("MM_CHANNEL_FIELD", cfg, "channel_field", "channel")
        self.user_field = _get_str("MM_USER_FIELD", cfg, "user_field", "toNodeId")

        self.extra_fields = _get_json("MM_EXTRA_JSON", cfg, "extra_json", {})

    def send(self, kind: str, dest: Any, message: str, ric_key: str, timestamp: Any) -> None:
        url = f"{self.base_url}{self.send_path}"
        payload: Dict[str, Any] = {self.message_field: message}
        if kind == "channel":
            payload[self.channel_field] = dest
        elif kind == "user":
            payload[self.user_field] = dest
        if isinstance(self.extra_fields, dict):
            payload.update(self.extra_fields)

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout, verify=self.verify_tls)
        if not resp.ok:
            raise RuntimeError(f"MeshMonitor send failed: {resp.status_code} {resp.text}")


class Forwarder:
    def __init__(
        self,
        routing: RoutingConfig,
        sender: MeshMonitorSender,
        runtime_cfg: Optional[Dict[str, Any]] = None,
        message_cfg: Optional[Dict[str, Any]] = None,
        dedupe_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.routing = routing
        self.sender = sender
        queue_max = _get_int("QUEUE_MAX", runtime_cfg, "queue_max", 1000)
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=queue_max)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, name="forwarder-worker", daemon=True)

        self.max_len = _get_int("MAX_MESSAGE_LEN", runtime_cfg, "max_message_len", 0)
        self.message_prefix = _get_str("MM_MESSAGE_PREFIX", message_cfg, "message_prefix", "") or ""
        self.message_suffix = _get_str("MM_MESSAGE_SUFFIX", message_cfg, "message_suffix", "") or ""
        self.include_ric = _get_bool("MM_INCLUDE_RIC", message_cfg, "include_ric", False)
        self.include_ric_name = _get_bool("MM_INCLUDE_RIC_NAME", message_cfg, "include_ric_name", False)
        self.include_timestamp = _get_bool("MM_INCLUDE_TIMESTAMP", message_cfg, "include_timestamp", False)

        self.dedupe = DedupeCache(dedupe_cfg or {})

    def start(self) -> None:
        self.worker.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=5)

    def enqueue(self, payload: Dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            logging.warning("Queue full, dropping message")

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(item)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed to process message: %s", exc)
            finally:
                self.queue.task_done()

    def _process(self, item: Dict[str, Any]) -> None:
        ric_raw = item.get("address")
        message = item.get("message")
        timestamp = item.get("timestamp")
        if message is None or ric_raw is None:
            return
        ric_int, ric_key = _normalize_ric(ric_raw)
        if not ric_key:
            return

        routes = self.routing.route_all(ric_key, ric_int)
        if not routes:
            return

        text = str(message)
        ric_label = self.routing.ric_phonebook.get(ric_key)
        include_ric_label = self.include_ric_name and bool(ric_label)

        def build_text(use_ric_label: bool) -> str:
            parts = []
            if self.message_prefix:
                parts.append(self.message_prefix)
            if text:
                parts.append(text)
            if self.include_ric:
                if use_ric_label and ric_label:
                    parts.append(f"[RIC {ric_key} {ric_label}]")
                else:
                    parts.append(f"[RIC {ric_key}]")
            elif use_ric_label and ric_label:
                parts.append(f"[RIC_NAME {ric_label}]")
            if self.include_timestamp and timestamp is not None:
                parts.append(f"[TS {timestamp}]")
            if self.message_suffix:
                parts.append(self.message_suffix)
            return " ".join(p for p in parts if p)

        if include_ric_label and self.max_len > 0:
            text_with_label = build_text(True)
            if len(text_with_label) <= self.max_len:
                text = text_with_label
            else:
                text = build_text(False)
        else:
            text = build_text(include_ric_label)
        if self.max_len > 0 and len(text) > self.max_len:
            text = text[: self.max_len]

        for kind, dest in routes:
            dedupe_key = self._dedupe_key(kind, dest, ric_key, text)
            if self.dedupe.is_duplicate(dedupe_key):
                logging.debug("Duplicate suppressed for RIC %s to %s=%s", ric_key, kind, dest)
                continue
            try:
                self.sender.send(kind, dest, text, ric_key, timestamp)
                self.dedupe.mark_sent(dedupe_key)
                logging.info("Forwarded RIC %s to %s=%s", ric_key, kind, dest)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed to send RIC %s to %s=%s: %s", ric_key, kind, dest, exc)

    @staticmethod
    def _dedupe_key(kind: str, dest: Any, ric_key: str, text: str) -> str:
        raw = f"{kind}|{dest}|{ric_key}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _setup_logging(runtime_cfg: Optional[Dict[str, Any]] = None) -> None:
    level = _get_str("LOG_LEVEL", runtime_cfg, "log_level", "INFO")
    level = (level or "INFO").upper()
    log_file = _get_str("LOG_FILE", runtime_cfg, "log_file", "/var/log/meshsag.log")

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level, logging.INFO))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.WatchedFileHandler(log_file)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception as exc:  # noqa: BLE001
            root.warning("Failed to set up log file %s: %s", log_file, exc)


def main() -> None:
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config_data = _load_config(config_path)

    runtime_cfg = config_data.get("runtime", {})
    mqtt_cfg = config_data.get("mqtt", {})
    mesh_cfg = config_data.get("meshmonitor", {})
    dedupe_cfg = config_data.get("dedupe", {})

    _setup_logging(runtime_cfg)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    config_data = _merge_ric_phonebook(config_data, config_dir)
    logging.info("=== MeshSAG starting ===")
    logging.info("Config path: %s", config_path)
    try:
        logging.info("Config (redacted):\n%s", yaml.safe_dump(_redact_config(config_data), sort_keys=False).strip())
    except Exception:  # noqa: BLE001
        logging.info("Config (redacted): %s", _redact_config(config_data))
    routing = RoutingConfig.from_dict(config_data)
    logging.debug(
        "Exclude parsed (global): singles=%s ranges=%s",
        sorted(routing.exclude_singles),
        routing.exclude_ranges,
    )
    for entry in routing.channel_filters:
        if entry["exclude_singles"] or entry["exclude_ranges"]:
            logging.debug(
                "Exclude parsed (channel %s): singles=%s ranges=%s",
                entry["channel"],
                sorted(entry["exclude_singles"]),
                entry["exclude_ranges"],
            )
    sender = MeshMonitorSender(mesh_cfg)
    forwarder = Forwarder(routing, sender, runtime_cfg, mesh_cfg, dedupe_cfg)
    forwarder.start()

    mqtt_host = _get_str("MQTT_HOST", mqtt_cfg, "host", "localhost")
    mqtt_port = _get_int("MQTT_PORT", mqtt_cfg, "port", 1883)
    mqtt_topic = _get_str("MQTT_TOPIC", mqtt_cfg, "topic", "owrx/POCSAG")
    mqtt_qos = _get_int("MQTT_QOS", mqtt_cfg, "qos", 0)
    mqtt_client_id = _get_str("MQTT_CLIENT_ID", mqtt_cfg, "client_id", "pocsag-forwarder")
    mqtt_username = _get_str("MQTT_USERNAME", mqtt_cfg, "username", None)
    mqtt_password = _get_str("MQTT_PASSWORD", mqtt_cfg, "password", None)

    client = mqtt.Client(client_id=mqtt_client_id, clean_session=True)
    if mqtt_username:
        client.username_pw_set(mqtt_username, mqtt_password)

    if _get_bool("MQTT_TLS", mqtt_cfg, "tls", False):
        client.tls_set(
            ca_certs=_get_str("MQTT_CA_CERT", mqtt_cfg, "ca_cert", None) or None,
            certfile=_get_str("MQTT_CLIENT_CERT", mqtt_cfg, "client_cert", None) or None,
            keyfile=_get_str("MQTT_CLIENT_KEY", mqtt_cfg, "client_key", None) or None,
        )

    def on_connect(_client, _userdata, _flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker %s:%s", mqtt_host, mqtt_port)
            _client.subscribe(mqtt_topic, qos=mqtt_qos)
        else:
            logging.error("MQTT connection failed with rc=%s", rc)

    def on_message(_client, _userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:  # noqa: BLE001
            logging.warning("Invalid JSON payload, skipping")
            return
        if isinstance(payload, dict):
            forwarder.enqueue(payload)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(mqtt_host, mqtt_port, keepalive=60)
    client.loop_start()

    stop_event = threading.Event()

    def _handle_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        client.loop_stop()
        forwarder.stop()


if __name__ == "__main__":
    main()
