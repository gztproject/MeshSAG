"""Pager pages must survive a second MQTT connection that uses the shipped client id.

The installed forwarder connects with the client id from config.example.yaml
(copied into the service config by scripts/install.sh) or, when that key is
omitted, with the default applied in pocsag_forwarder.main. These tests start
that process against a broker that accepts the example's anonymous connection,
then connect again with that same id and publish a QoS 0 page.

They fail while the second connection takes the forwarder's session. They pass
once a connection that only knows the shipped id can no longer drop the page.
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from paho.mqtt import client as mqtt

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config.example.yaml"
FORWARDER = REPO_ROOT / "pocsag_forwarder.py"
CLIENT_ID_RE = re.compile(r" as (.+) \(p\d+, c\d+, k\d+")

# paho-mqtt 1.6.1 waits at least this long before its first reconnect.
# Publishing inside the window is what makes a stolen session drop a QoS 0 page.
PAHO_MIN_RECONNECT_SECONDS = 1.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _load_example() -> dict:
    with EXAMPLE_CONFIG.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AssertionError("config.example.yaml must be a mapping")
    return data


class _MeshHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length)
        with self.server.records_lock:
            self.server.records.append((self.path, body))
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):  # noqa: A003
        return


class MeshSink:
    def __init__(self) -> None:
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MeshHandler)
        self._httpd.records = []
        self._httpd.records_lock = threading.Lock()
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def received(self, text: str) -> bool:
        needle = text.encode("utf-8")
        with self._httpd.records_lock:
            return any(needle in body for _path, body in self._httpd.records)

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


class Broker:
    def __init__(self) -> None:
        self.port = _free_port()
        self._dir = Path(os.environ.get("TMPDIR", "/tmp")) / f"meshsag-mqtt-{uuid.uuid4().hex}"
        self._dir.mkdir(parents=True)
        config_path = self._dir / "mosquitto.conf"
        # Anonymous listener matches config.example.yaml, which leaves username empty.
        config_path.write_text(
            "\n".join(
                [
                    f"listener {self.port} 127.0.0.1",
                    "allow_anonymous true",
                    "log_dest stdout",
                    "log_type all",
                    "connection_messages true",
                    "persistence false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        self.lines: list[str] = []
        self._lines_lock = threading.Lock()
        self._proc = subprocess.Popen(
            ["mosquitto", "-c", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(self._dir),
        )
        assert self._proc.stdout is not None
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        self._wait_until_listening()

    def _read_output(self) -> None:
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            with self._lines_lock:
                self.lines.append(raw.decode("utf-8", errors="replace"))

    def _wait_until_listening(self) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"mosquitto exited early: {''.join(self.client_ids_log())}")
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise RuntimeError("mosquitto did not accept connections")

    def client_ids_log(self) -> list[str]:
        with self._lines_lock:
            return list(self.lines)

    def connected_client_ids(self) -> list[str]:
        found = []
        for line in self.client_ids_log():
            match = CLIENT_ID_RE.search(line)
            if match and "New client connected" in line:
                found.append(match.group(1))
        return found

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)


class ForwarderProcess:
    def __init__(self, config_path: Path) -> None:
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("MQTT_", "MESHMONITOR_", "MM_", "DEDUPE_", "PAGER_", "REDIS_", "LOG_", "QUEUE_", "MAX_", "SEND_")):
                env.pop(key, None)
        env["CONFIG_PATH"] = str(config_path)
        self._proc = subprocess.Popen(
            [sys.executable, str(FORWARDER)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.output: list[str] = []
        self._lock = threading.Lock()
        assert self._proc.stdout is not None
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

    def _read_output(self) -> None:
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            with self._lock:
                self.output.append(raw.decode("utf-8", errors="replace"))

    def output_text(self) -> str:
        with self._lock:
            return "".join(self.output)

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)


def _write_config(data: dict, directory: Path) -> Path:
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _retarget(data: dict, broker: Broker, mesh: MeshSink, log_file: Path) -> dict:
    """Point the shipped config at the local broker and page sink.

    client id, qos, username, and password are left as the example or the
    code default under test. Only endpoints that must be local move.
    """
    mqtt_cfg = dict(data.get("mqtt") or {})
    mqtt_cfg["host"] = "127.0.0.1"
    mqtt_cfg["port"] = broker.port
    mesh_cfg = dict(data.get("meshmonitor") or {})
    mesh_cfg["base_url"] = mesh.base_url
    runtime_cfg = dict(data.get("runtime") or {})
    runtime_cfg["log_file"] = str(log_file)
    updated = dict(data)
    updated["mqtt"] = mqtt_cfg
    updated["meshmonitor"] = mesh_cfg
    updated["runtime"] = runtime_cfg
    return updated


def _publish(broker: Broker, topic: str, text: str, qos: int) -> None:
    payload = json.dumps(
        {"address": 123456, "message": text, "timestamp": int(time.time())}
    )
    publisher = mqtt.Client(client_id=f"owrx-publisher-{uuid.uuid4().hex}", clean_session=True)
    publisher.connect("127.0.0.1", broker.port, keepalive=30)
    publisher.loop_start()
    try:
        info = publisher.publish(topic, payload, qos=qos, retain=False)
        info.wait_for_publish(timeout=5)
    finally:
        publisher.disconnect()
        publisher.loop_stop()


def _wait_until(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class _HeldClient:
    """Anonymous client that connects with a chosen id and stays there."""

    def __init__(self, broker: Broker, client_id: str) -> None:
        self.client_id = client_id
        self._connected = threading.Event()
        self._rc = None
        self._client = mqtt.Client(client_id=client_id, clean_session=True, userdata=self)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.reconnect_delay_set(min_delay=0, max_delay=0)
        self._broker = broker

    def _on_connect(self, _client, _userdata, _flags, rc):
        self._rc = rc
        if rc == 0:
            self._connected.set()

    def _on_disconnect(self, client, _userdata, _rc):
        # Take the id back immediately, the way the attack keeps the session.
        if client is self._client:
            try:
                client.reconnect()
            except Exception:
                pass

    def connect(self) -> None:
        self._client.connect("127.0.0.1", self._broker.port, keepalive=60)
        self._client.loop_start()
        if not self._connected.wait(5):
            raise AssertionError(
                f"attacker connection with client id {self.client_id!r} was not accepted; "
                f"rc={self._rc}"
            )

    def close(self) -> None:
        try:
            self._client.disconnect()
        except Exception:
            pass
        self._client.loop_stop()


class MqttClientIdTakeoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = Path(os.environ.get("TMPDIR", "/tmp")) / f"meshsag-case-{uuid.uuid4().hex}"
        self._tmpdir.mkdir(parents=True)
        self.broker = Broker()
        self.mesh = MeshSink()
        self.forwarders: list[ForwarderProcess] = []
        self.attackers: list[_HeldClient] = []

    def tearDown(self) -> None:
        for attacker in self.attackers:
            attacker.close()
        for forwarder in self.forwarders:
            forwarder.close()
        self.broker.close()
        self.mesh.close()

    def _start_forwarder(self, data: dict) -> ForwarderProcess:
        config_path = _write_config(
            _retarget(data, self.broker, self.mesh, self._tmpdir / f"forwarder-{len(self.forwarders)}.log"),
            self._tmpdir,
        )
        forwarder = ForwarderProcess(config_path)
        self.forwarders.append(forwarder)
        return forwarder

    def _prove_subscription(self, topic: str, qos: int) -> str:
        token = f"baseline-{uuid.uuid4().hex}"
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            _publish(self.broker, topic, token, qos)
            if _wait_until(lambda: self.mesh.received(token), 0.4):
                return token
        forwarder_logs = "\n".join(item.output_text() for item in self.forwarders)
        self.fail(
            "forwarder never delivered a baseline page; broker log:\n"
            + "".join(self.broker.client_ids_log())
            + "\nforwarder log:\n"
            + forwarder_logs
        )

    def _assert_page_survives_takeover(self, topic: str, qos: int, attacker_id: str) -> None:
        self._prove_subscription(topic, qos)
        attacker = _HeldClient(self.broker, attacker_id)
        self.attackers.append(attacker)
        attacker.connect()
        # CONNACK is sent only after the broker has closed the previous session
        # for this id. Publish before paho's first reconnect delay elapses.
        token = f"dispatch-{uuid.uuid4().hex}"
        published_at = time.monotonic()
        _publish(self.broker, topic, token, qos=0)
        self.assertLess(
            time.monotonic() - published_at,
            PAHO_MIN_RECONNECT_SECONDS,
            "publish took longer than the client's minimum reconnect delay",
        )
        delivered = _wait_until(lambda: self.mesh.received(token), 2.5)
        if delivered:
            return
        forwarder_logs = "\n".join(item.output_text() for item in self.forwarders)
        self.fail(
            "QoS 0 page published while a second client held the shipped client id "
            f"{attacker_id!r} was not forwarded.\nbroker log:\n"
            + "".join(self.broker.client_ids_log())
            + "\nforwarder log:\n"
            + forwarder_logs
        )

    def test_example_config_client_id_cannot_drop_pages(self) -> None:
        """Install copies config.example.yaml. That client id must not be stealable."""
        data = _load_example()
        mqtt_cfg = data.get("mqtt") or {}
        shipped_id = mqtt_cfg.get("client_id")
        # Empty or missing means "broker assigns an id". A second client that
        # uses that same configured value must still not be able to take the session.
        attacker_id = "" if shipped_id is None else str(shipped_id)
        topic = str(mqtt_cfg.get("topic") or "owrx/POCSAG")
        qos = mqtt_cfg.get("qos")
        self._start_forwarder(data)
        self._assert_page_survives_takeover(topic, int(qos) if qos is not None else 0, attacker_id)

    def test_omitted_client_id_is_not_stable_across_processes(self) -> None:
        """README's minimal mqtt block omits client_id. The code default must not repeat."""
        data = _load_example()
        mqtt_cfg = dict(data.get("mqtt") or {})
        mqtt_cfg.pop("client_id", None)
        mqtt_cfg.pop("qos", None)
        data["mqtt"] = mqtt_cfg
        topic = str(mqtt_cfg.get("topic") or "owrx/POCSAG")

        first = self._start_forwarder(data)
        if not _wait_until(lambda: len(self.broker.connected_client_ids()) >= 1, 8):
            self.fail("first forwarder did not connect:\n" + first.output_text() + "".join(self.broker.client_ids_log()))
        previous_id = self.broker.connected_client_ids()[0]
        self.assertTrue(previous_id)
        first.close()
        self.forwarders.remove(first)

        self._start_forwarder(data)
        self._assert_page_survives_takeover(topic, 0, previous_id)


if __name__ == "__main__":
    unittest.main()
