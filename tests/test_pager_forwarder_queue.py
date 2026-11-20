"""Failing spec for pager-forwarder queue loss.

A client that can publish to the subscribed MQTT topic can put a distinct JSON
page on the forwarder's single queue. config.example.yaml turns duplicate
suppression off, caps that queue at queue_max, and sleeps send_delay_seconds
after every routed delivery attempt. These tests require a newer routed page
to be delivered, and not kept behind an older flood, including when an earlier
delivery is still in progress, the queue is already full, the flood is for
another address or is not routed at all, one address matches several
destinations, or the mesh send fails.

Run from the repository root:

    python3 -m unittest tests.test_pager_forwarder_queue -v
"""

import json
import logging
import os
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import yaml

import pocsag_forwarder as pf


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_CONFIG = os.path.join(ROOT, "config.example.yaml")

# Environment variables that override the config dicts passed into Forwarder.
_OVERRIDES = (
    "QUEUE_MAX",
    "MAX_MESSAGE_LEN",
    "SEND_DELAY_SECONDS",
    "LOG_LEVEL",
    "LOG_FILE",
    "DEDUPE_ENABLED",
    "DEDUPE_WINDOW_SECONDS",
    "DEDUPE_BACKEND",
    "DEDUPE_KEY_PREFIX",
    "REDIS_URL",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "REDIS_TLS",
    "MESHMONITOR_BASE_URL",
    "MESHMONITOR_API_TOKEN",
    "MESHMONITOR_API_STYLE",
    "MESHMONITOR_SOURCE_ID",
    "MESHMONITOR_SEND_PATH",
    "MESHMONITOR_TIMEOUT",
    "MESHMONITOR_VERIFY_TLS",
    "MM_MESSAGE_FIELD",
    "MM_CHANNEL_FIELD",
    "MM_USER_FIELD",
    "MM_MESSAGE_PREFIX",
    "MM_MESSAGE_SUFFIX",
    "MM_EXTRA_JSON",
    "MM_INCLUDE_RIC",
    "MM_INCLUDE_RIC_NAME",
    "MM_INCLUDE_TIMESTAMP",
    "PAGER_PHONEBOOK_API_ENABLED",
    "PAGER_PHONEBOOK_API_BASE_URL",
    "PAGER_PHONEBOOK_API_PATH",
    "PAGER_PHONEBOOK_API_TOKEN",
    "PAGER_PHONEBOOK_API_TIMEOUT",
    "PAGER_PHONEBOOK_API_VERIFY_TLS",
    "PAGER_PHONEBOOK_API_PREFER_LONG_NAME",
)

ROUTED_RIC = 123456
OTHER_RIC = 200001
LEGIT = "LEGIT-PAGE"
SEED = "SEED-IN-FLIGHT"


def load_example() -> Dict[str, Any]:
    with open(EXAMPLE_CONFIG, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise AssertionError("config.example.yaml must be a mapping")
    return data


def page(address: Any, message: str, stamp: str = "2024-06-01T00:00:00Z") -> Dict[str, Any]:
    return {"address": address, "message": message, "timestamp": stamp}


class _MeshHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        server = self.server
        text = ""
        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                text = str(payload.get(server.message_field, ""))
        except (UnicodeDecodeError, json.JSONDecodeError):
            text = ""
        hold = False
        with server.lock:
            server.texts.append(text)
            if server.hold_first and not server.held:
                server.held = True
                hold = True
        if hold:
            server.first_held.set()
            server.release_first.wait()
        status = int(server.response_status)
        body = b"{}"
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


class MeshStub:
    """Local stand-in for MeshMonitor. The forwarder still uses MeshMonitorSender."""

    def __init__(self, message_field: str = "text", status: int = 200, hold_first: bool = False) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MeshHandler)
        self.httpd.texts = []
        self.httpd.lock = threading.Lock()
        self.httpd.message_field = message_field
        self.httpd.response_status = status
        self.httpd.hold_first = hold_first
        self.httpd.held = False
        self.httpd.first_held = threading.Event()
        self.httpd.release_first = threading.Event()
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="mesh-stub", daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def texts(self) -> List[str]:
        with self.httpd.lock:
            return list(self.httpd.texts)

    def release(self) -> None:
        self.httpd.release_first.set()

    def close(self) -> None:
        self.release()
        self.httpd.shutdown()
        self.httpd.server_close()


class PagerQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)
        self._saved_env = {key: os.environ.get(key) for key in _OVERRIDES}
        for key in _OVERRIDES:
            os.environ.pop(key, None)
        self.example = load_example()
        self.forwarders = []
        self.stubs = []
        self.processes = []

    def tearDown(self) -> None:
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for forwarder in self.forwarders:
            forwarder.stop_event.set()
        for stub in self.stubs:
            stub.release()
        for forwarder in self.forwarders:
            forwarder.stop()
        for stub in self.stubs:
            stub.close()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        logging.disable(logging.NOTSET)

    def _runtime(self, **overrides: Any) -> Dict[str, Any]:
        runtime = dict(self.example["runtime"])
        runtime.update(overrides)
        return runtime

    def _dedupe(self) -> Dict[str, Any]:
        return dict(self.example["dedupe"])

    def _mesh_cfg(self, stub: MeshStub) -> Dict[str, Any]:
        mesh = dict(self.example["meshmonitor"])
        mesh["base_url"] = stub.base_url
        return mesh

    def _forwarder(
        self,
        stub: MeshStub,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        routing: Optional[pf.RoutingConfig] = None,
    ) -> pf.Forwarder:
        mesh = self._mesh_cfg(stub)
        if routing is None:
            routing = pf.RoutingConfig.from_dict(self.example)
        forwarder = pf.Forwarder(
            routing,
            pf.MeshMonitorSender(mesh),
            pf.PagerPhonebookResolver(self.example.get("pager_phonebook_api")),
            self._runtime(**(runtime_overrides or {})),
            mesh,
            self._dedupe(),
        )
        self.forwarders.append(forwarder)
        self.stubs.append(stub)
        return forwarder

    def _watch(
        self,
        stub: MeshStub,
        legit: str,
        seed: Optional[str],
        max_other_texts: int,
        timeout: float,
        forwarder: Optional[pf.Forwarder] = None,
    ) -> None:
        deadline = time.monotonic() + timeout
        seen: List[str] = []
        idle_since: Optional[float] = None
        last_len = -1
        while time.monotonic() < deadline:
            seen = stub.texts()
            if legit in seen:
                before = seen[: seen.index(legit)]
                others = {text for text in before if text != seed}
                if len(others) > max_other_texts:
                    self.fail(
                        "newer page was forwarded only after older flood traffic: "
                        f"allowed {max_other_texts} other text(s), saw {seen}"
                    )
                return
            others = {text for text in seen if text != seed}
            if len(others) > max_other_texts:
                self.fail(
                    "older flood traffic was forwarded while the newer page was waiting: "
                    f"allowed {max_other_texts} other text(s), saw {seen}"
                )
            drained = False
            if forwarder is not None:
                drained = forwarder.queue.qsize() == 0 and forwarder.queue.unfinished_tasks == 0
            if drained and len(seen) == last_len:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= 0.3:
                    self.fail(f"newer routed page was discarded; forwarded {seen}")
            else:
                idle_since = None
            last_len = len(seen)
            time.sleep(0.02)
        self.fail(f"newer routed page was not forwarded within {timeout:.1f}s; saw {seen}")

    def test_example_config_does_not_bury_a_newer_routed_page(self) -> None:
        """Distinct pages for a routed RIC, with the example queue and delay, must not
        keep a newer page behind the backlog. Duplicate suppression stays as shipped."""
        routing = pf.RoutingConfig.from_dict(self.example)
        self.assertEqual(routing.route_all(str(ROUTED_RIC), ROUTED_RIC), [("channel", 0)])

        stub = MeshStub(hold_first=True)
        forwarder = self._forwarder(stub)
        self.assertEqual(forwarder.queue.maxsize, int(self.example["runtime"]["queue_max"]))
        self.assertEqual(forwarder.send_delay_seconds, float(self.example["runtime"]["send_delay_seconds"]))

        forwarder.start()
        forwarder.enqueue(page(ROUTED_RIC, SEED))
        self.assertTrue(stub.httpd.first_held.wait(5), "seed page did not reach MeshMonitor")
        for index in range(8):
            forwarder.enqueue(page(ROUTED_RIC, f"flood-{index}", stamp=f"2024-06-01T00:00:{index:02d}Z"))
        forwarder.enqueue(page(ROUTED_RIC, LEGIT, stamp="2024-06-01T00:00:59Z"))
        stub.release()

        delay = float(self.example["runtime"]["send_delay_seconds"])
        # One in-flight send may finish. Nothing already queued ahead of the new page may go out first.
        self._watch(stub, LEGIT, SEED, max_other_texts=0, timeout=delay * 3 + 2)

    def test_full_queue_keeps_a_newer_routed_page(self) -> None:
        """Once queue_max messages are waiting, a newer routed page must still be delivered
        ahead of that backlog. send_delay is left at 0 so a drop fails immediately."""
        queue_max = int(self.example["runtime"]["queue_max"])
        stub = MeshStub()
        forwarder = self._forwarder(stub, {"send_delay_seconds": 0})
        for index in range(queue_max):
            forwarder.enqueue(page(ROUTED_RIC, f"flood-{index}"))
        forwarder.enqueue(page(ROUTED_RIC, LEGIT))
        forwarder.start()
        self._watch(stub, LEGIT, seed=None, max_other_texts=0, timeout=5, forwarder=forwarder)

    def test_unrouted_json_cannot_discard_a_newer_routed_page(self) -> None:
        """on_message queues every JSON object before routing. Filling the queue with
        objects that will not be forwarded must not discard a later routed page."""
        queue_max = int(self.example["runtime"]["queue_max"])
        stub = MeshStub()
        forwarder = self._forwarder(stub, {"send_delay_seconds": 0})
        for index in range(queue_max):
            forwarder.enqueue({"unused": index})
        forwarder.enqueue(page(ROUTED_RIC, LEGIT))
        forwarder.start()
        self._watch(stub, LEGIT, seed=None, max_other_texts=0, timeout=5, forwarder=forwarder)

    def test_flood_on_one_address_does_not_bury_another(self) -> None:
        routing = pf.RoutingConfig.from_dict(self.example)
        self.assertEqual(routing.route_all(str(OTHER_RIC), OTHER_RIC), [("channel", 1)])
        stub = MeshStub(hold_first=True)
        forwarder = self._forwarder(stub, {"send_delay_seconds": 0.2, "queue_max": 1000})
        forwarder.start()
        forwarder.enqueue(page(ROUTED_RIC, SEED))
        self.assertTrue(stub.httpd.first_held.wait(5), "seed page did not reach MeshMonitor")
        for index in range(12):
            forwarder.enqueue(page(ROUTED_RIC, f"flood-{index}"))
        forwarder.enqueue(page(OTHER_RIC, LEGIT))
        stub.release()
        # The in-flight page may finish, and a one-deep cap for the flooded address may
        # still send a single older page. The other address must not wait out the burst.
        self._watch(stub, LEGIT, SEED, max_other_texts=1, timeout=3)

    def test_multi_destination_fanout_does_not_bury_a_newer_page(self) -> None:
        """A RIC that matches both a user and a channel sleeps once per destination.
        A newer page for that address must still jump the unsent fan-out backlog."""
        routing = pf.RoutingConfig(
            [{"channel": 0, "rics": [ROUTED_RIC]}],
            {str(ROUTED_RIC): "ALPHA"},
            [],
        )
        self.assertEqual(
            routing.route_all(str(ROUTED_RIC), ROUTED_RIC),
            [("user", "ALPHA"), ("channel", 0)],
        )
        stub = MeshStub(hold_first=True)
        forwarder = self._forwarder(
            stub,
            {"send_delay_seconds": 0.2, "queue_max": 1000},
            routing=routing,
        )
        forwarder.start()
        forwarder.enqueue(page(ROUTED_RIC, SEED))
        self.assertTrue(stub.httpd.first_held.wait(5), "seed page did not reach MeshMonitor")
        for index in range(6):
            forwarder.enqueue(page(ROUTED_RIC, f"flood-{index}"))
        forwarder.enqueue(page(ROUTED_RIC, LEGIT))
        stub.release()
        self._watch(stub, LEGIT, SEED, max_other_texts=0, timeout=3)

    def test_failed_mesh_send_does_not_let_a_flood_bury_a_newer_page(self) -> None:
        """The per-destination delay also runs when MeshMonitor rejects the send."""
        stub = MeshStub(status=500, hold_first=True)
        forwarder = self._forwarder(stub, {"send_delay_seconds": 0.2, "queue_max": 1000})
        forwarder.start()
        forwarder.enqueue(page(ROUTED_RIC, SEED))
        self.assertTrue(stub.httpd.first_held.wait(5), "seed page did not reach MeshMonitor")
        for index in range(6):
            forwarder.enqueue(page(ROUTED_RIC, f"flood-{index}"))
        forwarder.enqueue(page(ROUTED_RIC, LEGIT))
        stub.release()
        self._watch(stub, LEGIT, SEED, max_other_texts=0, timeout=3)

    def test_mqtt_publish_cannot_crowd_out_a_newer_routed_page(self) -> None:
        """Publish the flood as MQTT JSON to the example topic. The running forwarder
        must still deliver the newer routed page."""
        queue_max = 8
        stub = MeshStub(hold_first=True)
        self.stubs.append(stub)
        broker_port = _free_port()
        config_dir = tempfile.TemporaryDirectory()
        self.addCleanup(config_dir.cleanup)
        config_path = os.path.join(config_dir.name, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(_mqtt_config(broker_port, stub.base_url, queue_max), handle)

        broker = _start_mosquitto(broker_port, config_dir.name)
        self.processes.append(broker)
        self.assertTrue(_wait_proc_line(broker, "running", 5), _proc_output(broker))

        env = os.environ.copy()
        for key in _OVERRIDES:
            env.pop(key, None)
        env["CONFIG_PATH"] = config_path
        env["PYTHONUNBUFFERED"] = "1"
        forwarder_proc = subprocess.Popen(
            ["python3", os.path.join(ROOT, "pocsag_forwarder.py")],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.processes.append(forwarder_proc)
        self.assertTrue(
            _wait_proc_line(forwarder_proc, "Connected to MQTT", 10),
            _proc_output(forwarder_proc),
        )
        # on_connect logs before the subscribe packet is acknowledged.
        time.sleep(0.4)

        from paho.mqtt import client as mqtt

        publisher = mqtt.Client(client_id="queue-flood-test")
        publisher.connect("127.0.0.1", broker_port, keepalive=60)
        publisher.loop_start()
        self.addCleanup(publisher.loop_stop)
        self.addCleanup(publisher.disconnect)

        def publish_page(body: Dict[str, Any]) -> None:
            info = publisher.publish("owrx/POCSAG", json.dumps(body), qos=1)
            info.wait_for_publish(timeout=5)

        publish_page(page(ROUTED_RIC, SEED))
        self.assertTrue(stub.httpd.first_held.wait(5), "seed publish did not reach MeshMonitor: " + _proc_output(forwarder_proc))
        for index in range(queue_max):
            publish_page(page(ROUTED_RIC, f"flood-{index}"))
        publish_page(page(ROUTED_RIC, LEGIT))
        time.sleep(0.3)
        stub.release()
        self._watch(stub, LEGIT, SEED, max_other_texts=0, timeout=5)


def _mqtt_config(broker_port: int, mesh_url: str, queue_max: int) -> Dict[str, Any]:
    return {
        "mqtt": {
            "host": "127.0.0.1",
            "port": broker_port,
            "topic": "owrx/POCSAG",
            "qos": 1,
            "client_id": "pocsag-forwarder",
            "username": "",
            "password": "",
            "tls": False,
        },
        "meshmonitor": {
            "base_url": mesh_url,
            "api_style": "sources",
            "source_id": "default",
            "api_token": "",
            "timeout": 10,
            "verify_tls": True,
            "message_field": "text",
            "channel_field": "channel",
            "user_field": "toNodeId",
        },
        "runtime": {
            "log_level": "INFO",
            "log_file": "",
            "queue_max": queue_max,
            "max_message_len": 233,
            "send_delay_seconds": 0,
        },
        "dedupe": {"enabled": False, "window_seconds": 60, "backend": "memory"},
        "channel_filters": [{"channel": 0, "rics": [ROUTED_RIC]}],
        "exclude_rics": [],
        "ric_to_user": {},
        "pager_phonebook_api": {"enabled": False},
    }


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_mosquitto(port: int, directory: str) -> subprocess.Popen:
    config_path = os.path.join(directory, "mosquitto.conf")
    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    f"listener {port} 127.0.0.1",
                    "allow_anonymous true",
                    "persistence false",
                    "log_dest stderr",
                    "",
                ]
            )
        )
    return subprocess.Popen(
        ["mosquitto", "-c", config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_proc_line(proc: subprocess.Popen, needle: str, timeout: float) -> bool:
    if proc.stdout is None:
        return False

    def _read() -> None:
        for line in proc.stdout:
            proc._captured.append(line)  # type: ignore[attr-defined]
            if needle in line:
                proc._found.set()  # type: ignore[attr-defined]

    proc._captured = []  # type: ignore[attr-defined]
    proc._found = threading.Event()  # type: ignore[attr-defined]
    threading.Thread(target=_read, name="proc-log", daemon=True).start()
    if proc._found.wait(timeout):  # type: ignore[attr-defined]
        return True
    return proc.poll() is not None and needle in "".join(proc._captured)  # type: ignore[attr-defined]


def _proc_output(proc: subprocess.Popen) -> str:
    captured = getattr(proc, "_captured", [])
    return "".join(captured)[-4000:]


if __name__ == "__main__":
    unittest.main()
