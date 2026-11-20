"""Spec tests for uncapped MQTT bodies in the POCSAG forwarder.

The running forwarder subscribes to the configured topic and, in
``main``'s ``on_message``, UTF-8-decodes and JSON-decodes the whole
packet before ``Forwarder.enqueue``. ``max_message_len`` is applied
later, inside the worker, and only to the text passed to
``MeshMonitorSender.send``.

These tests publish to that topic through a broker and the same
paho-mqtt client the program ships. They fail while a body larger than
``MAX_ACCEPTED_MQTT_BYTES`` is decoded or queued, and while a message
longer than a positive ``max_message_len`` is stored on the queue.
They pass once those bodies are rejected before decoding and the queue
only retains a message no longer than the configured cap. Short pages
must still be forwarded.

The pager-phonebook HTTP client and config/env JSON loaders are not
this path and are not covered here. QoS 1 and 2 use the same
``on_message`` callback after paho delivers ``message.payload``.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tracemalloc
import unittest
from typing import Any, Dict, List, Optional

import pocsag_forwarder as pf
from paho.mqtt import client as mqtt

# Largest MQTT body the forwarder may decode or retain. POCSAG text is
# short (the sample config caps forwarded text at 233). 256 KiB is
# headroom for the JSON envelope and still far below a publish that can
# exhaust process memory. Mosquitto 2.0.18 defaults to no packet limit.
MAX_ACCEPTED_MQTT_BYTES = 256 * 1024
OVERSIZE_BYTES = 1024 * 1024
UNDER_CAP_MESSAGE = "M" * 4000
SHORT_MESSAGE = "hello-route"
AFTER_MESSAGE = "still-up"

_PUBLISHER = r"""
import json
import os
import select
import struct
import sys
import time

from paho.mqtt import publish

port = int(sys.argv[1])
topic = sys.argv[2]
sub_fd = int(sys.argv[3])
ack_fd = int(sys.argv[4])
if sys.argv[5] not in ("capped", "unlimited"):
    raise SystemExit("unknown mode")


def read_exact(fd, size):
    buf = b""
    while len(buf) < size:
        chunk = os.read(fd, size - len(buf))
        if not chunk:
            raise SystemExit("ack pipe closed")
        buf += chunk
    return buf


def publish_and_wait(payload):
    deadline = time.time() + 20
    last_publish = 0.0
    while time.time() < deadline:
        now = time.time()
        if now - last_publish >= 0.25:
            try:
                publish.single(
                    topic,
                    payload=payload,
                    hostname="127.0.0.1",
                    port=port,
                    qos=0,
                    client_id="",
                )
                last_publish = now
            except Exception:
                time.sleep(0.05)
                continue
        ready, _, _ = select.select([ack_fd], [], [], 0.1)
        if not ready:
            continue
        (observed,) = struct.unpack("!I", read_exact(ack_fd, 4))
        if observed == len(payload):
            print(json.dumps({"published": len(payload), "observed": observed}), flush=True)
            return
    raise SystemExit("timed out waiting for %s-byte publish" % len(payload))


read_exact(sub_fd, 1)

pages = [
    json.dumps(
        {"address": 123456, "message": "hello-route", "timestamp": "2020-01-01T00:00:00Z"}
    ).encode("utf-8"),
    json.dumps(
        {
            "address": 123456,
            "message": "M" * 4000,
            "timestamp": "2020-01-01T00:00:00Z",
        }
    ).encode("utf-8"),
]

pages.append(
    json.dumps(
        {
            "address": 123456,
            "message": "A" * (1024 * 1024),
            "timestamp": "2020-01-01T00:00:00Z",
        }
    ).encode("utf-8")
)
pages.append(b"{" + (b"B" * (1024 * 1024)))
pages.append(json.dumps(["C" * (1024 * 1024)]).encode("utf-8"))
pages.append(
    json.dumps(
        {"address": 123456, "message": "still-up", "timestamp": "2020-01-01T00:00:00Z"}
    ).encode("utf-8")
)

for body in pages:
    publish_and_wait(body)
"""


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _max_str_len(obj: Any, ceiling: int) -> int:
    best = 0
    stack = [obj]
    steps = 0
    while stack:
        cur = stack.pop()
        steps += 1
        if steps > 10000:
            break
        if isinstance(cur, str):
            best = max(best, len(cur))
        elif isinstance(cur, (bytes, bytearray)):
            best = max(best, len(cur))
        elif isinstance(cur, dict):
            stack.extend(cur.keys())
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)
        if best > ceiling:
            return best
    return best


def _clear_forwarder_env() -> Dict[str, Optional[str]]:
    names = [
        "CONFIG_PATH",
        "QUEUE_MAX",
        "MAX_MESSAGE_LEN",
        "SEND_DELAY_SECONDS",
        "LOG_LEVEL",
        "LOG_FILE",
        "MQTT_HOST",
        "MQTT_PORT",
        "MQTT_TOPIC",
        "MQTT_QOS",
        "MQTT_CLIENT_ID",
        "MQTT_USERNAME",
        "MQTT_PASSWORD",
        "MQTT_TLS",
        "MQTT_CA_CERT",
        "MQTT_CLIENT_CERT",
        "MQTT_CLIENT_KEY",
        "MESHMONITOR_BASE_URL",
        "MESHMONITOR_API_TOKEN",
        "MESHMONITOR_SEND_PATH",
        "MESHMONITOR_API_STYLE",
        "MESHMONITOR_SOURCE_ID",
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
        "DEDUPE_ENABLED",
        "DEDUPE_WINDOW_SECONDS",
        "DEDUPE_BACKEND",
        "DEDUPE_KEY_PREFIX",
        "PAGER_PHONEBOOK_API_ENABLED",
    ]
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    return saved


def _restore_env(saved: Dict[str, Optional[str]]) -> None:
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class _Session:
    def __init__(self, max_message_len: int) -> None:
        self.max_message_len = max_message_len
        self.records: List[Dict[str, Any]] = []
        self.sent: List[Dict[str, Any]] = []
        self.publisher_rc: Optional[int] = None
        self.publisher_out = ""
        self.publisher_err = ""
        self.broker_err = ""


class MqttPayloadCapTests(unittest.TestCase):
    def test_positive_max_len_forwards_short_pages_without_retaining_oversize_bodies(self) -> None:
        session = self._run(max_message_len=233, mode="capped")
        self._assert_delivered_in_full(session)
        self._assert_short_pages_forwarded(session)
        self._assert_oversize_payloads_dropped_before_decode(session)
        self._assert_under_cap_message_not_queued_past_max_len(session, 233)
        for item in session.sent:
            self.assertLessEqual(
                item["len"],
                233,
                "sent text longer than max_message_len: %s" % item,
            )

    def test_unlimited_max_len_still_drops_oversize_payloads_before_decode(self) -> None:
        session = self._run(max_message_len=0, mode="unlimited")
        self._assert_delivered_in_full(session)
        self._assert_short_pages_forwarded(session)
        self._assert_oversize_payloads_dropped_before_decode(session)
        under_cap = [
            rec
            for rec in session.records
            if rec["raw_len"] <= MAX_ACCEPTED_MQTT_BYTES and b"MMMM" in rec["head"]
        ]
        self.assertTrue(under_cap, "4000-character page was not delivered to the forwarder")
        forwarded = [item for item in session.sent if item["text"] == UNDER_CAP_MESSAGE]
        self.assertTrue(
            forwarded,
            "with max_message_len disabled, a page under the packet cap must still be forwarded intact; sent=%s"
            % session.sent,
        )

    def _assert_delivered_in_full(self, session: _Session) -> None:
        self.assertEqual(
            session.publisher_rc,
            0,
            "publisher failed rc=%s\n%s\n%s\nbroker:\n%s"
            % (session.publisher_rc, session.publisher_out, session.publisher_err, session.broker_err),
        )
        reported = [json.loads(line) for line in session.publisher_out.splitlines() if line.strip()]
        self.assertGreaterEqual(len(reported), 5, session.publisher_out)
        for item in reported:
            self.assertGreaterEqual(
                item["observed"],
                item["published"],
                "broker or client truncated the publish before the forwarder saw it: %s" % item,
            )
        oversized = [rec for rec in session.records if rec["raw_len"] > MAX_ACCEPTED_MQTT_BYTES]
        self.assertGreaterEqual(
            len(oversized),
            3,
            "expected the JSON object, the invalid body, and the JSON array to arrive oversized: %s"
            % [(rec["raw_len"], rec["head"]) for rec in session.records],
        )

    def _assert_short_pages_forwarded(self, session: _Session) -> None:
        texts = [item["text"] for item in session.sent]
        self.assertIn(SHORT_MESSAGE, texts, session.sent)
        self.assertIn(AFTER_MESSAGE, texts, session.sent)

    def _assert_oversize_payloads_dropped_before_decode(self, session: _Session) -> None:
        oversized = [rec for rec in session.records if rec["raw_len"] > MAX_ACCEPTED_MQTT_BYTES]
        for rec in oversized:
            self.assertLess(
                rec["extra_peak"],
                MAX_ACCEPTED_MQTT_BYTES,
                "decoding MQTT payload of %s bytes allocated %s more bytes in the forwarder; "
                "drop packets larger than %s before UTF-8 or JSON decoding. loads=%s enqueued=%s"
                % (
                    rec["raw_len"],
                    rec["extra_peak"],
                    MAX_ACCEPTED_MQTT_BYTES,
                    rec["loads_lens"],
                    rec["enqueued_max_str"],
                ),
            )
            for length in rec["loads_lens"]:
                self.assertLessEqual(
                    length,
                    MAX_ACCEPTED_MQTT_BYTES,
                    "json.loads saw %s bytes from an MQTT payload of %s; check length before decoding"
                    % (length, rec["raw_len"]),
                )
            for length in rec["enqueued_max_str"]:
                self.assertLessEqual(
                    length,
                    MAX_ACCEPTED_MQTT_BYTES,
                    "queue retained a %s-byte string from an MQTT payload of %s" % (length, rec["raw_len"]),
                )
        for item in session.sent:
            if item["len"] > MAX_ACCEPTED_MQTT_BYTES:
                self.fail("forwarded a body larger than the MQTT accept cap: %s" % item)

    def _assert_under_cap_message_not_queued_past_max_len(self, session: _Session, max_len: int) -> None:
        matches = [
            rec
            for rec in session.records
            if rec["raw_len"] <= MAX_ACCEPTED_MQTT_BYTES and b"MMMM" in rec["head"]
        ]
        self.assertTrue(matches, "under-cap oversize page did not reach on_message: %s" % session.records)
        for rec in matches:
            # Rejecting the page, or queueing it only after the message field
            # has been cut to max_message_len, both close the retention hole.
            # Keeping the original body on the queue does not.
            for length in rec["enqueued_max_str"]:
                self.assertLessEqual(
                    length,
                    max_len,
                    "queue kept %s characters of a message whose send cap is %s; "
                    "truncate or reject the message field before enqueue" % (length, max_len),
                )
        sent_texts = [item["text"] for item in session.sent]
        self.assertNotIn(
            UNDER_CAP_MESSAGE,
            sent_texts,
            "the uncut 4000-character body was forwarded; max_message_len is %s" % max_len,
        )

    def _run(self, max_message_len: int, mode: str) -> _Session:
        session = _Session(max_message_len)
        saved_env = _clear_forwarder_env()
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        holder: Dict[str, Optional[Dict[str, Any]]] = {"bucket": None}
        real_loads = pf.json.loads
        real_decode = pf.json.JSONDecoder.decode
        real_raw_decode = pf.json.JSONDecoder.raw_decode
        real_put = pf.queue.Queue.put
        real_send = pf.MeshMonitorSender.send
        real_subscribe = mqtt.Client.subscribe
        real_on_message = mqtt.Client._handle_on_message
        real_sleep = pf.time.sleep
        stop = threading.Event()

        def _note_decoded(value: Any) -> None:
            bucket = holder["bucket"]
            if bucket is None:
                return
            if isinstance(value, (str, bytes, bytearray)):
                bucket["loads_lens"].append(len(value))

        def spy_loads(value, *args, **kwargs):
            _note_decoded(value)
            return real_loads(value, *args, **kwargs)

        def spy_decode(decoder, value, *args, **kwargs):
            _note_decoded(value)
            return real_decode(decoder, value, *args, **kwargs)

        def spy_raw_decode(decoder, value, *args, **kwargs):
            _note_decoded(value)
            return real_raw_decode(decoder, value, *args, **kwargs)

        def _note_queued(item: Any) -> None:
            bucket = holder["bucket"]
            if bucket is None or not isinstance(item, dict):
                return
            # Measure what the queue actually stores. Truncating inside
            # enqueue before put_nowait is enough; inspecting only the
            # argument before that truncation would reject a real fix.
            bucket["enqueued_max_str"].append(_max_str_len(item, MAX_ACCEPTED_MQTT_BYTES + 1))

        def spy_put(q, item, *args, **kwargs):
            # put_nowait() calls put(); this is the single handoff into the queue.
            _note_queued(item)
            return real_put(q, item, *args, **kwargs)

        def spy_send(_sender, _kind, _dest, message, _ric_key, _timestamp):
            text = message if isinstance(message, str) and len(message) <= 4100 else ""
            session.sent.append({"len": len(message), "text": text})

        sub_r, sub_w = os.pipe()
        ack_r, ack_w = os.pipe()
        broker = None
        publisher = None
        broker_err_path = ""
        tmp = tempfile.TemporaryDirectory()
        try:
            port = _free_port()
            config_path = os.path.join(tmp.name, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        """\
                        mqtt:
                          host: 127.0.0.1
                          port: {port}
                          topic: owrx/POCSAG
                          qos: 0
                          client_id: pocsag-forwarder-cap-{port}
                          username: ""
                          password: ""
                          tls: false
                        meshmonitor:
                          base_url: http://127.0.0.1:9
                          api_token: ""
                          api_style: sources
                          source_id: default
                        runtime:
                          log_level: WARNING
                          log_file: ""
                          queue_max: 10
                          max_message_len: {max_len}
                          send_delay_seconds: 0
                        dedupe:
                          enabled: false
                        pager_phonebook_api:
                          enabled: false
                        ric_to_user:
                          "123456": "ALPHA"
                        """
                    ).format(port=port, max_len=max_message_len)
                )
            os.environ["CONFIG_PATH"] = config_path

            broker_conf = os.path.join(tmp.name, "mosquitto.conf")
            broker_err_path = os.path.join(tmp.name, "mosquitto.err")
            with open(broker_conf, "w", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        """\
                        listener {port} 127.0.0.1
                        allow_anonymous true
                        persistence false
                        log_dest stderr
                        """
                    ).format(port=port)
                )
            broker_err = open(broker_err_path, "w", encoding="utf-8")
            broker = subprocess.Popen(
                ["mosquitto", "-c", broker_conf],
                stdout=subprocess.DEVNULL,
                stderr=broker_err,
                start_new_session=True,
            )
            broker_err.close()
            self._wait_for_port(port, broker)

            def spy_subscribe(client, *args, **kwargs):
                result = real_subscribe(client, *args, **kwargs)
                try:
                    os.write(sub_w, b"S")
                except OSError:
                    pass
                return result

            def spy_handle_on_message(client, message):
                bucket = {
                    "raw_len": len(message.payload),
                    "head": bytes(message.payload[:120]),
                    "loads_lens": [],
                    "enqueued_max_str": [],
                    "extra_peak": 0,
                }
                holder["bucket"] = bucket
                tracemalloc.reset_peak()
                before = tracemalloc.get_traced_memory()[0]
                try:
                    return real_on_message(client, message)
                finally:
                    _current, peak = tracemalloc.get_traced_memory()
                    bucket["extra_peak"] = peak - before
                    holder["bucket"] = None
                    session.records.append(bucket)
                    try:
                        os.write(ack_w, struct.pack("!I", bucket["raw_len"]))
                    except OSError:
                        pass

            pf.json.loads = spy_loads
            pf.json.JSONDecoder.decode = spy_decode
            pf.json.JSONDecoder.raw_decode = spy_raw_decode
            pf.queue.Queue.put = spy_put
            pf.MeshMonitorSender.send = spy_send
            def spy_sleep(seconds):
                # main() blocks in time.sleep until a signal. Stop it from the
                # test once the publisher has finished without racing SIGTERM
                # against the handler main() installs.
                if stop.is_set():
                    raise SystemExit(0)
                return real_sleep(seconds)

            mqtt.Client.subscribe = spy_subscribe
            mqtt.Client._handle_on_message = spy_handle_on_message
            pf.time.sleep = spy_sleep

            publisher = subprocess.Popen(
                [sys.executable, "-c", _PUBLISHER, str(port), "owrx/POCSAG", str(sub_r), str(ack_r), mode],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(sub_r, ack_r),
                text=True,
            )
            os.close(sub_r)
            os.close(ack_r)
            sub_r = ack_r = -1

            def _stop_after_publisher() -> None:
                try:
                    publisher.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    publisher.kill()
                stop.set()

            stopper = threading.Thread(target=_stop_after_publisher, name="stop-forwarder", daemon=True)
            stopper.start()
            try:
                pf.main()
            except SystemExit:
                pass
            finally:
                stopper.join(timeout=5)
        finally:
            pf.json.loads = real_loads
            pf.json.JSONDecoder.decode = real_decode
            pf.json.JSONDecoder.raw_decode = real_raw_decode
            pf.queue.Queue.put = real_put
            pf.MeshMonitorSender.send = real_send
            mqtt.Client.subscribe = real_subscribe
            mqtt.Client._handle_on_message = real_on_message
            pf.time.sleep = real_sleep
            for fd in (sub_r, ack_r, sub_w, ack_w):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            if publisher is not None:
                if publisher.poll() is None:
                    publisher.kill()
                out, err = publisher.communicate(timeout=5)
                session.publisher_rc = publisher.returncode
                session.publisher_out = out or ""
                session.publisher_err = err or ""
            if broker is not None:
                if broker.poll() is None:
                    os.killpg(broker.pid, signal.SIGTERM)
                    try:
                        broker.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(broker.pid, signal.SIGKILL)
            try:
                with open(broker_err_path, encoding="utf-8") as handle:
                    session.broker_err = handle.read()
            except OSError:
                session.broker_err = ""
            tmp.cleanup()
            _restore_env(saved_env)
        return session

    def _wait_for_port(self, port: int, broker: subprocess.Popen) -> None:
        deadline = time.time() + 5
        while time.time() < deadline:
            if broker.poll() is not None:
                break
            try:
                sock = socket.create_connection(("127.0.0.1", port), 0.2)
            except OSError:
                time.sleep(0.05)
                continue
            sock.close()
            return
        err = ""
        raise AssertionError("mosquitto did not listen on %s (rc=%s) %s" % (port, broker.poll(), err))


if __name__ == "__main__":
    unittest.main()
