"""Redis dedupe TLS must reject a trusted certificate issued for another name.

These tests fail while hostname matching is left off: the forwarder finishes
the handshake and sends the configured Redis password to that peer. They pass
once a rediss connection requires the certificate name to match the configured
host and still requires a trusted chain.
"""

import os
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pocsag_forwarder import DedupeCache, Forwarder, RoutingConfig

# Obvious placeholder. Not a real credential.
REDIS_PASSWORD_PLACEHOLDER = "placeholder"

CONFIGURED_HOST = "localhost"
OTHER_NAME = "not-redis.example"
RIC = "123456"

_ENV_KEYS = (
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
    "SEND_DELAY_SECONDS",
    "QUEUE_MAX",
    "MAX_MESSAGE_LEN",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


def _openssl(args, cwd):
    completed = subprocess.run(
        ["openssl", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )
    return completed


def _make_ca(directory):
    cert = os.path.join(directory, "ca.pem")
    key = os.path.join(directory, "ca.key")
    _openssl(
        [
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            key,
            "-out",
            cert,
            "-days",
            "2",
            "-subj",
            "/CN=MeshSAGTestCA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        ],
        directory,
    )
    return cert, key


def _make_leaf(directory, ca_cert, ca_key, name, filename):
    key = os.path.join(directory, f"{filename}.key")
    csr = os.path.join(directory, f"{filename}.csr")
    cert = os.path.join(directory, f"{filename}.pem")
    _openssl(
        [
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            key,
            "-out",
            csr,
            "-subj",
            f"/CN={name}",
            "-addext",
            f"subjectAltName=DNS:{name}",
            "-addext",
            "extendedKeyUsage=serverAuth",
            "-addext",
            "keyUsage=digitalSignature,keyEncipherment",
        ],
        directory,
    )
    _openssl(
        [
            "x509",
            "-req",
            "-in",
            csr,
            "-CA",
            ca_cert,
            "-CAkey",
            ca_key,
            "-CAcreateserial",
            "-out",
            cert,
            "-days",
            "2",
            "-copy_extensions",
            "copy",
        ],
        directory,
    )
    return cert, key


class _RedisPeer:
    """TLS endpoint that speaks enough RESP to record authentication."""

    def __init__(self, certfile, keyfile):
        self.certfile = certfile
        self.keyfile = keyfile
        self.tcp_accepted = False
        self.handshake_ok = False
        self.auth_attempted = False
        self.saw_configured_password = False
        self.command_names = []
        self.port = None
        self._thread = None
        self._listener = None
        self._ready = threading.Event()
        self._store = set()

    def start(self):
        listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        listener.bind(("::1", 0))
        listener.listen(1)
        listener.settimeout(5)
        self._listener = listener
        self.port = listener.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, name="redis-peer", daemon=True)
        self._thread.start()
        self._ready.wait(5)

    def close(self):
        listener = self._listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _serve(self):
        self._ready.set()
        try:
            conn, _addr = self._listener.accept()
        except OSError:
            return
        self.tcp_accepted = True
        conn.settimeout(5)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certfile, self.keyfile)
        try:
            tls = context.wrap_socket(conn, server_side=True)
        except (ssl.SSLError, OSError, socket.timeout):
            try:
                conn.close()
            except OSError:
                pass
            return
        self.handshake_ok = True
        tls.settimeout(2)
        try:
            reader = _Reader(tls)
            while True:
                try:
                    args = _read_array(reader)
                except (socket.timeout, EOFError, ssl.SSLError, OSError):
                    break
                if not args:
                    break
                name = args[0]
                self.command_names.append(name.decode("utf-8", "replace"))
                if name == b"AUTH":
                    self.auth_attempted = True
                    supplied = args[-1]
                    if supplied == REDIS_PASSWORD_PLACEHOLDER.encode("utf-8"):
                        self.saw_configured_password = True
                        tls.sendall(b"+OK\r\n")
                    else:
                        tls.sendall(b"-ERR no\r\n")
                    continue
                if name == b"EXISTS" and len(args) >= 2:
                    present = 1 if args[1] in self._store else 0
                    tls.sendall(f":{present}\r\n".encode("ascii"))
                    continue
                if name == b"SET" and len(args) >= 2:
                    self._store.add(args[1])
                    tls.sendall(b"+OK\r\n")
                    continue
                tls.sendall(b"+OK\r\n")
        finally:
            try:
                tls.close()
            except OSError:
                pass


class _Reader:
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""

    def _fill(self):
        chunk = self.sock.recv(4096)
        if not chunk:
            raise EOFError
        self.buf += chunk

    def read_exact(self, size):
        while len(self.buf) < size:
            self._fill()
        data = self.buf[:size]
        self.buf = self.buf[size:]
        return data

    def read_line(self):
        while b"\r\n" not in self.buf:
            self._fill()
        line, self.buf = self.buf.split(b"\r\n", 1)
        return line


def _read_array(reader):
    header = reader.read_line()
    if not header.startswith(b"*"):
        raise EOFError
    count = int(header[1:])
    args = []
    for _ in range(count):
        bulk = reader.read_line()
        if not bulk.startswith(b"$"):
            raise EOFError
        length = int(bulk[1:])
        if length < 0:
            args.append(b"")
            continue
        args.append(reader.read_exact(length))
        trailer = reader.read_exact(2)
        if trailer != b"\r\n":
            raise EOFError
    return args


class RedisTlsHostnameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.mkdtemp(prefix="meshsag-redis-tls-")
        cls.ca_cert, cls.ca_key = _make_ca(cls._directory)
        cls.other_cert, cls.other_key = _make_leaf(
            cls._directory, cls.ca_cert, cls.ca_key, OTHER_NAME, "other"
        )
        cls.match_cert, cls.match_key = _make_leaf(
            cls._directory, cls.ca_cert, cls.ca_key, CONFIGURED_HOST, "match"
        )
        # A name that matches the host, but signed by a different CA the
        # process is not told to trust.
        other_dir = os.path.join(cls._directory, "untrusted")
        os.makedirs(other_dir)
        untrusted_ca, untrusted_key = _make_ca(other_dir)
        cls.untrusted_cert, cls.untrusted_key = _make_leaf(
            other_dir, untrusted_ca, untrusted_key, CONFIGURED_HOST, "leaf"
        )

    def setUp(self):
        self._saved_env = {key: os.environ.get(key) for key in _ENV_KEYS}
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        os.environ["SSL_CERT_FILE"] = self.ca_cert

    def tearDown(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _url_config(self, port):
        password = quote(REDIS_PASSWORD_PLACEHOLDER, safe="")
        return {
            "enabled": True,
            "window_seconds": 60,
            "backend": "redis",
            "key_prefix": "pocsag",
            "redis_url": f"rediss://:{password}@{CONFIGURED_HOST}:{port}/0",
        }

    def _tls_config(self, port):
        return {
            "enabled": True,
            "window_seconds": 60,
            "backend": "redis",
            "key_prefix": "pocsag",
            "redis_host": CONFIGURED_HOST,
            "redis_port": port,
            "redis_db": 0,
            "redis_password": REDIS_PASSWORD_PLACEHOLDER,
            "redis_tls": True,
        }

    def _forward_messages(self, dedupe_cfg, times):
        routing = RoutingConfig([], {RIC: "NODE"}, [], {})
        sender = _Sender()
        app = Forwarder(
            routing,
            sender,
            None,
            {"send_delay_seconds": 0, "queue_max": 10, "max_message_len": 0},
            {},
            dedupe_cfg,
        )
        payload = {"address": RIC, "message": "alert", "timestamp": 1}
        try:
            for _ in range(times):
                app._process(payload)
        finally:
            client = app.dedupe._redis
            if client is not None:
                client.close()
        return sender

    def _mark_sent(self, dedupe_cfg):
        cache = DedupeCache(dedupe_cfg)
        try:
            cache.mark_sent("same-message")
        finally:
            if cache._redis is not None:
                cache._redis.close()
        return cache

    def test_rediss_url_rejects_trusted_certificate_for_another_name(self):
        peer = _RedisPeer(self.other_cert, self.other_key)
        peer.start()
        try:
            self._forward_messages(self._url_config(peer.port), times=1)
        finally:
            peer.close()
        self.assertTrue(peer.tcp_accepted)
        self.assertFalse(peer.auth_attempted)
        self.assertFalse(peer.saw_configured_password)

    def test_redis_tls_mark_sent_rejects_trusted_certificate_for_another_name(self):
        peer = _RedisPeer(self.other_cert, self.other_key)
        peer.start()
        try:
            self._mark_sent(self._tls_config(peer.port))
        finally:
            peer.close()
        self.assertTrue(peer.tcp_accepted)
        self.assertFalse(peer.auth_attempted)
        self.assertFalse(peer.saw_configured_password)

    def test_rediss_url_accepts_matching_name_and_dedupes(self):
        peer = _RedisPeer(self.match_cert, self.match_key)
        peer.start()
        try:
            sender = self._forward_messages(self._url_config(peer.port), times=2)
        finally:
            peer.close()
        self.assertTrue(peer.handshake_ok)
        self.assertTrue(peer.saw_configured_password)
        self.assertEqual(sender.count, 1)
        self.assertIn("EXISTS", peer.command_names)
        self.assertIn("SET", peer.command_names)

    def test_redis_tls_mark_sent_accepts_matching_name(self):
        peer = _RedisPeer(self.match_cert, self.match_key)
        peer.start()
        try:
            self._mark_sent(self._tls_config(peer.port))
        finally:
            peer.close()
        self.assertTrue(peer.handshake_ok)
        self.assertTrue(peer.saw_configured_password)
        self.assertIn("SET", peer.command_names)

    def test_rediss_url_rejects_untrusted_chain_for_matching_name(self):
        peer = _RedisPeer(self.untrusted_cert, self.untrusted_key)
        peer.start()
        try:
            self._forward_messages(self._url_config(peer.port), times=1)
        finally:
            peer.close()
        self.assertTrue(peer.tcp_accepted)
        self.assertFalse(peer.handshake_ok)
        self.assertFalse(peer.auth_attempted)
        self.assertFalse(peer.saw_configured_password)

    def test_redis_tls_mark_sent_rejects_untrusted_chain(self):
        peer = _RedisPeer(self.untrusted_cert, self.untrusted_key)
        peer.start()
        try:
            self._mark_sent(self._tls_config(peer.port))
        finally:
            peer.close()
        self.assertTrue(peer.tcp_accepted)
        self.assertFalse(peer.handshake_ok)
        self.assertFalse(peer.auth_attempted)
        self.assertFalse(peer.saw_configured_password)


class _Sender:
    def __init__(self):
        self.count = 0

    def send(self, kind, dest, message, ric_key, timestamp):
        self.count += 1


if __name__ == "__main__":
    unittest.main()
