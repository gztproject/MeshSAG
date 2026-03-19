#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: ./mqtt_integration_test.sh [-c config.yaml] <RIC> <message...>

Publishes a test POCSAG JSON payload to the MQTT broker configured in MeshSAG.

Examples:
  ./mqtt_integration_test.sh 123456 Test message from shell
  ./mqtt_integration_test.sh -c /etc/meshsag/config.yaml 123456 "Dispatch test"
EOF
}

CONFIG_PATH="${CONFIG_PATH:-config.yaml}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1" >&2
        usage
        exit 1
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

RIC="$1"
shift
MESSAGE="$*"

if [[ ! "$RIC" =~ ^[0-9]+$ ]]; then
  echo "RIC must be numeric: $RIC" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

TIMESTAMP="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

export TEST_CONFIG_PATH="$CONFIG_PATH"
export TEST_RIC="$RIC"
export TEST_MESSAGE="$MESSAGE"
export TEST_TIMESTAMP="$TIMESTAMP"

python3 <<'PY'
import json
import os
import sys

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency PyYAML: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    from paho.mqtt import publish
except ImportError as exc:  # pragma: no cover
    print(f"Missing dependency paho-mqtt: {exc}", file=sys.stderr)
    sys.exit(1)


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    sys.exit(1)


config_path = os.environ["TEST_CONFIG_PATH"]
ric = os.environ["TEST_RIC"]
message = os.environ["TEST_MESSAGE"]
timestamp = os.environ["TEST_TIMESTAMP"]

try:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
except FileNotFoundError:
    fail(f"Config file not found: {config_path}")
except Exception as exc:  # noqa: BLE001
    fail(f"Failed to read config {config_path}: {exc}")

mqtt_cfg = config.get("mqtt") or {}
host = str(mqtt_cfg.get("host") or "localhost")
port = int(mqtt_cfg.get("port") or 1883)
topic = str(mqtt_cfg.get("topic") or "owrx/POCSAG")
qos = int(mqtt_cfg.get("qos") or 0)
client_id = str(mqtt_cfg.get("client_id") or "meshsag-mqtt-test")
username = mqtt_cfg.get("username") or None
password = mqtt_cfg.get("password") or None
tls_enabled = bool(mqtt_cfg.get("tls") or False)

auth = None
if username:
    auth = {"username": str(username), "password": None if password is None else str(password)}

tls = None
if tls_enabled:
    tls = {}
    ca_cert = mqtt_cfg.get("ca_cert") or None
    client_cert = mqtt_cfg.get("client_cert") or None
    client_key = mqtt_cfg.get("client_key") or None
    if ca_cert:
        tls["ca_certs"] = str(ca_cert)
    if client_cert:
        tls["certfile"] = str(client_cert)
    if client_key:
        tls["keyfile"] = str(client_key)

payload = {
    "address": int(ric),
    "message": message,
    "timestamp": timestamp,
}

try:
    publish.single(
        topic=topic,
        payload=json.dumps(payload),
        hostname=host,
        port=port,
        qos=qos,
        auth=auth,
        tls=tls,
        client_id=f"{client_id}-test",
    )
except Exception as exc:  # noqa: BLE001
    fail(f"Failed to publish MQTT test message: {exc}")

print(f"Published MQTT test message to {host}:{port} topic={topic} ric={ric}")
print(json.dumps(payload))
PY
