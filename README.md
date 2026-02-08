# POCSAG -> MeshMonitor Forwarder

Listens to POCSAG JSON messages on MQTT and forwards matched RICs to MeshMonitor for delivery into Meshtastic.

## What it does
- Subscribes to `owrx/POCSAG` (JSON with `message`, `address`, `timestamp`)
- Routes by RIC:
  - `ric_to_user`: exact RIC -> short name (direct message)
  - `channel_filters`: list of RICs or ranges -> channel
  - A single RIC can match multiple entries; the message is sent to all matches.
  - `exclude_rics`: list of RICs or ranges to suppress entirely.
  - Per-channel exclusions: prefix a RIC or range with `!` inside `rics` (quote the value so YAML doesn't treat `!` as a tag).

MeshMonitor v2 provides an API with token auth and Swagger docs at `/api/v1/docs` on your instance. The release notes also mention a `POST /api/v1/messages` endpoint for programmatic sends.

## Files
- `pocsag_forwarder.py`
- `config.example.yaml`
- `requirements.txt`

## Setup
1. Create config:
   - Copy `config.example.yaml` to `config.yaml` and edit.
2. Install deps:
   - `python3 -m venv .venv`
   - `. .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Set environment or config:
   - Preferred: put settings in `config.yaml` under `mqtt`, `meshmonitor`, and `runtime`.
   - Environment variables are optional and override config values.
   - Minimal config example:
   ```yaml
   mqtt:
     host: localhost
     port: 1883
     topic: owrx/POCSAG

   meshmonitor:
     base_url: http://localhost:8080
     api_token: your_token_here
     # message_prefix, message_suffix, include_ric, include_timestamp also supported

   runtime:
     log_level: INFO
     log_file: /var/log/meshsag.log
 
   dedupe:
     enabled: true
     window_seconds: 60
     backend: memory
   ```
   - Optional env overrides:
   - `CONFIG_PATH` (default `config.yaml`)
   - `MQTT_HOST`, `MQTT_PORT`, `MQTT_TOPIC`, `MQTT_QOS`, `MQTT_CLIENT_ID`
   - `MQTT_USERNAME`, `MQTT_PASSWORD`
   - `MQTT_TLS=true`, `MQTT_CA_CERT`, `MQTT_CLIENT_CERT`, `MQTT_CLIENT_KEY`
   - `MESHMONITOR_BASE_URL`, `MESHMONITOR_API_TOKEN`, `MESHMONITOR_SEND_PATH`
   - `MESHMONITOR_TIMEOUT`, `MESHMONITOR_VERIFY_TLS`
   - `MM_MESSAGE_FIELD`, `MM_CHANNEL_FIELD`, `MM_USER_FIELD`
   - `MM_MESSAGE_PREFIX`, `MM_MESSAGE_SUFFIX`
   - `MM_EXTRA_JSON`, `MM_INCLUDE_RIC`, `MM_INCLUDE_TIMESTAMP`
   - `DEDUPE_ENABLED`, `DEDUPE_WINDOW_SECONDS`, `DEDUPE_BACKEND`, `DEDUPE_KEY_PREFIX`
   - `REDIS_URL` or `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`, `REDIS_TLS`
   - `LOG_FILE`
   - `QUEUE_MAX`, `MAX_MESSAGE_LEN`, `LOG_LEVEL`
4. Run:
   - `python pocsag_forwarder.py`

## MeshMonitor request mapping
By default, the sender posts to `/api/v1/messages` with JSON:
- channel: `{ "text": "...", "channel": <channel> }`
- user: `{ "text": "...", "toNodeId": "<shortName>" }`

If your MeshMonitor API expects different field names or extra params, adjust via env:
- `MM_MESSAGE_FIELD` (default `text`)
- `MM_CHANNEL_FIELD` (default `channel`)
- `MM_USER_FIELD` (default `toNodeId`)
- `MM_MESSAGE_PREFIX`, `MM_MESSAGE_SUFFIX`
- `MM_EXTRA_JSON` (raw JSON string merged into the payload)
- `MM_INCLUDE_RIC`, `MM_INCLUDE_TIMESTAMP` (append `[RIC ...]` / `[TS ...]` to message text)

Note: Verify the actual payload shape for your MeshMonitor version using the Swagger UI on your server.

## Deduplication
If enabled, the forwarder suppresses duplicate sends to the same destination when the same message text appears within the configured window. Keys are derived from `kind + destination + RIC + message text`. Backend can be `memory` or `redis` (optional).

## Example routing config
```yaml
channel_filters:
  - channel: 0
    rics:
      - 123456
      - "123460-123470"
      - "!123462"

exclude_rics:
  - 123499
  - "123500-123520"

ric_to_user:
  123499: "ALPHA"
```

## Optional MQTT TLS
Set `MQTT_TLS=true` and provide:
- `MQTT_CA_CERT`
- `MQTT_CLIENT_CERT`
- `MQTT_CLIENT_KEY`

## Systemd service
Recommended for servers. This repo includes a unit file at `systemd/meshsag.service`.

1. Create a service user:
   - `sudo useradd --system --no-create-home --shell /usr/sbin/nologin meshsag`
2. Install the repo and deps:
   - `sudo mkdir -p /opt/meshsag`
   - `sudo rsync -a ./ /opt/meshsag/`
   - `sudo python3 -m venv /opt/meshsag/.venv`
   - `sudo /opt/meshsag/.venv/bin/pip install -r /opt/meshsag/requirements.txt`
3. Place config:
   - `sudo mkdir -p /etc/meshsag`
   - `sudo cp /opt/meshsag/config.example.yaml /etc/meshsag/config.yaml`
   - `sudo chown -R meshsag:meshsag /etc/meshsag`
   - `sudo chmod 600 /etc/meshsag/config.yaml`
   - `sudo touch /var/log/meshsag.log`
   - `sudo chown meshsag:meshsag /var/log/meshsag.log`
   - `sudo chmod 640 /var/log/meshsag.log`
4. Install and enable the service:
   - `sudo cp /opt/meshsag/systemd/meshsag.service /etc/systemd/system/meshsag.service`
   - `sudo systemctl daemon-reload`
   - `sudo systemctl enable --now meshsag`
5. Logs:
   - `sudo journalctl -u meshsag -f`

## Updates (manual `git pull`)
Assuming the repo is installed at `/opt/meshsag` and running as the `meshsag` service.

1. Pull latest:
   - `cd /opt/meshsag`
   - `sudo -u meshsag git pull --ff-only`
2. Update Python deps (safe to run every time):
   - `sudo /opt/meshsag/.venv/bin/pip install -r /opt/meshsag/requirements.txt`
3. Restart the service:
   - `sudo systemctl restart meshsag`
4. Verify:
   - `sudo journalctl -u meshsag -n 100 --no-pager`
5. If `config.example.yaml` changed, review and merge updates into `/etc/meshsag/config.yaml`.

## Install/update scripts
If you prefer, use the bundled scripts (run as root):
- Install: `sudo bash scripts/install.sh`
- Update: `sudo bash scripts/update.sh`

You can override defaults via env vars:
- `INSTALL_DIR`, `CONFIG_DIR`, `SERVICE_NAME`, `SERVICE_USER`, `LOG_FILE`

## Log rotation
If you use the install script, it installs `logrotate/meshsag` to `/etc/logrotate.d/meshsag`.
It rotates `/var/log/meshsag.log` daily and keeps 14 compressed archives.
