#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must run as root (use sudo)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTALL_DIR="${INSTALL_DIR:-/opt/meshsag}"
CONFIG_DIR="${CONFIG_DIR:-/etc/meshsag}"
SERVICE_NAME="${SERVICE_NAME:-meshsag}"
SERVICE_USER="${SERVICE_USER:-meshsag}"
LOG_FILE="${LOG_FILE:-/var/log/meshsag.log}"

if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
fi

mkdir -p "${INSTALL_DIR}"
rsync -a "${ROOT_DIR}/" "${INSTALL_DIR}/"

python3 -m venv "${INSTALL_DIR}/.venv"
"${INSTALL_DIR}/.venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"

mkdir -p "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
  cp "${INSTALL_DIR}/config.example.yaml" "${CONFIG_DIR}/config.yaml"
fi
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${CONFIG_DIR}"
chmod 600 "${CONFIG_DIR}/config.yaml"

mkdir -p "$(dirname "${LOG_FILE}")"
touch "${LOG_FILE}"
chown "${SERVICE_USER}:${SERVICE_USER}" "${LOG_FILE}"
chmod 640 "${LOG_FILE}"

if [[ -f "${ROOT_DIR}/logrotate/meshsag" ]]; then
  cp "${ROOT_DIR}/logrotate/meshsag" "/etc/logrotate.d/${SERVICE_NAME}"
fi

cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=MeshSAG POCSAG forwarder
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${INSTALL_DIR}
Environment=CONFIG_PATH=${CONFIG_DIR}/config.yaml
ExecStart=${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/pocsag_forwarder.py
Restart=on-failure
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "Installed and started ${SERVICE_NAME}."
echo "Config: ${CONFIG_DIR}/config.yaml"
echo "Logs: journalctl -u ${SERVICE_NAME} -f"
