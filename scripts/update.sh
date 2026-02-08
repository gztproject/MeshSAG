#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must run as root (use sudo)." >&2
  exit 1
fi

INSTALL_DIR="${INSTALL_DIR:-/opt/meshsag}"
SERVICE_NAME="${SERVICE_NAME:-meshsag}"
SERVICE_USER="${SERVICE_USER:-meshsag}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  echo "No git repo found at ${INSTALL_DIR}. Aborting." >&2
  exit 1
fi

if command -v sudo >/dev/null 2>&1; then
  sudo -u "${SERVICE_USER}" git -C "${INSTALL_DIR}" pull --ff-only
else
  su -s /bin/sh -c "git -C '${INSTALL_DIR}' pull --ff-only" "${SERVICE_USER}"
fi

if [[ -x "${INSTALL_DIR}/.venv/bin/pip" ]]; then
  "${INSTALL_DIR}/.venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"
fi

systemctl restart "${SERVICE_NAME}"

echo "Updated and restarted ${SERVICE_NAME}."
echo "Check config changes in ${INSTALL_DIR}/config.example.yaml."
