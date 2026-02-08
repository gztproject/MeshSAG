#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must run as root (use sudo)." >&2
  exit 1
fi

INSTALL_DIR="${INSTALL_DIR:-/opt/meshsag}"
SERVICE_NAME="${SERVICE_NAME:-meshsag}"
SERVICE_USER="${SERVICE_USER:-meshsag}"

log() {
  echo "[meshsag-update] $*"
}

log "Starting update."
log "INSTALL_DIR=${INSTALL_DIR}"
log "SERVICE_NAME=${SERVICE_NAME}"
log "SERVICE_USER=${SERVICE_USER}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  echo "No git repo found at ${INSTALL_DIR}. Aborting." >&2
  exit 1
fi

log "Pulling latest changes."
if command -v sudo >/dev/null 2>&1; then
  sudo -u "${SERVICE_USER}" git -C "${INSTALL_DIR}" pull --ff-only
else
  su -s /bin/sh -c "git -C '${INSTALL_DIR}' pull --ff-only" "${SERVICE_USER}"
fi

if [[ -x "${INSTALL_DIR}/.venv/bin/pip" ]]; then
  log "Updating Python dependencies."
  "${INSTALL_DIR}/.venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"
fi

log "Restarting service ${SERVICE_NAME}."
systemctl restart "${SERVICE_NAME}"

log "Updated and restarted ${SERVICE_NAME}."
log "Check config changes in ${INSTALL_DIR}/config.example.yaml."
