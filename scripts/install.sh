#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must run as root (use sudo)." >&2
  exit 1
fi

umask 022

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

INSTALL_DIR="${INSTALL_DIR:-/opt/meshsag}"
CONFIG_DIR="${CONFIG_DIR:-/etc/meshsag}"
SERVICE_NAME="${SERVICE_NAME:-meshsag}"
SERVICE_USER="${SERVICE_USER:-meshsag}"
LOG_FILE="${LOG_FILE:-/var/log/meshsag.log}"

log() {
  echo "[meshsag-install] $*"
}

die() {
  echo "[meshsag-install] ERROR: $*" >&2
  exit 1
}

# Prints the first path under $1 that an account other than root could
# modify, or that is setuid/setgid.
find_unsafe() {
  find "$1" \( ! -user root -o \( ! -type l -perm /6022 \) \) -print -quit
}

# Fails unless every directory above $1 is owned by root and is either not
# writable by group/other or sticky, so no other account can swap $1.
check_parent_dirs() {
  local dir uid mode
  dir="$(dirname -- "$1")"
  while :; do
    if [[ -e "${dir}" ]]; then
      read -r uid mode < <(stat -c '%u %a' -- "${dir}")
      if [[ "${uid}" -ne 0 ]]; then
        die "${dir} is not owned by root."
      fi
      if (( (8#${mode} & 8#022) && !(8#${mode} & 8#1000) )); then
        die "${dir} is writable by accounts other than root."
      fi
    fi
    [[ "${dir}" == "/" ]] && break
    dir="$(dirname -- "${dir}")"
  done
}

INSTALL_DIR="$(realpath -m -- "${INSTALL_DIR}")"

log "Starting install."
log "INSTALL_DIR=${INSTALL_DIR}"
log "CONFIG_DIR=${CONFIG_DIR}"
log "SERVICE_NAME=${SERVICE_NAME}"
log "SERVICE_USER=${SERVICE_USER}"
log "LOG_FILE=${LOG_FILE}"

if ! id -u "${SERVICE_USER}" >/dev/null 2>&1; then
  log "Creating service user ${SERVICE_USER}."
  useradd --system --no-create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
else
  log "Service user ${SERVICE_USER} exists."
fi

log "Syncing repo to ${INSTALL_DIR}."
check_parent_dirs "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
unsafe="$(find_unsafe "${INSTALL_DIR}")"
if [[ -n "${unsafe}" ]]; then
  die "${unsafe} can be modified by an account other than root. Move ${INSTALL_DIR} aside and re-run the install."
fi
# The service runs this tree and root installs its requirements, so it must
# be owned by root and writable by no one else, whoever owns the checkout.
rsync -a --safe-links --exclude=/.venv --chown=root:root --chmod=go-w,ug-s \
  "${ROOT_DIR}/" "${INSTALL_DIR}/"
unsafe="$(find_unsafe "${INSTALL_DIR}")"
if [[ -n "${unsafe}" ]]; then
  die "${unsafe} can be modified by an account other than root after sync."
fi

log "Setting up virtualenv."
python3 -m venv "${INSTALL_DIR}/.venv"
"${INSTALL_DIR}/.venv/bin/pip" --version >/dev/null 2>&1 || true
log "Installing Python dependencies."
"${INSTALL_DIR}/.venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"

log "Preparing config in ${CONFIG_DIR}."
mkdir -p "${CONFIG_DIR}"
if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
  log "Config not found, copying example."
  cp "${INSTALL_DIR}/config.example.yaml" "${CONFIG_DIR}/config.yaml"
else
  log "Config exists, leaving in place."
fi

if [[ -f "${INSTALL_DIR}/ric_phonebook.example.yaml" && ! -f "${CONFIG_DIR}/ric_phonebook.yaml" ]]; then
  log "Phonebook not found, copying example."
  cp "${INSTALL_DIR}/ric_phonebook.example.yaml" "${CONFIG_DIR}/ric_phonebook.yaml"
fi
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${CONFIG_DIR}"
chmod 600 "${CONFIG_DIR}/config.yaml"
if [[ -f "${CONFIG_DIR}/ric_phonebook.yaml" ]]; then
  chmod 640 "${CONFIG_DIR}/ric_phonebook.yaml"
fi

log "Setting up log file ${LOG_FILE}."
mkdir -p "$(dirname "${LOG_FILE}")"
touch "${LOG_FILE}"
chown "${SERVICE_USER}:${SERVICE_USER}" "${LOG_FILE}"
chmod 640 "${LOG_FILE}"

if [[ -f "${INSTALL_DIR}/logrotate/meshsag" ]]; then
  log "Installing logrotate config."
  install -m 0644 -o root -g root "${INSTALL_DIR}/logrotate/meshsag" "/etc/logrotate.d/${SERVICE_NAME}"
fi

log "Writing systemd unit."
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
ProtectSystem=strict
ReadWritePaths=${LOG_FILE}

[Install]
WantedBy=multi-user.target
EOF

log "Reloading systemd and starting service."
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

log "Installed and started ${SERVICE_NAME}."
log "Config: ${CONFIG_DIR}/config.yaml"
log "Logs: journalctl -u ${SERVICE_NAME} -f"
