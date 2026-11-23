#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must run as root (use sudo)." >&2
  exit 1
fi

umask 022

INSTALL_DIR="${INSTALL_DIR:-/opt/meshsag}"
CONFIG_DIR="${CONFIG_DIR:-/etc/meshsag}"
SERVICE_NAME="${SERVICE_NAME:-meshsag}"
SERVICE_USER="${SERVICE_USER:-meshsag}"
req_copy=""

log() {
  echo "[meshsag-update] $*"
}

die() {
  echo "[meshsag-update] ERROR: $*" >&2
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
    read -r uid mode < <(stat -c '%u %a' -- "${dir}")
    if [[ "${uid}" -ne 0 ]]; then
      die "${dir} is not owned by root."
    fi
    if (( (8#${mode} & 8#022) && !(8#${mode} & 8#1000) )); then
      die "${dir} is writable by accounts other than root."
    fi
    [[ "${dir}" == "/" ]] && break
    dir="$(dirname -- "${dir}")"
  done
}

# Root runs pip from the virtualenv against requirements.txt, so both must be
# out of reach of every account except root. The requirements file is opened
# once, checked through that descriptor, and installed from a private copy
# read from the same descriptor.
install_requirements() {
  local venv="${INSTALL_DIR}/.venv" unsafe uid mode type req_fd
  read -r uid mode < <(stat -c '%u %a' -- "${INSTALL_DIR}")
  if [[ "${uid}" -ne 0 && "${uid}" -ne "${service_uid}" ]] || (( 8#${mode} & 8#022 )); then
    die "${INSTALL_DIR} can be modified by an account other than root or ${SERVICE_USER}; refusing to install dependencies as root."
  fi
  if [[ -L "${venv}" ]]; then
    die "${venv} is a symlink; refusing to install dependencies as root."
  fi
  unsafe="$(find_unsafe "${venv}")"
  if [[ -n "${unsafe}" ]]; then
    die "${unsafe} can be modified by an account other than root; refusing to install dependencies as root."
  fi

  exec {req_fd}< "${INSTALL_DIR}/requirements.txt"
  IFS='|' read -r uid mode type < <(stat -L -c '%u|%a|%F' -- "/dev/fd/${req_fd}")
  if [[ "${type}" != "regular file" || "${uid}" -ne 0 ]] || (( 8#${mode} & 8#6022 )); then
    exec {req_fd}<&-
    die "${INSTALL_DIR}/requirements.txt must be a root-owned file that no other account can write; refusing to install dependencies as root."
  fi
  req_copy="$(mktemp)"
  trap 'rm -f -- "${req_copy}"' EXIT
  cat <&"${req_fd}" > "${req_copy}"
  exec {req_fd}<&-

  "${venv}/bin/pip" install -r "${req_copy}"
}

INSTALL_DIR="$(realpath -e -- "${INSTALL_DIR}")"

log "Starting update."
log "INSTALL_DIR=${INSTALL_DIR}"
log "CONFIG_DIR=${CONFIG_DIR}"
log "SERVICE_NAME=${SERVICE_NAME}"
log "SERVICE_USER=${SERVICE_USER}"

if [[ ! -d "${INSTALL_DIR}/.git" ]]; then
  echo "No git repo found at ${INSTALL_DIR}. Aborting." >&2
  exit 1
fi

check_parent_dirs "${INSTALL_DIR}"
service_uid="$(id -u "${SERVICE_USER}")"
install_uid="$(stat -c '%u' -- "${INSTALL_DIR}")"

if [[ -z "$(find_unsafe "${INSTALL_DIR}")" ]]; then
  log "Pulling latest changes as root."
  git -c core.hooksPath=/dev/null -c core.fsmonitor=false -C "${INSTALL_DIR}" pull --ff-only
elif [[ "${install_uid}" -eq "${service_uid}" ]]; then
  log "Pulling latest changes as ${SERVICE_USER}."
  if command -v sudo >/dev/null 2>&1; then
    sudo -u "${SERVICE_USER}" git -C "${INSTALL_DIR}" pull --ff-only
  else
    su -s /bin/sh -c "git -C '${INSTALL_DIR}' pull --ff-only" "${SERVICE_USER}"
  fi
else
  die "$(find_unsafe "${INSTALL_DIR}") can be modified by an account other than root. Move ${INSTALL_DIR} aside and re-run scripts/install.sh."
fi

if [[ -x "${INSTALL_DIR}/.venv/bin/pip" ]]; then
  log "Updating Python dependencies."
  install_requirements
fi

if [[ -f "${CONFIG_DIR}/config.yaml" ]]; then
  phonebook_path="$(awk -F: '/^[[:space:]]*ric_phonebook_file:[[:space:]]*/ {print $2; exit}' "${CONFIG_DIR}/config.yaml" | sed 's/#.*$//' | xargs || true)"
  if [[ -n "${phonebook_path}" ]]; then
    if [[ "${phonebook_path}" != /* ]]; then
      phonebook_path="${CONFIG_DIR}/${phonebook_path}"
    fi
    if [[ ! -f "${phonebook_path}" ]]; then
      log "WARNING: ric_phonebook_file is set but missing: ${phonebook_path}"
    fi
  fi
fi

log "Restarting service ${SERVICE_NAME}."
systemctl restart "${SERVICE_NAME}"

log "Updated and restarted ${SERVICE_NAME}."
log "Check config changes in ${INSTALL_DIR}/config.example.yaml."
