#!/usr/bin/env bash
# Spec: local secret files must not be readable by other accounts after install.
# These tests fail while the install copy leaves operator credentials world-readable.
# They pass once that copy is removed or is not readable by other local accounts.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

failures=0
cleanup_paths=()
cleanup_users=()
cleanup_files=()

note() { printf '%s\n' "$*"; }

fail() {
  printf 'FAIL: %s\n' "$*"
  failures=$((failures + 1))
}

pass() { printf 'PASS: %s\n' "$*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'setup error: required command %s is not installed\n' "$1" >&2
    exit 2
  fi
}

new_placeholder() {
  local kind=$1
  local suffix
  suffix="$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')"
  printf 'placeholder-%s-%s' "$kind" "$suffix"
}

# Paths an unprivileged account can read that contain the exact secret value.
readable_secret_paths() {
  local dir=$1
  local secret=$2
  sudo -u nobody /bin/grep -R -I -F -l -- "$secret" "$dir" 2>/dev/null || true
}

assert_secrets_unreadable() {
  local label=$1
  local dir=$2
  shift 2
  local name secret hits hit
  while [[ $# -ge 2 ]]; do
    name=$1
    secret=$2
    shift 2
    hits="$(readable_secret_paths "$dir" "$secret")"
    if [[ -n "$hits" ]]; then
      fail "${label}: unprivileged account read ${name}"
      while IFS= read -r hit; do
        [[ -z "$hit" ]] && continue
        printf '  readable path: %s\n' "${hit#"${dir}/"}"
      done <<<"$hits"
    else
      pass "${label}: unprivileged account did not read ${name}"
    fi
  done
}

assert_tree_not_preexposed() {
  local label=$1
  local dir=$2
  shift 2
  local before=$failures
  assert_secrets_unreadable "$label" "$dir" "$@"
  if [[ "$failures" -ne "$before" ]]; then
    printf 'setup error: secret files were already readable before install\n' >&2
    exit 2
  fi
}

prepare_operator_tree() {
  local parent=$1
  mkdir -p "$parent/MeshSAG"
  chmod 700 "$parent"
  local src="$parent/MeshSAG"
  tar -C "$REPO_DIR" \
    --exclude .git \
    --exclude __pycache__ \
    --exclude .venv \
    -cf - . | tar -C "$src" -xf -
  chmod 755 "$src"
  umask 022
  cat >"$src/config.yaml" <<EOF
mqtt:
  host: localhost
  port: 1883
  username: placeholder-mqtt-user
  password: ${MQTT_PASSWORD}
meshmonitor:
  base_url: http://localhost:8080
  api_token: ${MESH_TOKEN}
dedupe:
  redis_password: ${REDIS_PASSWORD}
pager_phonebook_api:
  api_token: ${PHONEBOOK_TOKEN}
EOF
  printf '@apiToken = %s\n' "$HTTP_TOKEN" >"$src/API.http"
  # Operator-edited HTTP scratch file that ships in the tree.
  if [[ -f "$src/MeshMonotirRequests.http" ]]; then
    sed -i "s/^@apiToken = .*/@apiToken = ${HTTP_FILE_TOKEN}/" "$src/MeshMonotirRequests.http"
  else
    printf '@apiToken = %s\n' "$HTTP_FILE_TOKEN" >"$src/MeshMonotirRequests.http"
  fi
  chmod 644 "$src/config.yaml" "$src/API.http" "$src/MeshMonotirRequests.http"
  local cfg_mode src_mode
  cfg_mode="$(stat -c '%a' "$src/config.yaml")"
  src_mode="$(stat -c '%a' "$src")"
  if [[ "$cfg_mode" != "644" || "$src_mode" != "755" ]]; then
    printf 'setup error: expected config mode 644 and tree mode 755, got %s and %s\n' \
      "$cfg_mode" "$src_mode" >&2
    exit 2
  fi
  printf '%s' "$src"
}

track_tree() { cleanup_paths+=("$1"); }
track_user() { cleanup_users+=("$1"); }
track_file() { cleanup_files+=("$1"); }

cleanup() {
  local path user file
  for path in "${cleanup_paths[@]+"${cleanup_paths[@]}"}"; do
    case "$path" in
      /tmp/meshsag-exposure.*) rm -rf "$path" ;;
    esac
  done
  for file in "${cleanup_files[@]+"${cleanup_files[@]}"}"; do
    rm -f "$file"
  done
  for user in "${cleanup_users[@]+"${cleanup_users[@]}"}"; do
    userdel "$user" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

require_cmd rsync
require_cmd python3
require_cmd tar
require_cmd sudo
require_cmd od
if ! python3 -m venv /tmp/meshsag-exposure-venv-probe >/tmp/meshsag-exposure-venv-probe.log 2>&1; then
  printf 'setup error: python3 -m venv failed; install python3-venv\n' >&2
  cat /tmp/meshsag-exposure-venv-probe.log >&2 || true
  exit 2
fi
rm -rf /tmp/meshsag-exposure-venv-probe /tmp/meshsag-exposure-venv-probe.log
if ! id nobody >/dev/null 2>&1; then
  printf 'setup error: unprivileged account nobody does not exist\n' >&2
  exit 2
fi

MQTT_PASSWORD="$(new_placeholder mqtt)"
MESH_TOKEN="$(new_placeholder meshmonitor)"
REDIS_PASSWORD="$(new_placeholder redis)"
PHONEBOOK_TOKEN="$(new_placeholder phonebook)"
HTTP_TOKEN="$(new_placeholder http-scratch)"
HTTP_FILE_TOKEN="$(new_placeholder http-file)"
SECRETS=(
  "mqtt password" "$MQTT_PASSWORD"
  "meshmonitor token" "$MESH_TOKEN"
  "redis password" "$REDIS_PASSWORD"
  "phonebook token" "$PHONEBOOK_TOKEN"
  "gitignored http scratch token" "$HTTP_TOKEN"
  "http scratch token" "$HTTP_FILE_TOKEN"
)

# install.sh copies the tree with rsync -a, then chmod 600 only on the etc copy.
test_install_sh_copied_secrets_not_world_readable() {
  note "TEST test_install_sh_copied_secrets_not_world_readable"
  local work src inst cfg log service_user service_name
  work="$(mktemp -d /tmp/meshsag-exposure.XXXXXX)"
  # mktemp is mode 700. Other accounts must be able to traverse the install
  # parent, or a world-readable copy is hidden by the test harness.
  chmod 755 "$work"
  track_tree "$work"
  src="$(prepare_operator_tree "$work/src")"
  assert_tree_not_preexposed "install.sh source" "$work/src" "${SECRETS[@]}"

  inst="$work/opt-meshsag"
  cfg="$work/etc-meshsag"
  log="$work/meshsag.log"
  service_user="meshsaginst"
  service_name="meshsag-exposure-inst"
  track_user "$service_user"
  track_file "/etc/systemd/system/${service_name}.service"
  track_file "/etc/logrotate.d/${service_name}"

  local install_log install_rc
  install_log="$work/install.log"
  set +e
  env INSTALL_DIR="$inst" CONFIG_DIR="$cfg" SERVICE_NAME="$service_name" \
    SERVICE_USER="$service_user" LOG_FILE="$log" \
    bash "$src/scripts/install.sh" >"$install_log" 2>&1
  install_rc=$?
  set -e
  note "install.sh exited ${install_rc}"
  if ! grep -q "Syncing repo to ${inst}." "$install_log"; then
    fail "install.sh did not sync the tree"
    cat "$install_log"
    return
  fi
  if [[ ! -f "$inst/pocsag_forwarder.py" ]]; then
    fail "install.sh did not copy the forwarder into the install directory"
    return
  fi
  if [[ -f "$inst/config.yaml" ]]; then
    note "install copy config.yaml mode $(stat -c '%a %U:%G' "$inst/config.yaml")"
  else
    note "install copy config.yaml is absent"
  fi
  if [[ -f "$cfg/config.yaml" ]]; then
    note "etc config.yaml mode $(stat -c '%a %U:%G' "$cfg/config.yaml")"
  fi
  assert_secrets_unreadable "install.sh install dir" "$inst" "${SECRETS[@]}"
  assert_secrets_unreadable "install.sh config dir" "$cfg" "${SECRETS[@]}"
}

# README systemd section: rsync the tree, then chmod only the etc example copy.
test_readme_systemd_rsync_secrets_not_world_readable() {
  note "TEST test_readme_systemd_rsync_secrets_not_world_readable"
  local work src inst cfg log unit
  work="$(mktemp -d /tmp/meshsag-exposure.XXXXXX)"
  chmod 755 "$work"
  track_tree "$work"
  src="$(prepare_operator_tree "$work/src")"
  assert_tree_not_preexposed "readme source" "$work/src" "${SECRETS[@]}"

  inst="$work/opt-meshsag"
  cfg="$work/etc-meshsag"
  log="$work/meshsag.log"
  unit="$work/meshsag.service"
  track_user meshsag
  track_file /etc/systemd/system/meshsag.service
  track_file /etc/systemd/system/multi-user.target.wants/meshsag.service

  local -a commands=()
  local line cmd
  while IFS= read -r line; do
    commands+=("$line")
  done < <(
    awk '
      /^## Systemd service$/ { capture = 1; next }
      /^## / { capture = 0 }
      capture {
        rest = $0
        while (match(rest, /`[^`]+`/)) {
          cmd = substr(rest, RSTART + 1, RLENGTH - 2)
          # Inline paths such as systemd/meshsag.service are not commands.
          if (cmd ~ /^sudo[[:space:]]/) print cmd
          rest = substr(rest, RSTART + RLENGTH)
        }
      }
    ' "$REPO_DIR/README.md"
  )
  if [[ "${#commands[@]}" -eq 0 ]]; then
    fail "README systemd section has no install commands"
    return
  fi

  local saw_copy=0
  for cmd in "${commands[@]}"; do
    cmd="${cmd//\/opt\/meshsag/${inst}}"
    cmd="${cmd//\/etc\/meshsag/${cfg}}"
    cmd="${cmd//\/var\/log\/meshsag.log/${log}}"
    cmd="${cmd//\/etc\/systemd\/system\/meshsag.service/${unit}}"
    if [[ "$cmd" == *rsync* || "$cmd" == *"install.sh"* ]]; then
      saw_copy=1
    fi
    # journalctl -f follows forever and does not copy or chmod files.
    if [[ "$cmd" == *journalctl* && "$cmd" == *" -f"* ]]; then
      note "readme command skipped (does not exit): ${cmd}"
      continue
    fi
    note "readme command: ${cmd}"
    set +e
    (cd "$src" && bash -c "$cmd")
    local cmd_rc=$?
    set -e
    if [[ "$cmd_rc" -ne 0 ]]; then
      case "$cmd" in
        *systemctl*)
          note "readme systemctl exited ${cmd_rc}; continuing"
          ;;
        *)
          fail "documented install command exited ${cmd_rc}"
          ;;
      esac
    fi
  done
  if [[ "$saw_copy" -ne 1 ]]; then
    fail "README systemd install no longer copies the tree"
    return
  fi
  if [[ ! -f "$inst/pocsag_forwarder.py" ]]; then
    fail "README install did not copy the forwarder into the install directory"
    return
  fi
  if [[ -f "$inst/config.yaml" ]]; then
    note "readme install copy config.yaml mode $(stat -c '%a %U:%G' "$inst/config.yaml")"
  else
    note "readme install copy config.yaml is absent"
  fi
  assert_secrets_unreadable "readme install dir" "$inst" "${SECRETS[@]}"
  if [[ -d "$cfg" ]]; then
    assert_secrets_unreadable "readme config dir" "$cfg" "${SECRETS[@]}"
  fi
}

test_install_sh_copied_secrets_not_world_readable
test_readme_systemd_rsync_secrets_not_world_readable

if [[ "$failures" -ne 0 ]]; then
  note "RESULT: ${failures} failure(s)"
  exit 1
fi
note "RESULT: pass"
exit 0
