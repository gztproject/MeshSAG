#!/usr/bin/env bash
# Spec for the service-install ownership boundary.
#
# These checks fail while the documented root install keeps the checkout
# owner's write access to the program and requirements that root and the
# service account later execute. They pass once that write access is gone
# and a root package install refuses a requirements file another account
# can still change.
#
# Run from a machine with rsync, python3-venv, and sudo:
#   sudo bash tests/test_install_ownership.sh

set -u

if [[ "$(id -u)" -ne 0 ]]; then
  exec sudo -n bash "$0" "$@"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILURES=0
BASE=""
CHECKOUT=""
SERVICE_USER=""
SERVICE_NAME=""

fail() {
  echo "FAIL: $*" >&2
  FAILURES=$((FAILURES + 1))
}

cleanup() {
  set +e
  if [[ -n "${SERVICE_NAME}" ]]; then
    rm -f "/etc/systemd/system/${SERVICE_NAME}.service" \
      "/etc/logrotate.d/${SERVICE_NAME}"
  fi
  if [[ -n "${CHECKOUT}" ]]; then
    userdel "${CHECKOUT}" 2>/dev/null
  fi
  if [[ -n "${SERVICE_USER}" ]]; then
    userdel "${SERVICE_USER}" 2>/dev/null
  fi
  if [[ -n "${BASE}" && -d "${BASE}" ]]; then
    rm -rf "${BASE}"
  fi
}
trap cleanup EXIT

unit_field() {
  local key="$1" file="$2"
  awk -F= -v key="${key}" '$1 == key {print substr($0, index($0, "=") + 1); exit}' "${file}"
}

check_unit_hardening() {
  local file="$1" log_path="$2" label="$3"
  if [[ ! -f "${file}" ]]; then
    fail "${label} unit file is missing (${file})"
    return
  fi
  if ! grep -Eq '^ProtectSystem=strict([[:space:]]*#.*)?$' "${file}"; then
    fail "${label} does not set ProtectSystem=strict"
  fi
  local paths found=0
  paths="$(awk -F= '$1 == "ReadWritePaths" {print substr($0, index($0, "=") + 1)}' "${file}")"
  if [[ -z "${paths}" ]]; then
    fail "${label} does not grant ReadWritePaths for the log file only"
    return
  fi
  local line entry
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    for entry in ${line}; do
      found=1
      entry="${entry#-}"
      if [[ "${entry}" != "${log_path}" ]]; then
        fail "${label} ReadWritePaths includes ${entry}, not only ${log_path}"
      fi
    done
  done <<< "${paths}"
  if [[ "${found}" -eq 0 ]]; then
    fail "${label} does not grant ReadWritePaths for the log file only"
  fi
}

write_payload() {
  local payload="$1" result="$2"
  cat > "${payload}" <<EOF
import os
from pathlib import Path
marker = "placeholder-not-a-secret"
text = Path(os.environ["CONFIG_PATH"]).read_text(encoding="utf-8")
Path(${result@Q}).write_text("uid=%s\\nfound=%s\\n" % (os.getuid(), marker in text))
EOF
}

write_evil_package() {
  local pkg="$1" marker="$2"
  mkdir -p "${pkg}"
  cat > "${pkg}/pyproject.toml" <<'EOF'
[build-system]
requires = []
build-backend = "attacker_backend"
backend-path = ["."]

[project]
name = "attacker-pkg"
version = "0.0.1"
EOF
  cat > "${pkg}/attacker_backend.py" <<EOF
import os
from pathlib import Path
Path(${marker@Q}).write_text("uid=%s\\n" % os.getuid())

def get_requires_for_build_wheel(config_settings=None):
    return []

def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    raise RuntimeError("stop after marker")

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    raise RuntimeError("stop after marker")
EOF
  chown -R "${CHECKOUT}:${CHECKOUT}" "${pkg}"
}

assert_no_root_execution() {
  local marker="$1" label="$2"
  if [[ -f "${marker}" ]] && grep -q '^uid=0$' "${marker}"; then
    fail "${label} executed the checkout user's package as root"
  fi
}

# --- fixtures --------------------------------------------------------------

BASE="$(mktemp -d /tmp/meshsag-ownership.XXXXXX)"
chmod 755 "${BASE}"
CHECKOUT="msco${$}"
SERVICE_USER="mssv${$}"
SERVICE_NAME="msag${$}"
SRC="${BASE}/src"
INSTALL="${BASE}/opt"
CONFIG="${BASE}/etc"
LOG="${BASE}/log/meshsag.log"
RESULT_DIR="${BASE}/results"
mkdir -p "${SRC}" "${CONFIG}" "${RESULT_DIR}" "$(dirname "${LOG}")"
chmod 1777 "${RESULT_DIR}"

useradd --create-home --home-dir "${BASE}/home" --shell /bin/bash "${CHECKOUT}"
useradd --system --no-create-home --shell /usr/sbin/nologin "${SERVICE_USER}"
chmod 755 "${BASE}/home"

cp -a "${REPO_ROOT}/." "${SRC}/"
chown -R "${CHECKOUT}:${CHECKOUT}" "${SRC}" "${BASE}/home"

cat > "${CONFIG}/config.yaml" <<EOF
mqtt:
  host: 127.0.0.1
  port: 1883
  topic: owrx/POCSAG
  username: placeholder-user
  password: placeholder-not-a-secret
meshmonitor:
  base_url: http://127.0.0.1:9
  api_token: placeholder-not-a-secret
dedupe:
  redis_password: placeholder-not-a-secret
runtime:
  log_level: INFO
  log_file: ${LOG}
EOF

echo "=== install.sh from a checkout owned by ${CHECKOUT} ==="
set +e
env INSTALL_DIR="${INSTALL}" CONFIG_DIR="${CONFIG}" SERVICE_NAME="${SERVICE_NAME}" \
  SERVICE_USER="${SERVICE_USER}" LOG_FILE="${LOG}" \
  bash "${SRC}/scripts/install.sh" > "${BASE}/install.log" 2>&1
install_rc=$?
set -e
echo "install.sh exited ${install_rc} (system manager activation may fail in this environment)"

if [[ ! -f "${INSTALL}/pocsag_forwarder.py" || ! -f "${INSTALL}/requirements.txt" ]]; then
  echo "install.sh did not copy the program. Log:" >&2
  cat "${BASE}/install.log" >&2
  exit 1
fi

UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
if [[ ! -f "${UNIT}" ]]; then
  echo "install.sh did not write a unit. Log:" >&2
  cat "${BASE}/install.log" >&2
  exit 1
fi

echo "=== installed tree is not writable by the checkout account ==="
mapfile -t writable < <(sudo -u "${CHECKOUT}" find "${INSTALL}" -writable 2>/dev/null || true)
if [[ "${#writable[@]}" -gt 0 ]]; then
  fail "checkout account can write installed paths, including ${writable[0]}"
fi

forwarder="${INSTALL}/pocsag_forwarder.py"
requirements="${INSTALL}/requirements.txt"
if [[ "$(stat -c '%U' "${forwarder}")" != "root" ]]; then
  fail "pocsag_forwarder.py is owned by $(stat -c '%U:%G %a' "${forwarder}") after install"
fi
if [[ "$(stat -c '%U' "${requirements}")" != "root" ]]; then
  fail "requirements.txt is owned by $(stat -c '%U:%G %a' "${requirements}") after install"
fi
if [[ "$(stat -c '%U' "${INSTALL}")" != "root" ]]; then
  fail "install directory is owned by $(stat -c '%U:%G %a' "${INSTALL}") after install"
fi

echo "=== config stays readable by the service account only ==="
if sudo -u "${CHECKOUT}" python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).read_text()' "${CONFIG}/config.yaml" 2>/dev/null; then
  fail "checkout account can read the service config"
fi
if ! sudo -u "${SERVICE_USER}" python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).read_text()' "${CONFIG}/config.yaml" 2>/dev/null; then
  fail "service account cannot read the config the unit points at"
fi

echo "=== unit executes that program as the service account ==="
check_unit_hardening "${UNIT}" "${LOG}" "generated unit"
check_unit_hardening "${REPO_ROOT}/systemd/meshsag.service" "/var/log/meshsag.log" "systemd/meshsag.service"

unit_user="$(unit_field User "${UNIT}")"
unit_workdir="$(unit_field WorkingDirectory "${UNIT}")"
unit_exec="$(unit_field ExecStart "${UNIT}")"
unit_env="$(unit_field Environment "${UNIT}")"
if [[ "${unit_user}" != "${SERVICE_USER}" ]]; then
  fail "generated unit User=${unit_user}, expected the service account"
fi
if [[ "${unit_exec}" != *"${forwarder}"* ]]; then
  fail "generated unit ExecStart does not run ${forwarder}"
fi

payload="${BASE}/payload.py"
result="${RESULT_DIR}/service-read.txt"
write_payload "${payload}" "${result}"
chown "${CHECKOUT}:${CHECKOUT}" "${payload}"
replaced=0
if sudo -u "${CHECKOUT}" cp "${payload}" "${forwarder}" 2>/dev/null; then
  replaced=1
elif sudo -u "${CHECKOUT}" rm -f "${forwarder}" 2>/dev/null && sudo -u "${CHECKOUT}" cp "${payload}" "${forwarder}" 2>/dev/null; then
  replaced=1
fi
if [[ "${replaced}" -eq 1 ]]; then
  fail "checkout account replaced the installed forwarder after install"
  rm -f "${result}"
  set +e
  # Same steps systemd-executor takes for this unit: enter WorkingDirectory,
  # apply Environment=, switch to User=, and exec ExecStart. systemd is not
  # PID 1 here, so the uid switch is performed with sudo -u.
  timeout 20 sudo -u "${unit_user}" env "${unit_env}" \
    bash -c 'cd "$1" && shift && exec "$@"' _ "${unit_workdir}" ${unit_exec} \
    > "${BASE}/service.out" 2> "${BASE}/service.err"
  set -e
  if [[ -f "${result}" ]] && grep -q '^found=True$' "${result}"; then
    service_uid="$(id -u "${SERVICE_USER}")"
    if grep -q "^uid=${service_uid}$" "${result}"; then
      fail "service account executed the replaced forwarder and read the config"
    fi
  fi
fi

echo "=== root pip of a requirements file the checkout account can change ==="
pkg="${BASE}/home/evilpkg"
marker="${RESULT_DIR}/pip-uid.txt"
write_evil_package "${pkg}" "${marker}"
poisoned=0
if sudo -u "${CHECKOUT}" bash -c 'printf "%s\n" "$1" > "$2"' _ "${pkg}" "${requirements}" 2>/dev/null; then
  poisoned=1
elif sudo -u "${CHECKOUT}" rm -f "${requirements}" 2>/dev/null && sudo -u "${CHECKOUT}" bash -c 'printf "%s\n" "$1" > "$2"' _ "${pkg}" "${requirements}" 2>/dev/null; then
  poisoned=1
fi
if [[ "${poisoned}" -eq 1 ]]; then
  fail "checkout account replaced requirements.txt after install"
  set +e
  "${INSTALL}/.venv/bin/pip" install -r "${requirements}" > "${BASE}/pip.log" 2>&1
  set -e
  assert_no_root_execution "${marker}" "documented root pip install"
fi

echo "=== update.sh on the tree install.sh just produced ==="
installed_marker="${RESULT_DIR}/update-installed-uid.txt"
rm -f "${installed_marker}"
# Point the already-copied package at this marker and poison again if possible.
write_evil_package "${pkg}" "${installed_marker}"
sudo -u "${CHECKOUT}" bash -c 'printf "%s\n" "$1" > "$2"' _ "${pkg}" "${requirements}" 2>/dev/null || true
set +e
env INSTALL_DIR="${INSTALL}" CONFIG_DIR="${CONFIG}" SERVICE_NAME="${SERVICE_NAME}" \
  SERVICE_USER="${SERVICE_USER}" \
  bash "${SRC}/scripts/update.sh" > "${BASE}/update-installed.log" 2>&1
set -e
assert_no_root_execution "${installed_marker}" "update.sh on the installed tree"

prepare_service_repo() {
  local tree="$1" origin="$2"
  mkdir -p "${tree}"
  cp -a "${SRC}/." "${tree}/"
  chown -R root:root "${tree}"
  git -C "${tree}" config --unset-all core.hooksPath >/dev/null 2>&1 || true
  chown -R "${SERVICE_USER}:${SERVICE_USER}" "${tree}"
  sudo -u "${SERVICE_USER}" git -C "${tree}" checkout -B main >/dev/null 2>&1
  sudo -u "${SERVICE_USER}" git -C "${tree}" remote remove origin >/dev/null 2>&1 || true
  git init --bare "${origin}" >/dev/null 2>&1
  chown -R "${SERVICE_USER}:${SERVICE_USER}" "${origin}"
  sudo -u "${SERVICE_USER}" git -C "${tree}" remote add origin "${origin}"
  sudo -u "${SERVICE_USER}" git -C "${tree}" push -u origin refs/heads/main:refs/heads/main >/dev/null 2>&1
  python3 -m venv "${tree}/.venv"
}

assert_update_refuses_attacker_requirements() {
  local label="$1" tree="$2" origin="$3" mode="$4"
  prepare_service_repo "${tree}" "${origin}"
  local req="${tree}/requirements.txt"
  local pkg="${tree}-pkg"
  local marker="${tree}-uid.txt"
  write_evil_package "${pkg}" "${marker}"
  if [[ "${mode}" == "owned-by-checkout" ]]; then
    chown "${CHECKOUT}:${CHECKOUT}" "${req}"
    chmod 644 "${req}"
  elif [[ "${mode}" == "root-but-world-writable" ]]; then
    chown root:root "${req}"
    chmod 666 "${req}"
  else
    fail "unknown requirements mode ${mode}"
    return
  fi
  if ! sudo -u "${CHECKOUT}" bash -c 'printf "%s\n" "$1" > "$2"' _ "${pkg}" "${req}"; then
    fail "${label} fixture: checkout account could not write requirements.txt"
    return
  fi
  if ! sudo -u "${SERVICE_USER}" git -C "${tree}" pull --ff-only > "${tree}-pull.log" 2>&1; then
    fail "${label} fixture: service account could not pull an up-to-date repo"
    cat "${tree}-pull.log" >&2
    return
  fi
  # Re-assert the attacker's content immediately before the product runs,
  # which is the moment a check inside update.sh would have to still be true.
  if ! sudo -u "${CHECKOUT}" bash -c 'printf "%s\n" "$1" > "$2"' _ "${pkg}" "${req}"; then
    fail "${label} fixture: requirements.txt stopped being writable before update.sh"
    return
  fi
  set +e
  env INSTALL_DIR="${tree}" CONFIG_DIR="${CONFIG}" SERVICE_NAME="${SERVICE_NAME}" \
    SERVICE_USER="${SERVICE_USER}" \
    bash "${REPO_ROOT}/scripts/update.sh" > "${tree}-update.log" 2>&1
  set -e
  assert_no_root_execution "${marker}" "${label}"
}

echo "=== update.sh refuses a requirements file the checkout account owns ==="
assert_update_refuses_attacker_requirements \
  "update.sh with checkout-owned requirements.txt" \
  "${BASE}/upd-owned" "${BASE}/upd-owned.git" "owned-by-checkout"

echo "=== update.sh refuses a root-owned requirements file others can write ==="
assert_update_refuses_attacker_requirements \
  "update.sh with root-owned mode 666 requirements.txt" \
  "${BASE}/upd-open" "${BASE}/upd-open.git" "root-but-world-writable"

echo "=== update.sh still installs a root-owned, non-writable requirements file ==="
safe_tree="${BASE}/upd-safe"
prepare_service_repo "${safe_tree}" "${BASE}/upd-safe.git"
chown root:root "${safe_tree}/requirements.txt"
chmod 644 "${safe_tree}/requirements.txt"
if sudo -u "${CHECKOUT}" bash -c 'printf x >> "$1"' _ "${safe_tree}/requirements.txt" 2>/dev/null; then
  fail "fixture: checkout account could write a root-owned mode 644 requirements file"
elif ! sudo -u "${SERVICE_USER}" git -C "${safe_tree}" pull --ff-only > "${safe_tree}-pull.log" 2>&1; then
  fail "fixture: service account could not pull the safe repo"
  cat "${safe_tree}-pull.log" >&2
else
  set +e
  env INSTALL_DIR="${safe_tree}" CONFIG_DIR="${CONFIG}" SERVICE_NAME="${SERVICE_NAME}" \
    SERVICE_USER="${SERVICE_USER}" \
    bash "${REPO_ROOT}/scripts/update.sh" > "${safe_tree}-update.log" 2>&1
  set -e
  if ! grep -q "Updating Python dependencies" "${safe_tree}-update.log"; then
    fail "update.sh refused a root-owned mode 644 requirements file"
  elif ! grep -Eq "Successfully installed|Requirement already satisfied" "${safe_tree}-update.log"; then
    fail "update.sh did not install the root-owned requirements file"
    cat "${safe_tree}-update.log" >&2
  fi
fi

echo "=== documented rsync install does not leave the tree writable ==="
DOC_INSTALL="${BASE}/doc-opt"
DOC_SRC="${SRC}"
ran_rsync=0
in_section=0
while IFS= read -r line; do
  if [[ "${line}" == "## Systemd service" ]]; then
    in_section=1
    continue
  fi
  if [[ "${in_section}" -eq 1 && "${line}" == "## "* ]]; then
    break
  fi
  [[ "${in_section}" -eq 1 ]] || continue
  if [[ "${line}" =~ \`([^\`]+)\` ]]; then
    cmd="${BASH_REMATCH[1]}"
  else
    continue
  fi
  [[ "${cmd}" == *"/opt/meshsag"* ]] || continue
  case "${cmd}" in
    sudo\ mkdir*|sudo\ rsync*|sudo\ chown*|sudo\ chmod*)
      ;;
    *)
      continue
      ;;
  esac
  cmd="${cmd#sudo }"
  cmd="${cmd//\/opt\/meshsag/${DOC_INSTALL}}"
  if [[ "${cmd}" == rsync* ]]; then
    ran_rsync=1
  fi
  (
    cd "${DOC_SRC}"
    bash -c "${cmd}"
  ) > "${BASE}/readme-cmd.log" 2>&1 || fail "documented command failed: ${cmd}"
done < "${REPO_ROOT}/README.md"

if [[ "${ran_rsync}" -eq 0 ]]; then
  if [[ -d "${DOC_INSTALL}" ]]; then
    fail "documented install changed ${DOC_INSTALL} without an rsync the test could see"
  fi
else
  if [[ ! -f "${DOC_INSTALL}/pocsag_forwarder.py" ]]; then
    fail "documented rsync did not install pocsag_forwarder.py"
  else
    mapfile -t doc_writable < <(sudo -u "${CHECKOUT}" find "${DOC_INSTALL}" -writable 2>/dev/null || true)
    if [[ "${#doc_writable[@]}" -gt 0 ]]; then
      fail "documented install leaves ${doc_writable[0]} writable by the checkout account"
    fi
  fi
fi

echo "=== results ==="
if [[ "${FAILURES}" -ne 0 ]]; then
  echo "${FAILURES} check(s) failed" >&2
  exit 1
fi
echo "all ownership checks passed"
exit 0
