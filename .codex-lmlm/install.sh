#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
TOOL_NAME="codex"
VERSION="${VERSION:-1.0.0}"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# --- UI Formatting ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { printf "${BLUE}[INFO]${NC} %s\n" "$1"; }
log_success() { printf "${GREEN}[OK]${NC} %s\n" "$1"; }
log_warn()    { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
log_error()   { printf "${RED}[ERROR]${NC} %s\n" "$1" >&2; exit 1; }

# --- Clean Trap Handler ---
TMP_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT INT TERM

# --- Dependency Verification ---
need_cmd() {
    command -v "$1" >/dev/null 2>&1 || log_error "Required command '$1' is missing."
}

# --- Target Detection ---
detect_target() {
    local os arch
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"

    case "${os}" in
        linux)  os="linux" ;;
        darwin) os="darwin" ;;
        *)      log_error "Unsupported OS: ${os}" ;;
    esac

    case "${arch}" in
        x86_64|amd64)  arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *)             log_error "Unsupported architecture: ${arch}" ;;
    esac

    TARGET="${os}_${arch}"
}

# --- Idempotency Check ---
check_installed() {
    if command -v "${TOOL_NAME}" >/dev/null 2>&1; then
        local installed_version
        installed_version="$("${TOOL_NAME}" --version 2>&1 | awk '{print $NF}' || true)"
        if [ "${installed_version}" = "${VERSION}" ]; then
            log_success "${TOOL_NAME} version ${VERSION} is already installed."
            exit 0
        else
            log_warn "Upgrading ${TOOL_NAME} from ${installed_version} to ${VERSION}..."
        fi
    fi
}

# --- Main Installer ---
main() {
    need_cmd curl
    need_cmd tar

    detect_target
    check_installed

    log_info "Installing ${TOOL_NAME} v${VERSION} (${TARGET})..."

    local asset_url="https://github.com/example/${TOOL_NAME}/releases/download/v${VERSION}/${TOOL_NAME}_${TARGET}.tar.gz"
    local archive_path="${TMP_DIR}/${TOOL_NAME}.tar.gz"

    log_info "Downloading binary..."
    curl -fsSL "${asset_url}" -o "${archive_path}" || log_error "Download failed from ${asset_url}"

    log_info "Extracting payload..."
    tar -xzf "${archive_path}" -C "${TMP_DIR}"

    # Determine sudo requirements for installation target
    local sudo_cmd=""
    if [ ! -w "${INSTALL_DIR}" ]; then
        if [ "$(id -u)" -ne 0 ]; then
            log_warn "Write access to ${INSTALL_DIR} requires elevated privileges."
            need_cmd sudo
            sudo_cmd="sudo"
        fi
    fi

    ${sudo_cmd} mkdir -p "${INSTALL_DIR}"
    ${sudo_cmd} mv "${TMP_DIR}/${TOOL_NAME}" "${INSTALL_DIR}/${TOOL_NAME}"
    ${sudo_cmd} chmod 755 "${INSTALL_DIR}/${TOOL_NAME}"

    log_success "${TOOL_NAME} v${VERSION} successfully installed to ${INSTALL_DIR}/${TOOL_NAME}"

    # Verify target directory is inside system PATH
    case ":$PATH:" in
        *":${INSTALL_DIR}:"*) ;;
        *) log_warn "${INSTALL_DIR} is not in your current \$PATH environment variable." ;;
    esac
}

main "$@"
