#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.8.19"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PYENV_CMD=""
PYENV_ROOT_VALUE=""

echo "== pyenv setup =="
echo "Target Python version: ${PYTHON_VERSION}"
echo

# -----------------------------
# Helper: setup Python via pyenv
# -----------------------------
setup_python() {
    if [ -z "${PYENV_CMD:-}" ]; then
        echo "INTERNAL ERROR: PYENV_CMD not set."
        exit 1
    fi

    if [ -n "${PYENV_ROOT_VALUE:-}" ]; then
        export PYENV_ROOT="${PYENV_ROOT_VALUE}"
        export PATH="${PYENV_ROOT}/bin:${PATH}"
    fi

    echo "Using pyenv command: ${PYENV_CMD}"
    "${PYENV_CMD}" --version || true

    # Install Python if missing
    if ! "${PYENV_CMD}" versions --bare | grep -qx "${PYTHON_VERSION}"; then
        echo "Installing Python ${PYTHON_VERSION} via pyenv..."
        "${PYENV_CMD}" install "${PYTHON_VERSION}"
    else
        echo "Python ${PYTHON_VERSION} already installed in pyenv."
    fi

    echo "Setting local Python version for project..."
    ( cd "${PROJECT_DIR}" && "${PYENV_CMD}" local "${PYTHON_VERSION}" )

    echo
    echo "Done."
    echo ".python-version now contains: ${PYTHON_VERSION}"
}

# -----------------------------
# Step 1: detect existing pyenv
# -----------------------------
if command -v pyenv >/dev/null 2>&1; then
    echo "Found pyenv in PATH."
    PYENV_CMD="pyenv"
    # Let pyenv manage its own root
else
    if [ -x "${HOME}/.pyenv/bin/pyenv" ]; then
        echo "Found existing pyenv at ${HOME}/.pyenv/bin/pyenv."
        PYENV_CMD="${HOME}/.pyenv/bin/pyenv"
        PYENV_ROOT_VALUE="${HOME}/.pyenv"
    elif [ -x "${PROJECT_DIR}/.pyenv/bin/pyenv" ]; then
        echo "Found existing pyenv at ${PROJECT_DIR}/.pyenv/bin/pyenv."
        PYENV_CMD="${PROJECT_DIR}/.pyenv/bin/pyenv"
        PYENV_ROOT_VALUE="${PROJECT_DIR}/.pyenv"
    fi
fi

if [ -n "${PYENV_CMD:-}" ]; then
    echo "pyenv already installed; skipping installation steps."
    setup_python
    exit 0
fi

# -----------------------------
# Step 2: install pyenv if missing
# -----------------------------
if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required to install pyenv but was not found."
    exit 1
fi

echo "pyenv not found."

# Ask to install in ~/.pyenv
read -r -p "Install pyenv in \$HOME/.pyenv? [y/N] " ans
if [[ "${ans}" =~ ^[Yy] ]]; then
    if [ -d "${HOME}/.pyenv" ]; then
        echo "Directory ${HOME}/.pyenv already exists, assuming pyenv is there."
    else
        echo "Cloning pyenv into ${HOME}/.pyenv..."
        git clone https://github.com/pyenv/pyenv.git "${HOME}/.pyenv"
    fi
    PYENV_CMD="${HOME}/.pyenv/bin/pyenv"
    PYENV_ROOT_VALUE="${HOME}/.pyenv"

    # -----------------------------
    # Step 3: add ~/.pyenv to PATH
    # -----------------------------
    echo
    echo "pyenv installed at ${HOME}/.pyenv."
    read -r -p "Add pyenv at \$HOME/.pyenv to your PATH in your shell rc file? [y/N] " addpath

    if [[ "${addpath}" =~ ^[Yy] ]]; then
        # Pick an rc file (best-effort)
        if [ -n "${SHELL:-}" ] && [[ "${SHELL}" == *"bash" ]]; then
            RC_FILE="${HOME}/.bashrc"
        elif [ -n "${SHELL:-}" ] && [[ "${SHELL}" == *"zsh" ]]; then
            RC_FILE="${HOME}/.zshrc"
        else
            RC_FILE="${HOME}/.bashrc"
        fi
        # Append to rc file if not already present, with guards
        if ! grep -q "pyenv initialization (added by setup_pyenv_3_8.sh)" "${RC_FILE}"; then
            {
                echo ""
                echo "# pyenv initialization (added by setup_pyenv_3_8.sh)"
                echo 'export PYENV_ROOT="$HOME/.pyenv"'
                echo 'if ! echo "$PATH" | tr ":" "\n" | grep -qx "$PYENV_ROOT/bin"; then'
                echo '    PATH="$PYENV_ROOT/bin:$PATH"'
                echo 'fi'
                echo 'export PATH'
                echo 'eval "$(pyenv init -)"'
            } >> "${RC_FILE}"
            echo "Appended pyenv initialization to ${RC_FILE}."
        else
            echo "pyenv initialization snippet already present in ${RC_FILE}, not adding again."
        fi


        echo "Appended pyenv initialization to ${RC_FILE}."
        echo "It will take effect in new shells. For this script run, we'll set it manually."

        export PYENV_ROOT="${PYENV_ROOT_VALUE}"
        export PATH="${PYENV_ROOT}/bin:${PATH}"

        setup_python
        exit 0
    else
        echo "User chose not to modify PATH. Stopping here as requested."
        exit 0
    fi

else
    # Ask to install in repo-local ./.pyenv
    read -r -p "Install pyenv in this repository at ./.pyenv instead? [y/N] " ans_repo
    if [[ "${ans_repo}" =~ ^[Yy] ]]; then
        if [ -d "${PROJECT_DIR}/.pyenv" ]; then
            echo "Directory ${PROJECT_DIR}/.pyenv already exists, assuming pyenv is there."
        else
            echo "Cloning pyenv into ${PROJECT_DIR}/.pyenv..."
            git clone https://github.com/pyenv/pyenv.git "${PROJECT_DIR}/.pyenv"
        fi
        PYENV_CMD="${PROJECT_DIR}/.pyenv/bin/pyenv"
        PYENV_ROOT_VALUE="${PROJECT_DIR}/.pyenv"

        # For repo-local pyenv we do NOT touch user PATH
        setup_python
        exit 0
    else
        echo "No pyenv installation chosen. Aborting."
        exit 1
    fi
fi
