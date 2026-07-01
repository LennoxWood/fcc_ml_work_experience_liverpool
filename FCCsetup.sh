#!/bin/bash
# FCCsetup.sh
#
# FCC HH Analysis -- environment setup and JupyterLab launcher.
#
# Works on both Windows (via Git Bash, invoked by setup.bat) and Linux/macOS.
#
# USAGE (Windows):  double-click setup.bat  -- or --  open Git Bash and run:
#     bash FCCsetup.sh
#
# USAGE (Linux/macOS / cluster):
#     cd FCCHBBTAUTAU
#     source FCCsetup.sh
#
# First run:  installs miniforge conda locally and creates the FCCHH
#             environment from environment.yml. Takes ~10-15 minutes.
# Later runs: skips straight to activation and launches JupyterLab.

# ---------------------------------------------------------------------------
# Locate the repo root regardless of how/where this script was invoked
# ---------------------------------------------------------------------------
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(dirname "$SCRIPT_PATH")"

echo ""
echo "============================================================"
echo "  FCC HH Analysis -- Setup"
echo "  Project directory: ${PROJECT_DIR}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Detect operating system (for the miniforge installer filename)
# ---------------------------------------------------------------------------
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    OS="windows"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
else
    OS="linux"
fi

# ---------------------------------------------------------------------------
# Step 1: Install miniforge (conda + mamba) if not already present
# ---------------------------------------------------------------------------
CONDA_INSTALL="${PROJECT_DIR}/conda"

if [[ ! -d "${CONDA_INSTALL}" ]]; then
    echo ">>> conda not found -- installing miniforge to ${CONDA_INSTALL}..."
    echo "    This will take 2-3 minutes."
    echo ""

    REPO="https://github.com/conda-forge/miniforge/releases/latest/download"

    if [[ "$OS" == "windows" ]]; then
        INSTALLER="Miniforge3-Windows-x86_64.exe"
        curl -L -O "${REPO}/${INSTALLER}"
        ./"${INSTALLER}" /InstallationType=JustMe /RegisterPython=0 \
            /S /D="$(cygpath -w "${CONDA_INSTALL}")"
        rm "${INSTALLER}"
    elif [[ "$OS" == "mac" ]]; then
        ARCH="$(uname -m)"
        if [[ "$ARCH" == "arm64" ]]; then
            INSTALLER="Miniforge3-MacOSX-arm64.sh"
        else
            INSTALLER="Miniforge3-MacOSX-x86_64.sh"
        fi
        curl -L -O "${REPO}/${INSTALLER}"
        bash "${INSTALLER}" -b -p "${CONDA_INSTALL}"
        rm "${INSTALLER}"
    else
        INSTALLER="Miniforge3-Linux-x86_64.sh"
        curl -L -O "${REPO}/${INSTALLER}"
        bash "${INSTALLER}" -b -p "${CONDA_INSTALL}"
        rm "${INSTALLER}"
    fi

    if [[ $? -ne 0 ]]; then
        echo ""
        echo "ERROR: miniforge installation failed."
        echo "       Check your internet connection and try again."
        exit 1
    fi
    echo ""
    echo ">>> miniforge installed successfully."
else
    echo ">>> conda already installed."
fi

# Activate the base conda environment
source "${CONDA_INSTALL}/bin/activate"

# ---------------------------------------------------------------------------
# Step 2: Create the FCCHH environment if it doesn't exist yet
# ---------------------------------------------------------------------------
ENV_NAME="FCCHH"
ENV_YML="${PROJECT_DIR}/environment.yml"

if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo ">>> ${ENV_NAME} environment already exists -- skipping creation."
else
    echo ""
    echo ">>> Creating ${ENV_NAME} environment from environment.yml..."
    echo "    This will take around 10-15 minutes on first run."
    echo "    You'll see package names scrolling past -- that's normal."
    echo ""

    if [[ ! -f "${ENV_YML}" ]]; then
        echo "ERROR: environment.yml not found at ${ENV_YML}"
        echo "       Make sure you've cloned the full repository."
        exit 1
    fi

    mamba env create -f "${ENV_YML}"

    if [[ $? -ne 0 ]]; then
        echo ""
        echo "ERROR: environment creation failed."
        echo "       See the error messages above for details."
        echo "       Common fixes:"
        echo "         - Check your internet connection"
        echo "         - Try: mamba env create -f environment.yml"
        exit 1
    fi

    echo ""
    echo ">>> ${ENV_NAME} environment created successfully."
fi

# ---------------------------------------------------------------------------
# Step 3: Activate the FCCHH environment
# ---------------------------------------------------------------------------
conda activate "${ENV_NAME}"

if [[ $? -ne 0 ]]; then
    echo "ERROR: could not activate ${ENV_NAME} environment."
    exit 1
fi

# Make the sparticles package importable from notebooks
export PYTHONPATH="${PROJECT_DIR}/python/:$PYTHONPATH"

# ---------------------------------------------------------------------------
# Step 4: Launch JupyterLab
# ---------------------------------------------------------------------------
echo ""
echo ">>> Launching JupyterLab..."
echo ""
echo "    JupyterLab will open automatically in your browser."
echo "    If it doesn't, copy the URL shown below and paste it"
echo "    into your browser manually."
echo ""
echo "    To stop JupyterLab, press Ctrl+C in this window."
echo ""

# Auto-detect whether we're running locally or over SSH
if [[ "$OS" == "linux" ]] && [[ -n "$SSH_CLIENT" || -n "$SSH_TTY" ]]; then
    # Remote/cluster -- don't try to open a browser
    jupyter lab --no-browser --port 8888
else
    # Local -- open browser automatically
    jupyter lab --port 8888
fi
