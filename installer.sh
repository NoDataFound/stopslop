#!/usr/bin/env bash
set -euo pipefail

choose_python() {
  for b in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$b" >/dev/null 2>&1; then
      echo "$b"
      return 0
    fi
  done
  echo "No suitable python3 found" >&2
  exit 1
}

PYBIN="$(choose_python)"
PYVER="$($PYBIN -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
VENV_DIR=".venv-cyberslop-py${PYVER}"

$PYBIN -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Installed into $VENV_DIR"
echo
echo "Next steps:"
echo "1) source $VENV_DIR/bin/activate"
echo "2) Copy .env.example to .env and set your API keys if you use CLI"
echo "3) For Streamlit set secrets in .streamlit/secrets.toml"
echo "4) Run: streamlit run app_streamlit.py"
echo "   or CLI: python cli.py --help"
