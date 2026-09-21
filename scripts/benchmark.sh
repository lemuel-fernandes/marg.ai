#!/usr/bin/env bash
set -euo pipefail
python -m cProfile -s cumulative -m src.main