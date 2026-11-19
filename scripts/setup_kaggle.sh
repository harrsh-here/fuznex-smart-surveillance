#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -r ai/requirements/yolo.txt
echo 'Use only approved model paths downloaded from ai/model-manifest.json.'
