$ErrorActionPreference = 'Stop'
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ai\requirements\yolo.txt
Write-Host 'Install approved model files outside Git, then configure explicit paths.'
