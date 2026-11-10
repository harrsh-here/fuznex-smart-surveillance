# AI installation and configuration

Create a project-local virtual environment and install pinned development
dependencies:

```powershell
cd C:\Users\DELL\Documents\ds\fuznex-smart-surveillance
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ai\requirements.txt
```

Place only approved, verified artifacts outside Git, then configure absolute or
deployment-managed paths in `SFaceConfig`. Verify SHA-256 against
`ai/model-manifest.json` before loading. Do not copy lab enrollment files,
images, recordings, model weights, secrets, or databases into this repository.
