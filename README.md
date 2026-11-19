# FuzNex Smart Surveillance System

## Portable AI integration

The portable AI workspace is under [`ai/`](ai/README.md). Its migration guide,
platform setup, Kaggle notebook, model-safety requirements, test command, and
known limits are documented in [`docs/portable-ai-migration.md`](docs/portable-ai-migration.md).

Shared development repository for the FYP **AI-Based Smart Surveillance System with Face Detection and Activity Monitoring**.

## Phase
Phase 1 — Research and Define the Problem (development of approved single-camera pilot underway).

## Ownership
| Folder | Lead | Responsibility |
|---|---|---|
| `ai/` | Harsh | Face detection, alignment, enrollment, recognition, matching |
| `backend/` | Govind | API, database selection, persistent storage, event/alert integration |
| `frontend/` | Gaurav | Dashboard and enrollment interface |
| `camera/` | Gopal | Camera capture, phone input and future multi-camera research |
| `docs/` | Dixit | Requirements, architecture and privacy documentation |
| `contracts/` | Shared | Versioned interfaces agreed before implementation |
| `tests/` | Shared | Mock integration and contract tests |

## Local quick start (no third-party packages)
```bash
python -m unittest discover -s tests -v
python scripts/demo_mock.py
```

## Windows diagnostic console

Launch `Launch-Private-SFace-Tester.cmd` (the existing desktop shortcut uses
this entry point). It opens the single **SPECTRA AI Diagnostic Console** rather
than a terminal menu. Select and verify local model files once in Diagnostics,
then use Live test, guided consented enrollment, People, and Logs & sessions in
the same window. Local model paths, gallery data, biometric templates and
exports remain under ignored `local_data/` or `diagnostics/` directories.

The old `ai.console` and `ai.enrollment` commands remain available for
headless/Kaggle and recovery workflows.

## Migration note

The prior local prototype was preserved in place. Its small LBPH enrollment,
training, and removal utilities are archived under `research/legacy_prototype/`;
they are not imported by the active AI module. Lightweight face-detector
evidence is in `research/face_detection/`. Raw face photos, alert captures,
datasets, downloaded weights, model files, virtual environments, caches,
databases, and secrets are excluded from Git.

## Integration approach
1. AI works against a mock identity-template provider and returns recognition results; backend owns persistent data.
2. Frontend develops against agreed mock API payloads.
3. Govind selects storage technology after reviewing requirements from AI and UI.
4. Integrate on a single demo machine first; do not require teammates' PCs to remain online.

## Security and data handling
Never commit real enrollment images, biometric embeddings, personal records, credentials, video recordings, local databases or model weights. Use consented evaluation data outside Git and document retention/deletion policies. Recognition status is not an authorization or suspicion decision.

## Git workflow
Create a **private** GitHub repository. Protect `main`; develop on `feature/<owner>-<task>` branches and merge via reviewed pull requests. Each contributor owns their module; coordinate any change to `contracts/`.

## Model status
SCRFD-500M-KPS selected as detector after custom evaluation. Face recognition model and decision thresholds remain to be validated. No model weights are bundled.
