# AI integration change summary and unresolved issues

- Replaced the mock-only AI scaffold with reusable detection, alignment,
  recognition, tracking, configuration, data types, and orchestration modules.
- Added backend-neutral contracts and mock-based smoke tests.
- Did not add model files, biometric data, camera code, persistence, alerts, or
  frontend changes.

Unresolved: SCRFD model provenance/license; SFace pretrained-weight commercial
and redistribution permissions; an authorized evaluation dataset; backend
template transport/authentication; and a usable Git worktree (the supplied
folder's Git metadata cannot currently be used by `git status`).

Recommended approved document copies: research reports in `02 — Individual Research`,
this architecture/model/backend documentation in `04 — Architecture & Technical Design`,
and test reports/checklists in `05 — Development & Testing`.
