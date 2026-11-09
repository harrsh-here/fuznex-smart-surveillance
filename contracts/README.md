# Integration contracts (draft v0.1)

These are proposals for team review, **not a finalized backend API**.

## Enrollment
Input: `person_id` (assigned by backend) and 3–5 consented images. AI validates quality, aligns faces and returns embeddings with `model_version` and quality metadata. Backend persists embeddings and selected profile photo; temporary images are deleted after processing according to retention policy.

## Recognition
Input: a camera frame, camera ID, timestamp, and current enrolled template gallery. AI returns detected faces with bounding boxes, person ID or null, status (`known`, `unknown`, `uncertain`), similarity score or null, and model version.

## Template synchronization
Backend owns create/update/delete and active enrollment status. AI loads only active templates, with model-version checks. Do not assume face embeddings from different recognition models are compatible.

## Operational notes
- Match thresholds must be calibrated, not guessed.
- An unknown face is not inherently suspicious.
- Face recognition does not decide area authorization.
- Frame processing and template loading must work without remote network dependencies in the integrated demo.
