# AI module — Harsh

Owns: SCRFD face detection adapter, face quality checks, alignment, embeddings, enrollment validation, template matching, calibrated `known/unknown/uncertain` decisions and CPU evaluation.

Does **not** own: database persistence, main backend, UI, alerts or camera transport.

Current scaffold contains a mock template provider for interface development only. It does not perform face recognition or infer identities. Keep model weights outside Git.

The archived LBPH prototype is isolated in `research/legacy_prototype/` and is
never imported from this package. Keep production integrations behind
`RecognitionSystem`; do not import code that loads local biometric data, camera
devices, or model weights at module-import time.
