# Integration contracts (draft v0.2)

These are backend-neutral proposals for review, not a finalized API.

- `recognition_result.example.json` is the output of
  `SFacePipeline.process_frame(frame).to_dict()`.
- `template_gallery.example.json` describes the in-memory, active-template input
  supplied by the backend.

The backend owns people, template persistence, consent, authorization, alerts,
and camera transport. AI returns match states only: `known`, `unknown`,
`uncertain`, or `unevaluable`. `match_score` is cosine similarity, never a
probability. `unevaluable` must not be converted into unknown.

Only model-compatible normalized templates may be supplied. Do not put face
images, real embeddings, identity labels, or private gallery files in Git.

SPECTRA DIAG remains identity-disabled in its production contract. Its local
diagnostic mode is off by default and requires explicitly supplied local
SCRFD/SFace paths and a caller-owned, read-only gallery. Identity IDs,
candidates, embeddings, and templates are never emitted in diagnostic output.

`observation.example.json` is the provisional `1.1.0-draft` local observation
contract for independent consumers. `detection_trace` supports frame-level
duplicate-detection investigation and must not be used for identity inference.
images, embeddings, databases, or credentials in Git or frontend payloads.
