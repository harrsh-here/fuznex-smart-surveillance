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
images, embeddings, databases, or credentials in Git or frontend payloads.
