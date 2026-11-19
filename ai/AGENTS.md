# AI directory instructions

Before changing code or contracts in this directory, read
[`INTEGRATION_HANDOFF.md`](INTEGRATION_HANDOFF.md), [`README.md`](README.md),
and the relevant tests. Treat `ai/combined.py` as the authoritative source for
the current `1.1.0-draft` observation output.

- Do not add identity publication, face-to-person association, tracking
  publication, event/alert semantics, distance, or camera-health fields to the
  combined contract without a reviewed versioned contract and tests.
- Keep models, frames, recordings, credentials, galleries, embeddings, and
  local databases out of Git and frontend payloads.
- Preserve the separation between local diagnostic/enrollment tools and the
  identity-disabled combined observation path.
- Keep application changes and documentation changes separately reviewable;
  run the applicable tests before claiming contract behavior.
