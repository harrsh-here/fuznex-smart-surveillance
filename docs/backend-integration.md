# Backend integration guide

1. Backend retrieves active templates for one supported `model_version` and
   supplies `{identity_id: normalized_embedding}` to `pipeline.set_templates`.
2. Camera layer sends each decoded OpenCV BGR frame to `process_frame(frame)`.
3. Backend consumes `FrameResult.to_dict()` using `contracts/recognition_result.example.json`.
4. Backend alone persists templates, maps identity IDs to people, controls alerts,
   and applies authorization/business decisions.

`known`, `unknown`, and `uncertain` are experimental match states. `unevaluable`
means a fresh frame was too small, blurry, or misaligned; it must not be coerced
to `unknown`. `match_score` is cosine similarity, not probability.
