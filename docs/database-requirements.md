# Database requirements for Govind — draft for team review

## From AI (Harsh)
- Stable unique person ID and active/revoked enrollment status.
- Multiple embeddings per person, with template ID, recognition model/version, creation timestamp and optional quality metadata.
- Fetch active templates; create, replace and delete enrollment templates; update AI's local gallery after changes.
- Prevent mixing incompatible model versions.
- Restricted access, audit trail, secure deletion and backups appropriate for biometric information.

## From UI (Gaurav) — confirm exact fields
- Person profile data and one profile photograph (storage reference, not necessarily DB binary).
- Enrollment/upload workflow and status/error messages.
- Searchable event history with timestamps, camera IDs and recognition statuses.
- Dashboard access permissions and appropriate redaction.

## Integration requirements
- Backend accepts enrollment images temporarily and forwards them to AI; stores returned templates only after successful validation.
- Backend retrieves profile information from `person_id` returned by AI.
- Backend stores events; AI returns inference results, not database transactions.
- First demo must work on one computer; no dependence on other members' laptops.
- Mock data and API examples must remain compatible with `contracts/`.

## Decision owned by Govind
Compare suitable storage options, choose database and photo-storage strategy, and document backup, access control, migration and retention design. Do not finalize before Gaurav's field requirements are reviewed.
