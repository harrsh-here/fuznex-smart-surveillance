# Architecture — draft v0.1

Camera frame provider → AI recognition module → structured inference result → backend event/profile orchestration → frontend.

Enrollment: frontend → backend → AI quality/embedding pipeline → backend persists templates and one profile photo.

The AI module loads active templates through a provider abstraction and matches in memory. During remote development, each module uses mocks. During integration, code runs on one demo machine, and Govind's backend replaces the mock template provider.

Person tracking/body Re-ID are later research branches, not prerequisites for the single-camera identity pilot. Unknown identity is not evidence of suspicious activity.
