"""Proposed AI module interface. Real model integration is intentionally pending."""
from ai.mocks.template_store import load_active_templates

class RecognitionSystem:
    def __init__(self, template_provider=load_active_templates):
        self.template_provider = template_provider

    def load_templates(self):
        return self.template_provider()

    def enroll_person(self, images):
        """Future: validate 3–5 images and return embeddings, without saving photos."""
        raise NotImplementedError("Enrollment model and quality policy not integrated yet")

    def recognize_frame(self, frame, camera_id, timestamp):
        """Future: detect, align, embed, match, classify and return result contract."""
        raise NotImplementedError("Recognition model not integrated yet")
