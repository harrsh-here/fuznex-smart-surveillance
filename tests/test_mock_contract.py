import unittest
from ai.types import FaceResult, FrameResult

class TestRecognitionContract(unittest.TestCase):
    def test_serialized_contract_avoids_probability_claims(self):
        result=FrameResult("sface-2021dec-experimental",(FaceResult((1,2,3,4),(1,2,2,2),7,.9,"person-1","known",.61,0,"OpenCV SFace LFW cosine reference",None),))
        payload=result.to_dict(); face=payload["faces"][0]
        self.assertEqual(face["track_id"],7); self.assertEqual(face["state"],"known")
        self.assertIn("match_score",face); self.assertNotIn("confidence",face)

if __name__ == "__main__": unittest.main()
