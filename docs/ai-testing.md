# AI testing results and limitations

Run:

```powershell
python -m unittest discover -s tests -v
```

The integration tests use injected detector/recognizer doubles; no model weights,
camera, images, or biometric templates are required. The existing AI-lab webcam
workflow remains separate and was not modified by this integration preparation.

Limitations: no authorized identity-disjoint dataset is available in this repo,
so no FuzNex accuracy, FAR/FRR, unknown-rejection, low-light, or pose claim is
made. Real model loading needs externally supplied, verified weights and a
rights review.
