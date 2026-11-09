# Contributing
1. Work on a feature branch (`feature/harsh-recognition`, `feature/govind-storage`, etc.).
2. Do not commit secrets, raw biometric data, photos, embeddings, video or downloaded weights.
3. Run `python -m unittest discover -s tests -v` before opening a pull request.
4. Include a short summary, test results and any changes to data contracts.
5. Obtain agreement from affected module owners before modifying `contracts/`.
6. Keep backend database technology independent of the AI module; use the shared contract.
