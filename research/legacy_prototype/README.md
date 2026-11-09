# Legacy LBPH prototype

This copies the useful enrollment, training, and removal utilities from the
pre-migration prototype. It is research material only: the active FuzNex AI
interface remains `ai.pipeline.RecognitionSystem` and uses synthetic templates.

The scripts use repository-local `local_data/` rather than old project paths.
This directory, its images, model file, labels, alert captures, and any cascade
or model downloads are ignored by Git. Do not run these scripts with real faces
without the applicable consent, retention, and security controls.

Run modules explicitly after installing optional OpenCV and NumPy dependencies:

```bash
python -m research.legacy_prototype.lbph.enroll
python -m research.legacy_prototype.lbph.train
python -m research.legacy_prototype.lbph.remove_person
```

The live surveillance loop was not migrated because it was coupled to local
camera captures and alert images. The untouched original remains at
`C:\Users\DELL\Documents\ds\SurvillianceSystem`.
