# Baseline setup notes — DeCLIP and TruFor

Date: 2026-11-13

- DeCLIP official source cloned to `I:\icvgip workshop\baselines\DeCLIP`.
- Source: https://github.com/bit-ml/DeCLIP
- Commit: cb9de1077a3600653c7e896fe201535fb2b68c44
- Repository README says source code is CC BY-NC-SA 4.0 and references integrated third-party code; check each applicable license and model weight terms before distribution or commercial use.
- Official setup specifies Python 3.10.14, torch 2.2.2, torchvision 0.17.2; current system Python is 3.12.10. No global packages changed.
- The README's official COCO-SD model checkpoint link redirects to Google sign-in. No checkpoint downloaded; no DeCLIP inference or performance evaluation has been run.
- TruFor official README https://github.com/grip-unina/TruFor says downloading/using its files implicitly signifies agreement to license terms and limits use to nonprofit/informational purposes. Source and weights have not been downloaded by this setup.
- Next: decide baseline after reviewing applicable terms and checkpoint availability; use a separate, reproducible environment rather than changing global Python.

## VAAS alternative (diagnostic setup only)
- Project-local venv: `I:\icvgip workshop\baselines\vaas-env`, created with access to existing system site packages; global packages were not changed.
- MIT-licensed project and model card: https://github.com/OBA-Research/VAAS and https://huggingface.co/OBA-Research/vaas.
- `vaas==0.2.0`; adjusted local transformers from 5.22.2 to 4.57.6 after a PyTorch compatibility error. Existing system torch: 2.5.1+cu121.
- The model loader downloaded/located SegFormer config and checkpoint but failed before inference because transformers requires torch >=2.6 to load the pretrained PyTorch checkpoint securely. No model inference, scores or masks were produced.
- The venv inherits system packages and emitted dependency conflicts; it is not a clean reproducible environment. Next steps require an isolated torch >=2.6 environment and verification of checkpoint/dependency compatibility, or a different usable pretrained baseline. Do not bypass the torch security check.

## VAAS diagnostic completed — 13 November 2026
- Installed torch 2.6.0+cu124 and torchvision 0.21.0+cu124 inside vaas-env; system-wide torch was not changed. The environment inherits system packages and has reported unrelated dependency conflicts.
- Ran paper_work/run_vaas_diagnostic.py with OBA-Research/vaas on CUDA, using OpenSDI_train sd15 viewer row 0. The model loaded and returned three raw scores and a 224x224 map.
- Output: paper_work/diagnostic_01/result.json and anomaly_map.npy.
- This is a one-image execution check, not validation, an AI-edit probability, or a measured detection/localization performance. Viewer JPEG assets may be transcoded. VAAS is a general anomaly baseline, not proof of AI manipulation.
