# Open Stethoscope 🩺

**Open-source AI heart murmur detection for primary care.**

Cardiovascular disease is the leading cause of death in China. In county and township health centers, general practitioners often lack the training to interpret heart sounds. Open Stethoscope brings cardiology-grade auscultation to every village doctor — for free, open source, and on-device.

## Why

- CVD is the #1 cause of death in China and worldwide
- ~1.2 million heart sound recordings are needed at grassroots level each year, but there aren't enough cardiologists to read them
- Digital stethoscopes + AI can screen for heart murmurs (valvular disease) with accuracy approaching specialist auscultation
- Existing solutions are closed-source and enterprise-priced — this project is open and runs on edge hardware

## What

A deep learning system for heart murmur detection from phonocardiogram (PCG) recordings, built on the [CirCor DigiScope dataset](https://physionet.org/content/circor-heart-sound/) (PhysioNet Challenge 2022):

- **5,272 recordings** from **1,568 patients**, expert-annotated for murmurs
- Lightweight models designed to run on low-cost edge devices (Tesla P4 / CPU)
- Reproducible baseline reproducing the PhysioNet 2022 Challenge setup

## Roadmap

- [x] Reproduce PhysioNet 2022 Challenge baseline
- [ ] Train lightweight murmur classifier
- [ ] On-device inference demo (edge deployment)
- [ ] Chinese primary-care deployment guide

## Dataset

CirCor DigiScope v1.0.3 — open access, no application required:

```bash
wget https://physionet.org/static/published-projects/circor-heart-sound/circor-heart-sound-1.0.3.zip
```

## License

MIT (code). Dataset: Open Data Commons Attribution License v1.0 (see [PhysioNet](https://physionet.org/content/circor-heart-sound/)).
