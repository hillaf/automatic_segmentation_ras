# Automatic visual segmentation of fish in Recirculated Aquaculture Systems using the Segment Anything Model


_Fred  H., Krogh M. A., Bang Jensen B., Ruotsalainen L., Vielma J., Pastell M._,
2025

[[Paper](https://doi.org/10.1016/j.aquaculture.2026.744459)][[Preprint](https://dx.doi.org/10.2139/ssrn.5281709)] [[Dataset](https://doi.org/10.5281/zenodo.15528511)]

This repository contains scripts to reproduce the results presented in _Automatic visual segmentation of fish in Recirculated Aquaculture Systems using the Segment Anything Model_.

The mask data can also be found on [Zenodo](https://doi.org/10.5281/zenodo.15528511).


## Requirements

Python >= 3.9.6

`pip install -r requirements.txt`

## Generate tables

`python3 src/evaluate_detection.py --tables`
