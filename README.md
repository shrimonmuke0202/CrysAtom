# CrysAtom: Distributed Representation of Atoms for Crystal Property Prediction
# This work is accepted at the Third Learning on Graphs Conference (LoG 2024)
# We will make the necessary updates soon. Please stay tuned!
## Environment Setup
The dependency pakages can be installed using the command.
```python
pip install -r requirements.txt
```

## Dataset
A small subset of dataset present in the data directory. Upon acceptance we will resealse all datasets.
## Dense Vector Extraction
We will release our dense vector representation upon acceptance of this work. Please wait.
To get the dense vector representation please run the AtomVectorExtractor.py

## Downstream Property Prediction
After getting the 200 dimensional feature vector from the CrysAtom, run ```python3 train_jv.py``` for JARVIS and ```python3 train_mp.py``` for Materials Project.

## Citation  <a name="cite"></a>
Please cite our paper if it's helpful to you in your research.

```bibtext
@inproceedings{
mukherjee2024crysatom,
title={CrysAtom: Distributed Representation of Atoms for Crystal Property Prediction},
author={Shrimon Mukherjee and Madhusudan Ghosh and Partha Basuchowdhuri},
booktitle={The Third Learning on Graphs Conference},
year={2024},
url={https://openreview.net/forum?id=2AIVM5pWXz}
}
```
