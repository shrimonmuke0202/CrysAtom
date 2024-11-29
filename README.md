# CrysAtom: Distributed Representation of Atoms for Crystal Property Prediction
# This work is accepted at the Third Learning on Graphs Conference (LoG 2024)
# We will make the necessary updates soon. Please stay tuned!
## Environment Setup
The dependency pakages can be installed using the command.
```python
pip install -r requirements.txt
```

## Dataset
A small subset of dataset present in the data directory. To get the datasets used for creating dense vector is present in [[Zenodo](https://zenodo.org/records/14242239?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjA2NzVlMjVmLWM1ZWEtNDk4NC04ZmM5LTFkMWMzNjg0ZTRjMSIsImRhdGEiOnt9LCJyYW5kb20iOiIzYjdkNTFhMzFkYjc1ZWU3N2M2NGIzMmE4YWFiYjNlOSJ9.zG80nRXipzEdJ9MypJe2toA5UcvOmsZ1svlaKk-5qHtN937iHdOlKU1WeIBslatZFgXvCcNb7NrGRFIgPFXhBQ)]
## Dense Vector Extraction
Our 200-dimensional vector is located in the dense_vector directory. To use this dense vector representation in your model, please adjust the input dimension of our GNN model accordingly. If you use our vector, kindly cite our paper.
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
