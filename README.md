# CrysATOM

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
After getting the 200 dimensional feature vector from the CrysAtom, run ```python3 train_jv.py``` for JARVIS and ```python3 train_mp.py``` for Materials Project. We take the Matformer original code and slightly modified it to input our dense representation.
