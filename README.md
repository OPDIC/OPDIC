# OPDIC

This repository is for paper "Imputing Missing Values for Better Clustering over Incomplete Data".

## File Structure

* code: source code of algorithms. For MICE, HC, kPOD, CSDI, kCMM, GCID, MForest, GAIN, we use the open source implementations for them, i.e., 
  - MICE: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
  - HC: https://github.com/HoloClean/holoclean
  - CSDI: https://github.com/ermongroup/csdi
  - kPOD: https://github.com/iiradia/kPOD
  - kCMM: https://github.com/clarkdinh/k-cmm
  - GCID: https://github.com/ethan-yizhang/gmm-with-incomplete-data
  - MForest: https://hackage.haskell.org/package/MissingPy 
  - GAIN: https://github.com/jsyoon0823/GAIN 
* data: dataset source files of all seven public data collections used in experiments.

## Dataset

* Iris: http://archive.ics.uci.edu/ml/datasets/Iris
* Glass: http://archive.ics.uci.edu/ml/datasets/Glass+Identification
* Wine: http://archive.ics.uci.edu/ml/datasets/Wine
* DryBean: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset 
* UrbanGB: https://archive.ics.uci.edu/ml/datasets/urbangb
* Live: https://archive.ics.uci.edu/ml/datasets/Facebook+Live+Sellers+in+Thailand 
* SB: https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset
* Hepatitis: https://sci2s.ugr.es/keel/dataset.php?cod=100
* Market: https://sci2s.ugr.es/keel/dataset.php?cod=163
* The real incomplete dataset ECG needs an internal assessment in the company and is not available for download thus far.

## Dependencies
python 3.8
```
numpy==1.10.0
pandas==0.23.4
scikit-learn==0.24.1
gurobipy==9.5.0
```

## Instruction
``` sh
cd code
python main.py
```