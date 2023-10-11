# Introduction

This experiment is used for training the model on the <strong> something-something-v2 </strong> dataset using data augmentation methods (**mixup,cutmix,scrambmix** ) and comparing to baseline. 
# Steps to follow

1. Download and set the  <strong> something-something-v2 </strong> data set by navigating to ```setup/something-something-v2``` and following the README.

3. For  different data augmentation methods, execute the following commands in sequence:

```
python mmaction2/tools/train.py experiments/something-something-v2/baseline.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/something-something-v2/scrambmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/something-something-v2/cutmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/something-something-v2/mixup.py --validate --deterministic --seed 0
```

# Result
The result should like this:

| Group     | val/top1_acc | val/top5_acc   | val/mean_class_accuracy |
|-----------|--------------|----------------|--------------------------|
| mixup     | 0.5349265383 | 0.8284051994   | 0.4626973437             |
| cutmix    | 0.5730014361 | 0.8562801488   | 0.5030298959             |
| scrambmix | 0.56707295   | 0.85083036     | 0.4987165364             |
| baseline  | 0.5661523732 | 0.8545126487   | 0.5004892057             |

