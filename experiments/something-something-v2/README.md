# Introduction

This experiment is used for training the model on the <strong> something-something-v2 </strong> dataset using data augmentation methods (**mixup,cutmix,scrambmix,FloatFrameCutMix,FrameCutMix** ) and comparing to baseline. 
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

| Augment      | Top-1 (%) | Top-5 (%) | Mean Class (%) |
|----------- |------------|-----------|----------------|
| baseline           | 56.62      | 85.45     | 50.05          |
| mixup              | 53.49      | 82.84     | 46.27          |
| cutmix             | 57.30      | 85.63     | 50.30          |
| scrambmix          | 56.71      | 85.08     | 49.87          |
| FloatFrameCutMix   |           |           |                |
| FrameCutMix        |           |           |                |

