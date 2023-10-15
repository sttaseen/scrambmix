# Introduction

This experiment is used for training the model on the <strong> autsl </strong> dataset using data augmentation methods (**mixup,cutmix,scrambmix,FloatFrameCutMix,FrameCutMix** ) and comparing to baseline. 
# Steps to follow

1. Download and set the  <strong> autsl </strong> data set by navigating to ```setup/autsl``` and following the README.

2. For different data augmentation methods, execute the following commands in sequence:

```
python mmaction2/tools/train.py experiments/autsl/fframecutmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/autsl/framecutmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/autsl/scrambmix.py --validate --deterministic --seed 0
```


# Result
The result should like this:

| Augment               | Top-1 (%) | Top-5 (%) | Mean Class (%) |
|-----------------------|-----------|-----------|----------------|
| Baseline              | 92.78     | 99.28     | -              |
| MixUp                 | 90.99     | 98.98     | -              |
| CutMix                | 94.89     | 99.43     | 94.75          |
| FrameCutMix           | 95.43     | 99.71     | 95.43          |
| FloatFrameCutMix      | 95.56     | 99.6      | 95.54          |
| ScrambMix-v1          | 93.66     | 99.4      | -              |
| ScrambMix-v2          | 94.28     | 99.33     | 94.24          |
| ScrambMix-v3 (a=0.2)  | 94.68     | 99.55     | 94.68          |
| scrambmix-a=2         | 94.95     | 99.57     | 94.97          |
| ScrambMix-v3 (a=5)    | 95.11     | 99.47     | 95.11          |

Running time: 1d 15h(each) with RTX 4090
