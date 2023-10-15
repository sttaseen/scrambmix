# Introduction

This experiment is used for training the model on the <strong> hmdb </strong> dataset using data augmentation methods (**mixup,cutmix,scrambmix,FloatFrameCutMix,FrameCutMix** ) with artificially noisy and comparing to baseline. 
# Steps to follow

1. Download and set the  <strong> hmdb </strong> data set by navigating to ```setup/hmdb``` and running ```bash setup.sh```.

3. For different data augmentation methods, execute the following commands in sequence:

```
python mmaction2/tools/train.py experiments/label-noise/baseline.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/label-noise/mixup.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/label-noise/scrambmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/label-noise/cutmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/label-noise/framecutmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/label-noise/floatframecutmix.py --validate --deterministic --seed 0
```

# Result
The result should like this:

| Augment       | Top-1 (%)     | Top-5 (%)    | Mean Class (%) |
|---------------|--------------|--------------|--------------------------|
| Baseline      | 57.56       | 91.30      | 56.70                   |
| Cutmix        | 66.57       | 94.97      | 66.13                   |
| Mixup         | 57.81       | 90.99       | 57.01                   |
| FFrameCutmix  | 61.96       | 93.78       | 61.54                   |
| FrameCutmix   | 61.79       | 93.92       | 61.48                   |
| Scrambmix     | 67.03       | 95.63       | 66.04                   |

Running time : 6h 20m (each) with RTX 4090
