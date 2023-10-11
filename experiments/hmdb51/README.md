# Introduction

This experiment is used for training the model on the <strong> hmdb </strong> dataset using data augmentation methods (**mixup,cutmix,scrambmix,FloatFrameCutMix,FrameCutMix** ) and comparing to baseline. 
# Steps to follow

1. Download and set the  <strong> hmdb </strong> data set by navigating to ```setup/hmdb``` and running ```bash setup.sh```.

3. For different data augmentation methods, execute the following commands in sequence:

```
python mmaction2/tools/train.py experiments/hmdb51/baseline.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/hmdb51/cutmix.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/hmdb51/baseline.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/hmdb51/scrambmix.py --validate --deterministic --seed 0
```

# Result
The result should like this:

| Augment            | Top-1 (%) | Top-5 (%) | Mean Class (%) |
|--------------------|-----------|-----------|----------------|
| Baseline           | 71.37     | 91.9      | 71.37          |
| MixUp              | 69.35     | 91.05     | 69.35          |
| CutMix             | 68.82     | 90.72     | 68.82          |
| ScrambMix          | 72.22     | 92.55     | 72.22          |
| FloatFrameCutMix   |           |           |                |
| FrameCutMix        |           |           |                |
