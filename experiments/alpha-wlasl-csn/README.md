# Introduction

The files in this folder are used for training the <strong> CSN </strong> model on the <strong> wlasl </strong> dataset without data augmentation and searching for the optimal alpha value.

# Steps to follow

1. Download and set the <strong> wlasl </strong> data set by navigating to ```setup/wlasl``` and following the README.
2. For the <strong> baseline </strong> and different <strong> alpha </strong> values, execute the following commands in sequence:

```
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=0.1.py --resume work_dirs/a\=0.1/latest.pth --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=0.25.py --resume work_dirs/a\=0.25/latest.pth --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=0.5.py --resume work_dirs/a\=0.5/latest.pth --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=1.py --resume work_dirs/a\=1/latest.pth --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=2.py --resume work_dirs/a\=2/latest.pth --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=4.py --resume work_dirs/a\=4/latest.pth --validate --deterministic --seed 0
```

# Result

The result should like this:

| Alpha | Top-1 (%) | Top-5 (%) | Mean Class (%) |
|-------|-----------|-----------|----------------|
| 0.1   | 0.8295    | 0.9341    | 0.8317         |
| 0.25  | 0.8333    | 0.9612    | 0.8325         |
| 0.5   | 0.8333    | 0.9419    | 0.8338         |
| 1     | 0.8217    | 0.9496    | 0.8213         |
| 2     | 0.8372    | 0.9457    | 0.8342         |
| 4     | 0.8217    | 0.9535    | 0.8208         |

Running time : 2h 56m(each) with RTX 4090
