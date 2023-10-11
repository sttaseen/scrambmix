# Introduction

This experiment is used for training the model on the <strong> wlasl </strong> dataset with **reversed-scrambmix**. **Reversed-scrambmix** is also **scrambmix** but the labels were **reversed**.
For example, we have a label which is **<0.6 dog, 0.4 cat>**, after reversing, we have **<0.4 dog, 0.6 cat>**. And we use grid search to find the best alpha value.
# Steps to follow

1. Download and set the  <strong> wlasl </strong> data set by navigating to ```setup/wlasl``` and following the README.

3. For different alpha values, execute the following commands in sequence:

```
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=0.1.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=0.25.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=0.5.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=1.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=2.py --validate --deterministic --seed 0
```
```
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=4.py --validate --deterministic --seed 0
```

# Result
The result should like this:

| Alpha      | Top-1 (%)  | Top-5 (%)  | Mean Class (%) |
|------------|----------|----------|------------|
| 4    | 67.44    | 88.37    | 66.88      |
| 2    | 64.34    | 86.82    | 64.75      |
| 1    | 62.02    | 84.88    | 61.63      |
| 0.5  | 55.81    | 84.11    | 55.67      |
| 0.25 | 51.16    | 81.01    | 50.63      |
| 0.1  | 48.06    | 75.58    | 47.80      |
