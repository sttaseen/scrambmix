# Introduction

This experiment is used for training the <strong> CSN </strong> model on the <strong> hmdb </strong> dataset without data augmentation and searching for the optimal alpha value. 
# Steps to follow

1. Download and set the <strong> hmdb </strong> data set by navigating to ```setup/hmdb``` and running ```bash setup.sh```
2. For the <strong> baseline </strong> and different <strong> alpha </strong> values, execute the following commands in sequence:

```
bash automate.sh
```
```
bash automate1.sh
```
```
bash automate2.sh
```
# Result
The result should like this:

Model	<strong> CSN-50 </strong>	
Baseline	<strong> Top-1(%) : </strong> 76.14	  <strong> Top-5 (%) : </strong> 93.66

Grid Search result:

| Alpha | Top-1 (%)    | Top-5 (%)    | Mean Class (%) |
|-------|--------------|--------------|----------------|
| 0.1   | 0.76601307   | 0.94640523   | 0.76601307     |
| 0.25  | 0.75947712   | 0.93986928   | 0.75947712     |
| 0.5   | 0.75686275   | 0.93954248   | 0.75686275     |
| 1     | 0.75882353   | 0.9379085    | 0.75882353     |
| 2     | 0.76013072   | 0.9379085    | 0.76013072     |
| 4     | 0.75620915   | 0.94444444   | 0.75620915     |

