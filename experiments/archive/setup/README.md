## Experiment Setup

This is just a quick way to make sure that everyone is correctly set up so that we can run experiments with MMAction2.

## Steps to follow:
1. Setup the conda environment and the packages. Follow the [setup steps](https://github.com/sttaseen/scrambmix#setup) on the main page.

2. Download the WLASL100 dataset by using the script I made. This can be done by navigating to ```setup/wlasl/``` and running ```bash setup.sh```.

**Note:** Make sure you have a [kaggle key](https://www.kaggle.com/docs/api#:~:text=is%20%24PYTHON_HOME/Scripts.-,Authentication,-In%20order%20to) set up. 

3. Run the training script by running the following from the root directory of the repository:

```
python mmaction2/tools/train.py experiments/setup/slowfast-wandb.py
```

If there is a problem with wandb, you can run the following:

```
python mmaction2/tools/train.py experiments/setup/slowfast-base.py
```

Sorry for the poor formatting of this README. Use the main page whenever possible or contact me if you need help.

Good luck!
