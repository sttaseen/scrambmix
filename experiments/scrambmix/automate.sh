cd ../..

python mmaction2/tools/train.py experiments/scrambmix/configs/csn-50.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/scrambmix/configs/slowfast-wandb.py --validate --deterministic --seed 0
