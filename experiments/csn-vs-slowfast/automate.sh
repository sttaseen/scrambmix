cd ../..

python mmaction2/tools/train.py experiments/setup/csn-50.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/setup/slowfast-wandb.py --validate --deterministic --seed 0
