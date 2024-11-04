cd ../..

python mmaction2/tools/train.py experiments/csn-vs-slowfast/csn-50.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/csn-vs-slowfast/slowfast-wandb.py --validate --deterministic --seed 0
