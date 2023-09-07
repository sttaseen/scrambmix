cd ../..

# python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=1.py --validate --deterministic --seed 0

# python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=0.1.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=0.5-1.py --validate --deterministic --seed 0

# TODO: Do these later.
"""
python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=2.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=0.1.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=1.py --validate --deterministic --seed 0
"""