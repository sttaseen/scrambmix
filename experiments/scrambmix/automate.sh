cd ../..

# Rented GPU
python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=5.py  --validate --deterministic --seed 0 

# python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=2.py --validate --deterministic --seed 0

# python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=0.2.py --validate --deterministic --seed 0

# python mmaction2/tools/train.py experiments/scrambmix/configs/alpha=1.py --validate --deterministic --seed 0
