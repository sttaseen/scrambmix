cd ../..

# Sadat

python mmaction2/tools/train.py experiments/hmdb51/cutmix.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/hmdb51/mixup.py --validate --deterministic --seed 0
