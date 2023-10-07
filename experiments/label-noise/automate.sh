cd ../..

# python mmaction2/tools/train.py experiments/something-something-v2/baseline.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/something-something-v2/scrambmix.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/something-something-v2/cutmix.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/something-something-v2/mixup.py --validate --deterministic --seed 0




