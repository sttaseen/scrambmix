cd ../..

# python mmaction2/tools/train.py experiments/hmdb51/baseline.py --validate --deterministic --seed 1
# python mmaction2/tools/train.py experiments/hmdb51/fframecutmix.py --validate --deterministic --seed 1
# python mmaction2/tools/train.py experiments/hmdb51/framecutmix.py --validate --deterministic --seed 1
# python mmaction2/tools/train.py experiments/hmdb51/cutmix.py --validate --deterministic --seed 1
# python mmaction2/tools/train.py experiments/hmdb51/mixup.py --validate --deterministic --seed 1
python mmaction2/tools/train.py experiments/hmdb51/scrambmix.py --validate --deterministic --seed 1

# python mmaction2/tools/train.py experiments/hmdb51/baseline.py --validate --deterministic --seed 2
# python mmaction2/tools/train.py experiments/hmdb51/fframecutmix.py --validate --deterministic --seed 2
# python mmaction2/tools/train.py experiments/hmdb51/framecutmix.py --validate --deterministic --seed 2
# python mmaction2/tools/train.py experiments/hmdb51/cutmix.py --validate --deterministic --seed 2
# python mmaction2/tools/train.py experiments/hmdb51/mixup.py --validate --deterministic --seed 2
# python mmaction2/tools/train.py experiments/hmdb51/scrambmix.py --validate --deterministic --seed 2

