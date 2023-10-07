cd ../..

python mmaction2/tools/train.py experiments/label-noise/baseline.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/label-noise/mixup.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/label-noise/scrambmix.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/label-noise/cutmix.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/label-noise/framecutmix.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/label-noise/floatframecutmix.py --validate --deterministic --seed 0



