cd ../..

# python mmaction2/tools/train.py experiments/wlasl/fframecutmix.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/wlasl/framecutmix.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/wlasl/baseline.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/wlasl/mixup.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/wlasl/cutmix.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/wlasl/scrambmix.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/wlasl/fframecutmix.py --validate --deterministic --seed 1
python mmaction2/tools/train.py experiments/wlasl/framecutmix.py --validate --deterministic --seed 1
python mmaction2/tools/train.py experiments/wlasl/baseline.py --validate --deterministic --seed 1
python mmaction2/tools/train.py experiments/wlasl/mixup.py --validate --deterministic --seed 1
python mmaction2/tools/train.py experiments/wlasl/cutmix.py --validate --deterministic --seed 1
python mmaction2/tools/train.py experiments/wlasl/scrambmix.py --validate --deterministic --seed 1


