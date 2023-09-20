cd ../..

# GPU 1

python mmaction2/tools/train.py experiments/wlasl/mixup.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/wlasl/cutmix.py --validate --deterministic --seed 0


