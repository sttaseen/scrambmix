cd ../..

# GPU 2

python mmaction2/tools/train.py experiments/wlasl/baseline.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/wlasl/scrambmix.py --validate --deterministic --seed 0


