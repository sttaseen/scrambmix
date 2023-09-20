cd ../..

# Qi Xin

python mmaction2/tools/train.py experiments/hmdb51/baseline.py --validate --deterministic --seed 0

python mmaction2/tools/train.py experiments/hmdb51/scrambmix.py --validate --deterministic --seed 0


