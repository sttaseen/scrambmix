cd ../..

# Grid Search?
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=0.1.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=0.25.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=0.5.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=1.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=2.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-reversed-scrambmix/alpha\=4.py --validate --deterministic --seed 0





