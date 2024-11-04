cd ../..

# Grid Search?
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=0.1.py --resume work_dirs/a\=0.1/latest.pth --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=0.25.py --resume work_dirs/a\=0.25/latest.pth --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=0.5.py --resume work_dirs/a\=0.5/latest.pth --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=1.py --resume work_dirs/a\=1/latest.pth --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=2.py --resume work_dirs/a\=2/latest.pth --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/alpha-wlasl-csn/alpha\=4.py --resume work_dirs/a\=4/latest.pth --validate --deterministic --seed 0





