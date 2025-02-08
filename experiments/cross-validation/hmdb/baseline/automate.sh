cd ../../../..

# python mmaction2/tools/train.py experiments/cross-validation/hmdb/scrambmix/fold-4.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/cross-validation/hmdb/scrambmix/fold-5.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/cross-validation/hmdb/fold-3.py --validate --deterministic --seed 0

# python mmaction2/tools/train.py experiments/cross-validation/hmdb/baseline/fold-1.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/cross-validation/hmdb/baseline/fold-2.py --validate --deterministic --seed 0
python mmaction2/tools/train.py experiments/cross-validation/hmdb/baseline/fold-3.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/cross-validation/hmdb/baseline/fold-4.py --validate --deterministic --seed 0
# python mmaction2/tools/train.py experiments/cross-validation/hmdb/baseline/fold-5.py --validate --deterministic --seed 0




