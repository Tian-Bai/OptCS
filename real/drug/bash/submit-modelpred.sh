#!/bin/bash 

#now run normal batch commands

ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scikit-learn/1.5.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.4.1_py312

python3 modelpred.py --data $1 --model $2 --conformal_score $3 --seed $4 --alpha $5 --epoch 50 