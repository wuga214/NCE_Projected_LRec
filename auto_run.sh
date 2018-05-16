#!/bin/bash

ranks=(50 100)
lambdas=(0.1 1 10)
alphas=(1 5 10)
path=/media/wuga/Experiments/Recsys-18/IMPLEMENTATION_Projected_LRec/data/movielens/

for rank in "${ranks[@]}"; do
    for lamb in "${lambdas[@]}"; do
        for alpha in "${alphas[@]}"; do
            python main.py -d "$path" -t Rtrain.npz -v Rvalid.npz -i 4 -a "$alpha" -r "$rank" -l "$lamb" -m "WPLRec" --disable-item-item
        done
    done
done