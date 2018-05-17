#!/bin/bash

ranks=(50)
lambdas=(0.1)
alphas=(0 0.1 0.2 0.3)
path=/media/wuga/Experiments/Recsys-18/IMPLEMENTATION_Projected_LRec/data/movielens/

for rank in "${ranks[@]}"; do
    for lamb in "${lambdas[@]}"; do
        for alpha in "${alphas[@]}"; do
            python main.py -d "$path" -t Rtrain.npz -v Rvalid.npz -i 4 -a "$alpha" -r "$rank" -l "$lamb" -m "WPLRec" --disable-item-item
        done
    done
done
