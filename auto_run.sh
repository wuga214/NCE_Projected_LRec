#!/bin/bash

ranks=(100 200 300 500)
lambdas=(0.1 1 10)

for rank in "${ranks[@]}"; do
    for lamb in "${lambdas[@]}"; do
        python main.py -r "$rank" -l "$lamb" -i 7
    done
done