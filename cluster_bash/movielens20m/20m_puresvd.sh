#!/usr/bin/env bash
cd IF-VAE-Recommendationcd ~/IF-VAE-Recommendationpython tune_parameters.py -d data/movielens20m/ -n movielens20m/puresvd.csv -y config/puresvd.yml
