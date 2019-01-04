#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/yahoo/ -n yahoo/plrec.csv -y config/plrec.yml
