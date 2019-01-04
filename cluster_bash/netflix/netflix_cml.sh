#!/usr/bin/env bash
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/cml.csv -y config/cml.yml
