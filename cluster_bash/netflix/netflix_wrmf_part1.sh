#!/usr/bin/env bash
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/wrmf-part1.csv -y config/wrmf-part1.yml