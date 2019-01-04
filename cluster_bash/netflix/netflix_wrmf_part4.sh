#!/usr/bin/env bash
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/wrmf-part4.csv -y config/wrmf-part4.yml