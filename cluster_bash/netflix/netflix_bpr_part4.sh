#!/usr/bin/env bash
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/bpr-part4.csv -y config/bpr-part4.yml
