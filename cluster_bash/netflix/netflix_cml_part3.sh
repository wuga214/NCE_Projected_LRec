#!/usr/bin/env bash
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/cml-part3.csv -y config/cml-part3.yml
