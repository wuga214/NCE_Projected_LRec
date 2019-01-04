#!/usr/bin/env bash
cd ~/IF-VAE-Recommendation
python tune_parameters.py -d data/netflix/ -n netflix/vae-part3.csv -y config/vae-part3.yml
