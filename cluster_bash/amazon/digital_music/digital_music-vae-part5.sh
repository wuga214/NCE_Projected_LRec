#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/NCE_Projected_LRec
python tune_parameters.py -d data/amazon/digital_music/ -n amazon/digital_music/vae-part5.csv -y config/vae-part5.yml
